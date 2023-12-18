from dataset import cifar
from mosaic_core import engine
from torch.cuda.amp import autocast, GradScaler
from mosaic_core import registry
import numpy as np
import config
import torch
import os
import logging
import torch.nn as nn
from utils.utils import DataIter
from torch.utils.data import DataLoader
import torch.optim as optim
import utils.utils as utils
import copy
import random
from tqdm import tqdm
from torchvision import transforms



class OneTeacher:
    def __init__(self, student, teacher, generator, discriminator, args, writer):
        self.writer = writer
        self.args = args
        self.val_loader, self.priv_data, self.local_datanum, self.local_cls_datanum = self.gen_dataset(
            args)
        self.local_datanum = torch.FloatTensor(self.local_datanum).cuda()
        self.local_cls_datanum = torch.FloatTensor(
            self.local_cls_datanum).cuda()
        self.netG = generator
        self.normalizer = utils.Normalizer(args.dataset)
        self.netDS = utils.copy_parties(config.N_PARTIES, discriminator)
        # teacher/student
        self.netS = student
        self.init_netTS(teacher)

    def gen_dataset(self, args):
        num_classes, ori_training_dataset, val_dataset = registry.get_dataset(name=args.dataset,
                                                                              data_root=args.data_root)
        _, train_dataset, _ = registry.get_dataset(
                                name=args.unlabeled, data_root=args.data_root)
        # _, ood_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # see Appendix Sec 2, ood data is also used for training
        # ood_dataset.transforms = ood_dataset.transform = train_dataset.transform # w/o augmentation
        train_dataset.transforms = train_dataset.transform = val_dataset.transform  # w/ augmentation
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config.FED_BATCHSIZE, 
            num_workers=config.NUM_WORKERS)
        # return train_dataset,ood_dataset,num_classes,ori_training_dataset,val_dataset
        local_datanum = np.zeros(config.N_PARTIES)
        local_cls_datanum = np.zeros((config.N_PARTIES, args.N_class))
        for localid in range(config.N_PARTIES):
            # count
            local_datanum[localid] = 50000
            # class specific count
            for cls in range(args.N_class):
                local_cls_datanum[localid, cls] = 5000

        return val_loader, [train_dataset], local_datanum, local_cls_datanum

    def init_netTS(self, teacher):
        epochs = config.INIT_EPOCHS
        if self.args.from_teacher_ckpt == '':
            self.netTS = utils.copy_parties(
                config.N_PARTIES, teacher)  # can be different
        else:
            self.netTS = []
            ckpts = os.listdir(self.args.from_teacher_ckpt)
            for n in range(config.N_PARTIES):
                teacher = registry.get_model(
                    self.args.teacher, num_classes=self.args.N_class, pretrained=True).eval()
                teacher = teacher.cuda(self.args.gpu)
                cur_teacher = ''
                for ckpt in ckpts:
                    if str(n) == ckpt.split('.')[0]:
                        cur_teacher = os.path.join(self.args.from_teacher_ckpt, ckpt)
                        utils.load_dict(cur_teacher, teacher)
                        self.netTS.append(teacher)
                        break
                print(cur_teacher)
                
                
        ckpt_dir = f'{config.LOCAL_CKPTPATH}/{self.args.dataset}/a{self.args.alpha}+sd{self.args.seed}+e{epochs}+b{config.BATCHSIZE}'
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

    def init_training(self):
        self.local_dataloader = []
        self.local_ood_dataloader = []
        for n in range(config.N_PARTIES):
            tr_dataset = self.priv_data[n]
            local_loader = DataLoader(
                dataset=tr_dataset,
                batch_size=config.FED_BATCHSIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                sampler=None)
            self.local_dataloader.append(DataIter(local_loader))
        # FIXME! only limited data used
        # TODO min -> max
        self.steps_per_disc = [len(local_loader.dataloader) for local_loader in self.local_dataloader]
        self.cur_step_per_disc = [0 for _ in range(len(self.local_dataloader))]
        if self.args.use_maxIters:
            self.iters_per_round = max(
                [len(local_loader.dataloader) for local_loader in self.local_dataloader])
        else:
            self.iters_per_round = min(
                [len(local_loader.dataloader) for local_loader in self.local_dataloader])
        # print(f'alpha:{self.args.alpha} ===iters_per_round:{self.iters_per_round}')
        # with open('record.txt','a') as f:
            # f.write(f'===iters_per_round:{self.iters_per_round}, data_per_local : {[len(local_loader.dataloader) for local_loader in self.local_dataloader]}\n')
        
        steps_all = self.args.epochs * self.iters_per_round
        # init optim
        if config.OPTIMIZER == 'SGD':
            self.optim_s = optim.SGD(
                self.netS.parameters(), lr=config.STU_LR, momentum=0.9, weight_decay=config.STU_WD)
        else:
            self.optim_s = optim.Adam(
                self.netS.parameters(), lr=config.STU_LR, betas=(0.9, 0.999), weight_decay=config.STU_WD)

        if not self.args.fixed_lr:
            self.sched_s = optim.lr_scheduler.CosineAnnealingLR(
                self.optim_s, steps_all,eta_min=config.STU_LR_MIN)  # TODO, ,
        # for gen
        if self.args.use_pretrained_generator:
            loc = 'cuda:{}'.format(self.args.gpu)
            checkpoint = torch.load(self.args.use_pretrained_generator, 
                                    map_location=loc)
            self.netG.load_state_dict(checkpoint)
            
        self.optim_g = torch.optim.Adam(
            self.netG.parameters(), lr=config.GEN_LR, betas=[0.5, 0.999])

        if not self.args.fixed_lr:
            self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_g, T_max=steps_all)
        param_ds = []
        for n in range(config.N_PARTIES):
            param_ds += list(self.netDS[n].parameters())

        self.optim_d = torch.optim.Adam(
            param_ds, lr=config.DIS_LR, betas=[0.5, 0.999])


        if not self.args.fixed_lr:
            self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_d, T_max=steps_all)
        # TODO: not all complete steps_all if local_percetnt<1

        # criterion
        self.criterion_distill = nn.L1Loss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.criterion_bce = nn.functional.binary_cross_entropy_with_logits

        # netS training records
        self.bestacc = 0
        self.best_statdict = copy.deepcopy(self.netS.state_dict())
        # save path
        self.savedir = f'{config.CKPTPATH}/{self.args.logfile}'
        self.savedir_gen = f'{config.CKPTPATH}/{self.args.logfile}/gen'
        if not os.path.isdir(self.savedir_gen):
            os.makedirs(self.savedir_gen)

    def update(self):
        self.init_training()
        
        self.global_step = 0
        # default: local_percent =1
        selectN = list(range(0, config.N_PARTIES))
        self.localweight = self.local_datanum / \
            self.local_datanum.sum()  # nlocal*nclass
        self.localclsweight = self.local_cls_datanum / \
            self.local_cls_datanum.sum(dim=0)  # nlocal*nclass
        # resume trainning if args.resume exists

        self.resume_trainning()
        # use fp16 to accelerate computations
        if self.args.fp16:
            self.args.scaler_s = GradScaler() if self.args.fp16 else None
            self.args.scaler_g = GradScaler() if self.args.fp16 else None
            self.args.scaler_d = GradScaler() if self.args.fp16 else None
            self.args.autocast = autocast
        else:
            self.args.autocast = engine.utils.dummy_ctx


        for round in range(self.args.start_epoch, self.args.epochs):
            if config.LOCAL_PERCENT < 1:
                selectN = random.sample(selectN, int(config.LOCAL_PERCENT*config.N_PARTIES))

                countN = self.local_datanum[selectN]
                self.localweight = countN/countN.sum()  # nlocal
                countN = self.local_cls_datanum[selectN]
                self.localclsweight = countN/countN.sum(dim=0)  # nlocal*nclass
            logging.info(f'************Start Round {round} -->> {self.args.epochs}***************')
            self.current_round = round
            self.update_round(round, selectN)


        # save G,D in the end
        torch.save(self.netG.state_dict(), f'{self.savedir_gen}/generator.pt')
        for n in range(config.N_PARTIES):
            torch.save(self.netDS[n].state_dict(),
                       f'{self.savedir_gen}/discrim{n}.pt')
        ############################# exiting ####################################

    def resume_trainning(self):
        ############################################
        # Resume Train from checkpoints
        ############################################
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                if self.args.gpu is None:
                    checkpoint = torch.load(self.args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.args.gpu)
                    checkpoint = torch.load(self.args.resume, map_location=loc)

                try:
                    self.netS.module.load_state_dict(
                        checkpoint['s_state_dict'])
                    self.netG.module.load_state_dict(
                        checkpoint['g_state_dict'])
                    for n in range(config.N_PARTIES):
                        self.netDS[n].module.load_state_dict(
                            checkpoint['ds_state_dict'][n])
                except:
                    self.netS.load_state_dict(checkpoint['s_state_dict'])
                    self.netG.load_state_dict(checkpoint['g_state_dict'])
                    for n in range(config.N_PARTIES):
                        self.netDS[n].load_state_dict(
                            checkpoint['ds_state_dict'][n])
                best_acc1 = checkpoint['best_acc']
                try:
                    self.args.start_epoch = checkpoint['epoch']
                    self.bestacc = checkpoint['best_acc']
                    
                    self.optim_g.load_state_dict(checkpoint['optim_g'])
                    # self.sched_g.load_state_dict(checkpoint['sched_g'])
                    self.optim_s.load_state_dict(checkpoint['optim_s'])
                    # self.sched_s.load_state_dict(checkpoint['sched_s'])
                    # self.sched_d.load_state_dict(checkpoint['sched_d'])
                    self.optim_d.load_state_dict(checkpoint['optim_d'])

                    if self.args.modify_optim_lr:
                        for param_s in self.optim_s.param_groups:
                            print('ori_optim_s LR:', param_s['lr'])
                            param_s['lr'] = config.STU_LR
                            print('new_optim_s LR:', param_s['lr'])
                        for param_g in self.optim_g.param_groups:
                            print('ori_optim_g LR:', param_g['lr'])
                            param_g['lr'] = config.GEN_LR
                            print('new_optim_g LR:', param_g['lr'])
                        for param_d in self.optim_d.param_groups:
                            print('ori_optim_d LR:', param_d['lr'])
                            param_d['lr'] = config.DIS_LR
                            print('new_optim_d LR:', param_d['lr'])
                    self.optim_s.param_groups[0]['capturable'] = True
                    self.optim_g.param_groups[0]['capturable'] = True
                    self.optim_d.param_groups[0]['capturable'] = True
                except:
                    print("Fails to load additional model information")
                print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                      .format(self.args.resume, checkpoint['epoch'], best_acc1))
            else:
                print("[!] no checkpoint found at '{}'".format(self.args.resume))

    def change_status_to_train(self):
        self.netG.train()
        for n in range(config.N_PARTIES):
            self.netDS[n].train()
            self.netTS[n].eval()
        self.netS.train()

    def update_round(self, roundd, selectN):
        #
        self.change_status_to_train()
        bestacc_round = self.bestacc
        self.cur_step_per_disc = [0 for _ in range(len(self.local_dataloader))]
        # iters_per_round = len(self.local_dataloader[].dataset)
        for iter in tqdm(range(self.iters_per_round)):
            # 1. update D,G
            z = torch.randn(
                size=(config.FED_BATCHSIZE, config.GEN_Z_DIM)).cuda()
            
            syn_img = self.netG(z)
            syn_img = self.normalizer(syn_img)
            self.update_netDS_batch(syn_img, selectN)
            self.update_netG_batch(syn_img, selectN)
            # 2. Distill, update S
            self.update_netS_batch(selectN)
            if not self.args.fixed_lr:
                self.sched_d.step()
                self.sched_g.step()
                self.sched_s.step()
            # val
        
        
        acc = validate(self.optim_s.param_groups[0]['lr'], self.val_loader,
                       self.netS, self.criterion, self.args,
                       roundd)
        self.global_step += 1
        is_best = False
        if acc > bestacc_round:
            logging.info(f'Iter{iter}, best for now:{acc}')
            self.best_statdict = copy.deepcopy(self.netS.state_dict())
            bestacc_round = acc
            is_best = True
            # if iter % config.PRINT_FREQ == 0:
            #     logging.info(f'===R{roundd}, {iter}/{self.iters_per_round}, acc{acc}, best{self.bestacc}')

        # reload G,D?
        # self.netS.load_state_dict(self.best_statdict, strict=True)
        logging.info(
            f'=============Round{roundd}, BestAcc originate: {(self.bestacc):.2f}, to{(bestacc_round):.2f}====================')

        if bestacc_round > self.bestacc:
            print(f'selectN:{selectN}')
            savename = os.path.join(
                self.savedir, f'r{roundd}_{(bestacc_round):.2f}.pt')
            torch.save(self.best_statdict, savename)
            self.bestacc = bestacc_round

        if self.writer is not None:
            self.writer.add_scalar('BestACC', self.bestacc, roundd)

        checkpoints = {
            "epoch": roundd + 1,
            "arch": self.args.student,
            "s_state_dict": self.best_statdict,
            "g_state_dict": self.netG.state_dict(),
            "ds_state_dict": [self.netDS[i].state_dict() for i in range(config.N_PARTIES)],
            "best_acc": float(bestacc_round),
            "optim_s": self.optim_s.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            # "sched_d": self.sched_d.state_dict(),
            # "sched_s": self.sched_s.state_dict(),
            # "sched_g": self.sched_g.state_dict()
        }
        save_checkpoint(checkpoints,
                        is_best,
                        self.args.ckpt_path,
                        filename=f'E{roundd}-Dbs{config.FED_BATCHSIZE}-Ts{config.N_PARTIES}-ACC{round(float(self.bestacc), 2)}.pth')
        

    def update_netDS_batch(self, syn_img, selectN):
        loss = 0.
        with self.args.autocast():
            cnt_disc = 0
            for localid in selectN:
                if self.cur_step_per_disc[localid] >= self.steps_per_disc[localid]:
                    continue
                cnt_disc += 1
                
                d_out_fake = self.netDS[localid](syn_img.detach())
                real_img = self.local_dataloader[localid].next()[0].cuda()  # list [img, label, ?]
                self.cur_step_per_disc[localid] += 1
                d_out_real = self.netDS[localid](real_img.detach())

                loss_d = (self.criterion_bce(d_out_fake, torch.zeros_like(d_out_fake), reduction='sum') +
                          self.criterion_bce(d_out_real, torch.ones_like(d_out_real), reduction='sum')) / (
                    2 * len(d_out_fake))

                # print(f'loss: {loss}, loss_d: {loss_d}')
                loss += loss_d
            loss *= self.args.w_disc/cnt_disc
            # print(f'----loss{loss}----')

        self.optim_d.zero_grad()
        if self.args.fp16:
            scaler_d = self.args.scaler_d
            scaler_d.scale(loss).backward()
            scaler_d.step(self.optim_d)
            scaler_d.update()
        else:
            loss.backward()
            self.optim_d.step()
        #
        # loss.backward()
        # self.optim_d.step()
        # print(f'loss_item{loss.item()}')
        if self.writer is not None:
            self.writer.add_scalar('LossDiscriminator',
                                   loss.item(), self.global_step)

    def update_netG_batch(self, syn_img, selectN):
        # 1. gan loss
        loss_gan = []
        with self.args.autocast():
            for localid in selectN:
                d_out_fake = self.netDS[localid](syn_img)  # gradients in syn
                loss_gan.append(self.criterion_bce(d_out_fake, torch.ones_like(d_out_fake),
                                                   reduction='sum') / len(d_out_fake))
            # import pdb; pdb.set_trace()            
            if self.args.is_emsember_generator_GAN_loss == "y":
                loss_gan = self.ensemble_locals(torch.stack(loss_gan))
            else:
                loss_gan = torch.sum(torch.stack(loss_gan))

            # 2. adversarial distill loss
            logits_T = self.forward_teacher_outs(syn_img, selectN)
            
            ensemble_logits_T = self.ensemble_locals(logits_T)
            logits_S = self.netS(syn_img)
            
            if self.args.use_l1_loss:
                loss_adv = - self.criterion_distill(logits_S, ensemble_logits_T) #(bs, ncls) L1 loss
            else: 
                loss_adv = -engine.criterions.kldiv(logits_S,ensemble_logits_T, T=self.args.T)  
            
            if self.args.use_jsdiv:
                t_outs = [self.netTS[i](syn_img) for i in selectN]
                loss_js = engine.criterions.yxdiv(t_outs)
                # import pdb;pdb.set_trace()
            else:
                loss_js = 0.
            # 3.regularization for each t_out (not ensembled) #TO DISCUSS
            loss_align = []
            loss_balance = []
            for n in range(len(selectN)):
                t_out = logits_T[n]
                pyx = torch.nn.functional.softmax(t_out, dim=1)  # p(y|G(z))
                log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
                py = pyx.mean(0)
                # To generate distinguishable imgs
                loss_align.append(-(pyx * log_softmax_pyx).sum(1).mean())
                # Alleviating Mode Collapse for unconditional GAN
                loss_balance.append((py * torch.log2(py)).sum())

            if self.args.gen_loss_avg:
                loss_align = sum(loss_align)/len(loss_align)
                loss_balance = sum(loss_balance)/len(loss_balance)
            else:
                loss_align = self.ensemble_locals(torch.stack(loss_align))
                loss_balance = self.ensemble_locals(torch.stack(loss_balance))

            loss_gan = self.args.w_gan * loss_gan
            loss_adv = self.args.w_adv * loss_adv
            loss_align = self.args.w_algn * loss_align
            loss_balance = self.args.w_baln * loss_balance
            loss_js = self.args.w_js * loss_js

            # Final loss: L_align + L_local + L_adv (DRO) + L_balance
            loss = loss_gan + loss_adv + loss_align + loss_balance + loss_js

        self.optim_g.zero_grad()
        if self.args.fp16:
            scaler_g = self.args.scaler_g
            scaler_g.scale(loss).backward()
            scaler_g.step(self.optim_g)
            scaler_g.update()
        else:
            loss.backward()
            self.optim_g.step()
        # loss.backward()
        # self.optim_g.step()

        if self.writer is not None:
            self.writer.add_scalars('LossGen', {'loss_gan': loss_gan.item(),
                                                'loss_adv': loss_adv.item(),
                                                'loss_align': loss_align.item(),
                                                'loss_balance': loss_balance.item(),
                                                'loss_generator':loss.item()}, self.global_step)

    def forward_teacher_outs(self, images, localN=None):
        if localN is None:  # use central as teacher
            total_logits = self.netS(images).detach()
        else:  # update student
            # get local
            total_logits = []
            for n in localN:
                logits = self.netTS[n](images)
                total_logits.append(logits)
            total_logits = torch.stack(total_logits)  # nlocal*batch*ncls
        return total_logits

    def ensemble_locals(self, locals):
        """
        locals: (nlocal, batch, ncls) or (nlocal, batch/ncls) or (nlocal)
        """
        
        if len(locals.shape) == 3:
            localweight = self.localclsweight.unsqueeze(dim=1)  # nlocal*1*ncls
            ensembled = (locals * localweight).sum(dim=0)  # batch*ncls
        elif len(locals.shape) == 2:
            localweight = self.localweight[:, None]  # nlocal*1
            ensembled = (locals * localweight).sum(dim=0)  # batch/ncls
        elif len(locals.shape) == 1:
            ensembled = (locals * self.localweight).sum()  # 1
        return ensembled

    def update_netS_batch(self, selectN):
        for _ in range(5):
            with self.args.autocast():
                with torch.no_grad():
                    z = torch.randn(
                        size=(config.FED_BATCHSIZE, config.GEN_Z_DIM)).cuda()
                    syn_img_ori = self.netG(z)
                    syn_img = self.normalizer(syn_img_ori)
                    logits_T = self.ensemble_locals(
                        self.forward_teacher_outs(syn_img, selectN))
                #
                loigts_S = self.netS(syn_img.detach())

                if self.args.use_l1_loss:
                    loss = self.criterion_distill(loigts_S, logits_T.detach())
                else:
                    loss = engine.criterions.kldiv(loigts_S, logits_T.detach(), T=self.args.T)
                loss *= self.args.w_dist

            self.optim_s.zero_grad()
            if self.args.fp16:
                scaler_s = self.args.scaler_s
                scaler_s.scale(loss).backward()
                scaler_s.step(self.optim_s)
                scaler_s.update()
            else:
                loss.backward()
                self.optim_s.step()
            # loss.backward()
            # self.optim_s.step()
        if self.args.save_img:
            with self.args.autocast(), torch.no_grad():
                predict = logits_T[:config.FED_BATCHSIZE].max(1)[1]
                idx = torch.argsort(predict)
                vis_images = syn_img_ori[idx]
                path_to_image = './visualize/' + self.args.logfile
                os.makedirs(path_to_image, exist_ok=True)
                engine.utils.save_image_batch(self.args.normalizer(
                    self.local_dataloader[0].next()[0].cuda(), True), path_to_image+f'/real_{self.current_round}.png')
                engine.utils.save_image_batch(vis_images, path_to_image+f'/syn_{self.current_round}.png')
            
        if self.writer is not None:
            self.writer.add_scalar('LossDistill', loss.item(), self.global_step)


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if is_best:
        torch.save(state, os.path.join(save_dir, filename))
        print(f'[saved] ckpt saved to {os.path.join(save_dir, filename)}')
    else:
        torch.save(state, os.path.join(save_dir, 'latest.pth'))


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(current_lr, val_loader, model, criterion, args, current_epoch):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        if args.rank <= 0:
            args.logger.info(
                ' [Eval] Epoch={current_epoch} Acc@1={top1.avg:.4f} Acc@5={top5.avg:.4f} Loss={losses.avg:.4f} Lr={lr:.4f}'
                .format(current_epoch=current_epoch, top1=top1, top5=top5, losses=losses, lr=current_lr))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class MultiTeacher(OneTeacher):
    # def __init__(self,  args):
    #     pass

    def gen_dataset(self, args):
        priv_train_data, ori_training_dataset, test_dataset = cifar.dirichlet_datasplit(
            args, privtype=args.dataset)

        local_dataset = []
        for n in range(config.N_PARTIES):
            tr_dataset = cifar.Dataset_fromarray(priv_train_data[n]['x'],
                                                 priv_train_data[n]['y'],
                                                 train=True,
                                                 verbose=False)

            local_dataset.append(tr_dataset)

        num_classes, ori_training_dataset, val_dataset = registry.get_dataset(name=args.dataset,
                                                                              data_root=args.data_root)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config.FED_BATCHSIZE,
            num_workers=config.NUM_WORKERS)

        local_datanum = np.zeros(config.N_PARTIES)
        local_cls_datanum = np.zeros((config.N_PARTIES, self.args.N_class))
        for localid in range(config.N_PARTIES):
            # count
            local_datanum[localid] = priv_train_data[localid]['x'].shape[0]
            # class specific count
            for cls in range(args.N_class):
                local_cls_datanum[localid, cls] = (
                    priv_train_data[localid]['y'] == cls).sum()

        assert sum(local_datanum) == 50000
        assert sum(sum(local_cls_datanum)) == 50000
        # import pdb;pdb.set_trace()
        return val_loader, local_dataset, local_datanum, local_cls_datanum
