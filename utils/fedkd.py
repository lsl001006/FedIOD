import os
import numpy as np
import logging
from datetime import datetime
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from tensorboardX import SummaryWriter
import copy
import random
import config
import engine 
import utils.utils as utils
from utils.train import trainLocal
from utils.test import validate_model
from dataset.build_dataloader import build_dataloader
from utils.utils import DataIter
import dataset.cifar as cifar
from torch.utils.data import DataLoader
from mosaic_core import registry
class myFed:
    def __init__(self, student,teacher, generator, discriminator, writer, args):
        self.writer = writer
        self.args = args
        self.val_loader, self.priv_data,self.priv_ood_data, self.local_datanum, self.local_cls_datanum = build_dataloader(args)
        
        self.local_datanum = torch.FloatTensor(self.local_datanum).cuda()
        self.local_cls_datanum = torch.FloatTensor(self.local_cls_datanum).cuda()

        #model pretrain gen/dis?
        self.netG = generator
        self.normalizer = utils.Normalizer(args.dataset)
        self.netDS = utils.copy_parties(config.N_PARTIES, discriminator)
        #teacher/student
        self.netS = student
        self.init_netTS(teacher)
        

    def init_netTS(self,teacher):
        epochs = config.INIT_EPOCHS
        self.netTS = utils.copy_parties(config.N_PARTIES, teacher) #can be different
        ckpt_dir = f'{config.LOCAL_CKPTPATH}/{self.args.dataset}/a{self.args.alpha}+sd{self.args.seed}+e{epochs}+b{config.BATCHSIZE}' 
        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)
        
        if not self.args.logfile=='debug':
            for n in range(config.N_PARTIES):
                model = self.netTS[n]
                savename = f'{ckpt_dir}/{n}.pt'
                if os.path.exists(savename):
                    logging.info(f'Loading Local{n}......')
                    utils.load_dict(savename, self.netTS[n])
                    acc = validate_model(self.netTS[n], self.val_loader)
                else:
                    logging.info(f'Init Local{n}, Epoch={epochs}......')
                    tr_dataset=self.gen_dataset_fromarray(n)
                    train_loader = DataLoader(
                        dataset=tr_dataset, batch_size=config.BATCHSIZE, shuffle=True, num_workers=config.NUM_WORKERS, sampler=None) 
                    acc = trainLocal(model, train_loader, savename, test_loader=self.val_loader, writer=None)
                logging.info(f'Init Local{n}--Epoch={epochs}, Acc:{(acc):.2f}')

    def gen_dataset_fromarray(self,n):
        if config.N_PARTIES>1:
            tr_dataset = cifar.Dataset_fromarray(self.priv_data[n]['x'], self.priv_data[n]['y'], train=True, verbose=False)
        else:
            num_classes, tr_dataset, val_dataset = registry.get_dataset(name=self.args.dataset, data_root=config.MOSAIC_KD_DATA)
        return tr_dataset
    def init_training(self):
        self.local_dataloader = []
        self.local_ood_dataloader=[]
        for n in range(config.N_PARTIES):
            tr_dataset = self.gen_dataset_fromarray( n)
            local_loader = DataLoader(
                        dataset=tr_dataset, batch_size=config.FED_BATCHSIZE, shuffle=True, num_workers=config.NUM_WORKERS, sampler=None) 
            self.local_dataloader.append(DataIter(local_loader))
            ood_dataset = cifar.Dataset_fromarray(self.priv_ood_data[n]['x'], self.priv_ood_data[n]['y'], train=True, verbose=False)
            local_ood_loader = DataLoader(
                        dataset=ood_dataset, batch_size=config.FED_BATCHSIZE, shuffle=True, num_workers=config.NUM_WORKERS, sampler=None) 
            self.local_ood_dataloader.append(DataIter(local_ood_loader))

        self.iters_per_round = min([len(local_loader.dataloader) for local_loader in self.local_dataloader])#4~9
        # import ipdb; ipdb.set_trace()
        steps_all = config.ROUNDS*self.iters_per_round
        #init optim
        if config.OPTIMIZER == 'SGD':
            self.optim_s = optim.SGD(
                self.netS.parameters(), lr=config.STU_LR, momentum=0.9, weight_decay=config.STU_WD)
        else:    
            self.optim_s = optim.Adam(
                self.netS.parameters(), lr=config.STU_LR,  betas=(0.9, 0.999), weight_decay=config.STU_WD)
        self.sched_s = optim.lr_scheduler.CosineAnnealingLR(
                self.optim_s, steps_all, eta_min=config.STU_LR_MIN,)
        #for gen
        self.optim_g = torch.optim.Adam(self.netG.parameters(), lr=config.GEN_LR, betas=[0.5, 0.999])
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_g, T_max=steps_all )
        param_ds = []
        for n in range(config.N_PARTIES):
            param_ds += list(self.netDS[n].parameters()) 
        self.optim_d = torch.optim.Adam(param_ds, lr=config.DIS_LR, betas=[0.5, 0.999])
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_d, T_max=steps_all ) 
        ##TODO: not all complete steps_all if local_percetnt<1
        #
        # import ipdb; ipdb.set_trace()

        #criterion
        self.criterion_distill = nn.L1Loss(reduce='mean')
        self.criterion_bce = nn.functional.binary_cross_entropy_with_logits

        #netS training records
        self.bestacc = 0
        self.best_statdict = copy.deepcopy(self.netS.state_dict())
        #save path
        self.savedir = f'{config.CKPTPATH}/{self.args.logfile}'
        self.savedir_gen = f'{config.CKPTPATH}/{self.args.logfile}/gen'
        if not os.path.isdir(self.savedir_gen):
            os.makedirs(self.savedir_gen)
        
    def update(self):
        self.init_training()
        self.global_step = 0
        #default: local_percent =1
        selectN = list(range(0, config.N_PARTIES))
        self.localweight = self.local_datanum/self.local_datanum.sum()#nlocal*nclass
        self.localclsweight = self.local_cls_datanum/self.local_cls_datanum.sum(dim=0)#nlocal*nclass

        for round in range(config.ROUNDS):
            if config.LOCAL_PERCENT<1:    
                selectN = random.sample(selectN, 
                                        int(config.LOCAL_PERCENT*config.N_PARTIES))
                countN = self.local_datanum[selectN]
                self.localweight = countN/countN.sum() #nlocal
                countN = self.local_cls_datanum[selectN]
                self.localclsweight = countN/countN.sum(dim=0)#nlocal*nclass
                
            logging.info(f'************Start Round{round}***************')
            self.update_round(round, selectN)
        #save G,D
        torch.save(self.netG.state_dict(), f'{self.savedir_gen}/generator.pt') 
        for n in range(config.N_PARTIES):
             torch.save(self.netDS[n].state_dict(), f'{self.savedir_gen}/discrim{n}.pt') 

    def update_round(self, roundd, selectN):
        #
        bestacc_round = self.bestacc
        for iter in range(self.iters_per_round):
            #1. update D,G
            z = torch.randn(size=(config.FED_BATCHSIZE, config.GEN_Z_DIM)).cuda()
            syn_img = self.netG(z)
            syn_img = self.normalizer(syn_img)
            self.update_netDS_batch(syn_img, selectN)
            self.update_netG_batch(syn_img, selectN)
            #2. Distill, update S
            self.update_netS_batch(selectN)

            #val
        acc = validate_model(self.netS, self.val_loader)
        self.global_step += 1
        if acc>bestacc_round:
            logging.info(f'Iter{iter}, best for now:{acc}')
            self.best_statdict = copy.deepcopy(self.netS.state_dict())
            bestacc_round = acc
        if iter % config.PRINT_FREQ == 0:
            logging.info(f'===R{roundd}, {iter}/{self.iters_per_round}, acc{acc}, best{self.bestacc}')
                
        #reload G,D?
        # self.netS.load_state_dict(self.best_statdict, strict=True)
        logging.info(f'=============Round{roundd}, BestAcc originate: {(self.bestacc):.2f}, to{(bestacc_round):.2f}====================')
        
        if bestacc_round>self.bestacc:
            savename = os.path.join(self.savedir, f'r{roundd}_{(bestacc_round):.2f}.pt')
            torch.save(self.best_statdict, savename)
            self.bestacc = bestacc_round
        
        if self.writer is not None:
            self.writer.add_scalar('BestACC', self.bestacc, roundd)

    def update_netDS_batch(self, syn_img, selectN):
        loss = 0.
        for localid in selectN:
            d_out_fake = self.netDS[localid](syn_img.detach())
            real_img = self.local_dataloader[localid].next()[0].cuda()#list [img, label, ?]
            d_out_real = self.netDS[localid](real_img)
            loss_d = (self.criterion_bce(d_out_fake.detach(), torch.zeros_like(d_out_fake), reduction='sum') + \
                self.criterion_bce(d_out_real, torch.ones_like(d_out_real), reduction='sum'))/  (2*len(d_out_fake))  
            loss += loss_d
        loss *= self.args.w_disc

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()
        self.sched_d.step()

        if self.writer is not None:
            self.writer.add_scalar('LossDiscriminator', loss.item(), self.global_step)
        
    def update_netG_batch(self, syn_img, selectN):
        #1. gan loss
        loss_gan = 0.
        for localid in selectN:
            d_out_fake = self.netDS[localid](syn_img)#gradients in syn
            loss_gan += self.criterion_bce(d_out_fake, torch.ones_like(d_out_fake), reduction='mean')
        
        #2. adversarial distill loss
        logits_T = self.forward_teacher_outs(syn_img, selectN)
        ensemble_logits_T = self.ensemble_locals(logits_T)
        logits_S = self.netS(syn_img)
        loss_adv =- engine.criterions.kldiv(logits_S, ensemble_logits_T)  # - self.criterion_distill(logits_S, ensemble_logits_T) #(bs, ncls) 
        #3.regularization for each t_out (not ensembled) #TO DISCUSS
        loss_align = []
        loss_balance = []
        for n in range(len(selectN)):
            t_out = logits_T[n]
            pyx = torch.nn.functional.softmax(t_out, dim = 1) # p(y|G(z)
            log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
            py = pyx.mean(0)
            loss_align.append( -(pyx * log_softmax_pyx).sum(1).mean()) #To generate distinguishable imgs
            loss_balance.append( (py * torch.log2(py)).sum()) #Alleviating Mode Collapse for unconditional GAN
        
        loss_align = self.ensemble_locals(torch.stack(loss_align))
        loss_balance = self.ensemble_locals(torch.stack(loss_balance))

        loss_gan = self.args.w_gan*loss_gan
        loss_adv = self.args.w_adv*loss_adv
        loss_align = self.args.w_algn*loss_align
        loss_balance = self.args.w_baln*loss_balance

        # Final loss: L_align + L_local + L_adv (DRO) + L_balance
        loss = loss_gan + loss_adv + loss_align + loss_balance

        self.optim_g.zero_grad()
        loss.backward()
        self.optim_g.step()
        self.sched_g.step()

        if self.writer is not None:
            self.writer.add_scalars('LossGen', {'loss_gan': loss_gan.item(), 
                                                'loss_adv': loss_adv.item(), 
                                                'loss_align': loss_align.item(), 
                                                'loss_balance': loss_balance.item()}, self.global_step)

    def forward_teacher_outs(self, images, localN=None):
        if localN is None: #use central as teacher
            total_logits = self.netS(images).detach()
        else: #update student
            #get local
            total_logits = []
            for n in localN:
                tmodel = copy.deepcopy(self.netTS[n])
                logits = tmodel(images).detach()
                total_logits.append(logits)
                del tmodel
            total_logits = torch.stack(total_logits) #nlocal*batch*ncls       
        return total_logits
    
    def  ensemble_locals(self, locals):
        """
        locals: (nlocal, batch, ncls) or (nlocal, batch/ncls) or (nlocal)
        """
        if len(locals.shape)==3:
            localweight = self.localclsweight.unsqueeze(dim=1)#nlocal*1*ncls
            ensembled = (locals*localweight).sum(dim=0) #batch*ncls
        elif len(locals.shape)==2:
            localweight = self.localweight[:,None]#nlocal*1
            ensembled = (locals*localweight).sum(dim=0) #batch/ncls
        elif len(locals.shape)==1:
            ensembled = (locals*self.localweight).sum() #1
        return ensembled
    
    def update_netS_batch(self, selectN):
        for _ in range(5): 
            with torch.no_grad():
                z = torch.randn(size=(config.FED_BATCHSIZE, config.GEN_Z_DIM)).cuda()
                syn_img = self.netG(z)
                syn_img = self.normalizer(syn_img)
                ood_images = self.local_ood_dataloader[0].next()[0].cuda()
                images = torch.cat([syn_img, ood_images])
            #
            kd_img=syn_img
            self.netS.train()
            loigts_S = self.netS(kd_img)
            logits_T = self.ensemble_locals(self.forward_teacher_outs(kd_img, selectN))
            # loss =   self.criterion_distill(loigts_S, logits_T) #
            loss=engine.criterions.kldiv(loigts_S, logits_T.detach() )
            loss *= self.args.w_dist

            self.optim_s.zero_grad()
            loss.backward()
            self.optim_s.step()
            self.sched_s.step()

            if self.writer is not None:
                self.writer.add_scalar('LossDistill', loss.item(), self.global_step)