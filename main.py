import argparse
from ast import parse
import os
import random
import time
import warnings

from datetime import datetime
from mosaic_core import registry
from mosaic_core import engine
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import config
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time
import logging
from torch.utils.tensorboard import SummaryWriter

from PIL import PngImagePlugin
from mosaic_core.pipeline.FL import MultiTeacher, validate

from mosaic_core.pipeline.FL import OneTeacher
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

parser = argparse.ArgumentParser(description='MosaicKD for OOD data')
parser.add_argument('--data_root', default='mosaic_core/data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='resnet8')
parser.add_argument('--pipeline', default='mosaickd')
parser.add_argument('--N_class', type=int, default=10)  # 10
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--unlabeled', default='cifar10')
parser.add_argument('--log_tag', default='')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_g', default=1e-3, type=float)
parser.add_argument('--T', default=1.0, type=float,
                    help="Distillation temperature. T > 10000.0 will use MSELoss")
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--z_dim', default=100, type=int)
parser.add_argument('--output_stride', default=1, type=int)
parser.add_argument('--align', default=1, type=float)
parser.add_argument('--local', default=1, type=float)
parser.add_argument('--adv', default=1.0, type=float)

parser.add_argument('--w_disc', type=float, default=1.0)
parser.add_argument('--w_gan', type=float, default=1.0)
parser.add_argument('--w_adv', type=float, default=1.0)
parser.add_argument('--w_algn', type=float, default=1.0)
parser.add_argument('--w_baln', type=float, default=10.0)
parser.add_argument('--w_dist', type=float, default=1.0)
parser.add_argument('--w_js', type=float, default=1.0)

parser.add_argument('--balance', default=10.0, type=float)

parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')
parser.add_argument('--ood_subset', action='store_true',
                    help='use ood subset')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--logfile', default='', type=str)
parser.add_argument('--seed', default=20220819, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_emsember_generator_GAN_loss', default='y', type=str)
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--ckpt_path', default="./ckpt", type=str,
                    help="location to store checkpoint")
parser.add_argument('--from_teacher_ckpt', default="", type=str,
                    help='path used to load pretrained teacher ckpts')
parser.add_argument('--use_l1_loss', action='store_true',
                    help='default use kldiv, using this command will use l1_loss')
parser.add_argument('--use_pretrained_generator', default='', type=str,
                    help='use pretrained generator instead of conventional Generator')
parser.add_argument('--modify_optim_lr', action='store_true',
                    help='modify optimizer lr when resuming model-trainning process')
parser.add_argument('--fixed_lr', action='store_true',
                    help='Use fixed Learning Rate while training')
parser.add_argument('--gen_loss_avg', action='store_true',
                    help='use average instead of ensemble locals')
parser.add_argument('--use_maxIters', action='store_true',
                    help='use max to calculate iters_per_round instead of min')
parser.add_argument('--ngf', default=64, type=int,
                    help='modify the feature map size of generator')
parser.add_argument('--save_img', action='store_true',
                    help='save_img every time update student')
parser.add_argument('--use_jsdiv', action='store_true',
                    help='use js divergence')

# TODO warmup
# parser.add_argument('--warmup_epochs', default=0, type=int,
#                     help='learning rate warmup epochs before applying lr decay (linear)')
best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    #############################logs settings##############################
    handlers = [logging.StreamHandler()]
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if args.logfile:
        args.logfile = f'{datetime.now().strftime("%m%d%H%M")}'+args.logfile
        writer = SummaryWriter(comment=args.logfile)
        handlers.append(logging.FileHandler(
            f'./logs/{args.logfile}.txt', mode='a'))
    else:
        args.logfile = 'debug'
        writer = None
        handlers.append(logging.FileHandler(f'./logs/debug.txt', mode='a'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,
    )
    logging.info(args)
    for key in dir(config):
        if not key.startswith('_'):
            value = getattr(config, key)
            logging.info(f'{key}:{value}')
    ##############################main function##############################
    main_worker(args.gpu, ngpus_per_node, args, writer)


def main_worker(rank, ngpus_per_node, args, writer):
    args.gpu = rank
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    ###################################logs###################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = '%s-%s-%s' % (args.dataset, args.teacher,
                             args.student) if args.multiprocessing_distributed else '%s-%s-%s' % (args.dataset, args.teacher, args.student)
    args.logger = engine.utils.logger.get_logger(log_name,
                                                 output='checkpoints/MosaicKD/log-%s-%s-%s-%s%s.txt' % (args.dataset, args.unlabeled, args.teacher, args.student, args.log_tag))
    
    args.tb = SummaryWriter(log_dir=os.path.join(
        'tb_log', log_name+'_%s' % (time.asctime().replace(' ', '-'))))


    if args.rank <= 0:
        for k, v in engine.utils.flatten_dict(vars(args)).items():  # print args
            args.logger.info("%s: %s" % (k, v))
    
    ###############################setup models################################
    teacher, student, netG, netD, normalizer = setup_models(args.N_class, args)
    if args.pipeline == "mosaickd":
        # pass
        pipeline = OneTeacher(student, teacher, netG, netD, args, writer)
        pipeline.update()
    ###############################Entrance#####################################
    elif args.pipeline == "multi_teacher":
        pipeline = MultiTeacher(student, teacher, netG, netD, args, writer)
        pipeline.update()

    #################################deprecated##################################
    else:
        global best_acc1

        ############################################
        # Setup Dataset
        ############################################
        num_classes, ori_training_dataset, val_dataset = registry.get_dataset(
            name=args.dataset, data_root=args.data_root)
        _, train_dataset, _ = registry.get_dataset(
            name=args.unlabeled, data_root=args.data_root)
        # _, ood_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # see Appendix Sec 2, ood data is also used for training
        # ood_dataset.transforms = ood_dataset.transform = train_dataset.transform # w/o augmentation
        train_dataset.transforms = train_dataset.transform = val_dataset.transform

        ############################################
        # Setup dataset
        ############################################
        #cudnn.benchmark = False
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
            num_workers=args.workers, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

        ############################################
        # Setup optimizer
        ############################################
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optim_g = torch.optim.Adam(
            netG.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_g, T_max=args.epochs*len(train_loader))
        optim_d = torch.optim.Adam(
            netD.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_d, T_max=args.epochs*len(train_loader))

        optim_s = torch.optim.SGD(student.parameters(
        ), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_s, T_max=args.epochs*len(train_loader))

        ############################################
        # Train Loop
        ############################################
        args.autocast = engine.utils.dummy_ctx

        for epoch in range(args.start_epoch, args.epochs):

            args.current_epoch = epoch
            train(train_loader, val_loader, [student, teacher, netG, netD], criterion, [
                  optim_s, optim_g, optim_d], [sched_s, sched_g, sched_d], epoch, args)

            acc1 = validate(
                optim_s.param_groups[0]['lr'], val_loader, student, criterion, args)
            args.tb.add_scalar('acc@1', float(acc1), global_step=epoch)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            _best_ckpt = 'mosaic_core/checkpoints/MosaicKD/%s_%s_%s_%s.pth' % (
                args.dataset, args.unlabeled, args.teacher, args.student)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.student,
                    's_state_dict': student.state_dict(),
                    'g_state_dict': netG.state_dict(),
                    'd_state_dict': netD.state_dict(),
                    'best_acc1': float(best_acc1),
                    'optim_s': optim_s.state_dict(),
                    'sched_s': sched_s.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'sched_d': sched_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'sched_g': sched_g.state_dict(),
                }, is_best, _best_ckpt)
        if args.rank <= 0:
            args.logger.info("Best: %.4f" % best_acc1)


def setup_models(num_classes, args):
    ############################################
    # Setup Models
    ############################################
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(
        args.teacher, num_classes=num_classes, pretrained=True).eval()
    if args.from_teacher_ckpt == '':
        teacher.load_state_dict(torch.load('mosaic_core/checkpoints/pretrained/%s_%s.pth' %
                                (args.dataset, args.teacher), map_location='cpu')['state_dict'])
    normalizer = engine.utils.Normalizer(
        **registry.NORMALIZE_DICT[args.dataset])
    args.normalizer = normalizer

    
    netG = engine.models.generator.Generator(nz=args.z_dim, ngf=args.ngf, nc=3, img_size=32)
    netD = engine.models.generator.PatchDiscriminator(nc=3, ndf=128)

    ############################################
    # Device preparation
    ############################################
    torch.cuda.set_device(args.gpu)
    student = student.cuda(args.gpu)
    teacher = teacher.cuda(args.gpu)
    netG = netG.cuda(args.gpu)
    netD = netD.cuda(args.gpu)
    return teacher, student, netG, netD, normalizer


def prepare_ood_data(train_dataset, model, ood_size, args):
    model.eval()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    if os.path.exists('checkpoints/ood_index/%s-%s-%s-ood-index.pth' % (args.dataset, args.unlabeled, args.teacher)):
        ood_index = torch.load('checkpoints/ood_index/%s-%s-%s-ood-index.pth' %
                               (args.dataset, args.unlabeled, args.teacher))
    else:
        with torch.no_grad():
            entropy_list = []
            model.cuda(args.gpu)
            model.eval()
            for i, (images, target) in enumerate(tqdm(train_loader)):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(images)
                p = torch.nn.functional.softmax(output, dim=1)
                ent = -(p*torch.log(p)).sum(dim=1)
                entropy_list.append(ent)
            entropy_list = torch.cat(entropy_list, dim=0)
            ood_index = torch.argsort(entropy_list, descending=True)[
                :ood_size].cpu().tolist()
            model.cpu()
            os.makedirs('checkpoints/ood_index', exist_ok=True)
            torch.save(ood_index, 'checkpoints/ood_index/%s-%s-%s-ood-index.pth' %
                       (args.dataset, args.unlabeled, args.teacher))
    return ood_index


def train(train_loader, val_loader, model, criterion, optimizer, scheduler, epoch, args):
    global best_acc1
    student, teacher, netG, netD = model
    optim_s, optim_g, optim_d = optimizer
    train_loader = train_loader
    sched_s, sched_g, sched_d = scheduler
    student.train()
    teacher.eval()
    netD.train()
    netG.train()
    for i, (real, _) in enumerate(train_loader):
        if args.gpu is not None:
            real = real.cuda(args.gpu, non_blocking=True)

        ###############################
        # Patch Discrimination
        ###############################
        with args.autocast():
            z = torch.randn(size=(args.batch_size, args.z_dim),
                            device=args.gpu)
            images = netG(z)
            images = args.normalizer(images)
            d_out_fake = netD(images.detach())
            d_out_real = netD(real.detach())
            loss_d = (torch.nn.functional.binary_cross_entropy_with_logits(d_out_fake, torch.zeros_like(d_out_fake), reduction='sum') +
                      torch.nn.functional.binary_cross_entropy_with_logits(d_out_real, torch.ones_like(d_out_real), reduction='sum')) / (2*len(d_out_fake)) * args.local
        optim_d.zero_grad()
        loss_d.backward()
        optim_d.step()

        ###############################
        # Generation
        ###############################
        with args.autocast():
            t_out = teacher(images)
            s_out = student(images)

            pyx = torch.nn.functional.softmax(t_out, dim=1)  # p(y|G(z)
            log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
            py = pyx.mean(0)  # p(y)

            # Mosaicking to distill
            d_out_fake = netD(images)
            # (Eqn. 3) fool the patch discriminator
            loss_local = torch.nn.functional.binary_cross_entropy_with_logits(
                d_out_fake, torch.ones_like(d_out_fake), reduction='sum') / len(d_out_fake)
            # (Eqn. 4) label space aligning
            # torch.nn.functional.cross_entropy(t_out, t_out.max(1)[1])  #-(pyx * torch.log2(pyx)).sum(1).mean() # or torch.nn.functional.cross_entropy(t_out, t_out.max(1)[1])
            loss_align = -(pyx * log_softmax_pyx).sum(1).mean()
            # (Eqn. 7) fool the student
            loss_adv = - engine.criterions.kldiv(s_out, t_out)

            # Appendix: Alleviating Mode Collapse for unconditional GAN
            loss_balance = (py * torch.log2(py)).sum()

            # Final loss: L_align + L_local + L_adv (DRO) + L_balance
            loss_g = args.adv * loss_adv + loss_align * args.align + \
                args.local * loss_local + loss_balance * args.balance

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        ###############################
        # Knowledge Distillation
        ###############################
        for _ in range(5):
            with args.autocast():
                with torch.no_grad():
                    z = torch.randn(
                        size=(args.batch_size, args.z_dim), device=args.gpu)
                    vis_images = fake_images = netG(z)
                    fake_images = args.normalizer(fake_images)

                    # images = torch.cat([fake_images, ood_images]) # here we use both OOD data and synthetic data for training
                    t_out = teacher(fake_images)
                s_out = student(fake_images.detach())
                loss_s = engine.criterions.kldiv(
                    s_out, t_out.detach(), T=args.T)
            optim_s.zero_grad()
            loss_s.backward()
            optim_s.step()

        sched_s.step()
        sched_d.step()
        sched_g.step()

        if i == 0:
            with args.autocast(), torch.no_grad():
                predict = t_out[:args.batch_size].max(1)[1]
                idx = torch.argsort(predict)
                vis_images = vis_images[idx]
                engine.utils.save_image_batch(args.normalizer(
                    real, True), 'checkpoints/MosaicKD/%s-%s-%s-%s-ood-data.png' % (args.dataset, args.unlabeled, args.teacher, args.student))
                engine.utils.save_image_batch(vis_images, 'checkpoints/MosaicKD/%s-%s-%s-%s-mosaic-data.png' % (
                    args.dataset, args.unlabeled, args.teacher, args.student))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
