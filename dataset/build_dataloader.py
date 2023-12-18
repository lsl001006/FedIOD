from torch.utils.data import DataLoader
import os
import config
import numpy as np
import dataset.cifar as cifar
from utils import registry

def build_dataloader(args):
    # config.DATAPATH = os.path.expanduser(config.DATAPATH)
    assert args.dataset=='cifar10' or args.dataset=='cifar100'
    # publicdata = 'cifar100' if args.dataset=='cifar10' else 'imagenet'
    #TODO data filename should has $N_PARTIES
    if config.N_PARTIES==1:
        # _,_, test_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
        # _, train_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # _, ood_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # see Appendix Sec 2, ood data is also used for training
        # ood_dataset.transforms = ood_dataset.transform = train_dataset.transform # w/o augmentation
        # train_dataset.transforms = train_dataset.transform = test_dataset.transform # w/ augmentation
        priv_train_data,_, _ = cifar.dirichlet_datasplit(
            args, privtype=args.dataset)
        priv_ood_data,_, _ = cifar.dirichlet_datasplit(
            args, privtype=args.dataset)
        # _, train_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # priv_train_data=[train_dataset]
        # priv_ood_data=[train_dataset]
        num_classes, tr_dataset, test_dataset = registry.get_dataset(name= args.dataset, data_root=config.MOSAIC_KD_DATA) 
        
    else:
        priv_train_data,_, test_dataset = cifar.dirichlet_datasplit(
            args, privtype=args.dataset)
      
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config.FED_BATCHSIZE, shuffle=False, num_workers=config.NUM_WORKERS, sampler=None)
    # local 
    local_datanum = np.zeros(config.N_PARTIES)
    local_cls_datanum = np.zeros((config.N_PARTIES, args.N_class))
    for localid in range(config.N_PARTIES):
        #count
        local_datanum[localid] = priv_train_data[localid]['x'].shape[0]
        #class specific count
        for cls in range(args.N_class):
            local_cls_datanum[localid, cls] = (priv_train_data[localid]['y']==cls).sum()
        
        
        assert sum(local_datanum) == 50000
        assert sum(sum(local_cls_datanum)) == 50000

    
    

    return test_loader, priv_train_data, priv_ood_data, local_datanum, local_cls_datanum