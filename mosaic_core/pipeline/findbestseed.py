from torch.utils.data import DataLoader
import sys
print(sys.path)
from mosaic_core import registry
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from dataset import cifar
import config, os
from dataset.cifar import Cifar_Dataset

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data

def dirichlet_datasplit(privtype='cifar10'):
    datapath = os.path.expanduser(config.DATAPATH)  # data/fed
    print(f'use datapath:{datapath}')
    N_parties = config.N_PARTIES
    # private
    if privtype == 'cifar10':
        subpath = 'cifar-10-batches-py/'
        N_class = 10
    elif privtype == 'cifar100':
        subpath = 'cifar-100-python/'
        N_class = 100
    
    split_arr = np.random.dirichlet(
            [1.0]*N_parties, N_class)  # nclass*N_parties
        # np.save(splitname, split_arr)

    test_dataset = Cifar_Dataset(
        os.path.join(datapath, subpath), 
        privtype, 
        train=False, 
        verbose=False)
    train_dataset = Cifar_Dataset(
        os.path.join(datapath, subpath), 
        privtype, 
        train=True, 
        verbose=False)
    train_x, train_y = train_dataset.img, train_dataset.gt
    priv_data = [None] * N_parties
    for cls_idx in range(N_class):
        
        idx = np.where(train_y == cls_idx)[0]
        totaln = idx.shape[0]
        idx_start = 0
        for i in range(N_parties):
            if i == N_parties-1:
                cur_idx = idx[idx_start:]
            else:
                idx_end = idx_start + int(split_arr[cls_idx][i]*totaln)
                cur_idx = idx[idx_start: idx_end]
                idx_start = idx_end
            if cur_idx == ():
                continue
            if priv_data[i] is None:
                priv_data[i] = {}
                priv_data[i]['x'] = train_x[cur_idx]
                priv_data[i]['y'] = train_y[cur_idx]
                priv_data[i]['idx'] = cur_idx
            else:
                priv_data[i]['idx'] = np.r_[(priv_data[i]['idx'], cur_idx)]
                priv_data[i]['x'] = np.r_[
                    (priv_data[i]['x'], train_x[cur_idx])]
                priv_data[i]['y'] = np.r_[
                    (priv_data[i]['y'], train_y[cur_idx])]
    all_priv_data = {}
    all_priv_data['x'] = train_x
    all_priv_data['y'] = train_y
    return priv_data, train_dataset, test_dataset


def gen_dataset():
    
    priv_train_data, ori_training_dataset, test_dataset = dirichlet_datasplit(privtype='cifar10')

    local_dataset = []
    for n in range(20):
        tr_dataset = cifar.Dataset_fromarray(priv_train_data[n]['x'],
                                                 priv_train_data[n]['y'],
                                                 train=True,
                                                 verbose=False)

        local_dataset.append(tr_dataset)

    num_classes, ori_training_dataset, val_dataset = registry.get_dataset(name='cifar10',
                                                                              data_root='mosaic_core/data')
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=64,
            num_workers=6)

    local_datanum = np.zeros(20)
    local_cls_datanum = np.zeros((20, 10))
    for localid in range(20):
            # count
        local_datanum[localid] = priv_train_data[localid]['x'].shape[0]
            # class specific count
        for cls in range(10):
            local_cls_datanum[localid, cls] = (
                priv_train_data[localid]['y'] == cls).sum()

    assert sum(local_datanum) == 50000
    assert sum(sum(local_cls_datanum)) == 50000
    return val_loader, local_dataset, local_datanum, local_cls_datanum

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

    val_loader, priv_data, local_datanum, local_cls_datanum = gen_dataset()
    local_dataloader = []
    local_ood_dataloader = []
    for n in range(20):
        tr_dataset = priv_data[n]
        local_loader = DataLoader(
                    dataset=tr_dataset,
                    batch_size=64,
                    shuffle=True,
                    num_workers=6)
        local_dataloader.append(DataIter(local_loader))
    print(min([len(local_loader.dataloader) for local_loader in local_dataloader]))

