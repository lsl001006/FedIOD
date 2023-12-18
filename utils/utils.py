import torch
import logging
import copy
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def load_dict(savepath, model):
    pth = torch.load(savepath)
    # pth = pth['state_dict']
    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    new_pth = {}
    for k, v in pth.items():
        if 'module' in k:
            if is_data_parallel:  # saved multi-gpu, current multi-gpu
                new_pth[k] = v
            else:  # saved multi-gpu, current 1-gpu
                new_pth[k.replace('module.', '')] = v
        else:
            if is_data_parallel:  # saved 1-gpu, current multi-gpu
                new_pth['module.'+k] = v
            else:  # saved 1-gpu, current 1-gpu
                new_pth[k] = v
    m, u = model.load_state_dict(new_pth, strict=False)
    if m:
        logging.info('Missing: '+' '.join(m))
    if u:
        logging.info('Unexpected: '+' '.join(u))
    return


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def clone_model(model):
    my_model = copy.deepcopy(model)
    return my_model


def copy_parties(n_local, model):
    parties = []
    for n in range(n_local):
        model = clone_model(model)
        parties.append(model)
    return parties


def accuracy(output, target, topk=(1,)):
    """
    usage:
    prec1,prec5=accuracy(output,target,topk=(1,5))
    """
    maxk = max(topk)
    batchsize = target.size(0)
    if len(target.shape) == 2:  # multil label
        output_mask = output > 0.5
        correct = (output_mask == target).sum()
        return [100.0*correct.float() / target.numel()]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batchsize).item())
    return res


NORMALIZE_DICT = {
    'cifar100': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    #dict( mean=(0.5071, 0.4867, 0.4408),std=(0.2675, 0.2565, 0.2761) ),
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
}


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1/s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / \
        (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, dataset):
        self.mean = NORMALIZE_DICT[dataset]['mean']
        self.std = NORMALIZE_DICT[dataset]['std']

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


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


def extract_state_dict():
    dataset = "cifar10"
    teacher = "wrn40_2"
    torch.save(torch.load('ckpt/pretrained/%s_%s.pth' % (dataset,  teacher),
               map_location='cpu')['state_dict'], "ckpt/fed/cifar10/a1.0+sd1+e500+b16/0.pt")


def save_syn_img(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(
            x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(
            normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(
            normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(
        0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().cpu().numpy()*255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(
            img_tensor.shape[2]))

    return img
