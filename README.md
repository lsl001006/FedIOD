# Federated Learning via Input-Output Collaborative Distillation

[arXiv](https://arxiv.org/pdf/2312.14478.pdf)
The implementation for FedIOD on the CIFAR-10/CIFAR-100 datasets. We'll release all the code to public soon.

### Abstract

>
> Federated learning (FL) is a machine learning paradigm in which distributed local nodes collaboratively train a central model without sharing individually held private data. Existing FL methods either iteratively share local model parameters or deploy co-distillation. However, the former is highly susceptible to private data leakage, and the latter design relies on the prerequisites of task-relevant real data. Instead, we propose a data-free FL framework based on local-to-central collaborative distillation with direct input and output space exploitation. Our design eliminates any requirement of recursive local parameter exchange or auxiliary task-relevant data to transfer knowledge, thereby giving direct privacy control to local users. In particular, to cope with the inherent data heterogeneity across locals, our technique learns to distill input on which each local model produces consensual yet unique results to represent each expertise. We demonstrate that our proposed FL framework achieves state-of-the-art privacy-utility trade-offs with extensive experiments on image classification, segmentation, and reconstruction tasks under various real-world heterogeneous federated learning settings on both natural and medical images.
>

### Setup

```bash
conda create -n fediod
conda activate fediod
cd supp_code/
conda install --file requirements.txt
```

### Download pretrained teacher checkpoints

We upload the pretrained teacher checkpoints to an anonymous google drive, you can check them out for your reference. 

pretrained teachers for CIFAR-10/CIFAR-100 dataset: [Google Drive](https://drive.google.com/drive/folders/1qLHV_Y5VxovMDQ3YvEuJKfMboPQ54PZA?usp=share_link)

### Run scripts

For CIFAR-10 training: (alpha=1, seed=1)

```bash
bash scripts/train_a1_s1_c10.sh
```

For CIFAR-100 training: (alpha=1, seed=1)

```bash
bash scripts/train_a1_s1_c100.sh
```

**Note**: Please replace `--from_teacher_ckpt /path/to/teacher_ckpts/` to your pretrained teachers path in the shell scripts.
