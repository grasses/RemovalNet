#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import os
import os.path as osp
import numpy as np
import torch
import torchvision
import logging
from torchvision import transforms
from utils import ops
from . import inputx32, inputx64, inputx224
from dataset.inputx32 import CIFAR10, CINIC10, CelebA32, LFW32, SkinCancer, BCN20000, HAM10000, VGGFace2, GTSRB, TSRD
from dataset.inputx224 import LFW, CelebA, ImageNet

DATA_ROOT = osp.join(osp.abspath(osp.dirname(__file__)), "data")
logger = logging.getLogger('DataLoader')

task_list = {
    "CV32": ["CIFAR10", "CINIC10", "GTSRB", "GTSRB+1", "GTSRB+2", "TSRD", "CelebA32", "LFW32", "VGGFace2", "SkinCancer", "BCN20000", "HAM10000"],
    "CV224": ["ImageNet", "CelebA", "LFW"],
}
for i in range(40):
    task_list["CV32"].append(f"CelebA32+{i}")
    task_list["CV224"].append(f"CelebA32+{i}")


def load_cfg(dataset_id, arch_id=None):
    cfg = eval(f"inputx{get_size(dataset_id)}.cfg(dataset_id=dataset_id)")
    return cfg


def get_num_classess(dataset_id):
    NUM_CLASSES = {
        "CIFAR10": 10,
        "CINIC10": 10,
        "GTSRB": 43,
        "GTSRB+1": 43,
        "GTSRB+2": 43,
        "TSRD": 43,
        "SkinCancer": 7,
        "BCN20000": 7,
        "HAM10000": 7,
        "CelebA": 2,
        "CelebA32": 2,
        "VGGFace2": 2,
        "LFW": 2,
        "LFW32": 2,
        "ImageNet": 1000,
    }
    for i in range(40):
        NUM_CLASSES[f"CelebA+{i}"] = 2
        NUM_CLASSES[f"CelebA32+{i}"] = 2
    return NUM_CLASSES[dataset_id]


def get_size(dataset_id):
    INPUT_SIZE = {
        "CIFAR10": 32,
        "CINIC10": 32,
        "GTSRB": 32,
        "GTSRB+1": 32,
        "GTSRB+2": 32,
        "TSRD": 32,
        "CelebA32": 32,
        "LFW32": 32,
        "VGGFace2": 32,
        "SkinCancer": 32,
        "BCN20000": 32,
        "HAM10000": 32,
        "LFW": 224,
        "CelebA": 224,
        "Flower102": 224,
        "ImageNet": 224,
    }
    for i in range(40):
        INPUT_SIZE[f"CelebA32+{i}"] = 32
        INPUT_SIZE[f"CelebA+{i}"] = 224
    return INPUT_SIZE[dataset_id]


def unnormalize(tensor, mean, std, clamp=False):
    tmp = tensor.clone()
    for t, m, s in zip(tmp, mean, std):
        (t.mul_(s).add_(m))
    if clamp:
        tmp = torch.clamp(tmp, min=0.0, max=1.0)
    return tmp.double()


def get_bounds(mean, std):
    bounds = [-1, 1]
    if type(mean) == type(list([])):
        c = len(mean)
        _min = (np.zeros([c]) - np.array(mean)) / np.array([std])
        _max = (np.ones([c]) - np.array(mean)) / np.array([std])
        bounds = [np.min(_min).item(), np.max(_max).item()]
    elif type(mean) == float:
        bounds = [(0.0 - mean) / std, (1.0 - mean) / std]
    return bounds


def get_dataloader(dataset_id, split='train', batch_size=100, shuffle=True, subset_ratio=1.0, train_transforms=None, test_transforms=None):
    """
    Get dataloader.
    :param dataset_id:
    :param split:
    :param batch_size:
    :param shuffle:
    :param shot:
    :return:
    """
    shots = 0
    if "+" in dataset_id:
        dataset_id, shots = dataset_id.split("+")

    datapath = os.path.join(DATA_ROOT, dataset_id)
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"-> datapath: {datapath}")

    cfg = load_cfg(dataset_id=dataset_id, arch_id="")
    normalize = torchvision.transforms.Normalize(mean=cfg.mean, std=cfg.std)
    ops.set_default_seed(cfg.seed)

    if split == 'train':
        train_transforms = train_transforms if train_transforms is not None else transforms.Compose([
                transforms.Resize(cfg.resize_size),
                transforms.CenterCrop(cfg.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ])
        dataset = eval(dataset_id)(
            datapath, split, transform=train_transforms,
            shots=int(shots), seed=cfg.seed, preload=False
        )
    elif split == 'test' or split == 'val':
        test_transforms = test_transforms if test_transforms is not None else transforms.Compose([
            transforms.Resize(cfg.resize_size),
            transforms.CenterCrop(cfg.input_size),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = eval(dataset_id)(
            datapath, split, transform=test_transforms,
            shots=int(shots), seed=cfg.seed, preload=False
        )
    else:
        raise NotImplementedError()

    if subset_ratio < 1.0:
        size = len(dataset)
        idxs = np.random.choice(size, int(size * subset_ratio), replace=False).tolist()
        print(f"-> Load {int(subset_ratio*100)}% subset of {dataset_id} {len(idxs)}/{size}")
        dataset = torch.utils.data.Subset(dataset, idxs)
        dataset.num_classes = get_num_classess(dataset_id)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=False
    )
    data_loader.dataset_id = dataset_id
    data_loader.mean = cfg.mean
    data_loader.std = cfg.std
    data_loader.input_size = cfg.input_size
    data_loader.num_classes = dataset.num_classes
    data_loader.bounds = get_bounds(cfg.mean, cfg.std)
    data_loader.unnormalize = unnormalize
    logger.info(f'-> get_dataloader success: {dataset_id}_{split}, iter_size:{len(data_loader)} batch_size:{batch_size} num_classes:{data_loader.num_classes}')
    return data_loader


def get_seed_samples(dataset_id, batch_size, rand=False, shuffle=True, with_label=False, unormalize=False):
    """
    Return only $batch_size samples from train set
    :param dataset_id:
    :param batch_size:
    :param rand:
    :param shuffle:
    :param with_label:
    :param unormalize:
    :return:
    """
    datapath = os.path.join(DATA_ROOT, dataset_id)
    assert os.path.exists(datapath)

    cfg = load_cfg(dataset_id=dataset_id)
    if rand:
        batch_input_size = (batch_size, * cfg.input_shape)
        images = np.random.normal(size=batch_input_size).astype(np.float32)
    else:
        train_loader = get_dataloader(dataset_id=dataset_id, split='train', batch_size=batch_size, shuffle=shuffle)
        images, labels = next(iter(train_loader))
        unnormalize_images = train_loader.unnormalize(images).to('cpu').numpy()
        images = images.to('cpu').numpy()
        labels = labels.to('cpu').numpy()
        bounds = train_loader.bounds
        if not with_label:
            if unormalize:
                return images, unnormalize_images
            return images
        else:
            if unormalize:
                return images, unnormalize_images, bounds, labels
            return images, labels
    logger.info(f"-> get_seed_samples from:{dataset_id} batch_size:{batch_size}")
    return images


if __name__ == "__main__":
    train_loader = get_dataloader("Flower102")
    print(vars(train_loader))