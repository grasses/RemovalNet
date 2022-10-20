#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import argparse
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import logging
from torchvision import transforms
from . import inputx32, inputx64, inputx224
from dataset.inputx32 import CIFAR10, CINIC10
from dataset import MIT67, SDog120, Flower102, Caltech257Data, Stanford40Data, CUB200Data, ImageNet
DATA_ROOT = osp.join(osp.abspath(osp.dirname(__file__)), "data")
logger = logging.getLogger('DataLoader')

task_list = {
    "CV32": ["CIFAR10", "CIFAR100", "CINIC10"],
    "CV224": ["SDog120", "Flower102", "ImageNet"],
    "AUDIO": [],
}


def load_cfg(dataset_id, arch_id=None):
    cfg = eval(f"inputx{get_size(dataset_id)}.cfg()")
    if "resnet" in arch_id:
        cfg.lr /= 5.0
    return cfg


def get_num_classess(dataset_id):
    NUM_CLASSES = {
        "CIFAR10": 10,
        "CINIC10": 10,
        "CIFAR100": 100,
        "GSTB": 200,
        "MIT67": 67,
        "SDog120": 120,
        "Flower102": 102,
        "CUB200Data": 200,
        "ImageNet": 1000,
        "Caltech257Data": 256,
        "Stanford40Data": 40,
    }
    return NUM_CLASSES[dataset_id]


def get_size(dataset_id):
    INPUT_SIZE = {
        "CIFAR10": 32,
        "CINIC10": 32,
        "CIFAR100": 32,
        "GSTB": 64,
        "MIT67": 224,
        "SDog120": 224,
        "Flower102": 224,
        "CUB200Data": 224,
        "ImageNet": 224,
        "Caltech257Data": 224,
        "Stanford40Data": 224,
    }
    return INPUT_SIZE[dataset_id]


def unnormalize(tensor, mean, std):
    tmp_tensor = tensor.clone()
    for t, m, s in zip(tmp_tensor, mean, std):
        (t.mul_(s).add_(m)).mul_(255.0)
    return tmp_tensor.double()



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


def get_dataloader(dataset_id, split='train', batch_size=100, shuffle=True, shots=-1):
    """
    Get dataloader.
    :param dataset_id:
    :param split:
    :param batch_size:
    :param shuffle:
    :param shot:
    :return:
    """
    datapath = os.path.join(DATA_ROOT, dataset_id)
    assert os.path.exists(datapath)

    cfg = load_cfg(dataset_id=dataset_id, arch_id="")
    normalize = torchvision.transforms.Normalize(mean=cfg.mean, std=cfg.std)
    if split == 'train':
        dataset = eval(dataset_id)(
            datapath, True, transform=transforms.Compose([
                transforms.Resize(cfg.resize_size),
                transforms.CenterCrop(cfg.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            shots=shots, seed=cfg.seed, preload=False
        )
    else:
        dataset = eval(dataset_id)(
            datapath, False, transform=transforms.Compose([
                transforms.Resize(cfg.resize_size),
                transforms.CenterCrop(cfg.input_size),
                transforms.ToTensor(),
                normalize,
            ]),
            shots=shots, seed=cfg.seed, preload=False
        )

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
    logger.info(f'-> get_dataloader success: {dataset_id}_{split}, iter_size:{len(data_loader)} num_classes:{data_loader.num_classes}')
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
    logger.info(f"-> get_seed_samples from:{dataset_id} of size:{batch_size}")
    return images


if __name__ == "__main__":
    train_loader = get_dataloader("Flower102")
    print(vars(train_loader))