import os.path as osp
import random
import numpy as np
import torch
from torchvision.datasets import ImageFolder


class ImageNet(ImageFolder):
    def __init__(self, root, split="train", transform=None, target_transform=None, shots=-1, seed=100, preload=False):
        root = osp.join(root, split)
        self.transform = transform
        self.num_classes = 1000
        super(ImageNet, self).__init__(root=root, transform=transform)