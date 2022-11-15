import os.path as osp
import random
import numpy as np
import torch
from torchvision.datasets import ImageFolder


class ImageNet(ImageFolder):
    def __init__(self, root, split="train", transform=None, target_transform=None, shots=-1, seed=0, preload=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        root = osp.join(root, split)

        super(ImageNet, self).__init__(root=root, transform=transform)
        self.transform = transform
        self.num_classes = 1000