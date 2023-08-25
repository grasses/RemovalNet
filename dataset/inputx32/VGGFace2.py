#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2023/01/06, homeway'

# The VGGFace2 is the substitute dataset of CelebA+31
# We do not provide label of the dataset

import os
import torch
import os.path as osp
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset


class VGGFace2(VisionDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            shots: int = -1,
            seed: int = 0,
            preload: bool = False
    ) -> None:
        assert split in ["train", "test"]
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.num_classes = 7
        print(f"-> Load VGGFace2 dataset from:{self.root}")

        data = self._load_images()
        labels = torch.zeros(len(data)).long()
        self.data = data
        self.labels = labels.reshape(-1).long()

        '''
        x_train, x_eval, y_train, y_eval = train_test_split(data, labels, train_size=0.9, random_state=42)
        if split == "train":
            self.data = x_train
            self.labels = y_train.reshape(-1).long()
        elif split == "test":
            self.data = x_eval
            self.labels = y_eval.reshape(-1).long()
        '''

    def _load_images(self):
        files = os.listdir(osp.join(self.root, "images"))
        return files

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = osp.join(self.root, "images", self.data[index])
        x = Image.open(path).convert('RGB')
        y = self.labels[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = VGGFace2(root="../data/VGGFace2", split="test", transform=transform)
    print(len(dataset))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    for x, y in data_loader:
        print(x, y)
        exit(1)

























