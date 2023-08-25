#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2023/01/06, homeway'

# The HAM10000 dataset

import os
import csv
import numpy as np
import torch
import os.path as osp
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset
from imblearn.over_sampling import RandomOverSampler
from collections import namedtuple
CSV = namedtuple("CSV", ["header", "index", "data", "labels"])


class HAM10000(VisionDataset):
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
        print(f"-> Load HAM10000 dataset from:{self.root}")

        self.label_transform = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        self.metadata = self._load_csv(filename="HAM10000_metadata", header=1)
        data_array = np.array(self.metadata.data).reshape([-1, 1])
        labels_array = np.array(self.metadata.labels)
        data, labels = RandomOverSampler(random_state=42).fit_resample(data_array, labels_array)
        x_train, x_eval, y_train, y_eval = train_test_split(data, labels, train_size=0.8, random_state=42)

        if split == "train":
            self.data = x_train.reshape(-1).tolist()
            self.labels = torch.from_numpy(y_train.reshape(-1)).long()
        elif split == "test":
            self.data = x_eval.reshape(-1).tolist()
            self.labels = torch.from_numpy(y_eval.reshape(-1)).long()

    def _load_csv(
        self,
        filename: str = "HAM10000_metadata",
        header: Optional[int] = None
    ):
        path = osp.join(self.root, filename)
        with open(path) as csv_file:
            data = list(csv.reader(csv_file, delimiter=",", skipinitialspace=True))
        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []
        indices = [i for i in range(len(data))]
        labels = [self.label_transform[row[2]] for row in data]
        data = [row[1] for row in data]
        return CSV(headers, indices, data, labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = osp.join(self.root, "images", self.data[index] + ".jpg")
        x = Image.open(path)
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
    dataset = HAM10000(root="../data/HAM10000", split="train", transform=transform)
    print(len(dataset))
    exit(1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    for x, y in data_loader:
        print(x.shape, y)
        exit(1)

















