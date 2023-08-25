#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2023/01/06, homeway'

# The BCN20000 is the substitute dataset of SkinCancer
# We do not provide label of the dataset


import os
import csv
import torch
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset
from imblearn.over_sampling import RandomOverSampler
from collections import namedtuple
CSV = namedtuple("CSV", ["header", "index", "data", "labels"])


class BCN20000(VisionDataset):
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
        print(f"-> Load BCN20000 dataset from:{self.root}")

        path = osp.join(root, f"BCN20000_{split}.pt")
        if osp.exists(path):
            cache = torch.load(path)
            self.data, self.labels = cache["x"], cache["y"]
        else:
            self.metadata = self._load_csv(filename="ISIC_2019_Training_GroundTruth.csv", header=1)
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
        filename: str = "ISIC_2019_Training_GroundTruth.csv",
        header: Optional[int] = 1
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
        labels = []
        for row in data:
            y = [int(float(a)) for a in row[1:]]
            labels.append(int(torch.tensor(y).argmax(dim=0)))
        data = [row[0] for row in data]
        return CSV(headers, indices, data, labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if type(self.data[index]) is str:
            path = osp.join(self.root, "images", self.data[index] + ".jpg")
            x = Image.open(path)
            y = self.labels[index]
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y
        else:
            return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)

    def download(self):
        # TODO: download train, https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
        # TODO: download test, https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip
        pass


if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])
    split = "test"
    dataset = BCN20000(root="../data/BCN20000", split=split, transform=transform)
    print(len(dataset))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=8)
    phar = tqdm(data_loader)
    batch_x, batch_y = [], []
    for x, y in phar:
        batch_x.append(x)
        batch_y.append(y)
    cache = {
        "x": torch.cat(batch_x),
        "y": torch.cat(batch_y)
    }
    torch.save(cache, osp.join(dataset.root, f"BCN20000_{split}.pt"))












