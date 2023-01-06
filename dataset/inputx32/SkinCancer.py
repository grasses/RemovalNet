import os
import torch
import torchvision
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from typing import Optional, Callable, Tuple, Any
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class SkinCancer(VisionDataset):
    def __init__(
            self,
            root: str = os.path.expanduser("~/disk/datasets/SkinCancer"),
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            shots: int = -1,
            seed: int = 0,
            preload: bool = False
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        data = pd.read_csv(root + "/hmnist_28_28_RGB.csv", encoding='utf-8')
        labels = data['label']
        images = data.drop(columns=['label'])
        images, labels = RandomOverSampler(random_state=42).fit_resample(images, labels)
        X_train, X_eval, y_train, y_eval = train_test_split(images, labels, train_size=0.8, random_state=42)
        y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
        y_eval = torch.from_numpy(np.array(y_eval)).type(torch.LongTensor)
        self.num_classes = 7
        if split == "train":
            self.data = X_train.values.reshape(-1, 28, 28, 3).astype(np.uint8)
            self.labels = y_train
        elif split == "test":
            self.data = X_eval.values.reshape(-1, 28, 28, 3).astype(np.uint8)
            self.labels = y_eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        return self.transform(img), self.labels[index]




















