#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/30, homeway'


import os
import os.path as osp
import logging
from abc import ABC, abstractmethod
import torch


class Fingerprinting(ABC):
    def __init__(self, model1, model2, device, out_root):
        self.logger = logging.getLogger("Fingerprinting")
        self.model1 = model1
        self.model2 = model2
        self.device = device

        # init output directory
        self.out_root = out_root
        self.exp_root = osp.join(out_root, "exp")
        self.cache_root = osp.join(out_root, "cache")
        self.fingerprint_root = osp.join(out_root, "fingerprint")
        self.ckpt_root = osp.join(out_root, "ckpt")
        for path in [self.out_root, self.exp_root, self.cache_root, self.fingerprint_root, self.ckpt_root]:
            if not osp.exists(path):
                os.makedirs(path)

    @abstractmethod
    def extract(self, **kwargs):
        """extract fingerprinting samples for model1 & model2"""

    @abstractmethod
    def verify(self, **kwargs):
        """verify model1 & model2 ownership based on fingerprinting samples"""


class Watermarking(ABC):
    def __init__(self, model1, model2):
        self.logger = logging.getLogger('Watermarking')
        self.model1 = model1
        self.model2 = model2

    @abstractmethod
    def embed(self):
        """Takes a watermarking key K, and then train a watermarked model"""""

    @abstractmethod
    def extract(self):
        """Takes a watermarking key T, and then extract message from inspected model"""