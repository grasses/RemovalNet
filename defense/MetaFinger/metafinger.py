#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


import os.path as osp
from defense import Fingerprinting


class MetaFinger(Fingerprinting):
    def __init__(self, model1, model2, test_loader, device, out_root,):
        super().__init__(model1, model2, device=device, out_root=out_root)
        self.test_loader = test_loader
        self.weights_path = osp.join(out_root, "weights")

    def extract(self, **kwargs):
        pass

    def verify(self, fingerprint, **kwargs):
        pass

    def compare(self, **kwargs):
        return self.verify(self.extract(**kwargs))

