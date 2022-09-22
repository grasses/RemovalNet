#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/25, homeway'


"""
code for: 
    Deep Neural Network Fingerprinting by Conferrable Adversarial Examples
    
this script is a wrapper of attack.finetuner.Finetuner.
"""

import os
import os.path as osp
import logging
import torch
from attack.finetuner import Finetuner
from . import ops


class Trainer(Finetuner):
    def __init__(self, args, student, teacher, train_loader, test_loader, torch_model_path, seed):
        super().__init__(args, student, teacher=teacher, train_loader=train_loader, test_loader=test_loader, init_models=False)
        self.args = args
        self.model = student
        self.num_classes = train_loader.dataset.num_classes

        # load logger
        self.logger = logging.getLogger('ConferAE')
        self.torch_model_path = torch_model_path
        if not os.path.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)
        self.ckpt_path = os.path.join(self.torch_model_path, 'final_ckpt.pth')
        ops.set_default_seed(seed)


    def load_torch_model(self):
        self.test_path = os.path.join(self.torch_model_path, 'test.tsv')

        if osp.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt['state_dict'])
            with open(self.test_path, "r") as fp:
                for line in fp:
                    pass
                print(f"-> read from cache: {line}")
        else:
            self.train()
            self.save_torch_model(self.model)
        return self.model

    def save_torch_model(self, torch_model):
        self.logger.info(f"-> save model to:{self.ckpt_path}")
        torch.save(
            {
                "args": self.args,
                'state_dict': torch_model.to("cpu").state_dict()
            },
            self.ckpt_path,
        )

















