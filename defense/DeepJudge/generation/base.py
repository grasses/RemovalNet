#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/22, homeway'


import os
import os.path as osp
import numpy as np
import torch
import logging
ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))
cache_root = osp.join(ROOT, "cache")


class BaseSeeding:
    def __init__(self, model, task, test_loader, dataset, out_root):
        self.task = task
        self.model = model
        self.dataset = dataset
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
        self.logger = logging.getLogger('DeepJudge')
        self.out_root = out_root

    def load_test_samples(self, tag=""):
        fpath = osp.join(self.out_root, f"{self.dataset}_{self.task}_{tag}.pt")
        if osp.exists(fpath):
            self.logger.info(f"-> load test samples from: {fpath}")
            return torch.load(fpath, map_location="cpu")
        print("-> [DeepJudge] file not found!", fpath)
        return None

    def save_test_samples(self, seed_x, seed_y, test_x, test_y, tag=""):
        data = {
            "test_x": test_x,
            "test_y": test_y,
            "seed_x": seed_x,
            "seed_y": seed_y
        }
        fpath = osp.join(self.out_root, f"{self.dataset}_{self.task}_{tag}.pt")
        self.logger.info(f"-> save test samples to:{fpath}")
        return torch.save(data, fpath)

    def load_seed_samples(self, num=1000, order="max"):
        self.logger.info(f"-> generate base seed samples...")
        seed_x, seed_y, pred_list = [], [], []

        loader = iter(self.test_loader)
        steps = min(len(loader), 100)
        with torch.no_grad():
            for step in range(steps):
                try:
                    x, y = next(loader)
                except Exception as e:
                    loader = iter(self.test_loader)
                    x, y = next(loader)

                x = x.to(self.device)
                pred = self.model(x).cpu().detach()
                true_idx = torch.where(y == pred.argmax(dim=1))[0]
                pred_list.append(pred[true_idx])
                seed_x.append(x[true_idx].cpu())
                seed_y.append(y[true_idx].cpu())
            seed_x = torch.cat(seed_x)
            seed_y = torch.cat(seed_y)
            ginis = np.sum(np.square(torch.cat(pred_list).numpy()), axis=1)
            assert len(seed_x > 0)
            if order == "max":
                ranks = np.argsort(-ginis)
            else:
                ranks = np.argsort(ginis)
            return seed_x[ranks[:num]], seed_y[ranks[:num]]