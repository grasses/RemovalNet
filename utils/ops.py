#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'

"""system DNN tools"""

import math
import numpy as np
import torch
import random


def set_default_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"<--------------------------- seed:{seed} --------------------------->")


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def batch_forward(model, x, batch_size=200, argmax=False):
    """
    split x into batch_size, torch.cat result to return
    :param model:
    :param x:
    :param batch_size:
    :param argmax:
    :return:
    """
    device = next(model.parameters()).device
    steps = math.ceil(len(x)/batch_size)
    pred = []
    with torch.no_grad():
        for step in range(steps):
            off = int(step * batch_size)
            batch_x = x[off: off+batch_size].to(device)
            pred.append(model(batch_x).cpu().detach())
    pred = torch.cat(pred)
    return pred.argmax(dim=1) if argmax else pred