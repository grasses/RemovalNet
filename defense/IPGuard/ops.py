#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/24, homeway'

import torch
import math
import random
import numpy as np


def set_default_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def batch_mid_forward(model, x, layer_index, batch_size=200):
    steps = math.ceil(len(x) / batch_size)
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for step in range(steps):
            off = (step * batch_size)
            batch_x = x[off: off+batch_size].clone().to(device)
            batch_out = model.mid_forward(batch_x, layer_index=layer_index).detach().cpu()
            outputs.append(batch_out)
        del batch_x, batch_out, x
        outputs = torch.cat(outputs).detach().cpu()
        torch.cuda.empty_cache()
    return outputs