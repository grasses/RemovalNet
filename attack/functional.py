import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import functools
import random
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import math


def loss_kd(preds, labels, teacher_preds, alpha, T):
    loss = F.kl_div(
        F.log_softmax(preds / T, dim=1),
        F.softmax(teacher_preds / T, dim=1), reduction='batchmean') * T * T * alpha + \
           F.cross_entropy(preds, labels) * (1. - alpha)
    return loss


def l2sp(model, reg):
    reg_loss = 0
    dist = 0
    for m in model.modules():
        if hasattr(m, 'weight') and hasattr(m, 'old_weight'):
            diff = (m.weight - m.old_weight).pow(2).sum()
            dist += diff
            reg_loss += diff

        if hasattr(m, 'bias') and hasattr(m, 'old_bias'):
            diff = (m.bias - m.old_bias).pow(2).sum()
            dist += diff
            reg_loss += diff

    if dist > 0:
        dist = dist.sqrt()

    loss = (reg * reg_loss)
    return loss, dist


def advtest_fast(model, loader, adversary, args):
    advDataset = torch.load(args.adv_data_dir)
    test_loader = torch.utils.data.DataLoader(
        advDataset,
        batch_size=4, shuffle=False,
        num_workers=0, pin_memory=False)
    model.eval()

    total_ce = 0
    total = 0
    top1 = 0

    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for i, (batch, label, adv_batch, adv_label) in enumerate(test_loader):
        batch, label = batch.to('cuda'), label.to('cuda')
        adv_batch = adv_batch.to('cuda')

        total += batch.size(0)
        out_clean = model(batch)

        # if 'mbnetv2' in args.network:
        #     y = torch.zeros(batch.shape[0], model.classifier[1].in_features).cuda()
        # else:
        #     y = torch.zeros(batch.shape[0], model.fc.in_features).cuda()

        # y[:,0] = args.m
        # advbatch = adversary.perturb(batch, y)

        out_adv = model(adv_batch)

        _, pred_clean = out_clean.max(dim=1)
        _, pred_adv = out_adv.max(dim=1)

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

        # print('{}/{}...'.format(i+1, len(test_loader)))
    print(f"Finish adv test fast")
    del test_loader
    del advDataset
    return float(top1_clean) / total * 100, float(top1_adv) / total * 100, float(
        adv_trial - adv_success) / adv_trial * 100


def lazy_property(func):
    attribute = '_lazy_' + func.__name__
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper