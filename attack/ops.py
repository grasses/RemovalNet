#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/22, homeway'

"""important function for attacks"""

import os
import os.path as osp
import functools
import torch, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def batch_fed_forward(model, x, layer_index, batch_size=200):
    steps = math.ceil(len(x) / batch_size)
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for step in range(steps):
            off = (step * batch_size)
            batch_x = x[off: off+batch_size].clone().to(device)
            batch_out = model.fed_forward(batch_x, layer_index=layer_index).detach().cpu()
            outputs.append(batch_out)
        del batch_x, batch_out, x
        outputs = torch.cat(outputs).detach().cpu()
        torch.cuda.empty_cache()
    return outputs


def init_weights(model):
    for m in model.modules():
        if type(m) in [nn.Linear, nn.BatchNorm2d, nn.Conv2d]:
            m.reset_parameters()

def numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif type(x) is type([]):
        return np.array(x)
    return x


def loss_at(x, y):
    def attention(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    return (attention(x) - attention(y)).pow(2).mean()


def loss_kd(logit, labels, teacher_logit, alpha=0.5, T=1):
    loss = F.kl_div(
        F.log_softmax(logit / T, dim=1),
        F.softmax(teacher_logit / T, dim=1), reduction='batchmean') * T * T * alpha + \
           F.cross_entropy(logit, labels) * (1. - alpha)
    return loss


def linear_l2(model, beta_lmda):
    beta_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            beta_loss += (m.weight).pow(2).sum()
            beta_loss += (m.bias).pow(2).sum()
    return 0.5 * beta_loss * beta_lmda, beta_loss


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


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(1)
        return loss.mean()