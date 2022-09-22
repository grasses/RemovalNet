#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/07/06, homeway'


import torch
import torch.nn.functional as F


_best_topk_acc = {
    "top-1": 0,
    "top-3": 0,
    "top-5": 0,
}
def topk_test(model, test_loader, device, epoch=0, debug=False):
    global _best_topk_acc
    test_loss = 0.0
    correct = {
        "top-1": 0,
        "top-3": 0,
        "top-5": 0,
    }
    topk_acc = {
        "top-1": 0,
        "top-3": 0,
        "top-5": 0,
    }
    model.to(device)
    size = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            test_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct["top-1"] += torch.eq(y.view_as(pred), pred).sum().item()
            _, tk = torch.topk(logits, k=3, dim=1)
            correct["top-3"] += torch.eq(y[:, None, ...], tk).any(dim=1).sum().item()
            _, tk = torch.topk(logits, k=5, dim=1)
            correct["top-5"] += torch.eq(y[:, None, ...], tk).any(dim=1).sum().item()
            size += len(x)
    test_loss /= (1.0 * size)

    topk_acc["top-1"] = round(100.0 * correct["top-1"] / size, 5)
    topk_acc["top-3"] = round(100.0 * correct["top-3"] / size, 5)
    topk_acc["top-5"] = round(100.0 * correct["top-5"] / size, 5)
    for k, v in topk_acc.items():
        if v > _best_topk_acc[k]:
            _best_topk_acc[k] = v

    msg = "-> For E{:d}, [Test] loss={:.5f}, top-1={:.3f}%, top-3={:.3f}%, top-5={:.3f}%".format(
            int(epoch),
            test_loss,
            topk_acc["top-1"],
            topk_acc["top-3"],
            topk_acc["top-5"]
    )
    if debug: print(msg)
    return _best_topk_acc, topk_acc, test_loss

class MovingAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', momentum=0.9):
        self.name = name
        self.fmt = fmt
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.momentum * self.avg + (1 - self.momentum) * val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
