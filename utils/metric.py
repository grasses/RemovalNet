#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/07/06, homeway'


import os.path as osp
import torch
import torch.nn.functional as F


_best_topk_acc = {
    "top1": 0,
    "top3": 0,
    "top5": 0,
}
def topk_test(model, test_loader, device, epoch=0, debug=False):
    global _best_topk_acc
    test_loss = 0.0
    correct = {
        "top1": 0,
        "top3": 0,
        "top5": 0,
    }
    topk_acc = {
        "top1": 0,
        "top3": 0,
        "top5": 0,
    }
    model.to(device)
    size = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            test_loss += loss.item()
            pred = logits.argmax(dim=1)

            top1 = torch.eq(y.view_as(pred), pred).sum().item()
            correct["top1"] += top1

            top3 = top1
            if logits.shape[1] >= 3:
                _, tk = torch.topk(logits, k=3, dim=1)
                top3 = torch.eq(y[:, None, ...], tk).any(dim=1).sum().item()
            correct["top3"] += top3

            top5 = top1
            if logits.shape[1] >= 5:
                _, tk = torch.topk(logits, k=5, dim=1)
                top5 = torch.eq(y[:, None, ...], tk).any(dim=1).sum().item()
            correct["top5"] += top5

            size += len(x)
    test_loss /= (1.0 * size)
    topk_acc["top1"] = round(100.0 * correct["top1"] / size, 5)
    topk_acc["top3"] = round(100.0 * correct["top3"] / size, 5)
    topk_acc["top5"] = round(100.0 * correct["top5"] / size, 5)
    for k, v in topk_acc.items():
        if v > _best_topk_acc[k]:
            _best_topk_acc[k] = v
    msg = "-> For E{:d}, [Test] loss={:.5f}, top-1={:.3f}%, top-3={:.3f}%, top-5={:.3f}%".format(
            int(epoch),
            test_loss,
            topk_acc["top1"],
            topk_acc["top3"],
            topk_acc["top5"]
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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", output_dir=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        if output_dir is not None:
            self.filepath = osp.join(output_dir, "progress")

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log_str = '\t'.join(entries)
        print(log_str)
        # if self.filepath is not None:
        #     with open(self.filepath, "a") as f:
        #         f.write(log_str+"\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'