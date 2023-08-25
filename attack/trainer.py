#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'

"""
independent train process
"""

import math
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from utils import helper
import random
from utils import metric
import numpy as np
best_acc = 0.0


class Trainer:
    def __init__(
            self,
            args,
            model,
            teacher,
            train_loader,
            test_loader
    ):
        helper.set_default_seed(args.seed)
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.teacher = teacher.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.reg_layers = {}
        self.epoch = args.iterations
        if args.reinit:
            for m in model.modules():
                if type(m) in [nn.Linear, nn.BatchNorm2d, nn.Conv2d]:
                    m.reset_parameters()

    def train_epoch(self, model, optimizer, train_loader, step_start, step_end, device, backends):
        model.train()
        train_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()

        count = 0
        iter_loader = iter(train_loader)
        for step in range(step_start, step_end):
            try:
                inputs, targets = next(iter_loader)
            except Exception as e:
                iter_loader = iter(train_loader)
                inputs, targets = next(iter_loader)

            if backends:
                inputs = inputs.cuda()
                targets = targets.cuda()
            else:
                inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            count += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            helper.progress_bar(step, step_end,
                                'Step: {:d} | Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d}) | Lr: {:.6f}'.format(
                                step, train_loss / count, 100. * correct / count, correct, count,
                                float(current_lr)
                )
            )
        return model

    def test(self, model, test_loader, device, backends):
        global best_acc
        test_loss = 0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if backends:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                else:
                    inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                helper.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% Best_Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, best_acc, correct, total))
        # Save checkpoint.
        acc = 100.*correct/total
        return acc

    def train(self):
        global best_acc
        args = self.args
        # load model & dataset
        model = self.model
        train_loader, test_loader = self.train_loader, self.test_loader

        if args.backends:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            cudnn.benchmark = True
            print("-> running using backends")
        else:
            model = model.to(args.device)

        # simulate: add noise to iterations
        iterations = int(self.args.iterations * (0.8 + np.random.randint(10, 20) / 100.0))

        start_step = 0
        alpha = np.random.randint(20, 40) / 100.0
        split_step = [math.ceil(iterations * alpha), math.ceil(iterations * (1-alpha)), iterations]
        print("-> split_step", split_step)

        print(f"-> Task: {args.task_str}")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * (0.8 + 0.2 * random.random()), momentum=0.9, weight_decay=args.weight_decay)
        self.train_epoch(model, optimizer, train_loader, step_start=start_step, step_end=split_step[0], device=args.device, backends=args.backends)
        print(f"-> Task: {args.task_str}")
        metric.topk_test(model=model, test_loader=test_loader, device=args.device, epoch=split_step[0], debug=True)

        print(f"-> Task: {args.task_str}")
        ratio = 10.0 if "CIFAR10" in args.task_str else 5.0
        optimizer = torch.optim.SGD(model.parameters(), lr=(0.8 + 0.2 * random.random()) * args.lr/ratio, momentum=0.9, weight_decay=args.weight_decay)
        self.train_epoch(model, optimizer, train_loader, step_start=split_step[0], step_end=split_step[1],
                         device=args.device, backends=args.backends)
        print(f"-> Task: {args.task_str}")
        metric.topk_test(model=model, test_loader=test_loader, device=args.device, epoch=split_step[1], debug=True)

        print(f"-> Task: {args.task_str}")
        ratio = 100.0 if "CIFAR10" in args.task_str else 50.0
        optimizer = torch.optim.SGD(model.parameters(), lr=(0.8 + 0.2 * random.random()) * args.lr/ratio, momentum=0.9, weight_decay=args.weight_decay)
        self.train_epoch(model, optimizer, train_loader, step_start=split_step[1], step_end=split_step[2],
                         device=args.device, backends=args.backends)
        print(f"-> Task: {args.task_str}")
        metric.topk_test(model=model, test_loader=test_loader, device=args.device, epoch=split_step[2], debug=True)
        return model
























