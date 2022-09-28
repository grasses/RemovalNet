#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'

"""
independent train process
"""

import torch
import torch.backends.cudnn as cudnn
from utils import helper
best_acc = 0.0


class Trainer:
    def __init__(
            self,
            args,
            model,
            teacher,
            train_loader,
            test_loader,
            init_models=True
    ):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.teacher = teacher.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.reg_layers = {}
        helper.set_default_seed(args.seed)
        self.epoch = int(args.iterations / len(self.train_loader))

    def train_epoch(self, model, optimizer, train_loader, epoch, device, backends):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()

        for batch_idx in range(len(train_loader)):
            iter_loader = iter(train_loader)
            try:
                inputs, targets = next(iter_loader)
            except Exception as e:
                print(f"-> [Error] load data error {e}")
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
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            helper.progress_bar(batch_idx, len(train_loader),
                                'Epoch: {:d} | Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d}) | Lr: {:.5f}'.format(
                                epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
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

        start_epoch = 0
        split_step = [int(self.epoch * 0.25), int(self.epoch * 0.5), self.epoch]
        print("-> split_step", split_step)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        for epoch in range(start_epoch, start_epoch + split_step[0]):
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("\n-> {:s}_s{:d} LR={:6.4f}\n".format(args.task_str, args.seed, current_lr))
            self.train_epoch(model, optimizer, train_loader, epoch, device=args.device, backends=args.backends)
            self.test(model, test_loader, device=args.device, backends=args.backends)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr/10.0, momentum=0.9, weight_decay=args.weight_decay)
        for epoch in range(start_epoch + split_step[0], start_epoch + split_step[1]):
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("\n-> {:s}_s{:d} LR={:6.4f}\n".format(args.task_str, args.seed, current_lr))
            self.train_epoch(model, optimizer, train_loader, epoch, device=args.device, backends=args.backends)
            self.test(model, test_loader, device=args.device, backends=args.backends)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr/100.0, momentum=0.9, weight_decay=args.weight_decay)
        for epoch in range(start_epoch + split_step[1], start_epoch + split_step[2]):
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("\n-> {:s}_s{:d} LR={:6.4f}\n".format(args.task_str, args.seed, current_lr))
            self.train_epoch(model, optimizer, train_loader, epoch, device=args.device, backends=args.backends)
            self.test(model, test_loader, device=args.device, backends=args.backends)
        return model
























