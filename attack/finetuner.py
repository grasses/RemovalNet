#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/22, homeway'

"""Fine-tuning, code from: https://github.com/yuanchun-li/ModelDiff"""

import time, math
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import metric, helper
from attack import ops
from tqdm import tqdm
import random


class Finetuner(object):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
        init_models=True,
        debug=False
    ):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.teacher = teacher.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        helper.set_default_seed(args.seed)
        self.reg_layers = {}
        if init_models:
            self.init_models()
        self.debug = debug

    def init_models(self):
        args = self.args
        model = self.model
        teacher = self.teacher

        # Used to matching features
        def record_act(self, input, output):
            self.out = output

        if 'mobilenet' in args.network:
            reg_layers = {0: [model.features[16], teacher.features[16]],
                          1: [model.features[18], teacher.features[18]]}
            model.features[16].register_forward_hook(record_act)
            model.features[18].register_forward_hook(record_act)
        elif 'resnet' in args.network:
            reg_layers = {0: [model.layer3, teacher.layer3], 1: [model.layer4, teacher.layer4]}
            model.layer3.register_forward_hook(record_act)
            model.layer4.register_forward_hook(record_act)
        elif ('vgg' in args.network) or ('alexnet' in args.network):
            reg_layers = {0: [model.layer4, teacher.layer4], 1: [model.layer5, teacher.layer5]}
            model.layer4.register_forward_hook(record_act)
            model.layer5.register_forward_hook(record_act)
        elif 'densenet' in args.network:
            reg_layers = {0: [model.features.denseblock3, teacher.features.denseblock3],
                          1: [model.features.denseblock4, teacher.features.denseblock4]}
            model.features.denseblock3.register_forward_hook(record_act)
            model.features.denseblock4.register_forward_hook(record_act)
        elif 'inception' in args.network:
            reg_layers = {0: [model.Mixed_6e, teacher.Mixed_6e],
                          1: [model.Mixed_7c, teacher.Mixed_7c]}
            model.Mixed_6e.register_forward_hook(record_act)
            model.Mixed_7c.register_forward_hook(record_act)
        elif "vit" in args.network:
            reg_layers = {0: [model.blocks, teacher.blocks]}
            model.blocks.register_forward_hook(record_act)
        else:
            raise NotImplementedError(f"-> Not implemented for:{args.network}")

        # Stored pre-trained weights for computing L2SP
        for m in model.modules():
            if hasattr(m, 'weight') and not hasattr(m, 'old_weight'):
                m.old_weight = m.weight.data.clone().detach()
                # all_weights = torch.cat([all_weights.reshape(-1), m.weight.data.abs().reshape(-1)], dim=0)
            if hasattr(m, 'bias') and not hasattr(m, 'old_bias') and m.bias is not None:
                m.old_bias = m.bias.data.clone().detach()

        if args.reinit:
            for m in model.modules():
                if type(m) in [nn.Linear, nn.BatchNorm2d, nn.Conv2d]:
                    m.reset_parameters()
                    torch.nn.init.xavier_uniform(m.weight)

        retrain_linear = self.args.retrain_linear if 'retrain_linear' in self.args else None
        if retrain_linear:
            modules = []
            for m in model.modules():
                if type(m) == nn.Linear:
                    modules.append(m)
            num_tune_modules = math.ceil(len(modules) * retrain_linear)
            for m in modules[-num_tune_modules:]:
                m.reset_parameters()

        if 'mobilenet' in args.network:
            teacher.features[16].register_forward_hook(record_act)
            teacher.features[18].register_forward_hook(record_act)
        elif 'resnet' in args.network:
            teacher.layer3.register_forward_hook(record_act)
            teacher.layer4.register_forward_hook(record_act)
        elif ('vgg' in args.network) or ('alexnet' in args.network):
            teacher.layer4.register_forward_hook(record_act)
            teacher.layer5.register_forward_hook(record_act)
        elif 'densenet' in args.network:
            teacher.features.denseblock3.register_forward_hook(record_act)
            teacher.features.denseblock4.register_forward_hook(record_act)
        elif 'inception' in args.network:
            teacher.Mixed_6e.register_forward_hook(record_act)
            teacher.Mixed_7c.register_forward_hook(record_act)
        elif "vit" in args.network:
            teacher.blocks.register_forward_hook(record_act)
        else:
            raise NotImplementedError(f"-> Not implemented for:{args.network}")
        self.reg_layers = reg_layers


    def compute_steal_loss(self, batch, label, teacher_out):
        """
        Args:
            batch:
            label:
            teacher_logits:
        Returns: KD_loss = soft_loss + hard_loss
        """
        teacher_labels = teacher_out.argmax(dim=1)
        out = self.model(batch)
        _, pred = out.max(dim=1)

        alpha = self.args.steal_alpha
        T = self.args.temperature
        soft_loss = nn.KLDivLoss()(
            F.log_softmax(out / T, dim=1),
            F.softmax(teacher_out / T, dim=1)
        ) * (alpha * T * T)
        hard_loss = F.cross_entropy(out, teacher_labels) * (1. - alpha)
        KD_loss = soft_loss + hard_loss

        top1 = float(pred.eq(label).sum().item()) / label.shape[0] * 100.
        return KD_loss, top1, soft_loss, hard_loss


    def compute_negative_loss(self, batch, label, ce):
        # negative model: independent training
        model = self.model
        teacher = self.teacher
        args = self.args
        feat_loss, l2sp_loss = 0, 0
        out = model(batch)
        _, pred = out.max(dim=1)

        top1 = float(pred.eq(label).sum().item()) / label.shape[0] * 100.
        loss = 0.
        loss += ce(out, label)
        ce_loss = loss.item()
        linear_loss = 0
        total_loss = loss.item()
        return loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss
        
    def compute_loss(self, batch, label, ce, featloss, teacher_out):
        model = self.model
        args = self.args
        l2sp_lmda = self.args.l2sp_lmda
        reg_layers = self.reg_layers
        feat_loss, l2sp_loss = 0, 0

        out = model(batch)
        _, pred = out.max(dim=1)
        top1 = float(pred.eq(label).sum().item()) / label.shape[0] * 100.

        loss = 0.
        loss += ce(out, label)
        ce_loss = loss.item()

        # Compute the feature distillation loss only when needed
        if args.feat_lmda != 0:
            regloss = 0
            for key in reg_layers.keys():
                src_x = reg_layers[key][0].out
                tgt_x = reg_layers[key][1].out
                regloss += featloss(src_x, tgt_x.detach())
            regloss = args.feat_lmda * regloss
            loss += regloss
            feat_loss = regloss.item()

        beta_loss, linear_norm = ops.linear_l2(model, args.beta)
        loss = loss + beta_loss
        linear_loss = beta_loss.item()

        if l2sp_lmda != 0:
            reg, _ = ops.l2sp(model, l2sp_lmda)
            l2sp_loss = reg.item()
            loss = loss + reg
        total_loss = loss.item()
        return loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss

    def steal_test(self):
        model = self.model
        teacher = self.teacher
        loader = self.test_loader
        alpha = self.args.steal_alpha
        T = self.args.temperature
        
        with torch.no_grad():
            model.eval()
            teacher.eval()
            total_soft, total_hard, total_kd = 0, 0, 0
            total = 0
            top1 = 0
            for i, (batch, label) in enumerate(loader):
                batch, label = batch.to(self.device), label.to(self.device)
                total += batch.size(0)
                
                teacher_out = teacher(batch)
                out = model(batch)
                _, pred = out.max(dim=1)
                
                soft_loss = nn.KLDivLoss()(
                    F.log_softmax(out/T, dim=1),
                    F.softmax(teacher_out/T, dim=1)
                ) * (alpha * T * T)
                hard_loss = F.cross_entropy(out, label) * (1. - alpha)
                KD_loss = soft_loss + hard_loss
                
                total_soft += soft_loss.item()
                total_hard += hard_loss.item()
                total_kd += KD_loss.item()
                top1 += int(pred.eq(label).sum().item())
        return float(top1)/total*100, total_kd/(i+1), total_soft/(i+1), total_hard/(i+1)
        
    def test(self, model, teacher, test_loader, teststep=500, loss=True, train=True):
        reg_layers = self.reg_layers
        args = self.args
        split = "Train" if train else "Test"

        with torch.no_grad():
            model.eval()
            if loss:
                teacher.eval()
                ce = ops.CrossEntropyLabelSmooth(test_loader.dataset.num_classes, args.label_smoothing).to(self.device)
                featloss = torch.nn.MSELoss(reduction='none')

            total_ce = 0
            total_feat_reg = np.zeros(len(reg_layers))
            total_l2sp_reg = 0
            total = 0
            top1 = 0
            teststep = min(teststep, len(test_loader))
            phar = tqdm(range(teststep))
            loader = iter(test_loader)
            for step in phar:
                try:
                    batch, label = next(loader)
                except:
                    loader = iter(test_loader)
                    batch, label = next(loader)
                batch, label = batch.to(self.device), label.to(self.device)
                total += batch.size(0)
                out = model(batch)
                _, pred = out.max(dim=1)
                top1 += int(pred.eq(label).sum().item())
                if loss:
                    total_ce += ce(out, label).item()
                    if teacher is not None:
                        with torch.no_grad():
                            tout = teacher(batch)
                        for i, key in enumerate(reg_layers):
                            # print(key, len(reg_layers[key]))
                            src_x = reg_layers[key][0].out
                            tgt_x = reg_layers[key][1].out
                            # print(src_x.shape, tgt_x.shape)
                            regloss = featloss(src_x, tgt_x.detach()).mean()
                            total_feat_reg[i] += regloss.item()
                    _, unweighted = ops.l2sp(model, 0)
                    try:
                        total_l2sp_reg += unweighted.item()
                    except:
                        pass
                phar.set_description(f"-> Eval {split} [{step}/{teststep}] Acc@1:{round(float(top1)/total*100, 3)}%")
                phar.update(1)
                if step > teststep:
                    break
        return float(top1)/total*100, total_ce/(step+1), np.sum(total_feat_reg)/(step+1), total_l2sp_reg/(step+1), total_feat_reg/(step+1)


    def get_fine_tuning_parameters(self):
        model = self.model
        parameters = []
        ft_begin_module = self.args.ft_begin_module
        ft_ratio = self.args.ft_ratio if 'ft_ratio' in self.args else None

        if ft_ratio:
            all_params = [param for param in model.parameters()]
            num_tune_params = int(len(all_params) * ft_ratio)
            for v in all_params[-num_tune_params:]:
                parameters.append({'params': v})

            all_names = [name for name, _ in model.named_parameters()]
            with open(osp.join(self.args.output_dir, "finetune.log"), "w") as f:
                f.write(f"Fixed layers:\n")
                for name in all_names[:-num_tune_params]:
                    f.write(name+"\n")
                f.write(f"\n\nFinetuned layers:\n")
                for name in all_names[-num_tune_params:]:
                    f.write(name+"\n")
            return parameters

        if not ft_begin_module:
            return model.parameters()

        add_flag = False
        for k, v in model.named_parameters():
            # if ft_begin_module == k:
            if ft_begin_module in k:
                add_flag = True

            if add_flag:
                # print(k)
                parameters.append({'params': v})
        if ft_begin_module and not add_flag:
            raise RuntimeError("wrong ft_begin_module, no module to finetune")

        return parameters
            
    def train(self):
        model = self.model
        train_loader = self.train_loader
        test_loader = self.test_loader
        iterations = self.args.iterations
        lr = self.args.lr
        output_dir = self.args.output_dir
        l2sp_lmda = self.args.l2sp_lmda
        teacher = self.teacher
        reg_layers = self.reg_layers
        args = self.args

        model_params = self.get_fine_tuning_parameters()
        if l2sp_lmda == 0:
            optimizer = torch.optim.SGD(
                model_params, 
                lr=lr * (1 + random.random()),
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model_params, 
                lr=lr, 
                momentum=args.momentum, 
                weight_decay=0,
            )
        end_iter = iterations
        if args.const_lr:
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                end_iter,
            )

        teacher.eval()
        ce = ops.CrossEntropyLabelSmooth(train_loader.dataset.num_classes, args.label_smoothing).to(self.device)
        featloss = torch.nn.MSELoss()
        batch_time = metric.MovingAverageMeter('Time', ':6.3f')
        data_time = metric.MovingAverageMeter('Data', ':6.3f')
        ce_loss_meter = metric.MovingAverageMeter('CE Loss', ':6.3f')
        feat_loss_meter  = metric.MovingAverageMeter('Feat. Loss', ':6.3f')
        l2sp_loss_meter  = metric.MovingAverageMeter('L2SP Loss', ':6.3f')
        linear_loss_meter  = metric.MovingAverageMeter('LinearL2 Loss', ':6.3f')
        total_loss_meter  = metric.MovingAverageMeter('Total Loss', ':6.3f')
        top1_meter  = metric.MovingAverageMeter('Acc@1', ':6.2f')

        train_path = osp.join(output_dir, "train.tsv")
        with open(train_path, 'a') as wf:
            columns = ['time', 'seed', 'iter', 'Acc', 'celoss', 'featloss', 'l2sp']
            wf.write('\t'.join(columns) + '\n')
        test_path = osp.join(output_dir, "test.tsv")
        with open(test_path, 'a') as wf:
            columns = ['time', 'seed', 'iter', 'Acc', 'celoss', 'featloss', 'l2sp']
            wf.write('\t'.join(columns) + '\n')
        adv_path = osp.join(output_dir, "adv.tsv")
        with open(adv_path, 'a') as wf:
            columns = ['time', 'seed', 'iter', 'Acc', 'AdvAcc', 'ASR']
            wf.write('\t'.join(columns) + '\n')
        
        dataloader_iterator = iter(train_loader)
        warmup_iter = [500, 1000]

        for i in range(iterations):
            model.train()
            optimizer.zero_grad()
            end = time.time()
            try:
                batch, label = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                batch, label = next(dataloader_iterator)
            batch, label = batch.to(self.device), label.to(self.device)
            data_time.update(time.time() - end)

            with torch.no_grad():
                teacher_out = self.teacher(batch)
            if args.steal:
                loss, top1, soft_loss, hard_loss = self.compute_steal_loss(batch, label, teacher_out=teacher_out)
                total_loss = loss
                ce_loss = hard_loss
                feat_loss = soft_loss
                linear_loss, l2sp_loss = 0, 0
            elif args.negative:
                loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss = self.compute_negative_loss(
                    batch, label, ce,
                )
            else:
                loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss = self.compute_loss(
                    batch, label,
                    ce, featloss,
                    teacher_out
                )
            top1_meter.update(top1)
            ce_loss_meter.update(ce_loss)
            feat_loss_meter.update(feat_loss)
            linear_loss_meter.update(linear_loss)
            l2sp_loss_meter.update(l2sp_loss)
            total_loss_meter.update(total_loss)
            loss.backward()

            #-----------------------------------------
            for k, m in enumerate(model.modules()):
                if isinstance(m, nn.Conv2d):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().to(self.device)
                    m.weight.grad.data.mul_(mask)
                if isinstance(m, nn.Linear):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().to(self.device)
                    m.weight.grad.data.mul_(mask)

            #-----------------------------------------
            optimizer.step()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            if scheduler is not None:
                scheduler.step()

            batch_time.update(time.time() - end)
            if (i % args.print_freq == 0) or (i == iterations-1):
                progress = metric.ProgressMeter(
                    iterations,
                    [batch_time, data_time, top1_meter, total_loss_meter, ce_loss_meter, feat_loss_meter, l2sp_loss_meter, linear_loss_meter],
                    prefix="\n-> {:s}_s{:d} LR={:6.5f}\n".format(args.task_str, args.seed, current_lr),
                    output_dir=output_dir,
                )
                progress.display(i)

            if False: #and (i % args.test_interval == 0 and i > 0) or (i == iterations-1) or (i in warmup_iter):
                if self.args.steal:
                    test_top1, test_ce_loss, test_feat_loss, test_weight_loss = self.steal_test(
                        # model, teacher, test_loader, teststep=500, loss=True
                    )
                    train_top1, train_ce_loss, train_feat_loss, train_weight_loss = self.steal_test(
                        # model, teacher, train_loader, teststep=500, loss=True
                    )
                    test_feat_layer_loss, train_feat_layer_loss = 0, 0
                else:
                    test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = self.test(
                        model, teacher, train_loader, teststep=500, loss=True, train=False
                    )
                    train_top1, train_ce_loss, train_feat_loss, train_weight_loss, train_feat_layer_loss = self.test(
                        model, teacher, test_loader, teststep=500, loss=True, train=True
                    )
                
                print(
                    '-> Eval Train | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i, iterations, train_top1, train_ce_loss, train_feat_loss, train_weight_loss))
                print(
                    '-> Eval Test | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i, iterations, test_top1, test_ce_loss, test_feat_loss, test_weight_loss))
                localtime = time.asctime( time.localtime(time.time()) )[4:-6]
                with open(train_path, 'a') as af:
                    train_cols = [
                        localtime,
                        args.seed,
                        i,
                        round(train_top1,2),
                        round(train_ce_loss,2), 
                        round(train_feat_loss,2),
                        round(train_weight_loss,2),
                    ]
                    af.write('\t'.join([str(c) for c in train_cols]) + '\n')
                with open(test_path, 'a') as af:
                    test_cols = [
                        localtime,
                        args.seed,
                        i,
                        round(test_top1,2), 
                        round(test_ce_loss,2), 
                        round(test_feat_loss,2),
                        round(test_weight_loss,2),
                    ]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')
                if not args.no_save:
                    ckpt_path = osp.join(
                        args.output_dir,
                        "ckpt.pth"
                    )
                    torch.save(
                        {
                            'state_dict': model.state_dict(),
                            'acc_top1': round(test_top1, 3),
                            'test_ce_loss': round(test_ce_loss, 3),
                            'test_feat_loss': round(test_feat_loss, 3),
                            'test_weight_loss': round(test_weight_loss, 3)
                        },
                        ckpt_path,
                    )
                if i == (iterations-1):
                    acc_path = osp.join(
                        args.output_dir,
                        f"final_{round(test_top1, 3)}.tsv"
                    )
                    torch.save({}, acc_path)

            if ( hasattr(self, "iterative_prune") and i % args.prune_interval == 0 ):
                self.iterative_prune(i)

        return model.to('cpu')

    def countWeightInfo(self):
        ...
