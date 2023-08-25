#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/23, homeway'


import math
import torch
from tqdm import tqdm
from torchattacks.attack import Attack
from . import BaseSeeding
from .. import ops


class BlackboxSeeding(Attack, BaseSeeding):
    """code from: https://github.com/Harry24k/adversarial-attacks-pytorch"""
    def __init__(self, model, task, test_loader, dataset, batch_size, out_root):
        Attack.__init__(self, "BlackboxSeeding", model)
        BaseSeeding.__init__(self, model, task, test_loader, dataset, out_root)
        self.batch_size = batch_size
        self.bounds = test_loader.bounds
        self._supported_mode = ['default', 'targeted']

    def generate(self, seed_x, seed_y, method="FGSM"):
        """
        Generate blackbox testing samples
        :param method: [FGSM, PGD, CW, Random]
        :param batch_size: 200
        :return:
        """
        self.logger.info(f"-> generate blackbox test samples...seed size:{len(seed_x)}")
        cache = self.load_test_samples(tag=f"blackbox-{method}")
        if cache is not None:
            return cache["test_x"], cache["test_y"]

        if method == "FGSM":
            obj = self.FGSM
        elif method == "PGD":
            obj = self.PGD
        elif method == "CW":
            obj = self.CW
        elif method == "IPGuard":
            obj = self.IPGuard
        else:
            obj = self.Random

        test_x, test_y = [], []
        steps = math.ceil(len(seed_x) / self.batch_size)
        for step in range(steps):
            off = int(step * self.batch_size)
            x = seed_x[off: off + self.batch_size].to(self.device)
            y = seed_y[off: off + self.batch_size].to(self.device)
            _x, _y = obj(x, y)
            test_x.append(_x.cpu().detach())
            test_y.append(_y.cpu().detach())
        test_x = torch.cat(test_x)
        test_y = torch.cat(test_y)
        assert test_x.shape[0] > 0
        self.save_test_samples(seed_x=seed_x, seed_y=seed_y, test_x=test_x, test_y=test_y, tag=f"blackbox-{method}")
        self.logger.info(f"-> generate blackbox test samples done! size:{len(test_y)}")
        return test_x, test_y

    def Random(self, images, labels):
        images = images.clone().detach().to(self.device)
        with torch.no_grad():
            labels = self.model(images).detach().to(self.device)
            return images, labels

    def FGSM(self, images, labels, eps=0.05):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        loss = torch.nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.model(images)
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
        return self.targeted_samples(x=adv_images, y=labels, targeted=False)

    def PGD(self, images, labels, eps=0.1, alpha=10.0 / 255.0, steps=40, random_start=True, _targeted=False):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if _targeted:
            target_labels = self._get_target_label(images, labels)
        loss = torch.nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        if random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
        for _ in range(steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            if _targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=self.bounds[0], max=self.bounds[1]).detach()
        return self.targeted_samples(x=adv_images, y=labels, targeted=False)

    def IPGuard(self, images, labels, steps=1000, lr=0.01, k=0.1, target="least_likely"):
        assert target in ["least_likely", "random"]
        adv_images = images.clone().detach().to(self.device)
        true_labels = labels.clone().detach().to(self.device)
        with torch.no_grad():
            if target == "least_likely":
                target_labels = self.model(adv_images).argmin(dim=1)
            else:
                target_labels = torch.randint(0, torch.max(true_labels), labels.shape)

        relu = torch.nn.ReLU()
        init_labels = self.model(adv_images).argmax(dim=1).detach().to(self.device)
        target_labels = target_labels.clone().detach().to(self.device)
        phar = tqdm(range(target_labels.size(0)))
        cost = 0
        for idx in phar:
            adv_images = adv_images.detach().to(self.device)
            adv_image = adv_images[idx].clone().unsqueeze(0)
            for _ in range(steps):
                adv_image.requires_grad = True
                outs = self.model(adv_image)[0]
                pred_label = outs.argmax(dim=0)
                phar.set_description(
                    f"-> [IPGuard] idx:{idx} target:{target_labels[idx]} init:{init_labels[idx]} pred:{pred_label} cost:{cost}")
                if pred_label == target_labels[idx]: break
                cost = -relu(outs[int(target_labels[idx])] - outs[int(init_labels[idx])] + k) \
                       - relu(outs[int(pred_label)] - outs[int(target_labels[idx])])
                grad = torch.autograd.grad(cost, adv_image, retain_graph=False, create_graph=False)[0]
                adv_image = (adv_image + lr * grad.sign())
                adv_image = torch.clamp(adv_image, min=self.bounds[0], max=self.bounds[1]).detach().to(self.device)
            adv_images[idx] = adv_image.squeeze(0)
        adv_x = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach().to(self.device)
        return self.targeted_samples(x=adv_x, y=target_labels, targeted=True)

    def CW(self, images, labels, c=1e-4, kappa=0, steps=1000, lr=0.001):
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)
        MSELoss = torch.nn.MSELoss(reduction='none')
        Flatten = torch.nn.Flatten()
        optimizer = torch.optim.Adam([w], lr=self.lr)
        for step in range(self.steps):
            adv_images = self.tanh_space(w)
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.model(adv_images)
            if self._targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()
            cost = L2_loss + self.c * f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()
            # filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        with torch.no_grad():
            pred = self.model(best_adv_images)
            y_pred = pred.argmax(dim=1).detach().cpu()
            y_true = labels.cpu()
            success_idx = torch.where(y_pred != y_true)[0]
        return best_adv_images[success_idx].detach(), pred[success_idx].detach()

    def targeted_samples(self, x, y, targeted=False):
        with torch.no_grad():
            pred = self.model(x)
            y_pred = pred.argmax(dim=1).detach().cpu()
            y_target = y.detach().cpu()
            if not targeted:
                success_idx = torch.where(y_pred != y_target)[0]
            else:
                success_idx = torch.where(y_pred == y_target)[0]
        self.logger.info(f"-> success:{len(success_idx)}")
        return x[success_idx].detach(), pred[success_idx].detach()

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
        if self._targeted:
            return torch.clamp((i - j), min=-self.kappa)
        else:
            return torch.clamp((j - i), min=-self.kappa)