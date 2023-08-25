#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


"""
From https://github.com/Harry24k/adversarial-attacks-pytorch
"""


import os
import torch
import torch.nn


def fgsm(model, x, y, bounds, eps=5./255, targeted=False):
    device = next(model.parameters()).device
    images = x.clone().detach().to(device)
    labels = y.clone().detach().to(device)
    loss = torch.nn.CrossEntropyLoss()
    images.requires_grad = True
    outputs = model(images)
    # Calculate loss
    if targeted:
        cost = -loss(outputs, labels)
    else:
        cost = loss(outputs, labels)

    # Update adversarial images
    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=bounds[0], max=bounds[1]).detach()
    return adv_images


def pgd(model, x, y, bounds, eps=5./255., alpha=2./255., steps=10, targeted=False, random_start=True):
    device = next(model.parameters()).device
    images = x.clone().detach().to(device)
    labels = y.clone().detach().to(device)

    loss = torch.nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=bounds[0], max=bounds[1]).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        # Calculate loss
        if targeted:
            cost = -loss(outputs, labels)
        else:
            cost = loss(outputs, labels)
        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=bounds[0], max=bounds[1]).detach()
    return adv_images


def deepfool(model, x, y, bounds, steps=40, overshoot=0.02, return_target_labels=False):
    def _forward_indiv(model, image, label):
        image.requires_grad = True
        fs = model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = _construct_jacobian(fs, image)
        image = image.detach()
        f_0 = fs[label]
        w_0 = ws[label]
        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(torch.nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L]) * w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2) ** 2))
        target_label = hat_L if hat_L < label else hat_L + 1
        adv_image = image + (1 + overshoot) * delta
        adv_image = torch.clamp(adv_image, min=bounds[0], max=bounds[1]).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)

    device = next(model.parameters()).device
    images = x.clone().detach().to(device)
    labels = y.clone().detach().to(device)
    batch_size = len(images)
    correct = torch.tensor([True] * batch_size)
    target_labels = labels.clone().detach().to(device)
    curr_steps = 0
    adv_images = []
    for idx in range(batch_size):
        image = images[idx:idx + 1].clone().detach()
        adv_images.append(image)

    while (True in correct) and (curr_steps < steps):
        for idx in range(batch_size):
            if not correct[idx]: continue
            early_stop, pre, adv_image = _forward_indiv(model, adv_images[idx], labels[idx])
            adv_images[idx] = adv_image
            target_labels[idx] = pre
            if early_stop:
                correct[idx] = False
        curr_steps += 1
    adv_images = torch.cat(adv_images).detach()
    if return_target_labels:
        return adv_images, target_labels
    return adv_images


def cw(model, x, y, c=1, kappa=0, steps=50, lr=0.01, targeted=False):
    def tanh_space(x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(x):
        # torch.atanh is only for torch >= 1.7.0
        return atanh(x * 2 - 1)

    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

        # f-function in the paper

    def f(outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0])).to(device)[labels]

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit

        if targeted:
            return torch.clamp((i - j), min=-kappa)
        else:
            return torch.clamp((j - i), min=-kappa)

    device = next(model.parameters()).device
    images = x.clone().detach().to(device)
    labels = y.clone().detach().to(device)

    # w = torch.zeros_like(images).detach() # Requires 2x times
    w = inverse_tanh_space(images).detach()
    w.requires_grad = True

    best_adv_images = images.clone().detach()
    best_L2 = 1e10 * torch.ones((len(images))).to(device)
    prev_cost = 1e10
    dim = len(images.shape)

    MSELoss = torch.nn.MSELoss(reduction='none')
    Flatten = torch.nn.Flatten()

    optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(steps):
        # Get adversarial images
        adv_images = tanh_space(w)

        # Calculate loss
        current_L2 = MSELoss(Flatten(adv_images),
                             Flatten(images)).sum(dim=1)
        L2_loss = current_L2.sum()

        outputs = model(adv_images)
        if targeted:
            f_loss = f(outputs, labels).sum()
        else:
            f_loss = f(outputs, labels).sum()

        cost = L2_loss + c * f_loss

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Update adversarial images
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
        if step % max(steps // 10, 1) == 0:
            if cost.item() > prev_cost:
                return best_adv_images
            prev_cost = cost.item()

    return best_adv_images
