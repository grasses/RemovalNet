#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/23, homeway'


import random
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torchattacks.attack import Attack


class Adv(Attack):
    def __init__(self, model, bounds):
        super().__init__("Adv", model)
        self.bounds = bounds
        self._supported_mode = ['default', 'targeted']

    def IPGuard(self, images, labels, k, targeted, steps=1000, lr=1e-2):
        assert targeted in ["L", "R"]
        batch_x = images.clone().detach().to(self.device)

        # We find k should be very small, e.g., k=0.01, since logit is always small than 10
        k = Variable(torch.Tensor([k]), requires_grad=True)[0].detach()

        adv_x = []
        self.model.eval()
        batch_size = len(labels)
        phar = tqdm(range(batch_size))
        ReLU = torch.nn.ReLU()
        for idx in phar:
            x = batch_x[[idx]].clone()
            z = self.model(x)
            i = z.argmax(dim=1)[0]
            if targeted == "L":       # least-like
                j = z.argmin(dim=0)[0]
            else:                     # random
                ll = list(range(z.shape[1]))
                ll.remove(int(i))
                j = random.choice(ll)

            for step in range(steps):
                x.requires_grad = True
                z = self.model(x)
                z[0][i] = -1000
                z[0][j] = -1000
                t = z.argmax(dim=1)[0]

                z = self.model(x)[0]
                cost = ReLU(z[i] - z[j] + k) + ReLU(z[t] - z[i])
                grad = torch.autograd.grad(cost, x)[0]
                x = (x - lr * grad.sign()).detach()
                phar.set_description(
                    f"-> [IPGuard] idx{idx}-step{step} i:{int(i)} j:{int(j)} t:{int(t)} "
                    f"z_i:{round(float(self.model(x)[0][i]), 6)} z_j:{round(float(self.model(x)[0][j]), 6)} cost:{cost}")

                # ReLU(z_i - z_j + k) ≈ 0
                if cost <= 1e-4:
                    break

            z = self.model(x)[0]
            print(f"-> max_logit={round(float(torch.max(z)), 5)}, "
                  f"z_j={round(float(z[j]), 4)} ≥ z_i={round(float(z[i]), 4)} + {k} \n")
            adv_x.append(x.detach().cpu())
        batch_x = torch.cat(adv_x).to(self.device)
        batch_y = self.model(batch_x).argmax(dim=1)
        return batch_x.cpu().detach(), batch_y.cpu().detach()
















