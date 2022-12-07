#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/23, homeway'


import random
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torchattacks.attack import Attack
from . import ops


class Adv(Attack):
    def __init__(self, model, bounds):
        super().__init__("Adv", model)
        self.bounds = bounds
        self._supported_mode = ['default', 'targeted']

    def IPGuard(self, images, labels, k, targeted, steps=1000, lr=1e-2):
        assert targeted in ["L", "R"]
        batch_x = images.clone().detach()

        # We find k should be very small, e.g., k=0.01, since logit is always small than 10
        k = Variable(torch.Tensor([k]), requires_grad=True)[0].detach()

        adv_x = []
        self.model.eval()
        batch_size = len(labels)
        phar = tqdm(range(batch_size))
        ReLU = torch.nn.ReLU()

        for idx in phar:
            x = batch_x[[idx]].clone().to(self.device)
            z = self.model(x)
            i = z.argmax(dim=1)[0]
            if targeted == "L":       # least-like
                j = z.argmin(dim=1)[0]
            else:                     # random
                ll = list(range(z.shape[1]))
                ll.remove(int(i))
                j = random.choice(ll)

            for step in range(steps):
                x = x.detach()
                x.requires_grad = True
                optimizer = torch.optim.Adam([x], lr=lr)
                optimizer.zero_grad()

                if z.shape[1] > 2:
                    z = self.model(x)
                    z[0][i] = -1000
                    z[0][j] = -1000
                    t = z.argmax(dim=1)[0]
                    z = self.model(x)
                    loss = ReLU(z[0][i] - z[0][j] + k) + ReLU(z[0][t] - z[0][i])
                else:
                    # compatible binary classifier
                    z = self.model(x)
                    loss = ReLU(z[0][i] - z[0][j] + k)
                    t = j
                loss.backward()
                optimizer.step()
                phar.set_description(
                    f"-> [IPGuard] idx{idx}-step{step} i:{int(i)} j:{int(j)} t:{int(t)} "
                    f"z_i:{round(float(self.model(x)[0][i]), 4)} z_j:{round(float(self.model(x)[0][j]), 4)} loss:{round(float(loss.data), 4)}")

                # loss ≈ 0
                if loss <= 1e-5:
                    break

            z = self.model(x)[0]
            print(f"-> max_logit={round(float(torch.max(z)), 5)}, "
                  f"z_j={round(float(z[j]), 4)} ≥ z_i={round(float(z[i]), 4)} + {k} \n")
            adv_x.append(x.detach().cpu())

        batch_x = torch.cat(adv_x).to(self.device)
        batch_y = ops.batch_forward(self.model, batch_x, batch_size=200, argmax=True)
        return batch_x.cpu().detach(), batch_y.cpu().detach()
















