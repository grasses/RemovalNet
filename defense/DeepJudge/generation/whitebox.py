#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/22, homeway'


import numpy as np
import math
import torch
from tqdm import tqdm
from . import BaseSeeding


class WhiteboxSeeding(BaseSeeding):
    def __init__(self, model, task, test_loader, dataset, batch_size, out_root):
        BaseSeeding.__init__(self, model, task, test_loader, dataset, out_root)
        self.batch_size = batch_size
        self.bounds = test_loader.bounds

    @staticmethod
    def getKs(outputs, x, times):
        """ calculate threshold k for each neuron
        """
        shape = (outputs.shape[0], -1, outputs.shape[-1])
        outputs = torch.mean(torch.reshape(outputs, shape), dim=1).detach().cpu().numpy()
        layer_maxs = np.max(outputs, axis=0)
        return (times * layer_maxs)

    def generate(self, seed_x, seed_y, layer_index, m, iters=1000, alpha=0.2, lr=0.1, target_idx=None):
        """
        args:
            seeds: seeds for the generation
            layer_index: target layer index
            m: hyper-parameter
            iters: iteration budget
            step: optimization step
            target_idx: target neuron (optional)
            X: training data (optional)

        return:
            a dictionary of generated test cases {(layer_index, neuron_index): [test cases...]}
        """
        self.logger.info(f"-> generate whitebox test samples... layer_index:{layer_index}")
        data = self.load_test_samples(tag=f"whitebox-L{layer_index}_m{round(float(m), 1)}")
        if data is not None:
            return data["test_x"]

        steps = math.ceil(len(seed_x) / self.batch_size)
        outputs = []
        for step in range(steps):
            off = (step * self.batch_size)
            batch_x = seed_x[off: off + self.batch_size].clone().to(self.device)
            batch_out = self.model.fed_forward(batch_x, layer_index=layer_index).detach().cpu()
            outputs.append(batch_out.clone())
        outputs = torch.cat(outputs)
        num_neurons = min(outputs.shape[-1], 32)
        Ks = WhiteboxSeeding.getKs(outputs, seed_x, m)
        del outputs
        torch.cuda.empty_cache()

        if target_idx is None:
            neurons_idxs = list(range(num_neurons))
        else:
            neurons_idxs = [target_idx]
        tests = {}
        for idx in neurons_idxs:
            tests[(layer_index, idx)] = []

        self.model.eval()
        phar = tqdm(range(len(seed_x)))
        for sample_idx in phar:
            x = seed_x[[sample_idx]].detach().to(self.device)
            for idx in neurons_idxs:
                k = Ks[idx]
                x_ = torch.autograd.Variable(x).clone().to(self.device)
                optimizer = torch.optim.Adam([x_], lr=lr)
                t_cost, t_neurons = [], []
                for iter in range(iters):
                    x_.requires_grad = True
                    out = self.model.fed_forward(x_, layer_index=layer_index)
                    optimizer.zero_grad()
                    cost = -torch.mean(out.view(1, -1, num_neurons), dim=1)[0][idx]
                    cost.backward()
                    optimizer.step()

                    t_cost.append(-cost.item())
                    t_neurons.append(x_.detach().cpu())
                    if -cost.item() > k + alpha:
                        tests[(layer_index, idx)].append(x_.detach().cpu())
                        phar.set_description(f"-> Whitebox-L{layer_index} [{sample_idx + 1}/{len(seed_x)}], iter:{iter} neurons:{idx} cost:{-cost.item()} > k:{k+alpha}")
                        break

                if len(tests[(layer_index, idx)]) == 0:
                    _, top = torch.topk(torch.tensor(t_cost), k=1)
                    print(f"-> top:{top} largest_cost:{_} for layer:{layer_index} neurons:{idx}")
                    tests[(layer_index, idx)].append(t_neurons[top[0]].detach().cpu())

        for idx in neurons_idxs:
            sample = torch.cat(tests[(layer_index, idx)])
            tests[(layer_index, idx)] = sample

        min_size = min([len(v) for v in tests.values()])
        for k, v in tests.items():
            tests[k] = v[:min_size]

        self.save_test_samples(seed_x=seed_x, seed_y=seed_y, test_x=tests, test_y=[], tag=f"whitebox-L{layer_index}_m{round(float(m), 1)}")
        self.logger.info(f"-> generate whitebox test samples done! size:{len(tests)}")
        return tests
