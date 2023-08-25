#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/31, homeway'


import os
import math
import argparse
import os.path as osp
import logging
import torch
import random, json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from attack import ops
from torch import optim
from utils import metric, helper, vis
from torchmetrics.functional import pairwise_cosine_similarity



class RemovalNet:
    def __init__(self, model_T, model_t, cfg, train_loader, test_loader):
        self.cfg = cfg
        self.seed = cfg.seed
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.torch_model_path = cfg.models_root
        self.logger = logging.getLogger('RemovalNet')
        for path in [cfg.exp_root, cfg.proj_root, cfg.out_root]:
            if not osp.exists(path):
                os.makedirs(path)

        self.learning_data = {
            "t": [],
            "acc": [],
            "loss_dist": [],
            "loss_kd": [],
            "loss_kl": [],
            "loss_ce": [],
            "keys": ["acc", "loss_dist", "loss_kl", "loss_ce"]
        }
        self.task = cfg.task
        self.model_T = model_T.to(self.device)
        self.model_t = model_t.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = test_loader.num_classes

    def save_torch_model(self, torch_model, step=0):
        if not osp.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)
        ckpt_path = osp.join(self.torch_model_path, f'final_ckpt_s{self.seed}_t{step}.pth')
        if step == 0:
            ckpt_path = osp.join(self.torch_model_path, f'final_ckpt_s{self.seed}.pth')
        torch.save(
            {'state_dict': torch_model.cpu().state_dict()},
            ckpt_path,
        )
        torch_model.to(self.device)
        self.logger.info(f"-> save model to: {self.torch_model_path}")

    @staticmethod
    def normalize(vs):
        return [(v - torch.min(v)) / (torch.max(v) - torch.min(v) + 1e-6) for v in vs]

    @staticmethod
    def distance_cost(a, b, ydist):
        batch_size = len(a)
        if ydist == "l2":
            loss_dist = torch.norm(a - b, p=2)
        elif ydist == "cosine":
            loss_dist = F.cosine_similarity(a.view(batch_size, -1), b.view(batch_size, -1), dim=1).sum()
        elif ydist == "kl":
            loss_dist = F.kl_div(a, b, reduction="sum")
        elif ydist == "angle":
            loss_dist = (a / a.norm() - b / b.norm()).norm()
        else:
            raise NotImplementedError()
        return loss_dist

    def shuffle_features(self, outputs, ratio=1.0):
        """
        Shuffle intermediate feature space
        Args:
            outputs: Tensor, layer outputs
            ratio: float, [0-1], shuffle ratio

        Returns: outputs
        """
        channels = outputs[0].shape[0]
        size = math.ceil(channels * ratio)
        for i, layer in enumerate(outputs):
            idx = np.arange(0, channels)
            np.random.shuffle(idx)
            idx_rnd = idx[:size]
            idx_ord = np.sort(idx_rnd)
            outputs[i][idx_ord] = layer[idx_rnd].clone()
        return outputs.detach()

    def feature_maximize(self, t, model, x, y, l, lr=0.01, shuffle_ratio=0.1, poison_steps=20, ydist="l2"):
        # z = x -> f(x)^{l}  // z is latent space, maximize z & z_prime
        z = ops.batch_fed_forward(model, x, layer_index=l, batch_size=self.batch_size).detach().contiguous().to(self.device)

        # random shuffle intermediate features
        z_prime = z.clone() + torch.tan(torch.rand(z.shape, device=self.device))
        self.shuffle_features(z_prime, ratio=shuffle_ratio)
        for step in range(poison_steps):
            z_prime = self.shuffle_features(z_prime, ratio=shuffle_ratio).detach()
            z_prime.requires_grad = True
            logit = model.mid_forward(z_prime, layer_index=l)

            # minimize CELoss & maximize cosine dist
            loss_dist = 0.1 * torch.log(self.distance_cost(z_prime, z, ydist))
            loss_ce = F.cross_entropy(logit, y)
            loss = loss_ce - loss_dist
            grad = torch.autograd.grad(loss, [z_prime], retain_graph=False, create_graph=False)[0]
            z_prime = z_prime - lr * grad.sign()
            z_prime = z_prime.detach()
        return z_prime.clone().detach()

    def logit_maximize(self, t, model, x, y, ydist="l2"):
        with torch.no_grad():
            logits = F.softmax(model(x), dim=1)
            batch_size = len(logits)
            if ydist == "l2":
                dist = torch.cdist(logits, logits)
                idxs = torch.argmax(dist, dim=1).tolist()
            elif ydist == "cosine":
                dist = pairwise_cosine_similarity(logits, logits)
                idxs = torch.argmin(dist, dim=1).tolist()
            else:
                raise NotImplementedError()
            beta_list = []
            logits_prime = logits.clone()
            h = lambda a, b, alpha: beta * a + (1 - beta) * b
            for i in range(batch_size):
                for beta in np.arange(0.3, 1, 0.1):
                    out = h(logits[i], logits[idxs[i]], beta)
                    if out.argmax(dim=0) == y[i]:
                        logits_prime[i] = out
                        beta_list.append(round(beta, 2))
                        break
            return logits_prime.detach()

    def deepremoval_step(self, model_T, model_t, optimizer, x, y, l, t=0):
        cfg = self.cfg
        model_T.eval()
        model_t.eval()

        # remove feature-level fingerprints
        feats_prime = self.feature_maximize(t, model=model_t, x=x, y=y, l=l, ydist=self.cfg.ydist, shuffle_ratio=self.cfg.shuffle_ratio, poison_steps=self.cfg.poison_steps).detach()
        # remove logit-level fingerprints
        logit_prime = self.logit_maximize(t, model=model_t, x=x, y=y, ydist=self.cfg.ydist).detach()

        model_t.train()
        optimizer.zero_grad()
        logit = model_t(x)

        feats = model_t.fed_forward(x, layer_index=l)
        loss_dist = F.mse_loss(feats, feats_prime, reduction="mean")
        loss_kl = cfg.T * cfg.T * F.kl_div(
                F.log_softmax(logit / self.cfg.T, dim=1),
                F.softmax(logit_prime / self.cfg.T, dim=1), reduction='batchmean')
        loss_ce = F.cross_entropy(logit, y)

        loss = cfg.alpha * loss_kl + self.cfg.beta * loss_dist + cfg.gamma * loss_ce
        loss.backward()
        optimizer.step()
        return loss, loss_dist, loss_kl, loss_ce

    def deepremoval(self):
        cfg = self.cfg
        iterations = cfg.iterations + 1

        optimizer = optim.SGD(
            self.model_t.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            int(cfg.iterations * 1.5),
        )

        # prepare LayerCAM preview image
        eval_batch = []
        def get_eval_batch(model, loader, device, preview_off=30, preview_size=30):
            eval_x, eval_y = next(iter(loader))
            eval_x = eval_x[preview_off:preview_off+preview_size]
            mean, std = self.test_loader.mean, self.test_loader.std
            eval_ori_x = self.test_loader.unnormalize(eval_x, mean, std, clamp=True)[:preview_size]
            with torch.no_grad():
                eval_y = model(eval_x.to(device)).argmax(dim=1).detach().cpu()
                return [eval_x, eval_y, eval_ori_x]
        eval_batch.append(get_eval_batch(model=self.model_T, loader=self.train_loader, device=self.device))
        eval_batch.append(get_eval_batch(model=self.model_T, loader=self.test_loader, device=self.device))

        for step in range(0, 1+iterations):
            try:
                batch, label = next(loader)
            except:
                loader = iter(self.train_loader)
                batch, label = next(loader)
            x, y = batch.to(self.device), label.to(self.device)
            y = self.model_T(x).argmax(dim=1).long().detach()

            l = int(self.cfg.layer)
            loss, loss_dist, loss_kl, loss_ce = self.deepremoval_step(self.model_T, self.model_t, optimizer, x=x, y=y, l=l, t=step)
            scheduler.step()

            # testing & visualization
            if step == 0 or step % cfg.test_interval == 0 or (step == iterations - 1):
                _best_topk_acc, topk_acc, test_loss = metric.topk_test(self.model_t, self.test_loader, device=self.device, epoch=step, debug=True)
                self.learning_data["t"].append(step)
                self.learning_data["acc"].append(topk_acc["top1"])
                self.learning_data["loss_dist"].append(ops.numpy(loss_dist))
                self.learning_data["loss_kl"].append(ops.numpy(loss_kl))
                self.learning_data["loss_ce"].append(ops.numpy(loss_ce))
                vis.view_learning_state(self.learning_data, file_path=osp.join(self.cfg.exp_root, "LR"))
                torch.save(self.learning_data, osp.join(self.cfg.exp_root, f"learning_state.pt"))
        self.save_torch_model(self.model_t.cpu())


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", help="model 1.")
    parser.add_argument("-subset", type=str, required=True, help="substitute dataset of removalnet")
    parser.add_argument("-subset_ratio", type=float, required=True, help="substitute dataset rate of removalnet")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-ydist",  default="l2", type=str, choices=["l2", "cosine", "kl"], help="distance of adv logits")
    parser.add_argument("-layer", default=4, type=int, choices=[1, 2, 3, 4, 5], help="distance of adv logits")
    parser.add_argument("-batch_size", action="store", default=100, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-tag", type=str, required=True, help="Some words on this work")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.format_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.proj_root = osp.join(args.out_root, "Removal")

    args.alpha = 0.05
    args.T = 10
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.shuffle_ratio = 0.02

    args.layers = ["layer1", "layer2", "layer3", "layer4", "layer5"]
    if "CIFAR10" in args.model1:
        args.T = 20
        args.alpha = 0.2
        args.beta = 2.0
        args.gamma = 0.6
        args.lr = 8e-3
        args.poison_steps = 20
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000

    elif "CINIC10" in args.model1:
        args.T = 20
        args.alpha = 0.2
        args.beta = 2.0
        args.lr = 6e-3
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000
        if "mobile" in args.model1 or "densenet" in args.model1:
            args.lr /= 4.0
            args.shuffle_ratio = 0.1

    elif "CelebA" in args.model1:
        args.T = 20
        args.alpha = 0.2
        args.beta = 2.0
        args.gamma = 0.8
        args.lr = 1e-2
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000

    elif "HAM10000" in args.model1:
        args.T = 20
        args.alpha = 0.2
        args.beta = 2.0
        args.gamma = 0.8
        args.lr = 1e-2
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000

    if "BCN20000" in args.subset:
        args.T = 20
        args.alpha = 0.22
        args.beta = 2.0
        args.gamma = 0.8
        args.lr = 6e-3
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000

    if "VGGFace" in args.subset:
        args.lr = 6e-3
        args.alpha = 0.2
        args.beta = 2.0
        args.T = 20
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000

    if "LFW" in args.subset:
        args.lr = 3e-3
        args.alpha = 0.2
        args.beta = 2.0
        args.T = 20
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000

    elif "ImageNet" in args.model1:
        args.lr = 1e-2
        args.alpha = 0.4
        args.beta = 4.0
        args.gamma = 0.8
        args.T = 20
        args.shuffle_ratio = 0.2
        args.poison_steps = 30
        args.test_interval = 50
        args.plot_interval = 200
        args.iterations = 2000

    if "GTSRB" in args.model1:
        args.T = 10
        args.alpha = 0.2
        args.beta = 2.0
        args.gamma = 0.8
        args.lr = 1e-2
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000

    elif "TSRD" in args.model1:
        args.T = 10
        args.alpha = 0.2
        args.beta = 1.0
        args.gamma = 0.8
        args.lr = 1e-3
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000


    if "densenet" in args.model1:
        args.layers = ["features.pool0", "features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"]
    elif "mobile" in args.model1:
        args.layers = ["features.3", "features.6", "features.10", "features.16", "features.18"]
    elif "resnet" in args.model1:
        args.layers = ["layer1", "layer2", "layer3", "layer4", "layer3"]
    elif "vit" in args.model1.lower():
        args.layers = ["blocks", "blocks", "blocks", "blocks", "blocks"]
    elif "inception" in args.model1.lower():
        args.layers = ["Conv2d_1a_3x3", "Mixed_5d", "Mixed_6a", "Mixed_6e", "Mixed_7c"]


    helper.set_default_seed(args.seed)
    if args.subset_ratio < 0.9:
        args.lr = args.lr * (1.0 - args.subset_ratio)
    else:
        args.lr = args.lr * (0.9 + 0.2 * random.random())

    if round(args.subset_ratio, 2) - round(args.subset_ratio, 1) != 0:
        args.subset_ratio = round(args.subset_ratio, 2)
    else:
        args.subset_ratio = round(args.subset_ratio, 1)
    return args


def main():
    import benchmark
    from dataset import loader

    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    arch, dataset = args.model1.split("(")[1].split(")")[0].split(",")
    benchmk = benchmark.ImageBenchmark(datasets=dataset, archs=[arch], datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model0 = benchmk.load_wrapper(args.model1).load_torch_model()
    model1 = benchmk.load_wrapper(args.model1).load_torch_model() # a deepcopy of model1

    # substitute dataset to train model
    train_loader = loader.get_dataloader(args.subset, split="train", batch_size=args.batch_size, subset_ratio=args.subset_ratio)
    test_loader = loader.get_dataloader(dataset, split="test", shuffle=False, batch_size=args.batch_size)
    removalnet = f"removalnet({args.subset},{args.subset_ratio},{args.alpha},{args.beta},{args.T},l{args.layer})-"

    args.task = model1.task
    args.attack_name = f"{str(model1.task)}{removalnet}"
    args.models_root = osp.join(args.models_dir, f'{args.attack_name}')
    args.exp_root = osp.join(helper.ROOT, "output/Removal/exp", args.attack_name)

    removal = RemovalNet(model0, model1, args, train_loader=train_loader, test_loader=test_loader)
    data = {
        "tag": args.tag,
        "poison_steps": args.poison_steps,
        "plot_interval": args.plot_interval,
        "test_interval": args.test_interval,
        "arch": arch,
        "dataset": dataset,
        "subset": args.subset,
        "lr": args.lr,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "layers": args.layers,
        "namespace": str(args.namespace)
    }
    for k, v in vars(args).items():
        if type in [list, int, str]:
            data[k] = v
    with open(osp.join(args.exp_root, "conf.json"), "w") as f:
        json.dump(data, f)
    removal.deepremoval()


if __name__ == "__main__":
    main()
    """
        Example command:
        <===========================  train(vgg19_bn,CIFAR10)-  ===========================>
        python -m attack.RemovalNet.removalnet -model1 "train(vgg19_bn,CIFAR10)-" -subset CIFAR10 -subset_ratio 1.0 -layer 2 -batch_size 128 -device 0 -tag ''
    """


















