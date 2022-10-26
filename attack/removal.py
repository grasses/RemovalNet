#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/31, homeway'


import os, pytz
import argparse
import os.path as osp
import logging
import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
from attack import ops
from torch import optim
from utils import metric, helper, vis as exp_F
from torchmetrics.functional import pairwise_cosine_similarity
format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S"))
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class DeepRemoval:
    def __init__(self, model_0, model_t, cfg, train_loader, test_loader):
        self.cfg = cfg
        self.seed = cfg.seed
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.torch_model_path = cfg.models_dir
        self.logger = logging.getLogger('RemovalNet')
        self.output_dir = osp.join(helper.ROOT, "output/Removal", str(cfg.task))
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.learning_data = {
            "t": [],
            "acc": [],
            "loss_dist": [],
            "loss_kd": [],
            "keys": ["acc", "loss_dist", "loss_kd"]
        }
        self.task = cfg.task
        self.model_0 = model_0.to(self.device)
        self.model_t = model_t.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

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

    def normalize(self, vs):
        return [(v - torch.min(v)) / (torch.max(v) - torch.min(v) + 1e-6) for v in vs]

    def feature_poison(self, model, x, y, k=2, lr=0.1, alpha=0.2, poison_steps=20, theta=0.8, ydist="l2"):
        # z = x -> f(x)^{l}  // z is latent space, maximize z & z_prime
        z = ops.batch_fed_forward(model, x, layer_index=k, batch_size=self.batch_size).detach().contiguous().to(self.device)
        z_prime = z.clone() + torch.empty_like(z).uniform_(-1, 1)
        normalized_z = self.normalize(z)

        batch_size = len(z)
        for step in range(poison_steps):
            z_prime = z_prime.detach()
            z_prime.requires_grad = True
            logit_t = model.mid_forward(z_prime, layer_index=k)
            if ydist == "l2":
                # (0, 2), ->0 similar; ->2 dissimilar
                u = z_prime / z_prime.norm()
                v = z / z.norm()
                loss_dist = (u - v).norm()
            elif ydist == "cosine":
                # -cosine ∈ (-1, 1), cosine->1 dissimilar; cosine->-1 similar
                loss_dist = -F.cosine_similarity(z_prime.view(batch_size, -1), z.view(batch_size, -1), dim=1).mean()
            elif ydist == "kl":
                # -kl_loss < 0, kl_loss->0 similar
                loss_dist = -F.kl_div(z, z_prime, reduction="mean")
            else:
                raise NotImplementedError()
            # minimize CELoss & maximize cosine dist
            loss_ce = F.cross_entropy(logit_t, y)
            loss = alpha * loss_ce - (1 - alpha) * loss_dist
            grad = torch.autograd.grad(loss, [z_prime], retain_graph=False, create_graph=False)[0]
            z_prime = z_prime.detach() - lr * grad.sign()
            for idx in range(z.shape[0]):
                z_prime[idx][normalized_z[idx] > theta] = 0.0
        print(f"-> feature_poison() ce_loss:{loss_ce.item()} {ydist}_dist:{loss_dist.item()}")
        return z_prime.clone().detach()

    def boundary_poison(self, model, x, y, ydist="l2"):
        with torch.no_grad():
            logits = model(x).detach()
            batch_size = len(logits)
            if ydist == "l2":
                dist = torch.cdist(logits, logits)
                idxs = torch.argmax(dist, dim=1).tolist()
            elif ydist == "cosine":
                dist = pairwise_cosine_similarity(logits, logits)
                idxs = torch.argmin(dist, dim=1).tolist()

            beta_list = []
            logits_prime = logits.clone()
            h = lambda a, b, alpha: beta * a + (1 - beta) * b
            for i in range(batch_size):
                for beta in np.arange(0, 1, 0.1):
                    out = h(logits[i], logits[idxs[i]], beta)
                    if out.argmax(dim=0) == y[i]:
                        logits_prime[i] = out
                        beta_list.append(round(beta, 1))
                        break
            norm = (logits_prime-logits).norm(p=2).mean().detach().cpu()
            print(f"-> beta:{beta_list[:8]} norm:{norm}")
            return logits_prime.detach()

    def deepremoval(self):
        fig_path = os.path.join(self.output_dir, f"{self.cfg.ydist}_d{self.train_loader.dataset_id}")
        cfg = self.cfg
        cfg.lr = 3e-3
        cfg.poison_steps = 20
        cfg.test_interval = 10
        iterations = cfg.iterations + 1

        model_0 = self.model_0
        model_t = self.model_t
        optimizer = optim.SGD(
            model_t.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )

        loader = iter(self.train_loader)
        val_x, val_y = next(loader)
        val_x, val_y = val_x.to(self.device), val_y.to(self.device)
        for step in range(1, iterations+1):
            try:
                batch, label = next(loader)
            except:
                loader = iter(self.train_loader)
                batch, label = next(loader)
            x, y = batch.to(self.device), label.to(self.device)
            k = random.randint(2, 4)

            model_0.eval()
            model_t.eval()
            logit_0 = model_0(x).detach()
            labels = logit_0.argmax(dim=1).detach()

            # remove feature-level fingerprints
            feats_prime = self.feature_poison(model=model_t, x=x, y=labels, ydist=cfg.ydist, k=k, poison_steps=cfg.poison_steps).detach()
            # remove logit-level fingerprints
            logit_prime = self.boundary_poison(model=model_t, x=x, y=labels, ydist=cfg.ydist).detach()

            model_t.train()
            optimizer.zero_grad()
            feats = model_t.fed_forward(x, layer_index=k)
            loss_dist = F.mse_loss(feats, feats_prime, reduction="mean")
            loss_kd = ops.loss_kd(logit=model_t(x), labels=labels, teacher_logit=logit_prime, alpha=cfg.alpha, T=cfg.T)
            loss = loss_dist + loss_kd
            loss.backward()
            optimizer.step()

            if (step % cfg.test_interval == 0):
                acc1, acc2 = exp_F.plot_logist_embedding(model_0, model_t, self.test_loader, out_root=self.output_dir, file_name=f"RemovalNet_{cfg.ydist}_{step}.pdf")
                print(f"-> For T:{step} [Train] model_0:{acc1}% model_t:{acc2}% loss:{loss.item()} loss_dist:{loss_dist.item()} loss_kd:{loss_kd.item()}")
                self.save_torch_model(model_t.cpu(), step=step)

            # testing & exp visualization
            if step % 1 == 0 or (step == iterations - 1):
                _best_topk_acc, topk_acc, test_loss = metric.topk_test(model_t, self.test_loader, device=self.device, epoch=step, debug=True)
                self.learning_data["t"].append(step)
                self.learning_data["acc"].append(topk_acc["top1"])
                self.learning_data["loss_dist"].append(ops.numpy(loss_dist))
                self.learning_data["loss_kd"].append(ops.numpy(loss_kd))
                exp_F.plot_learning_curve(self.learning_data, file_path=fig_path + "accuracy")

            if step % cfg.test_interval == 0:
                for k in [1, 2, 3, 4]:
                    fig_path_cam = fig_path + f"_LayerCam_layer{k}_t{step}"
                    exp_F.plot_layer_cam(model_0, model_t, x=val_x, y=val_y, target_layer=f"layer{k}", fig_path=fig_path_cam, device=self.device)
                self.save_torch_model(model_t.cpu(), step=10*int(step/10))
            print()
        print()


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", required=True, help="model 1.")
    parser.add_argument("-subset", type=str, required=True, help="substitute dataset of removalnet")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-ydist", action="store", default="l2", type=str, choices=["l2", "cosine", "kl"], help="distance of adv logits")

    parser.add_argument("-batch_size", action="store", default=100, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args.namespace = format_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.removal_root = osp.join(args.out_root, "RemovalNet")
    return args



import benchmark
from dataset import loader
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    bench = benchmark.ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model0 = bench.load_wrapper(args.model1).load_torch_model()

    bench = benchmark.ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = bench.load_wrapper(args.model1).load_torch_model()

    # substitute dataset to train model
    train_loader = loader.get_dataloader(args.subset, split="train", batch_size=100)
    test_loader = loader.get_dataloader(model1.dataset_id, split="test", batch_size=500)
    cfg = loader.load_cfg(dataset_id=args.subset, arch_id="")
    cfg.layer_index = 5
    cfg.alpha = 0.9
    cfg.T = 20
    cfg.ydist = args.ydist
    cfg.iterations = 300
    cfg.task = model1.task
    cfg.seed = args.seed

    removalnet = f"removalnet({args.subset},{cfg.alpha},{cfg.T},{args.ydist})-"
    model_name = f"{str(model1.task)}{removalnet}"
    cfg.models_dir = osp.join(args.models_dir, f'{model_name}')

    removal = DeepRemoval(model0, model1, cfg, train_loader=train_loader, test_loader=test_loader)
    removal.deepremoval()


if __name__ == "__main__":
    main()
    """
        Example command:
        <===========================  Flower102-resnet18:20220901_Test  ===========================>
        SCRIPT="attack.deepremoval"
        python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-" -model2 "" -device 2
    """


















