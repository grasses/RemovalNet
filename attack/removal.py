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
from tqdm import tqdm
from attack.tap import tap_sgd
from attack import ops
from torch import optim
from utils import metric, helper, vis
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
        self.output_dir = cfg.output_dir
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
    def dist_loss(a, b, ydist):
        N = len(a)
        if ydist == "l2":
            return ((a / a.norm()) - (b / b.norm())).norm()
        elif ydist == "cosine":
            return F.cosine_similarity(a.view(N, -1), b.view(N, -1), dim=1).mean()
        elif ydist == "kl":
            return F.kl_div(a, b, reduction="mean")
        else:
            raise NotImplementedError(f"-> {ydist} not implemented!")

    @staticmethod
    def normalize(vs):
        return [(v - torch.min(v)) / (torch.max(v) - torch.min(v) + 1e-6) for v in vs]

    @staticmethod
    def distance(a, b, ydist):
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


    def feature_poison(self, model, x, y, l=2, lr=0.1, poison_steps=20, theta=0.5, ydist="l2"):
        # z = x -> f(x)^{l}  // z is latent space, maximize z & z_prime
        z = ops.batch_fed_forward(model, x, layer_index=l, batch_size=self.batch_size).detach().contiguous().to(self.device)
        z_prime = torch.rand(z.shape, device=self.device, requires_grad=True)
        normalized_z = self.normalize(z)

        for step in range(poison_steps):
            z_prime = z_prime.detach()
            z_prime.requires_grad = True
            logit_t = model.mid_forward(z_prime, layer_index=l)

            # minimize CELoss & maximize cosine dist
            loss_dist = torch.log(self.distance(z, z_prime, ydist))
            loss_ce = F.cross_entropy(logit_t, y)

            loss = loss_ce - loss_dist
            grad = torch.autograd.grad(loss, [z_prime], retain_graph=False, create_graph=False)[0]
            z_prime = z_prime.detach() - lr * grad.sign()

            # selected clip & random activate
            for idx in range(z.shape[0]):
                z_prime[idx][normalized_z[idx] > theta] = 0.0

        pred = logit_t.argmax(dim=1).detach()
        acc = (100.0 * pred.eq(y).sum() / len(y)).item()
        print(f"-> [Feature-Level] acc:{acc}% y:{y[:5].tolist()} pred:{pred[:5].tolist()} loss:{loss.item()} = ce_loss:{loss_ce.item()} + {ydist}_dist:{loss_dist.item()}")
        return z_prime.clone().detach()

    def boundary_poison(self, model, x, y, ydist="l2"):
        with torch.no_grad():
            logits = model(x)
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
                for beta in np.arange(0.2, 1, 0.05):
                    out = h(logits[i], logits[idxs[i]], beta)
                    if out.argmax(dim=0) == y[i]:
                        logits_prime[i] = out
                        beta_list.append(round(beta, 2))
                        break
            norm = (logits_prime-logits).norm(p=2).mean().detach().cpu()
            print(f"-> [Boundary-Level] beta:{beta_list[:8]} logits_norm:{norm}")
            return logits_prime.detach()

    def data_augmentation(self, model_0, model_t, x, y, l, lr=0.01, steps=20):
        adv_x = x.clone().to(self.device)
        logits_t = model_t(adv_x)
        probs, idxs = torch.topk(logits_t, k=2, dim=1)
        adv_y = (idxs[:, 1]).long()

        for step in range(steps):
            adv_x = adv_x.detach()
            adv_x.requires_grad = True
            layer_0 = model_0.fed_forward(adv_x, layer_index=l)
            layer_t = model_t.fed_forward(adv_x, layer_index=l)
            logits_0 = model_0(adv_x)
            logits_t = model_t(adv_x)

            loss_ce = F.cross_entropy(logits_t, adv_y)
            loss_dist = -F.mse_loss(layer_0, layer_t, reduction="mean")
            loss_miss = -F.kl_div(logits_t, logits_0)
            loss = loss_dist + loss_ce + loss_miss
            grad = torch.autograd.grad(loss, [adv_x], retain_graph=False, create_graph=False)[0]
            adv_x = adv_x - lr * grad.sign()

        a = model_0(adv_x).argmax(dim=1)
        b = model_t(adv_x).argmax(dim=1)
        acc = 100.0 * a.eq(b.view_as(a)).sum() / len(a)
        print(f"-> [Data Augmention] acc:{acc.item()} loss:{loss.item()} x_norm:{torch.norm(x-adv_x, p=2)}")
        return adv_x.detach()

    def generate_uaps(self, model_0, data_loader):
        from attack.uap import uap_sgd
        model_0.eval()
        model_0.to(self.device)
        x_val, y_val = next(iter(data_loader))

        uaps = []
        for c in range(self.num_classes):
            uap_init = torch.randn(*x_val.shape[1:])
            uap, losses = uap_sgd(model_0, data_loader, nb_epoch=1, y_target=c, uap_init=uap_init, device=self.device)
            uaps.append(uap)

            # for test
            batch_delta = torch.zeros_like(x_val)
            batch_delta.data = uaps[c].data.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
            perturbed = (x_val + batch_delta).to(self.device)
            preds = model_0(perturbed).argmax(dim=1).detach().cpu()
            print(f"-> UAP for:{c} losses:{torch.mean(torch.tensor(losses)).item()} preds:{preds[:10]}")
        return uaps


    def deepremoval_step(self, model_0, model_t, optimizer, x, y, l, t=0):
        cfg = self.cfg
        model_0.eval()
        model_t.eval()

        # remove feature-level fingerprints
        feats_prime = self.feature_poison(model=model_t, x=x, y=y, l=l, ydist=self.cfg.ydist, poison_steps=self.cfg.poison_steps).detach()
        # remove logit-level fingerprints
        logit_prime = self.boundary_poison(model=model_t, x=x, y=y, ydist=self.cfg.ydist).detach()

        model_t.train()
        optimizer.zero_grad()
        logit = model_t(x)
        loss_dist = F.mse_loss(model_t.fed_forward(x, layer_index=l), feats_prime, reduction="mean")
        loss_kd = cfg.T * cfg.T * cfg.alpha * F.kl_div(
                F.log_softmax(logit / self.cfg.T, dim=1),
                F.softmax(logit_prime / self.cfg.T, dim=1), reduction='batchmean') + \
            (1. - cfg.alpha) * F.cross_entropy(logit, y)
        loss = loss_dist + loss_kd
        loss.backward()
        optimizer.step()
        print(f"-> [Train] iters:{t} loss:{loss.item()}=loss_dist:{loss_dist.item()}+loss_kd:{loss_kd.item()}")
        return loss, loss_dist, loss_kd

    def deepremoval(self):
        cfg = self.cfg
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

        for step in range(1, 1+iterations):
            try:
                batch, label = next(loader)
            except:
                loader = iter(self.train_loader)
                batch, label = next(loader)
            x, y = batch.to(self.device), label.to(self.device)
            l = random.randint(2, 3)

            """
            if step > 20:
                adv_x = tap_sgd(model_0, model_t, x=val_x, y=val_y, lr=0.01, step_decay=0.8, steps=20, device=self.device)
                if len(adv_x) > 1:
                    adv_x = adv_x.to(self.device)
                    model_t.train()
                    optimizer.zero_grad()
                    logit_prime = model_0(adv_x)
                    adv_y = logit_prime.argmax(dim=1).long().detach()
                    '''
                    logit = model_t(adv_x)
                    loss = -cfg.T * cfg.T * cfg.alpha * F.kl_div(
                        F.log_softmax(logit / self.cfg.T, dim=1),
                        F.softmax(logit_prime / self.cfg.T, dim=1), reduction='batchmean') + \
                              (1. - cfg.alpha) * F.cross_entropy(logit, adv_y)
                    loss.backward()
                    optimizer.step()
                    '''
                    self.deepremoval_step(model_0, model_t, optimizer, x=adv_x, y=adv_y, l=l)
            """
            pred_y = model_0(x).argmax(dim=1).long().detach()
            loss, loss_dist, loss_kd = self.deepremoval_step(model_0, model_t, optimizer, x=x, y=pred_y, l=l)

            # testing & visualization
            if step % cfg.test_interval == 0 or (step == iterations - 1):
                _best_topk_acc, topk_acc, test_loss = metric.topk_test(model_t, self.test_loader, device=self.device, epoch=step, debug=True)
                self.learning_data["t"].append(step)
                self.learning_data["acc"].append(topk_acc["top1"])
                self.learning_data["loss_dist"].append(ops.numpy(loss_dist))
                self.learning_data["loss_kd"].append(ops.numpy(loss_kd))
                vis.view_learning_state(self.learning_data, file_path=osp.join(self.output_dir, "LR"))

            if step % cfg.plot_interval == 0:
                # plot LayerCAM
                for l in np.arange(4):
                    fig_path = osp.join(self.output_dir, f"LayerCam_{cfg.layers[l]}_t{step}")
                    vis.view_layer_activation(model_0, model_t, x=val_x, y=val_y, target_layer=cfg.layers[l], fig_path=fig_path, device=self.device)
                # plot decision boundary
                fig_path = osp.join(self.output_dir, f"Boundary")
                acc1, acc2 = vis.view_decision_boundary(model_0, model_t, self.test_loader, fig_path=fig_path, step=step)
                print(f"-> For T:{step} [Train] model_0:{acc1}% model_t:{acc2}% loss:{loss.item()} loss_dist:{loss_dist.item()} loss_kd:{loss_kd.item()}")
                self.save_torch_model(model_t.cpu(), step=10*int(step/10))
            print()
        self.save_torch_model(model_t.cpu())


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", help="model 1.")
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
    args.output_dir = ""

    args.alpha = 0.05
    args.T = 10
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.layers = ["layer1", "layer2", "layer3", "layer4", "layer5"]
    if "CIFAR10" in args.model1:
        args.alpha = 0.1
        args.T = 10
        args.lr = 1e-2
        args.poison_steps = 20
        args.test_interval = 1
        args.plot_interval = 10
        args.iterations = 1000
    elif "ImageNet" in args.model1:
        args.lr = 1e-2
        args.poison_steps = 30
        args.plot_interval = 10
        args.test_interval = 20
        args.iterations = 1000
    if "densenet" in args.model1:
        args.layers = ["features.pool0", "features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"]

    helper.set_default_seed(args.seed)
    args.lr = args.lr * (1 + 0.2 * random.random())
    return args


import benchmark
from dataset import loader
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    benchmk = benchmark.ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model0 = benchmk.load_wrapper(args.model1).load_torch_model()
    model1 = benchmk.load_wrapper(args.model1).load_torch_model() # a deepcopy of model1

    # substitute dataset to train model
    train_loader = loader.get_dataloader(args.subset, split="train", batch_size=100)
    test_loader = loader.get_dataloader(model1.dataset_id, split="test", batch_size=500)

    removalnet = f"removalnet({args.subset},{args.alpha},{args.T},{args.ydist})-"
    model_name = f"{str(model1.task)}{removalnet}"

    args.task = model1.task
    args.models_dir = osp.join(args.models_dir, f'{model_name}')
    args.output_dir = osp.join(helper.ROOT, "output/Removal", model_name)

    removal = DeepRemoval(model0, model1, args, train_loader=train_loader, test_loader=test_loader)
    removal.deepremoval()


if __name__ == "__main__":
    main()
    """
        Example command:
        <===========================  Flower102-resnet18:20220901_Test  ===========================>
        SCRIPT="attack.deepremoval"
        python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-" -model2 "" -device 2
    """


















