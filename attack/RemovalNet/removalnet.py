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

    def data_augmentation(self, model_T, model_t, x, y, l, lr=0.01, steps=20):
        adv_x = x.clone().to(self.device)
        logits_t = model_t(adv_x)
        probs, idxs = torch.topk(logits_t, k=2, dim=1)
        adv_y = (idxs[:, 1]).long()
        for step in range(steps):
            adv_x = adv_x.detach()
            adv_x.requires_grad = True
            layer_0 = model_T.fed_forward(adv_x, layer_index=l)
            layer_t = model_t.fed_forward(adv_x, layer_index=l)
            logits_0 = model_T(adv_x)
            logits_t = model_t(adv_x)
            loss_ce = F.cross_entropy(logits_t, adv_y)
            loss_dist = -F.mse_loss(layer_0, layer_t, reduction="mean")
            loss_miss = -F.kl_div(logits_t, logits_0)
            loss = loss_dist + loss_ce + loss_miss
            grad = torch.autograd.grad(loss, [adv_x], retain_graph=False, create_graph=False)[0]
            adv_x = adv_x - lr * grad.sign()
        a = model_T(adv_x).argmax(dim=1)
        b = model_t(adv_x).argmax(dim=1)
        acc = 100.0 * a.eq(b.view_as(a)).sum() / len(a)
        print(f"-> [Data Augmention] acc:{acc.item()} loss:{loss.item()} x_norm:{torch.norm(x-adv_x, p=2)}")
        return adv_x.detach()

    def data_augmentation_uaps(self, model_T, data_loader, layer_name=None):
        from .uap import uap_sgd
        cfg = self.cfg
        model_T.eval()
        model_T.to(self.device)
        x_val, y_val = next(iter(data_loader))
        uaps_path = osp.join(self.cfg.proj_root, f"cache/uaps_{self.task}_sub{cfg.subset}.pth")
        if not osp.exists(uaps_path):
            uaps = []
            for c in range(self.num_classes):
                uap_init = torch.randn(*x_val.shape[1:])
                uap, losses = uap_sgd(model_T, data_loader, nb_epoch=8, y_target=c, uap_init=uap_init, device=self.device)
                uaps.append(uap)
            torch.save(uaps, uaps_path)
        uaps = torch.load(uaps_path, map_location="cpu")
        for c in range(self.num_classes):
            # for test
            batch_delta = torch.zeros_like(x_val)
            batch_delta.data = uaps[c].data.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
            perturbed = (x_val + batch_delta).to(self.device)
            preds = model_T(perturbed).argmax(dim=1).detach().cpu()
            target = torch.ones([len(x_val)], dtype=preds.dtype, device=preds.device) * c
            acc = 100.0 * (target.eq(preds).sum() / len(x_val))
            print(f"-> UAP:{c} preds_acc:{acc}%")
        return uaps

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

    def shuffle_intermediate_features(self, outputs, ratio=1.0):
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
        self.shuffle_intermediate_features(z_prime, ratio=shuffle_ratio)
        batch_size = len(x)

        size = math.ceil(z_prime[0].shape[0] * shuffle_ratio)
        for step in range(poison_steps):
            z_prime = self.shuffle_intermediate_features(z_prime, ratio=shuffle_ratio).detach()
            z_prime.requires_grad = True
            logit = model.mid_forward(z_prime, layer_index=l)

            # minimize CELoss & maximize cosine dist
            loss_dist = 0.1 * torch.log(self.distance_cost(z_prime, z, ydist)) #+ torch.std(z_prime.view(batch_size, -1), dim=1).mean())
            loss_ce = F.cross_entropy(logit, y)
            loss = loss_ce - loss_dist
            grad = torch.autograd.grad(loss, [z_prime], retain_graph=False, create_graph=False)[0]
            z_prime = z_prime - lr * grad.sign()
            z_prime = z_prime.detach()

        '''
        for step in range(poison_steps):
            for i in range(batch_size):
                y_i = y[i].unsqueeze(0).detach()
                z_i = z[i].unsqueeze(0).detach()
                z_prime_i = z_prime[i].unsqueeze(0)
                z_prime_i = self.shuffle_intermediate_features(z_prime_i, ratio=0.3).detach()
                z_prime_i.requires_grad = True
                logit_i = model.mid_forward(z_prime_i, layer_index=l)

                # minimize CELoss & maximize cosine dist
                loss_dist = 0.5 * torch.log(self.distance_cost(z_prime_i, z_i, ydist))
                pred_y_i = logit_i.argmax(dim=1)
                loss = -loss_dist
                if pred_y_i != y_i:
                    loss_ce = F.cross_entropy(logit_i, y_i)
                    loss += loss_ce
                grad = torch.autograd.grad(loss, [z_prime_i], retain_graph=False, create_graph=False)[0]
                z_prime_i = z_prime_i - lr * grad.sign()
                z_prime[i] = z_prime_i.squeeze(0).data
            z_prime = z_prime.detach()
        '''

        if t % 10 == 0:
            logit_t = model.mid_forward(z_prime, layer_index=l)
            loss_dist = torch.log(self.distance_cost(z_prime, z, ydist))
            loss_ce = F.cross_entropy(logit_t, y)
            loss = loss_ce - loss_dist
            pred = logit_t.argmax(dim=1).detach()
            acc = round(float((100.0 * pred.eq(y).sum() / len(y))), 2)
            std1 = round(float(torch.std(z.view(batch_size, -1), dim=1).mean().float()), 4)
            std2 = round(float(torch.std(z_prime.view(batch_size, -1), dim=1).mean().float()), 4)
            print(
                f"-> [Feature-Level] feat_acc:{acc}% std:{std1} vs {std2} "
                f"loss:{round(float(loss.item()), 4)}=ce_loss:{round(float(loss_ce.item()), 4)}-{ydist}_dist:{round(float(loss_dist.item()), 4)} "
                f"shuffle size:[{size}/{z_prime[0].shape[0]}]")
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
                for beta in np.arange(0.2, 1, 0.05):
                    out = h(logits[i], logits[idxs[i]], beta)
                    if out.argmax(dim=0) == y[i]:
                        logits_prime[i] = out
                        beta_list.append(round(beta, 2))
                        break
            if t % 10 == 0:
                norm = (logits_prime-logits).norm(p=2).mean().detach().cpu()
                print(f"-> [Logit - Level] beta:{beta_list[:8]} logits_norm:{norm}")
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
        loss_dist = self.cfg.beta * F.mse_loss(feats, feats_prime, reduction="mean")
        loss_kl = cfg.T * cfg.T * cfg.alpha * F.kl_div(
                F.log_softmax(logit / self.cfg.T, dim=1),
                F.softmax(logit_prime / self.cfg.T, dim=1), reduction='batchmean')
        loss_ce = (1. - cfg.alpha) * F.cross_entropy(logit, y)

        loss = loss_dist + loss_kl + loss_ce
        loss.backward()
        optimizer.step()
        if t % 10 == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print(f"-> [Train] iters:{t} LR:{round(float(current_lr), 5)} loss:{loss.item()}=loss_dist:{loss_dist.item()}+loss_kl:{loss_kl.item()}+loss_ce:{loss_ce.item()} \n")
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

        # uap data augmentation
        #uaps = self.data_augmentation_uaps(self.model_T, data_loader=self.train_loader)
        for step in range(0, 1+iterations):
            try:
                batch, label = next(loader)
            except:
                loader = iter(self.train_loader)
                batch, label = next(loader)
            x, y = batch.to(self.device), label.to(self.device)
            y = self.model_T(x).argmax(dim=1).long().detach()

            l = int(self.cfg.layer)
            #l = int(np.random.choice(np.arange(1, self.cfg.layer + 1)))
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

            flag = (step > 800 and step <= 1000 and step % 20 == 0)
            if step % 50 == 0 or flag:
                self.save_torch_model(self.model_t.cpu(), step=step)

            if step == 0 or step == 50 or step % cfg.plot_interval == 0 or (step == iterations - 1):
                # plot LayerCAM
                choice = random.randint(0, 1)
                eval_x, eval_y, eval_ori_x = eval_batch[choice]
                for l in np.arange(self.cfg.layer-2, self.cfg.layer):
                    fig_path = osp.join(self.cfg.exp_root, f"LayerCam_{cfg.layers[l]}_t{step}")
                    vis.view_layer_activation(self.model_T, self.model_t, x=eval_x.clone(), y=eval_y.clone(), ori_x=eval_ori_x.clone(),
                                              size=len(eval_x), target_layer=cfg.layers[l], fig_path=fig_path, device=self.device)
                # plot decision boundary
                fig_path = osp.join(self.cfg.exp_root, f"Boundary")
                acc1, acc2 = vis.view_decision_boundary(self.model_T, self.model_t, self.test_loader, fig_path=fig_path, step=step)
                print(f"-> For T:{step} [Train] model_T:{acc1}% model_t:{acc2}% loss:{loss.item()} loss_dist:{loss_dist.item()} loss_kl:{loss_kl.item()} loss_ce:{loss_ce.item()}")
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
        args.alpha = 0.5
        args.beta = 1.0
        args.lr = 1e-2
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000
        if "mobile" in args.model1 or "densenet" in args.model1:
            args.lr /= 2.0
            args.shuffle_ratio = 0.1

    elif "CINIC10" in args.model1:
        args.T = 20
        args.alpha = 0.2
        args.beta = 2.0
        args.lr = 1e-2
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000
        if "mobile" in args.model1 or "densenet" in args.model1:
            args.lr /= 4.0
            args.shuffle_ratio = 0.1

    elif "CelebA" in args.model1:
        args.T = 10
        args.alpha = 0.2
        args.beta = 1.0
        args.lr = 5e-3
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000
        if "mobile" in args.model1 or "densenet" in args.model1:
            args.lr /= 4.0
            args.shuffle_ratio = 0.1

    elif "Skin" in args.model1:
        args.T = 20
        args.alpha = 0.2
        args.beta = 1.0
        args.lr = 5e-3
        args.shuffle_ratio = 0.1
        args.poison_steps = 30
        args.test_interval = 20
        args.plot_interval = 200
        args.iterations = 1000
        if "mobile" in args.model1 or "densenet" in args.model1:
            args.lr /= 4.0
            args.shuffle_ratio = 0.1

    elif "ImageNet" in args.model1:
        args.lr = 1e-2
        args.alpha = 0.2
        args.beta = 1.0
        args.T = 20
        args.poison_steps = 30
        args.plot_interval = 20
        args.test_interval = 200
        args.iterations = 1000

    if "LFW" in args.subset:
        args.lr = 5e-3
        args.alpha = 0.2
        args.beta = 0.5
        args.T = 10
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

    helper.set_default_seed(args.seed)
    args.lr = args.lr * (0.9 + 0.2 * random.random())
    return args


import benchmark
from dataset import loader
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    arch, dataset = args.model1.split("(")[1].split(")")[0].split(",")
    benchmk = benchmark.ImageBenchmark(datasets=dataset, archs=[arch], datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model0 = benchmk.load_wrapper(args.model1).load_torch_model()
    model1 = benchmk.load_wrapper(args.model1).load_torch_model() # a deepcopy of model1

    # substitute dataset to train model
    train_loader = loader.get_dataloader(args.subset, split="train", batch_size=args.batch_size, subset_ratio=args.subset_ratio)
    test_loader = loader.get_dataloader(dataset, split="test", shuffle=False, batch_size=args.batch_size)
    removalnet = f"removalnet({args.subset},{'{:.1f}'.format(args.subset_ratio)},{args.alpha},{args.beta},{args.T},l{args.layer})-"

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
        <===========================  Flower102-resnet18:20220901_Test  ===========================>
        SCRIPT="attack.RemovalNet.removalnet"
        python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-" -model2 "" -device 2
    """


















