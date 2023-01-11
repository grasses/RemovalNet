#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


"""
This script is used to evaluate benchmark of DeepJudge
"""

import os
import os.path as osp
import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import helper
from benchmark import ImageBenchmark
from dataset import loader as dloader


class Tensor2Dataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-mask", action="store", dest="mask", default="",
                        help="The mask to filter the models to generate, split with +")
    parser.add_argument("-phase", action="store", dest="phase", type=str, default="",
                        help="The phase to run. Use a prefix to filter the phases.")
    parser.add_argument("-regenerate", action="store_true", dest="regenerate", default=False,
                        help="Whether to regenerate the models.")
    parser.add_argument("-model1", action="store", dest="model1", default="pretrain(resnet18,ImageNet)-",
                        required=False, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", default="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-",
                        required=False, help="model 2.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-seed_method", action="store", default="PGD", type=str, choices=["FGSM", "PGD", "CW"],
                        help="Type of blackbox generation")
    parser.add_argument("-batch_size", required=False, type=int, default=100, help="tag of script.")
    parser.add_argument("-dataset", required=False, type=str, default="CIFAR10", help="model archtecture")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.deepjudge_root = osp.join(args.out_root, "DeepJudge", "exp")
    args.archs = {
        "CIFAR10": ["resnet50"],
        "ImageNet": ["resnet50"],
    }
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def view_decision_boudary(model1, model2, test_loader, fingerprints, fig_path, selects=[], device=torch.device("cpu")):
    """
    Args:
        model:
        test_loader:
        fingerprints:
    Returns: new testloader with fingerprints & test_loader
    """
    model1.to(device)
    model2.to(device)
    fingerprints = fingerprints.to(device)

    probs1 = {}
    probs2 = {}
    cnt1 = torch.zeros(len(selects)+1)
    cnt2 = torch.zeros(len(selects) + 1)
    for c in range(len(selects)):
        probs1[c] = []
        probs2[c] = []

    with torch.no_grad():
        phar = tqdm(test_loader)
        for x, y in phar:
            x = x.to(device)
            pred1 = F.softmax(model1(x), dim=1).detach().cpu()
            pred_y1 = pred1.argmax(dim=1)
            for i, c in enumerate(selects):
                idx = ((pred_y1 == c).nonzero(as_tuple=True)[0])
                cnt1[i] += len(idx)
                probs1[i].append(pred1[idx])

            pred2 = F.softmax(model2(x), dim=1).detach().cpu()
            pred_y2 = pred2.argmax(dim=1)
            for i, c in enumerate(selects):
                idx = ((pred_y2 == c).nonzero(as_tuple=True)[0])
                cnt2[i] += len(idx)
                probs2[i].append(pred2[idx])

        for i in range(len(selects)):
            probs1[i] = torch.cat(probs1[i])
            probs2[i] = torch.cat(probs2[i])

        # fingerprints
        fp_pred1 = F.softmax(model1(fingerprints), dim=1).detach().cpu()
        fp_pred2 = F.softmax(model2(fingerprints), dim=1).detach().cpu()

    labels_cnt = {}
    probs = torch.cat(list(probs1.values()) + list(probs2.values()) + list([fp_pred1, fp_pred2]))

    for idx in range(len(selects)):
        labels_cnt[f"model0_c{selects[idx]}"] = int(cnt1[idx])
    for idx in range(len(selects)):
        labels_cnt[f"modelt_c{selects[idx]}"] = int(cnt2[idx])
    labels_cnt["model0_fp"] = len(fingerprints)
    labels_cnt["modelt_fp"] = len(fingerprints)
    xy = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(probs)
    plot_embedding(xy, labels_cnt, fig_path=f"output/Removal/exp/{fig_path}")


def plot_embedding(xy, labels_cnt, fig_path, lims=120, fontsize=30):
    plt.figure(figsize=(16, 16), dpi=100)
    plt.cla()
    off = 0
    for label, cnt in labels_cnt.items():
        plt.scatter(xy[off:off + cnt, 0], xy[off:off + cnt, 1], lw=6, s=60, label=label)
        off += cnt

    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="upper right", numpoints=1, fontsize=50, prop={'size': 10})
    plt.savefig(fig_path)
    print(f"-> saving fig: {fig_path}")


def main():
    args = get_args()
    benchmk = ImageBenchmark(
        archs=args.archs[args.dataset][0],
        datasets=[args.dataset],
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir)

    args.device = torch.device("cuda:1")
    torch.set_printoptions(precision=2)

    arch = args.archs[args.dataset][0]
    dataset = args.dataset

    model1 = benchmk.load_wrapper(args.model1, seed=args.seed).load_torch_model()
    model2 = benchmk.load_wrapper(args.model2, seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=1000, shuffle=False)

    fp_path = f"/home/Hongwei/project/Model-Reuse/MetaFinger/output/fingerprints/{dataset}_{arch}.pth"
    data = torch.load(fp_path, map_location="cpu")

    x = data["x"].to(args.device)
    model1.to(args.device)
    model2.to(args.device)

    torch.set_printoptions(precision=2)

    pred1 = F.softmax(model1(x), dim=1).detach().cpu() * 100.0
    print(pred1.int())
    pred2 = F.softmax(model2(x), dim=1).detach().cpu() * 100.0
    print(pred2.int())

    print(pred1.argmax(dim=1).detach().cpu())
    print(pred2.argmax(dim=1).detach().cpu())
    exit(1)

    fig_path = f"boundary_{args.model2}.pdf"
    print(f"-> model:{args.model1} arch:{arch} dataset:{dataset}")
    view_decision_boudary(model1, model2, test_loader=test_loader, fingerprints=data["x"], fig_path=fig_path, selects=[8, 9], device=args.device)





main()