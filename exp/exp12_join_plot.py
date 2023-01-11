#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


"""
This script is used to evaluate benchmark of ZEST
"""


import copy
import os, argparse, logging
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from . import comm
from utils import helper
from . import vis, ops


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-dataset", choices=["CIFAR10", "CINIC10", "CelebA32+20", "CelebA32+31", "ImageNet"], help="Dataset")
    parser.add_argument("-batch_size", required=False, type=int, default=100, help="tag of script.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-start", default=100, type=int, help="Gap between two pretrained model")
    parser.add_argument("-gap", default=100, type=int, help="Gap between two pretrained model")
    parser.add_argument("-epsilon", action="store", default=0.2, type=float, help="Epsilon of ModelDiff")
    parser.add_argument("-k", action="store", default=0.1, type=float, help="k of IPGuard")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.proj_root = osp.join(args.out_root, "DeepJudge")
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def purify_result(results):
    keys = []
    for k in results.keys():
        if "0.2" in k and ("prune" in k or "fine" in k):
            keys.append(k)
    for k in keys:
        del results[k]
    return results


def main():
    args = get_args()
    methods = ["distill", "finetune", "prune", "negative"]
    metrics = ["LOD", "LAD", "L2", "cosine"]
    task_list = comm.task_dict[args.dataset]

    for model2 in task_list:
        # step1: join data & normalize
        arch, dataset = model2.split("-")[0].split("(")[1].split(")")[0].split(",")
        # DeepJudge
        layer_index = model2.split("-")[-2].split("(")[1].split(")")[0].split(",")[-1][1]
        tag = f"{dataset}_{arch}_L{layer_index}"
        pth1 = osp.join(args.out_root, f"DeepJudge/exp/exp11_{tag}.pt")
        results_dj = torch.load(pth1)
        pth2 = osp.join(args.out_root, f"DeepJudge/exp/exp11_{tag}_r{model2}.pt")
        results = torch.load(pth2)
        results.update(results_dj)
        data_dj = ops.exp11_normalize(purify_result(results), methods=["removalnet"] + methods, metrics=["LOD", "LAD"], defense_method="DeepJudge")

        # ZEST
        pth1 = osp.join(args.out_root, f"ZEST/exp/exp11_{dataset}_{arch}.pt")
        results_zest = torch.load(pth1)
        pth2 = osp.join(args.out_root, f"ZEST/exp/exp11_{dataset}_{arch}_r{model2}.pt")
        results = torch.load(pth2)
        results.update(results_zest)
        data_zest = ops.exp11_normalize(purify_result(results), methods=["removalnet"] + methods, metrics=["L2", "cosine"], defense_method="ZEST")

        # step2: plot
        # data[dists_nz] = [metics, models, 10]
        dists_nz = np.concatenate([data_dj["dists_nz"], data_zest["dists_nz"]])
        legends = data_dj["legends"] + data_zest["legends"]
        fpath = osp.join(args.out_root, f"pdf/exp12_distance_{dataset}_{arch}_r{model2}.pdf")
        vis.boxplot_distance(dists_nz, metrics=legends, xticks=data_dj["xticks"],
                             ylabel=data_dj["ylabel"], fpath=fpath)

        # ModelDiff
        tag = f"{dataset}_{arch}_eps{args.epsilon}"
        pth1 = osp.join(args.out_root, f"ModelDiff/exp/exp11_{tag}.pt")
        results_dj = torch.load(pth1)
        pth2 = osp.join(args.out_root, f"ModelDiff/exp/exp11_{tag}_r{model2}.pt")
        results = torch.load(pth2)
        results.update(results_dj)
        data_diff = ops.exp11_normalize(purify_result(results), methods=["removalnet"] + methods, metrics=["DDM"], defense_method="ModelDiff")

        # IPGuard
        k = comm.params_dict[args.dataset][arch][0]
        tag = f"{dataset}_{arch}_tLk{k}"
        pth1 = osp.join(args.out_root, f"IPGuard/exp/exp11_{tag}.pt")
        results_dj = torch.load(pth1)
        pth2 = osp.join(args.out_root, f"IPGuard/exp/exp11_{tag}_r{model2}.pt")
        results = torch.load(pth2)
        results.update(results_dj)
        data_ipguard = ops.exp11_normalize(purify_result(results), methods=["removalnet"] + methods, metrics=["MR"], defense_method="IPGuard")

        dists_nz = np.concatenate([data_diff["dists_nz"], data_ipguard["dists_nz"]])
        legends = data_diff["legends"] + data_ipguard["legends"]
        fpath = osp.join(args.out_root, f"pdf/exp12_similarity_{dataset}_{arch}_r{model2}.pdf")
        vis.boxplot_distance(dists_nz, metrics=legends, xticks=data_diff["xticks"],
                             ylabel=data_dj["ylabel"], fpath=fpath)




if __name__ == "__main__":
    main()


























