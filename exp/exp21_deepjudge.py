#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


"""
This script is used to evaluate benchmark of DeepJudge
"""


import math
import os, argparse
import os.path as osp
import torch
import numpy as np
from exp import vis, ops
from tqdm import tqdm
from utils import helper, metric as metric_fun, vis
from defense.DeepJudge import DeepJudge
from benchmark import ImageBenchmark
from dataset import loader as dloader


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", required=True, help="model 2.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-seed_method", action="store", default="PGD", type=str, choices=["FGSM", "PGD", "CW"],
                        help="Type of blackbox generation")
    parser.add_argument("-layer_index", action="store", default=4, type=int, help="GPU device id")
    parser.add_argument("-djm", required=False, type=float, default=3, help="m of DeepJudge")
    parser.add_argument("-batch_size", required=False, type=int, default=200, help="tag of script.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-start", default=100, type=int, help="Gap between two pretrained model")
    parser.add_argument("-gap", default=100, type=int, help="Gap between two pretrained model")
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
    args.djm = round(float(args.djm), 1)
    return args


def normalize(vs, min_v, max_v):
    return [min(1., (v - min_v) / (max_v - min_v + 1e-6)) for v in vs]


def exp21_eval(args, neg_dist, metrics, min_v, max_v):
    result = {"neg_acc": [], "neg_plot": {}, "acc": [], "dist": {}, "step": [], "plot":{}}
    for metric in metrics:
        result["dist"][metric] = []
        result["plot"][metric] = []

    arch, dataset = args.model1.split("(")[1].split(")")[0].split(",")
    exp_path = osp.join(args.proj_root, f"exp/exp21_{dataset}_{arch}_L{args.layer_index}_m{args.djm}_r{args.model2}.pt")

    benchmk = ImageBenchmark(archs=[arch], datasets=[dataset],
                             datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = benchmk.load_wrapper(args.model1, seed=1000).load_torch_model()
    model2 = benchmk.load_wrapper(args.model2, seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=dataset, split="test", batch_size=args.batch_size)


    if not osp.exists(exp_path):
        result["step"] = np.arange(args.start, 1001, args.gap).tolist()

        # step2.1: load DeepJudge
        deepjudge = DeepJudge(model1, model2, test_loader=test_loader,
                              device=args.device, seed=args.seed, out_root=args.proj_root,
                              batch_size=args.batch_size, m=args.djm,
                              layer_index=args.layer_index, seed_method=args.seed_method)
        fingerprint = deepjudge.extract()

        # step2.2: eval DeepJudge & Acc of source model
        phar = tqdm(np.arange(args.start, 1001, args.gap))
        for step in phar:
            ckpt = osp.join(args.models_dir, model2.task, f'final_ckpt_s{args.seed}_t{step}.pth')
            if not osp.exists(ckpt):
                raise FileNotFoundError(f"-> model not found:{ckpt}")
            weights = torch.load(ckpt)["state_dict"]
            deepjudge.model2.load_state_dict(weights)
            deepjudge.model2.to(args.device)
            result["step"].append(int(step))

            # step2.2: eval DeepJudge
            item = deepjudge.verify(fingerprint)
            for metric in item.keys():
                if metric not in metrics: continue
                result["dist"][metric].append(item[metric])
            # step2.2: eval Acc
            _, topk_acc, _ = metric_fun.topk_test(model2, test_loader, device=args.device, epoch=0)
            result["acc"].append(round(float(topk_acc["top1"]), 4))
            phar.set_description(f"-> Removal({step}): DeepJudge(L{args.layer_index}) {item} Acc:{topk_acc['top1']}")

        # eval accuracy for negative
        result["neg_acc"] = []
        result["neg_dist"] = neg_dist
        result["neg_plot"] = {}
        for seed in np.arange(100, 1001, 100):
            ckpt = osp.join(args.models_dir, model1.task + f"negative({arch})-", f'final_ckpt_s{seed}.pth')
            if not osp.exists(ckpt):
                raise FileNotFoundError(f"-> model not found:{ckpt}")
            weights = torch.load(ckpt)["state_dict"]
            model1.load_state_dict(weights)
            model1.eval()
            _, topk_acc, _ = metric_fun.topk_test(model1, test_loader, device=args.device, epoch=0)
            result["neg_acc"].append(round(float(topk_acc["top1"]), 4))
            print(f"-> step negative acc:{result['neg_acc']}")
        torch.save(result, exp_path)
    result = torch.load(exp_path)

    # step2.3: normalize & reformat data
    for idx, (metric, dist) in enumerate(result["neg_dist"].items()):
        result["neg_plot"][metric] = []
        for x, y in zip(result["neg_acc"], dist):
            x = round(x * 100.0, 2) if x < 1 else x
            result["neg_plot"][metric].append([x, y])
        result["neg_plot"][metric] = np.array(result["neg_plot"][metric], dtype=np.float32)

    # step2.3: normalize & reformat data
    model1 = benchmk.load_wrapper(args.model1, seed=1000).load_torch_model()
    _, topk_acc, _ = metric_fun.topk_test(model1, test_loader, device=args.device, epoch=0)
    source_acc = topk_acc["top1"]
    source_dist = 0.0
    for idx, (metric, dist) in enumerate(result["dist"].items()):
        result["plot"][metric] = [[source_acc, source_dist]]
        dist = normalize(dist, min_v=min_v[idx], max_v=max_v[idx])
        for x, y in zip(result["acc"], dist):
            x = round(x * 100.0, 2) if x < 1 else x
            result["plot"][metric].append([x, y])
        result["plot"][metric] = np.array(result["plot"][metric], dtype=np.float32)
    torch.save(result, exp_path)
    return result


def main():
    args = get_args()
    metrics = ["LOD", "LAD"]
    methods = ["distill", "finetune", "prune", "negative", "steal"]
    arch, dataset = args.model1.split("(")[1].split(")")[0].split(",")
    if dataset in ["CIFAR10", "CINIC10", "CelebA32+20", "CelebA32+31"]:
        args.batch_size = 500

    # step1: read normalized data
    tag = f"{dataset}_{arch}_L{args.layer_index}_m{args.djm}"
    fpath = osp.join(args.proj_root, f"exp/exp11_{tag}.pt")
    results = torch.load(fpath, map_location="cpu")

    rpath = osp.join(args.proj_root, f"exp/exp11_{tag}_r{args.model2}.pt")
    rresults = torch.load(rpath, map_location="cpu")
    rresults.update(results)

    cache = ops.exp11_normalize(rresults, methods=["removalnet"]+methods, metrics=metrics, defense_method="DeepJudge")
    min_v, max_v = cache["min_v"], cache["max_v"]
    print(f"-> min:{min_v} max:{max_v}")

    neg_dist = {}
    for idx, metric in enumerate(metrics):
        neg_dist[metric] = cache["dists_nz"][idx, 1].tolist()

    # step2: eval DeepJudge & Acc
    result = exp21_eval(args, neg_dist, metrics, min_v, max_v)

    # step3: plot curve
    for idx, metric in enumerate(metrics):
        metrics[idx] = f"DeepJudge-{metric}"

    exp_path = osp.join(args.proj_root, f"pdf/exp21_{tag}_r{args.model2}.pt")
    vis.plot_accuracy_dist_curve(result["plot"], result["neg_plot"], steps=result["step"], legends=metrics, path=exp_path.replace(".pt", ".pdf"))


if __name__ == "__main__":
    main()
















