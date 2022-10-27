#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


"""
This script is used to evaluate benchmark of DeepJudge
"""


import os, argparse, logging
import os.path as osp
import torch
import numpy as np
from utils import helper
from defense.ZEST.zest import ZEST
from benchmark import ImageBenchmark
from dataset import loader as dloader
from . import vis


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
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.zest_root = osp.join(args.out_root, "ZEST", "exp")
    args.archs = {
        "CIFAR10": ["resnet34", "vgg16_bn"],
        "ImageNet": ["vgg16_bn"],
    }
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def exp11_eval(args):
    methods = ["negative", "finetune", "distill", "steal", "prune"]
    out_root = osp.join(args.out_root, "ZEST")

    for arch in args.archs[args.dataset]:
        result = {}
        tag = f"{args.dataset}_{arch}"
        path = osp.join(args.zest_root, f"exp11_{tag}.pt")
        if not osp.exists(path):
            cfg = dloader.load_cfg(dataset_id=args.dataset)
            bench = ImageBenchmark(
                archs=[arch],
                datasets=[args.dataset],
                datasets_dir=args.datasets_dir,
                models_dir=args.models_dir)
            models = bench.list_models(cfg=cfg, methods=methods)
            model1, test_loader, fingerprint = None, None, None
            for idx, model in enumerate(models):
                print(f"-> idx:{idx} runing for model:{model} seed:{model.seed}")
                if idx == 0:
                    model1 = model.torch_model(seed=1000)
                    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=128)
                    continue

                key = f"{model1.task}_{str(model)}"
                if key not in result.keys():
                    result[key] = {
                        "1": [],
                        "2": [],
                        "cosine": [],
                        "inf": []
                    }
                model2 = model.torch_model(seed=model.seed)
                zest = ZEST(model1, model2, test_loader=test_loader,
                                 device=args.device,
                                 out_root=out_root)
                fingerprint = zest.extract(cache=False)
                dist = zest.verify(fingerprint)
                for metric in dist.keys():
                    result[key][metric].append(dist[metric])
                print()
                torch.save(result, path)
        result = torch.load(path, map_location="cpu")
        yield tag, result


def plot_boxplot(result, tag, fpath):
    xticklabels = []
    data = {}
    metrics = ["2"]
    for label, item in result.items():
        xticklabels.append(label.split("-")[-2])
        for metric in item.keys():
            if metric not in metrics:
                continue
            if metric not in data.keys():
                data[metric] = []
            data[metric].append(item[metric])
    vis.plot_boxplot(data, xticklabels=xticklabels, fpath=fpath)


def main():
    args = get_args()
    print(f"-> Running with config:{args}")
    for tag, result in exp11_eval(args):
        fpath = osp.join(args.zest_root, f"exp11_{tag}_boxplot.pdf")
        plot_boxplot(result, tag, fpath)









if __name__ == "__main__":
    main()









