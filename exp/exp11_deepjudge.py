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
from exp import vis
from utils import helper
from defense.DeepJudge import DeepJudge
from benchmark import ImageBenchmark
from dataset import loader as dloader


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
    args.proj_root = osp.join(args.out_root, "DeepJudge")
    args.archs = {
        "CIFAR10": ["resnet50"],
        "ImageNet": ["resnet50"],
    }
    for attr_idx in range(40):
        args.archs[f"CelebA+{attr_idx}"] = ["vgg19_bn"]

    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def exp11_eval(args):
    methods = ["distill", "quantize", "finetune", "prune", "negative", "steal"]
    out_root = osp.join(args.out_root, "DeepJudge")
    for arch in args.archs[args.dataset]:
        result = {}
        tag = f"{args.dataset}_{arch}"
        path = osp.join(args.proj_root, f"exp/exp11_{tag}.pt")
        if not osp.exists(path):
            cfg = dloader.load_cfg(dataset_id=args.dataset, arch_id=arch)
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
                    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test")
                    continue

                key = f"{model1.task}_{str(model)}"
                if key not in result.keys():
                    result[key] = {
                        "ROB": [],
                        "JSD": [],
                        "LOD": [],
                        "LAD": [],
                        "NOD": [],
                        "NAD": [],
                        "MR": []
                    }
                device = torch.device("cpu") if "quantize" in str(model) else args.device
                model2 = model.torch_model(seed=model.seed)
                deepjudge = DeepJudge(model1, model2, test_loader=test_loader,
                                      device=device, seed=args.seed, out_root=out_root,
                                      batch_size=args.batch_size,
                                      layer_index=3, seed_method=args.seed_method)
                if fingerprint is None:
                    fingerprint = deepjudge.extract()
                deepjudge.model2 = model2
                dist = deepjudge.verify(fingerprint)
                for metric in dist.keys():
                    result[key][metric].append(dist[metric])

            print()
            torch.save(result, path)
        result = torch.load(path)
        yield tag, result


def plot_boxplot(result, tag, fpath):
    xticklabels = []
    data = {}
    metrics = ["LAD", "LOD"]
    metrics_str = ""
    for metric in metrics:
        metrics_str += f"_{metric}"
    metrics_str = metrics_str[1:]

    print()
    for label, item in result.items():
        xticklabels.append(label.split("-")[-2])
        print(f"-> {label}")
        for metric in item.keys():
            min_v = round(min(item[metric]), 4)
            max_v = round(max(item[metric]), 4)
            med_v = round(((max_v + min_v) / 2), 4)
            mean_v = round(np.mean(item[metric]), 4)
            std_v = round(np.std(item[metric]), 4)
            print(f"-> metric: {metric} med:{med_v}±{max_v-med_v} mean:{mean_v} std:{std_v}")
            if metric not in metrics:
                continue
            if metric not in data.keys():
                data[metric] = []
            data[metric].append(item[metric])
        print()
    vis.plot_boxplot(data, xticklabels=xticklabels, fpath=fpath+f"_m{metrics_str}.pdf")


def exp11_eval_removalnet(args):
    out_root = osp.join(args.out_root, "DeepJudge")
    cfg = dloader.load_cfg(dataset_id=args.dataset, arch_id=args.archs[args.dataset][0])
    benchmk = ImageBenchmark(
        archs=args.archs[args.dataset][0],
        datasets=[args.dataset],
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir)
    model1 = benchmk.load_wrapper(args.model1, seed=1000).load_torch_model()
    model2 = benchmk.load_wrapper(args.model2, seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=128)
    deepjudge = DeepJudge(model1, model2, test_loader=test_loader,
                                      device=args.device, seed=args.seed, out_root=out_root,
                                      batch_size=args.batch_size,
                                      layer_index=3, seed_method=args.seed_method)
    fingerprint = deepjudge.extract()
    item = deepjudge.verify(fingerprint)

    print(item)
    for metric in item.keys():
        min_v = np.min(item[metric])
        max_v = np.max(item[metric])
        med_v = (max_v + min_v) / 2
        mean_v = np.mean(item[metric])
        std_v = np.std(item[metric])
        print(f"-> Removal metric: {metric} med:{med_v}±{max_v - med_v} mean:{mean_v} std:{std_v}")


def main():
    args = get_args()
    print(f"-> Running with config:{args}")
    for tag, result in exp11_eval(args):
        fpath = osp.join(args.proj_root, f"exp/exp11_{tag}_boxplot")
        plot_boxplot(result, tag, fpath)
    exp11_eval_removalnet(args)


if __name__ == "__main__":
    main()









