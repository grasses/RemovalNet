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
from utils import helper
from defense.ZEST.zest import ZEST
from benchmark import ImageBenchmark
from dataset import loader as dloader
from . import vis, ops


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="train(resnet50,CIFAR10)-",
                        required=False, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", default="train(resnet50,CIFAR10)-prune(0.5)-",
                        required=False, help="model 2.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-test_size", required=False, type=int, default=128, help="tag of script.")
    parser.add_argument("-dataset", required=False, type=str, default="CIFAR10", help="Dataset for testing")
    parser.add_argument("-arch", required=True, type=str, help="Model archtecture",
                        choices=["resnet50", "densenet121", "mobilenet_v2", "vgg16_bn", "vgg19_bn"])
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-removal", default=0, type=int, choices=[0, 1], help="Eval Removal?")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.proj_root = osp.join(args.out_root, "ZEST")
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def rename_metric(key):
    if key == "2":
        key = "L2"
    elif key == "1":
        key = "L1"
    elif key == "inf":
        key = "Linf"
    return key


def exp11_eval(args, methods, debug=False):
    result = {}
    tag = f"{args.dataset}_{args.arch}"
    path = osp.join(args.proj_root, f"exp/exp11_{tag}.pt")
    if debug or not osp.exists(path):
        cfg = dloader.load_cfg(dataset_id=args.dataset, arch_id=args.arch)
        benchmk = ImageBenchmark(
            archs=[args.arch], datasets=[args.dataset],
            datasets_dir=args.datasets_dir, models_dir=args.models_dir
        )
        models = benchmk.list_models(cfg=cfg, methods=methods)
        model1, test_loader, fingerprint = None, None, None
        for idx, model in enumerate(models):
            device = torch.device("cpu") if "quantize" in str(model) else args.device
            torch.set_num_threads(24)
            print(f"-> idx:{idx} runing for model:{model} seed:{model.seed}")
            if idx == 0:
                model1 = model.torch_model(seed=1000)
                test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=args.test_size)
                continue

            key = f"{model1.task}_{str(model)}"
            if key not in result.keys():
                result[key] = {
                    "L1": [],
                    "L2": [],
                    "cosine": [],
                    "Linf": []
                }
            model2 = model.torch_model(seed=model.seed)
            zest = ZEST(model1, model2, test_loader=test_loader, device=device, out_root=args.proj_root, seed=model.seed)
            fingerprint = zest.extract(cache=True)
            dist = zest.verify(fingerprint)
            for metric in dist.keys():
                result[key][metric].append(dist[metric])
            print(f"-> key:{key} dist:{dist}\n")
        print()
        if debug:
            cache = torch.load(path, map_location="cpu")
            result.update(cache)
        torch.save(result, path)
    result = torch.load(path, map_location="cpu")
    return tag, result


def exp11_eval_removalnet(args, metrics, steps):
    result = {}
    exp_path = osp.join(args.proj_root, f"exp/exp11_{args.dataset}_{args.arch}_r{args.model2}.pt")
    if not osp.exists(exp_path):
        benchmk = ImageBenchmark(
            archs=[args.arch], datasets=[args.dataset],
            datasets_dir=args.datasets_dir, models_dir=args.models_dir)
        model1 = benchmk.load_wrapper(args.model1, seed=1000).load_torch_model()
        model2 = benchmk.load_wrapper(args.model2, seed=args.seed).load_torch_model()
        test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=args.test_size)
        zest = ZEST(model1, model2, test_loader=test_loader, device=args.device, out_root=args.proj_root, seed=args.seed)

        key = f"{model1.task}_{model2.task}"
        result[key] = {}
        for metric in metrics:
            result[key][metric] = []

        zest_path = zest.fp_path
        phar = tqdm(steps)
        for step in phar:
            ckpt = osp.join(args.models_dir, model2.task, f'final_ckpt_s{args.seed}_t{step}.pth')
            if not osp.exists(ckpt):
                raise FileNotFoundError(f"-> model not found:{ckpt}")

            zest.fp_path = zest_path.split(".pt")[0] + f"_t{step}.pt"
            weights = torch.load(ckpt)["state_dict"]
            zest.model2.load_state_dict(weights)
            zest.model2.to(args.device)
            fingerprint = zest.extract(cache=True)
            item = zest.verify(fingerprint)
            for metric in item.keys():
                if metric not in metrics: continue
                result[key][metric].append(item[metric])
            phar.set_description(f"-> Removal({step}): ZEST {item}")
        print(f"-> Removal: {result[key]}")
        torch.save(result, exp_path)
    result = torch.load(exp_path, map_location="cpu")
    return result



def main():
    args = get_args()
    print(f"-> Running with config:{args}")

    metrics = ["L2", "cosine"]
    methods = ["distill", "finetune", "prune", "negative", "steal"]

    # eval baseline
    tag, results = exp11_eval(args, methods)
    fpath = osp.join(args.proj_root, f"pdf/exp11_{tag}_r{args.model2}.pdf")

    if args.removal:
        # eval removal attack
        steps = np.arange(800, 1000, 20)
        r_results = exp11_eval_removalnet(args, metrics=metrics, steps=steps)
        r_results.update(results)
        results = r_results
        methods = ["removalnet"] + methods

    # view via boxplot
    data = ops.exp11_normalize(results, methods=methods, metrics=metrics, defense_method="ZEST")
    vis.boxplot_distance(data["dists_nz"], metrics=data["legends"], xticks=data["xticks"], ylabel=data["ylabel"], fpath=fpath)



if __name__ == "__main__":
    main()









