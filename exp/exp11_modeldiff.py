#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


"""
This script is used to evaluate benchmark of Modeldiff
"""


import os, argparse, logging
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from utils import helper
from defense.ModelDiff.modeldiff import ModelDiff
from benchmark import ImageBenchmark
from dataset import loader as dloader
from . import vis, ops



def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="train(resnet50,CIFAR10)-", required=False,
                        help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", default="pretrain(resnet50,CIFAR10)-prune(0.5)-",
                        required=False, help="model 2.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-batch_size", required=False, type=int, default=100, help="tag of script.")
    parser.add_argument("-dataset", required=False, type=str, default="CIFAR10", help="Dataset for testing")
    parser.add_argument("-arch", required=True, type=str, help="Model archtecture",
                        choices=["resnet50", "densenet121", "mobilenet_v2", "vgg16_bn", "vgg19_bn"])
    parser.add_argument("-epsilon", action="store", default=0.2, type=float, help="Epsilon of ModelDiff")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-removal", default=0, type=int, choices=[0, 1], help="Eval Removal?")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.proj_root = osp.join(args.out_root, "ModelDiff")
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def exp11_eval(args, methods, debug=False):
    result = {}
    tag = f"{args.dataset}_{args.arch}_eps{args.epsilon}"
    path = osp.join(args.proj_root, f"exp/exp11_{tag}.pt")
    if debug or not osp.exists(path):
        cfg = dloader.load_cfg(dataset_id=args.dataset, arch_id=args.arch)
        benchmk = ImageBenchmark(
            archs=[args.arch], datasets=[args.dataset],
            datasets_dir=args.datasets_dir, models_dir=args.models_dir)
        models = benchmk.list_models(cfg=cfg, methods=methods)
        model1, test_loader, fingerprint = None, None, None

        for idx, model in enumerate(models):
            print(f"-> run:{str(model)} seed:{model.seed}")
            device = args.device
            torch.set_num_threads(16)
            if "quantize" in str(model):
                device = torch.device("cpu")
                torch.set_num_threads(24)

            if idx == 0:
                model1 = model.torch_model(seed=args.seed)
                test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=args.batch_size)
                continue

            key = f"{model1.task}_{str(model)}"
            if key not in result.keys():
                result[key] = {
                    "DDV": [],
                    "DDM": [],
                    "MR": [],
                    "WS": [],
                    "WS_abs": []
                }
            model2 = model.torch_model(seed=model.seed)
            modeldiff = ModelDiff(model1, model2, test_loader=test_loader, device=device, out_root=args.proj_root, seed=model.seed, epsilon=float(args.epsilon))
            dist = modeldiff.verify(modeldiff.extract())
            for metric in dist.keys():
                result[key][metric].append(dist[metric])
            print()

        if debug:
            cache = torch.load(path, map_location="cpu")
            result.update(cache)
        torch.save(result, path)
    result = torch.load(path, map_location="cpu")
    return tag, result


def exp11_eval_removalnet(args, metrics, steps):
    out_root = osp.join(args.out_root, "ModelDiff")
    exp_path = osp.join(args.proj_root, f"exp/exp11_{args.dataset}_{args.arch}_eps{args.epsilon}_r{args.model2}.pt")
    if osp.exists(exp_path):
        return torch.load(exp_path, map_location="cpu")

    benchmk = ImageBenchmark(
        archs=[args.arch],
        datasets=[args.dataset],
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir)
    model1 = benchmk.load_wrapper(args.model1, seed=1000).load_torch_model()
    model2 = benchmk.load_wrapper(args.model2, seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=args.batch_size)
    modeldiff = ModelDiff(model1, model2, test_loader=test_loader, device=args.device, out_root=out_root, seed=args.seed, epsilon=float(args.epsilon))

    key = f"{model1.task}_{model2.task}"
    result = {}
    result[key] = {}
    for metric in metrics:
        result[key][metric] = []

    fp_path = modeldiff.fp_path
    phar = tqdm(steps)
    for step in phar:
        ckpt = osp.join(args.models_dir, model2.task, f'final_ckpt_s{args.seed}_t{step}.pth')
        if not osp.exists(ckpt):
            raise FileNotFoundError(f"-> model not found:{ckpt}")

        modeldiff.fp_path = fp_path.split(".pt")[0] + f"_t{step}.pt"
        weights = torch.load(ckpt)["state_dict"]
        modeldiff.model2.load_state_dict(weights)
        modeldiff.model2.to(args.device)

        fingerprint = modeldiff.extract()
        item = modeldiff.verify(fingerprint)
        for metric in item.keys():
            if metric not in metrics: continue
            result[key][metric].append(item[metric])
        phar.set_description(f"-> Removal({step}): ModelDiff({args.epsilon}) {item}")
    print(f"-> Removal: {result[key]}")
    torch.save(result, exp_path)
    return result


def main():
    args = get_args()
    print(f"-> Running with config:{args}")

    metrics = ["DDM"]
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
    data = ops.exp11_normalize(results, methods=methods, metrics=metrics, defense_method="ModelDiff")
    vis.boxplot_distance(data["dists_nz"], metrics=data["legends"], xticks=data["xticks"], ylabel=data["ylabel"], fpath=fpath)



if __name__ == "__main__":
    main()









