#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


"""
This script is used to evaluate benchmark of ZEST
"""


import os, argparse, logging
import os.path as osp
import torch
import numpy as np
from exp import vis
from tqdm import tqdm
from utils import helper, metric, vis
from defense.ZEST.zest import ZEST
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
    parser.add_argument("-batch_size", required=False, type=int, default=100, help="tag of script.")
    parser.add_argument("-dataset", required=True, type=str, default="CIFAR10", help="model archtecture")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-gap", default=50, type=int, help="Gap between two pretrained model")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.zest_root = osp.join(args.out_root, "ZEST", "exp")
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


def rename_metric(key):
    if key == "2":
        key = "L2"
    elif key == "1":
        key = "L1"
    elif key == "inf":
        key = "Linf"
    elif key == "cosine":
        key = "Cosine"
    return key

def normalize(vs):
    return [(v - np.min(v)) / (np.max(v) - np.min(v) + 1e-6) for v in vs]


def exp21_eval(args):
    out_root = osp.join(args.out_root, "ZEST")
    benchmk = ImageBenchmark(archs=args.archs, datasets=[args.dataset],
        datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = benchmk.load_wrapper(args.model1, seed=args.seed).load_torch_model()

    benchmk = ImageBenchmark(archs=args.archs, datasets=[args.dataset],
                             datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model2 = benchmk.load_wrapper(args.model2, seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=128)

    path = osp.join(args.zest_root, f"exp21_{args.dataset}_{args.model2}.pt")
    device = torch.device("cpu") if "quantize" in str(args.model2) else args.device

    results = {
        # distance of different metric
        "dist": {
            "L1": [],
            "L2": [],
            "Cosine": [],
            "Linf": []
        },
        # accuracy of model
        "topk": {
            "top1": [],
            "top3": [],
            "top5": [],
        },
        # plot (x, y) pairs
        "plot": {
            "L1": [],
            "L2": [],
            "Cosine": [],
            "Linf": []
        }
    }

    print("-> path", path)
    if not osp.exists(path):
        phar = tqdm(np.arange(args.gap, 1000+args.gap, args.gap))
        for t in phar:
            # load intermediate model
            fname = f"final_ckpt_s{args.seed}_t{t}.pth"
            ckpt = osp.join(args.models_dir, args.model2, fname)
            if osp.exists(ckpt):
                model2.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
                model2.eval()
                model2.to(args.device)
            else:
                raise FileNotFoundError(f"-> FileNotFound: {ckpt}")

            # eval ZEST
            zest = ZEST(model1, model2, test_loader=test_loader, device=device, out_root=out_root, seed=args.seed)
            fingerprint = zest.extract(cache=False)
            dist = zest.verify(fingerprint)
            phar.set_description(f"-> model:{fname} dist:{dist}")
            for k, v in dist.items():
                results["dist"][rename_metric(k)].append(float(v))
            # eval accuracy
            _, topk_acc, _ = metric.topk_test(model2, test_loader, device=args.device, epoch=0)
            for k, v in topk_acc.items():
                results["topk"][k].append(round(float(v)/100.0, 4))
        # rectify results
        for k, v in results["topk"].items():
            results["topk"][k] = np.array(v, dtype=np.float)
        for k, v in results["dist"].items():
            k = rename_metric(k)
            results["dist"][k] = np.array(v, dtype=np.float)
        torch.save(results, path)
    else:
        results = torch.load(path, map_location="cpu")

    for k, v in results["dist"].items():
        k = rename_metric(k)
        results["dist"][k] = np.array(v, dtype=np.float)
        results["plot"][k] = []
        for x, y in zip(results["topk"]["top1"], v):
            results["plot"][k].append([x, y])
        results["plot"][k] = np.array(results["plot"][k], dtype=np.float)
        #item = np.array(results["plot"][k], dtype=np.float)
        #results["plot"][k] = item[item[:, 1].argsort()]
    vis.plot_accuracy_dist_curve(results["plot"], metrics=["Cosine"], path=path.replace(".pt", ".pdf"))


def main():
    args = get_args()
    exp21_eval(args)


if __name__ == "__main__":
    main()
















