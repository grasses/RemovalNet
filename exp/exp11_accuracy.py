#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/10/08, homeway'


"""
This script is used to evaluate accuracy of pretrained model
"""


from torchsummary import summary
import os, argparse
import os.path as osp
import torch
from utils import helper, metric
from benchmark import ImageBenchmark
from model import loader as mloader
from dataset import loader as dloader


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-dataset", required=False, type=str, default="CIFAR10", help="dataset")
    parser.add_argument("-device", required=False, type=int, default=1, help="GPU device id")
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
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


def main():
    args = get_args()
    bench = ImageBenchmark(
        archs=args.archs[args.dataset],
        datasets=[args.dataset],
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir)
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test")

    cfg = dloader.load_cfg(dataset_id=args.dataset)
    for arch in args.archs[args.dataset]:
        model = bench.load_wrapper(name=f"train({arch},{args.dataset})-")
        torch_model = model.load_torch_model(seed=1000)
        _, _, _ = metric.topk_test(torch_model, test_loader=test_loader, device=args.device, epoch=0, debug=True)
        summary(torch_model, (3, cfg.input_size, cfg.input_size))
        print()


if __name__ == "__main__":
    main()





















