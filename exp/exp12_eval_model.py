import os.path as osp
import torch
import numpy as np
import argparse, pytz, os, datetime
from benchmark import ImageBenchmark
from utils import helper, metric
from dataset import loader as dloader
from model import loader as mloader
from torchmetrics.functional import pairwise_cosine_similarity
format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S"))
ROOT = os.path.abspath(os.path.dirname(__file__))


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model", type=str, required=True, help="model")
    parser.add_argument("-batch_size", default=1000, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    args, unknown = parser.parse_known_args()
    return args




args = get_args()
arch, dataset = args.model.split("(")[1].split(")")[0].split(",")
print(args.model, arch, dataset, args.seed)

test_loader = dloader.get_dataloader(dataset_id=dataset, split="test", batch_size=args.batch_size)
benchmk = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir, archs=[arch], datasets=dataset)
model = benchmk.load_wrapper(args.model, seed=args.seed).load_torch_model()
metric.topk_test(model, test_loader, device=args.device, epoch=0, debug=True)

exit(1)

path = f"/home/Hongwei/project/Model-Reuse/RemovalNet/model/ckpt/train(resnet50,CIFAR10)-finetune(CIFAR10,0.5)-/final_ckpt_s{args.seed}.pth"
weights = torch.load(path)["state_dict"]
model.load_state_dict(weights, strict=True)
metric.topk_test(model, test_loader, device=args.device, epoch=0, debug=True)


