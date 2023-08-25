#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/25, homeway'


"""
code for: 
    IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary
"""

import argparse
import os, datetime, pytz
import os.path as osp
import logging
import torch
from defense import Fingerprinting
from .adv import Adv
from . import ops
format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S"))
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class IPGuard(Fingerprinting):
    def __init__(self, model1, model2, test_loader, out_root, device, k, targeted="L",
                 steps=1000, test_size=200, seed=100):
        super().__init__(model1, model2, device=device, out_root=out_root)
        self.k = k
        self.steps = steps
        self.test_size = test_size
        self.targeted = targeted

        # init logger
        self.logger = logging.getLogger('IPGuard')
        self.logger.info(f'-> comparing {model1.task} vs {model2.task}')

        # init dataset
        self.seed = seed
        self.dataset = model1.dataset_id
        self.test_loader = test_loader
        self.bounds = self.test_loader.bounds

        # init model
        self.task1 = model1.task
        self.task2 = model2.task
        self.model1 = model1.to(self.device)
        self.model2 = model2.to(self.device)

    def extract(self):
        """
        extract fingerprint samples.
        :return:
        """
        self.logger.info("-> extract fingerprint...")
        path = osp.join(self.fingerprint_root, f"{self.dataset}_{self.task1}_t{self.targeted}k{self.k}.pt")
        print("-> load path", path)
        if osp.exists(path):
            self.logger.info(f"-> load from cache:{path}")
            return torch.load(path, map_location="cpu")

        adv = Adv(self.model1, bounds=self.bounds)
        count = 0
        test_x, test_y = [], []
        for x, y in self.test_loader:
            x, y = adv.IPGuard(x, y, k=self.k, targeted=self.targeted, steps=self.steps)
            test_x.append(x)
            test_y.append(y)
            count += len(x)
            if count >= self.test_size:
                break

        fingerprint = {
            "test_x": torch.cat(test_x)[:self.test_size],
            "test_y": torch.cat(test_y)[:self.test_size]
        }
        torch.save(fingerprint, path)
        return fingerprint

    def verify(self, fingerprint):
        """
        verify ownership between model1 & model2
        :return:
        """
        self.logger.info("-> verify ownership...")
        test_x, test_y = fingerprint["test_x"], fingerprint["test_y"]
        y1, y2 = test_y.cpu(), []
        with torch.no_grad():
            y2.append(ops.batch_forward(model=self.model2, x=test_x, argmax=True))
        y2 = torch.cat(y2)
        matching_rate = round(float(y1.eq(y2.view_as(y1)).sum()) / len(y1), 5)
        self.logger.info(f"-> {self.task1} vs {self.task2} matching_rate:{matching_rate}")
        return {"MR": matching_rate}

    def compare(self):
        return self.verify(self.extract())


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="train(resnet50,CIFAR10)-", required=False, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", default="pretrain(resnet50,CIFAR10)-prune(0.5)-", required=False, help="model 2.")
    parser.add_argument("-k", action="store", default=0.01, type=float, help="k of IPGuard")
    parser.add_argument("-targeted", action="store", default="L", type=str, help="L:lest-likely R:random", choices=["L","R"])
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-test_size", action="store", default=100, type=int, help="GPU device id")
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args.namespace = format_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.fingerprint_root = osp.join(args.out_root, "IPGuard")
    return args


def main():
    from benchmark import ImageBenchmark
    from dataset import loader as dloader
    args = get_args()
    ops.set_default_seed(args.seed)

    filename = str(osp.basename(__file__)).split(".")[0]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                        )#filename=f"{args.logs_root}/{filename}_{args.namespace}.txt")
    benchmark = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = benchmark.load_wrapper(args.model1, seed=1000).torch_model(seed=1000)
    model2 = benchmark.load_wrapper(args.model2, seed=args.seed).torch_model(seed=args.seed)

    if "quantize" in model1.task or "quantize" in model2.task:
        args.device = torch.device("cpu")

    test_loader = dloader.get_dataloader(dataset_id=model1.dataset_id, split="test", batch_size=args.test_size, shuffle=True)
    ipguard = IPGuard(model1=model1, model2=model2, test_loader=test_loader, device=args.device, out_root=args.fingerprint_root, targeted=args.targeted, k=args.k, seed=args.seed)
    dist = ipguard.compare()
    print(f"-> IPGuard dist: {dist}")



if __name__ == "__main__":
    main()

    """
        Example command:
        <===========================  Flower102-resnet18  ===========================>
        k=1
        t="R"
        model1="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.IPGuard.ipguard" -model1 $model1 -model2 "train(resnet18,Flower102)-" -k $k -targeted $t -device 1
        python -m "defense.IPGuard.ipguard" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-distill()-" -k $k -targeted $t -device 1
        python -m "defense.IPGuard.ipguard" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-stealthnet(0.7,20)-" -k $k -targeted $t -device 1
        
        
        <===========================  Flower102-mbnetv2  ===========================>
        model1="pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.IPGuard.ipguard" -model1 $model1 -model2 "train(mbnetv2,Flower102)-" -device 1
        python -m "defense.IPGuard.ipguard" -model1 $model1 -model2 "pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-distill()-" -device 1
        
    """






















