#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/25, homeway'


"""
code for: 
    Fingerprinting Deep Neural Networks - A DeepFool Approach
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


class DeepFoolFP(Fingerprinting):
    def __init__(self, model1, model2, out_root, device, batch_size=100,
                 k=20, steps=1000, test_size=100, target_label="least_likely"):
        super().__init__(model1, model2, device=device, out_root=out_root)
        self.k = k
        self.steps = steps
        self.test_size = test_size
        self.batch_size = batch_size
        self.target_label = target_label

        # init logger
        self.logger = logging.getLogger('DeepFoolFP')
        self.logger.info(f'-> comparing {model1} vs {model2}')

        # init dataset
        self.dataset = model1.dataset_id
        self.test_loader = model1.get_test_loader(batch_size=batch_size, shuffle=True)
        self.bounds = self.test_loader.bounds

        # init model
        self.arch1 = str(model1)
        self.arch2 = str(model2)
        self.model1 = model1.torch_model.to(self.device)
        self.model2 = model2.torch_model.to(self.device)

    def extract(self):
        """
        extract fingerprint samples.
        :return:
        """
        self.logger.info("-> extract fingerprint...")
        path = osp.join(self.fingerprint_root, f"{self.arch1}_{self.dataset}.pt")
        if osp.exists(path):
            self.logger.info(f"-> load from cache:{path}")
            return torch.load(path, map_location="cpu")

        adv = Adv(self.model1, bounds=self.bounds)
        count = 0
        test_x, test_y = [], []
        for x, y in self.test_loader:
            x, y = adv.deepfool(x, y, k=self.k, target=self.target_label, steps=self.steps)
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
        self.logger.info(f"-> {self.arch1} vs {self.arch2} matching_rate:{matching_rate}")
        return matching_rate

    def compare(self):
        return self.verify(self.extract())


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="pretrain(resnet18,ImageNet)-",
                        required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2",
                        default="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-", required=True, help="model 2.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-batch_size", action="store", default=200, type=int, help="GPU device id")
    parser.add_argument("-seed", default=999, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args.namespace = format_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.fingerprint_root = osp.join(args.out_root, "DeepFoolFP")
    return args


def main():
    from benchmark import ImageBenchmark
    args = get_args()
    ops.set_default_seed(args.seed)

    filename = str(osp.basename(__file__)).split(".")[0]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                        )#filename=f"{args.logs_root}/{filename}_{args.namespace}.txt")
    benchmark = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = benchmark.get_model_wrapper(args.model1)
    model2 = benchmark.get_model_wrapper(args.model2)
    if "quantize" in str(model1) or "quantize" in str(model2):
        args.device = torch.device("cpu")

    deepfoolfp = DeepFoolFP(model1=model1, model2=model2, device=args.device, out_root=args.fingerprint_root)
    rate = deepfoolfp.compare()
    print(f"-> IPGuard matching_rate: {rate}")



if __name__ == "__main__":
    main()

    """
        Example command:
        <===========================  Flower102-resnet18  ===========================>
        model1="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.DeepFoolFP.deepfool" -model1 $model1 -model2 "train(resnet18,Flower102)-" -device 1
        
        
        <===========================  Flower102-mbnetv2  ===========================>
        model1="pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.DeepFoolFP.deepfool" -model1 $model1 -model2 "pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-stealthnet(0.7,20)-" -device 1
    """






















