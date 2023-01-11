#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'

"""
Official code of paper: 
https://github.com/testing4ai/deepjudge/blob/main/DeepJudge/metrics.py
"""

import os
import argparse
import logging
import torch
import scipy
import datetime
import pytz
import numpy as np
import os.path as osp
from defense import Fingerprinting
from dataset import loader
from . import ops
from .generation import BlackboxSeeding, WhiteboxSeeding
from utils import helper
format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S"))
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
out_root = osp.join(ROOT, "output")


class DeepJudge(Fingerprinting):
    def __init__(self, model1, model2, test_loader, device, out_root,
                 blackbox=False, seed_method="PGD", layer_index=4, m=3,
                 test_size=1000, batch_size=200, DIGISTS=4, seed=1000,):
        super().__init__(model1, model2, device=device, out_root=out_root)
        assert seed_method in ["FGSM", "PGD", "CW", "Random", "IPGuard"]
        self.metrics = ["ROB", "JSD", "LOD", "LAD", "NOD", "NAD", "MR"]
        if blackbox:
            self.metrics = ["ROB", "JSD", "MR"]
        self.m = round(float(m), 1)
        self.DIGISTS = DIGISTS
        self.layer_index = layer_index
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed_method = seed_method
        helper.set_default_seed(seed)

        # init logger
        self.logger = logging.getLogger('DeepJudge')
        self.logger.info(f'-> comparing {model1.task} vs {model2.task}')

        # init dataset
        self.test_loader = test_loader
        self.bounds = test_loader.bounds
        self.dataset = test_loader.dataset_id

        # init model
        self.task1 = model1.task
        self.task2 = model2.task
        self.model1 = model1.to(self.device)
        self.model2 = model2.to(self.device)

    def compare(self):
        return self.verify(self.extract())

    def extract(self):
        # step1: generate seed samples
        Bseed = BlackboxSeeding(self.model1, task=self.task1, test_loader=self.test_loader, dataset=self.dataset,
                                batch_size=self.batch_size, out_root=self.fingerprint_root)
        Wseed = WhiteboxSeeding(self.model1, task=self.task1, test_loader=self.test_loader, dataset=self.dataset,
                                batch_size=self.batch_size, out_root=self.fingerprint_root)
        seed_x, seed_y = Bseed.load_seed_samples(num=self.test_size)

        # step2: generate test samples
        test_x, test_y = Bseed.generate(seed_x=seed_x, seed_y=seed_y, method=self.seed_method)
        tests = Wseed.generate(seed_x=seed_x, seed_y=seed_y, layer_index=self.layer_index, m=self.m)
        min_size = min([len(v) for v in tests.values()])
        for k, v in tests.items():
            tests[k] = v[:min_size]

        fingerprint_data = {
            "whitebox": tests,
            "blackbox": [test_x, test_y],
        }
        return fingerprint_data

    def verify(self, fingerprint_data):
        """
        target_model: 受害者拥有的模型
        attack_model: 攻击者拥有的模型
        :return:
        """
        assert "blackbox" in fingerprint_data.keys()
        assert "whitebox" in fingerprint_data.keys()

        tests = fingerprint_data["whitebox"]
        test_x, test_y = fingerprint_data["blackbox"]

        # step3: multiple metrics testing
        target_model = self.model1
        attack_model = self.model2
        target_model.eval()
        attack_model.eval()
        result = {}
        for method in self.metrics:
            # run for black-box scenario
            if method == "ROB":
                result["ROB"] = self.metric_ROB(attack_model, test_x, test_y)
            elif method == "JSD":
                result["JSD"] = self.mertic_JSD(target_model, attack_model, test_x)
            elif method == "MR":
                result["MR"] = self.mertic_MR(target_model, attack_model, test_x)

            # run for white-box scenario
            elif method == "LOD":
                result["LOD"] = self.mertic_LOD(target_model, attack_model, tests=tests)
            elif method == "NOD":
                result["NOD"] = self.mertic_NOD(target_model, attack_model, tests=tests)
            elif method == "LAD":
                result["LAD"] = self.mertic_LAD(target_model, attack_model, tests=tests)
            elif method == "NAD":
                result["NAD"] = self.mertic_NAD(target_model, attack_model, tests=tests)
        self.logger.info(f"-> {self.task1} vs {self.task2} res:{result}")
        del test_x, test_y, tests
        return result

    def _numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def mertic_MR(self, model1, model2, advx, DIGISTS=4):
        advx = advx.clone().to(self.device)
        pred1 = ops.batch_forward(model1, x=advx, batch_size=self.batch_size, argmax=True).numpy()
        pred2 = ops.batch_forward(model2, x=advx, batch_size=self.batch_size, argmax=True).numpy()

        consist = int((pred1 == pred2).sum())
        dist = round(1.0 * consist / advx.size(0), DIGISTS)
        self.logger.info(f"-> mertic_MR() dist={dist}")
        return dist

    def metric_ROB(self, model, advx, advy, DIGISTS=4):
        """ Robustness (empirical)
        args:
            model: suspect model
            advx: black-box test cases (adversarial examples)
            advy: ground-truth labels

        return:
            Rob value
        """
        _advx = advx.clone().to(self.device)
        _advy = advy.clone().argmax(dim=1).numpy()
        _predy = ops.batch_forward(model, x=_advx, batch_size=self.batch_size, argmax=True)
        _predy = self._numpy(_predy)
        dist = round(np.sum(_predy == _advy) / _advy.shape[0], DIGISTS)
        self.logger.info(f"-> metric_ROB() dist={dist}")
        return dist

    def mertic_JSD(self, model1, model2, advx):
        """ Jensen-Shanon Distance
        args:
            model1 & model2: victim model and suspect model
            advx: black-box test cases

        return:
            JSD value
        """
        pred1 = ops.batch_forward(model1, x=advx, batch_size=self.batch_size, argmax=False).numpy()
        pred2 = ops.batch_forward(model2, x=advx, batch_size=self.batch_size, argmax=False).numpy()
        vectors1 = scipy.special.softmax(pred1, axis=1)
        vectors2 = scipy.special.softmax(pred2, axis=1)
        assert vectors1.shape[1] == vectors2.shape[1]
        mid = (vectors1 + vectors2) / 2
        distances = (scipy.stats.entropy(vectors1, mid, axis=1) + scipy.stats.entropy(vectors2, mid, axis=1))/2
        dist = round(np.average(distances), self.DIGISTS)
        self.logger.info(f"-> mertic_JSD() dist={dist}")
        return dist

    def mertic_LOD(self, model1, model2, tests, order=2):
        """ Layer Outputs Distance
        args:
            model1 & model2: victim model and suspect model
            tests: white-box test cases
            order: distance norm

        return:
            LOD value
        """
        lods = []
        for loc in tests.keys():
            samples = tests[loc].detach().clone().to(self.device)
            layer_index, idx = loc[0], loc[1]
            outputs1 = ops.batch_mid_forward(model1, samples, layer_index=layer_index)
            outputs2 = ops.batch_mid_forward(model2, samples, layer_index=layer_index)
            outputs1 = torch.mean(outputs1.view(outputs1.shape[0], -1, outputs1.shape[-1]), dim=1).numpy()
            outputs2 = torch.mean(outputs2.view(outputs1.shape[0], -1, outputs2.shape[-1]), dim=1).numpy()
            assert outputs1.shape == outputs2.shape
            lods.append(np.linalg.norm(outputs1 - outputs2, axis=1, ord=order))
        del samples, outputs1, outputs2
        dist = round(np.average(np.array(lods)), self.DIGISTS)
        self.logger.info(f"-> mertic_LOD() dist={dist}")
        return dist

    def mertic_LAD(self, model1, model2, tests, theta=0.5):
        """ Layer Activation Distance
        args:
            model1 & model2: victim model and suspect model
            tests: white-box test cases
            theta: activation threshold

        return:
            LAD value
        """

        def normalize(vs):
            return [(v - np.min(v)) / (np.max(v) - np.min(v) + 1e-6) for v in vs]

        lads = []
        for loc in tests.keys():
            samples = tests[loc].detach().clone().to(self.device)
            layer_index, idx = loc[0], loc[1]
            outputs1 = ops.batch_mid_forward(model1, samples, layer_index=layer_index)
            outputs2 = ops.batch_mid_forward(model2, samples, layer_index=layer_index)
            assert outputs1.shape == outputs2.shape

            outputs1 = torch.mean(torch.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), dim=1)
            outputs2 = torch.mean(torch.reshape(outputs2, (outputs1.shape[0], -1, outputs2.shape[-1])), dim=1)
            outputs1_normlized = normalize(self._numpy(outputs1))
            outputs2_normlized = normalize(self._numpy(outputs2))
            activations1 = np.array([np.where(i > theta, 1, 0) for i in outputs1_normlized])
            activations2 = np.array([np.where(i > theta, 1, 0) for i in outputs2_normlized])
            lads.append(np.linalg.norm(activations1 - activations2, axis=1, ord=1))
        del samples, outputs1, outputs2
        dist = round(np.average(np.array(lads)), self.DIGISTS)
        self.logger.info(f"-> mertic_LAD() dist={dist}")
        return dist

    def mertic_NOD(self, model1, model2, tests):
        """ Neuron Output Distance
        args:
            model1 & model2: victim model and suspect model
            tests: white-box test cases

        return:
            NOD value
        """
        nods = []
        for loc in tests.keys():
            samples = tests[loc].detach().clone().to(self.device)
            layer_index, idx = loc[0], loc[1]
            outputs1 = ops.batch_mid_forward(model1, samples, layer_index=layer_index)
            outputs2 = ops.batch_mid_forward(model2, samples, layer_index=layer_index)
            assert outputs1.shape == outputs2.shape
            outputs1 = torch.mean(torch.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), dim=1).numpy()
            outputs2 = torch.mean(torch.reshape(outputs2, (outputs1.shape[0], -1, outputs2.shape[-1])), dim=1).numpy()
            nods.append(np.abs(outputs1[:, idx] - outputs2[:, idx]))
        del samples, outputs1, outputs2
        dist = round(np.average(np.array(nods)), self.DIGISTS)
        self.logger.info(f"-> mertic_NOD() dist={dist}")
        return dist

    def mertic_NAD(self, model1, model2, tests, theta=0.5):
        """ Neuron Activation Distance
        args:
            model1 & model2: victim model and suspect model
            tests: white-box test cases
            theta: activation threshold

        return:
            NAD value
        """
        def normalize(vs):
            return [(v - np.min(v)) / (np.max(v) - np.min(v) + 1e-6) for v in vs]

        nads = []
        for loc in tests.keys():
            samples = tests[loc].detach().clone().to(self.device)
            layer_index, idx = loc[0], loc[1]
            outputs1 = ops.batch_mid_forward(model1, samples, layer_index=layer_index)
            outputs2 = ops.batch_mid_forward(model2, samples, layer_index=layer_index)
            assert outputs1.shape == outputs2.shape
            outputs1 = torch.mean(torch.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), dim=1)
            outputs2 = torch.mean(torch.reshape(outputs2, (outputs2.shape[0], -1, outputs2.shape[-1])), dim=1)
            outputs1_normlized = normalize(self._numpy(outputs1))
            outputs2_normlized = normalize(self._numpy(outputs2))
            activations1 = np.array([np.where(i > theta, 1, 0) for i in outputs1_normlized])
            activations2 = np.array([np.where(i > theta, 1, 0) for i in outputs2_normlized])
            nads.append(np.abs(activations1[:, idx] - activations2[:, idx]))
        del samples, outputs1, outputs2
        dist = round(np.average(np.array(nads)) * len(tests), self.DIGISTS)
        self.logger.info(f"-> mertic_NAD() dist={dist}")
        return dist



def benchmark_testing(args, victim):
    from benchmark import ImageBenchmark
    benchmark = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    benchmark.list_models()

    result_dict = {}
    test_cache = None
    out_path = osp.join(args.out_root, f"DeepJudge/exp/results_{victim}.pt")
    cache_path = osp.join(args.out_root, f"DeepJudge/exp/samples_{victim}.pt")
    if osp.exists(cache_path):
        test_cache = torch.load(cache_path)

    for model_wrapper in benchmark.list_models():
        if not model_wrapper.torch_model_exists():
            continue
        device = args.device
        name = model_wrapper.__str__()
        if (victim in name) and (len(name) > len(victim)):
            print(f"-> DeepJudge compare for {victim} vs {name}")
            model1 = benchmark.get_model_wrapper(victim)
            if "quantize" in str(model1) or "quantize" in str(name):
                device = torch.device("cpu")
            try:
                comparison = DeepJudge(model1, model_wrapper, device=device)
                result_dict[f"{victim}_{name}"] = comparison.compare(test_cache=test_cache)
                torch.save(result_dict, out_path)
            except Exception as e:
                print(f"-> error: {e}")


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
    parser.add_argument("-seed_method", action="store", default="PGD", type=str, choices=["FGSM", "PGD", "CW"], help="Type of blackbox generation")
    parser.add_argument("-batch_size", action="store", default=200, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args.namespace = format_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.deepjudge_root = osp.join(args.out_root, "DeepJudge")
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

    model1 = benchmark.load_wrapper(args.model1, seed=args.seed).torch_model(seed=args.seed)
    model2 = benchmark.load_wrapper(args.model2, seed=args.seed).torch_model(seed=args.seed)
    test_loader = loader.get_dataloader(model1.dataset_id)

    layer_index = 3
    if "quantize" in str(model1) or "quantize" in str(model2):
        args.device = torch.device("cpu")

    out_root = osp.join(args.out_root, "DeepJudge")
    deepjudge = DeepJudge(model1, model2, test_loader=test_loader,
                          device=args.device, seed=args.seed, out_root=out_root, batch_size=args.batch_size,
                          layer_index=layer_index, seed_method=args.seed_method)
    dist = deepjudge.verify(deepjudge.extract())
    print(f"-> DeepJudge dist: {dist}")


if __name__ == "__main__":
    main()

    """
    Example command:
    <===========================  Flower102-resnet18:20220901_Test  ===========================>
    SCRIPT="defense.DeepJudge.deepjudge"
    
    model1="train(resnet18,Flower102)-"
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-" -device 0
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-stealthnet(0.7,20)-" -device 0
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-" -device 0
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-stealthnet(0.7,20)-" -device 0
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-" -device 0  
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-stealthnet(0.7,20)-" -device 0
    
    
    model1="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-"
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-" -device 1
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-stealthnet(0.7,20)-" -device 1
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.5)-" -device 1
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.5)-stealthnet(0.7,20)-" -device 1
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.8)-" -device 1
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.8)-stealthnet(0.7,20)-" -device 1
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-distill()-" -device 1
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-distill()-stealthnet(0.7,20)-" -device 1
    
    
    model1="pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-"
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.2)-" -device 2
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.2)-stealthnet(0.7,20)-" -device 2
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.5)-" -device 2
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.5)-stealthnet(0.7,20)-" -device 2
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.8)-" -device 2
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.8)-stealthnet(0.7,20)-" -device 2
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-distill()-" -device 2
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-distill()-stealthnet(0.7,20)-" -device 2
    
    
    model1="pretrain(resnet18,ImageNet)-transfer(Flower102,1)-"
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.2)-" -device 3
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.2)-stealthnet(0.7,20)-" -device 3    
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.5)-" -device 3
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.5)-stealthnet(0.7,20)-" -device 3
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.8)-" -device 3
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.8)-stealthnet(0.7,20)-" -device 3
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-distill()-" -device 3
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-distill()-stealthnet(0.7,20)-" -device 3
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(resnet18)-" -device 0
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(resnet18)-stealthnet(0.7,20)-" -device 0
    
    
    
    <===========================  Flower102-mbnetv2  ===========================>
    model1="train(mbnetv2,Flower102)-"
    
    
    model1="pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-"
    python -m $SCRIPT -model1 $model1 -model2 "pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-stealthnet(0.7,20)-" -device 0
    
    
    
    <===========================  SDog120-resnet18  ===========================>
    model1="train(resnet18,SDog120)-"
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-stealthnet(0.7,20)-" -device 0
    
    
    model1="pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-"
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "train(resnet18,SDog120)-" -device 0
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-prune(0.2)" -device 0
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-stealthnet(0.7,20)-" -device 0
    
    
    
    model1="pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-"
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "train(resnet18,SDog120)-" -device 0
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-" -device 0
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-stealthnet(0.7,20)-" -device 0
    
    
    <===========================  SDog120-mbnetv2  ===========================>
    model1="pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-"
    python -m "defense.DeepJudge.deepjudge" -model1 $model1 -model2 "pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-stealthnet(0.7,20)-" -device 0
    
    
    """



    '''
    #benchmark_testing(victim="pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-")
    benchmark_testing(victim="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-")
    benchmark_testing(victim="train(mbnetv2,Flower102)-")
    benchmark_testing(victim="train(mbnetv2,SDog120)-")
    benchmark_testing(victim="train(resnet18,Flower102)-")
    benchmark_testing(victim="train(resnet18,SDog120)-")
    '''
