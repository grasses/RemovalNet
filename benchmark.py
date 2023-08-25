#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'

import re
import os
import logging
import torch
import shutil
import numpy as np
from model import loader as mloader
from dataset import loader as dloader
from utils import helper, metric
from attack.finetuner import Finetuner
from attack.trainer import Trainer
from attack.weight_pruner import WeightPruner
import argparse
import os.path as osp
CONTINUE_TRAIN = False

class ModelWrapper:
    def __init__(self, benchmark, teacher_wrapper, trans_str,
                 seed=1000, arch_id=None, dataset_id=None, iters=100, fc=True, **kwargs):
        self.logger = logging.getLogger('ModelWrapper')
        self.benchmark = benchmark
        self.teacher_wrapper = teacher_wrapper
        self.trans_str = trans_str
        self.arch_id = arch_id if arch_id else teacher_wrapper.arch_id
        self.dataset_id = dataset_id if dataset_id else teacher_wrapper.dataset_id
        self.torch_model_path = os.path.join(benchmark.models_dir, f'{self.__str__()}')
        self.iters = iters
        self.fc = fc
        self.seed = 1000 if (teacher_wrapper is None) else int(seed)
        self.ckpt_path = os.path.join(self.torch_model_path, f'final_ckpt_s{seed}.pth')
        self.cfg = dloader.load_cfg(self.dataset_id, self.arch_id)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if "quantize" in trans_str:
            self.cfg.device = torch.device("cpu")
        assert self.arch_id is not None
        assert self.dataset_id is not None

    def __str__(self):
        teacher_str = '' if self.teacher_wrapper is None else self.teacher_wrapper.__str__()
        return f'{teacher_str}{self.trans_str}-'

    def __call__(self, *args, **kwargs):
        return self.torch_model(*args, **kwargs)

    def batch_forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        m = re.match(r'(\S+)\((\S*)\)', self.trans_str)
        method = m.group(1)
        if method == "quantize":
            inputs = inputs.to("cpu")
        else:
            inputs = inputs.to(self.cfg.device)
        self.torch_model_on_device.eval()
        with torch.no_grad():
            return self.torch_model_on_device(inputs)

    @helper.lazy_property
    def torch_model_on_device(self):
        m = re.match(r'(\S+)\((\S*)\)', self.trans_str)
        method = m.group(1)
        if method == "quantize":
            return self.torch_model.to("cpu")
        else:
            print(f"-> model on device:{self.cfg.device}")
            return self.torch_model.to(self.cfg.device)

    def eval(self, torch_model):
        topk_acc = {
            "top1": 0,
            "top3": 0,
            "top5": 0
        }
        if self.dataset_id == "ImageNet":
            return topk_acc
        test_loader = dloader.get_dataloader(self.dataset_id, split='test')
        _, topk_acc, _ = metric.topk_test(torch_model, test_loader, epoch=0, debug=True, device=self.cfg.device)
        return topk_acc

    def torch_model_exists(self, **kwargs):
        return os.path.exists(self.ckpt_path)

    def save_torch_model(self, torch_model, **kwargs):
        if not os.path.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)
        topk_acc = self.eval(torch_model)
        torch.save(
            {
                'top1_acc': topk_acc["top1"],
                'top3_acc': topk_acc["top3"],
                'top5_acc': topk_acc["top5"],
                'iters': self.iters,
                'seed': self.seed,
                'state_dict': torch_model.state_dict()
            },
            self.ckpt_path,
        )

    def load_saved_weights(self, torch_model, **kwargs):
        """
        load weights in the latest checkpoint to torch_model
        """
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            torch_model.load_state_dict(ckpt['state_dict'], state_dict=True)
            self.logger.info('load_saved_weights: loaded a previous checkpoint')
        else:
            self.logger.info('load_saved_weights: no previous checkpoint found')
        return torch_model

    def load_torch_model(self, **kwargs):
        """
        load the model object from torch_model_path
        :return: torch.nn.Module object
        """
        torch_model = mloader.load_model(self.dataset_id, self.arch_id, pretrained=False)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")

        m = re.match(r'(\S+)\((\S*)\)', self.trans_str)
        method = m.group(1)
        params = m.group(2).split(',')

        if method == "negative" and "vit" in self.trans_str:
            from model.inputx224.ViT import vit_base_patch32_224_sam
            torch_model = vit_base_patch32_224_sam(num_classes=1000, pretrained=True)

        if method == 'quantize':
            dtype = params[0]
            dtype = torch.qint8 if dtype == 'qint8' else torch.float16
            # load from teacher model & quantize
            self.teacher_wrapper.gen_model(seed=1000)
            teacher_model = self.teacher_wrapper.load_torch_model()
            torch_model.load_state_dict(teacher_model.state_dict(), strict=True)
            torch_model = torch.quantization.quantize_dynamic(torch_model, qconfig_spec={torch.nn.Linear}, inplace=True, dtype=dtype)
            print("-> load model from: quantize!!!!!")
        else:
            torch_model.load_state_dict(ckpt['state_dict'], strict=True)
            print(f"-> load model from:{self.ckpt_path}")
        torch_model.seed = self.seed
        torch_model.task = self.__str__()
        torch_model.arch_id = self.arch_id
        torch_model.dataset_id = self.dataset_id
        return torch_model

    @helper.lazy_property
    def torch_model(self):
        return self.load_torch_model

    def gen_model(self, seed=1000, regenerate=False, **kwargs):
        """
        TODO: Rewrite this function!!!
        generate the torch model, seed=1000 is the default seed of teacher model
        :return:
        """
        self.seed = seed
        helper.set_default_seed(self.seed)
        trans_str = self.trans_str
        if not regenerate and self.torch_model_exists():
            self.logger.info(f'-> model already exists: {self.__str__()}')
            return

        self.logger.info(f'-> generating model for: {self.__str__()}')
        m = re.match(r'(\S+)\((\S*)\)', trans_str)
        method = m.group(1)
        params = m.group(2).split(',')

        if regenerate and os.path.exists(self.torch_model_path) and (method != 'quantize'):
            shutil.rmtree(self.torch_model_path)
        if not os.path.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)

        teacher_model = None
        if self.teacher_wrapper:
            self.teacher_wrapper.gen_model(seed=1000)
            teacher_model = self.teacher_wrapper.load_torch_model()
        train_loader = dloader.get_dataloader(self.dataset_id, split='train')
        test_loader = dloader.get_dataloader(self.dataset_id, split='test')

        cfg = dloader.load_cfg(self.dataset_id, self.arch_id)
        cfg.iterations = self.iters
        cfg.output_dir = self.torch_model_path
        cfg.seed = self.seed
        cfg.task_str = str(self.__str__() + f"_seed{cfg.seed}")
        if method == 'pretrain':
            # load pretrained model as specified by arch_id and save it to model path
            arch_id = params[0]
            dataset_id = params[1]
            if dataset_id != 'ImageNet':
                self.logger.warning(f'gen_model: pretrained model on {dataset_id} not supported')
                exit(1)

            torch_model = mloader.load_model(
                dataset_id=dataset_id,
                arch_id=arch_id,
                pretrained=False,
                pretrain="imagenet"
            )
            self.save_torch_model(torch_model)

        elif method == 'train':
            # train the model from scratch
            arch_id = params[0]
            dataset_id = params[1]
            torch_model = mloader.load_model(
                dataset_id=dataset_id,
                arch_id=arch_id,
                pretrained=False
            )
            cfg.network = self.arch_id
            cfg.ft_ratio = 1
            cfg.reinit = True
            cfg.weight_decay = 5e-3
            cfg.momentum = 0.8
            if CONTINUE_TRAIN:
                torch_model = self.load_saved_weights(torch_model)  # continue training
            finetuner = Finetuner(
                cfg,
                torch_model, torch_model,
                train_loader, test_loader,
                init_models=False
            )
            finetuner.train()
            self.save_torch_model(torch_model)

        elif method == 'quantize':
            dtype = params[0]
            dtype = torch.qint8 if dtype == 'qint8' else torch.float16
            student_model = mloader.load_model(dataset_id=self.dataset_id, arch_id=self.arch_id)
            student_model.load_state_dict(teacher_model.state_dict(), strict=True)
            student_model = torch.quantization.quantize_dynamic(student_model, dtype=dtype, inplace=True)
            self.save_torch_model(student_model, seed=seed)

        elif method == 'prune':
            prune_ratio = float(params[0])
            student_model = mloader.load_model(dataset_id=self.dataset_id, arch_id=self.arch_id)
            student_model.load_state_dict(teacher_model.state_dict(), strict=True)
            #cfg.method = "weight"
            cfg.network = self.arch_id
            cfg.weight_ratio = prune_ratio
            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training
            finetuner = WeightPruner(
                cfg,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            self.save_torch_model(student_model, seed=seed)
            finetuner.final_check_param_num()

        elif method == 'finetune':
            dataset_id = params[0]
            tune_ratio = float(params[1])
            cfg.ft_ratio = tune_ratio
            cfg.network = self.arch_id
            cfg.lr = cfg.finetune_lr
            student_model = mloader.load_model(dataset_id=dataset_id, arch_id=self.arch_id)
            student_model.load_state_dict(teacher_model.state_dict(), strict=True)
            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training
            finetuner = Finetuner(
                cfg,
                student_model, teacher_model,
                train_loader, test_loader,
                init_models=False
            )
            finetuner.train()
            self.save_torch_model(student_model, seed=seed)

        elif method == 'retraining':
            dataset_id = params[0]
            tune_ratio = float(params[1])
            cfg.ft_ratio = tune_ratio
            cfg.retrain_linear = True
            cfg.network = self.arch_id
            student_model = mloader.load_model(dataset_id=dataset_id, arch_id=self.arch_id)
            student_model.load_state_dict(teacher_model.state_dict(), strict=True)
            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training
            finetuner = Finetuner(
                cfg,
                student_model, teacher_model,
                train_loader, test_loader,
                init_models=True
            )
            finetuner.train()
            self.save_torch_model(student_model, seed=seed)

        elif method == 'distill':
            cfg.feat_lmda = 0.1
            cfg.network = self.arch_id
            cfg.weight_decay = 1e-4
            cfg.momentum = 0.8
            cfg.lr = 1e-3
            cfg.reinit = False
            #cfg.retrain_linear = float(params[0])
            student_model = mloader.load_model(
                dataset_id=self.dataset_id,
                arch_id=self.arch_id,
                pretrained=False
            )
            student_model.load_state_dict(teacher_model.state_dict(), strict=True)
            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training
            finetuner = Finetuner(
                cfg,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            self.save_torch_model(student_model, seed=seed)

        elif method == 'steal':
            arch_id = params[0]
            student_model = mloader.load_model(
                dataset_id=self.dataset_id,
                arch_id=self.arch_id,
                pretrained=False
            )
            cfg.network = arch_id
            cfg.steal = True
            cfg.reinit = True
            cfg.retrain_linear = 1.0
            cfg.steal_alpha = 0.5
            cfg.temperature = 1.0
            cfg.weight_decay = 5e-3
            cfg.momentum = 0.9
            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training
            finetuner = Finetuner(
                cfg,
                student_model, teacher_model,
                train_loader, test_loader,
                init_models=False
            )
            finetuner.train()
            self.save_torch_model(student_model, seed=seed)

        elif method == "negative":
            arch_id = params[0]
            # use output distillation to transfer teacher knowledge to another architecture
            student_model = mloader.load_model(
                dataset_id=self.dataset_id,
                arch_id=self.arch_id,
                pretrained=False
            )
            cfg.network = arch_id
            cfg.negative = True
            cfg.reinit = True
            cfg.weight_decay = 5e-3
            cfg.momentum = 0.9
            cfg.backends = False
            finetuner = Trainer(
                cfg,
                student_model, teacher_model,
                train_loader, test_loader
            )
            finetuner.train()
            self.save_torch_model(student_model, seed=seed)
        else:
            raise RuntimeError(f'unknown transformation: {method}')

    def knockoff(self, arch, subset, seed=1000, **kwargs):
        trans_str = f'knockoff({arch},{subset})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def quantize(self, dtype='qint8', seed=1000, **kwargs):
        """
        do post-training quantization on the model
        :param dtype: qint8 or float16
        :return:
        """
        trans_str = f'quantize({dtype})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def prune(self, prune_ratio=0.1, iters=10000, seed=1000, **kwargs):
        trans_str = f'prune({prune_ratio})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            iters=iters,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def finetune(self, dataset_id, tune_ratio=0.1, iters=10000, seed=1000, **kwargs):
        trans_str = f'finetune({dataset_id},{tune_ratio})'
        # model_wrapper is the wrapper of the student model
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            dataset_id=dataset_id,
            iters=iters,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def retraining(self, dataset_id, tune_ratio=0.1, iters=10000, seed=1000, **kwargs):
        trans_str = f'retraining({dataset_id},{tune_ratio})'
        # model_wrapper is the wrapper of the student model
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            dataset_id=dataset_id,
            iters=iters,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def distill(self, retrain_ratio, iters=10000, seed=1000, **kwargs):
        trans_str = f'distill({retrain_ratio})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            iters=iters,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def steal(self, arch_id, iters=10000, seed=1000, **kwargs):
        trans_str = f'steal({arch_id})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            arch_id=arch_id,
            iters=iters,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def negative(self, arch_id, iters=10000, seed=1000, **kwargs):
        trans_str = f'negative({arch_id})'
        # init param & retrain the model using ground-truth label
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            arch_id=arch_id,
            iters=iters,
            seed=seed,
            **kwargs
        )
        return model_wrapper

    def removalnet(self, dataset_id, iters=10000, seed=1000, **kwargs):
        """TODO: RemovalNet, what to save as params"""
        keyword = ""
        for k in kwargs.keys():
            keyword += f"{kwargs[k]},"
        trans_str = f'removalnet({dataset_id},{keyword[:-1]})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            iters=iters,
            seed=seed,
            dataset_id=dataset_id,
            **kwargs
        )
        return model_wrapper


class ImageBenchmark:
    def __init__(self, datasets, archs, datasets_dir='dataset/data', models_dir='model/ckpt'):
        self.logger = logging.getLogger('ImageBench')
        self.archs = [archs] if type(archs) == str else archs
        self.datasets = [datasets] if type(datasets) == str else datasets
        self.datasets_dir = datasets_dir
        self.models_dir = models_dir

    def load_pretrained(self, arch_id, seed=1000, fc=True):
        """
        Get the model pretrained on imagenet
        :param arch_id: the name of the arch
        :return: a ModelWrapper instance
        """
        model_wrapper = ModelWrapper(
            benchmark=self,
            teacher_wrapper=None,
            trans_str=f'pretrain({arch_id},ImageNet)',
            arch_id=arch_id,
            dataset_id='ImageNet',
            fc=fc,
            seed=1000
        )
        return model_wrapper

    def load_trained(self, arch_id, dataset_id, seed=1000, iters=10000, fc=True):
        """
        Get the model with architecture arch_id trained on dataset dataset_id
        :param arch_id: the name of the arch
        :param dataset_id: the name of the dataset
        :param iters: number of iterations
        :return: a ModelWrapper instance
        """
        model_wrapper = ModelWrapper(
            benchmark=self,
            teacher_wrapper=None,
            trans_str=f'train({arch_id},{dataset_id})',
            arch_id=arch_id,
            dataset_id=dataset_id,
            iters=iters,
            fc=fc,
            seed=1000
        )
        self.logger.info(f"-> load trained model:{str(model_wrapper)}")
        return model_wrapper

    def load_wrapper(self, name, seed=1000, fc=True, **kwargs):
        """
        Get model by name.
        :param name:
        :param fc:
        :param kwargs:
        :return:
        """
        m = name.split("-")[:-1]
        def extract(name):
            gen_type = str(name.split("(")[0])
            params = name.split("(")[1].split(")")[0].split(",")
            return gen_type, params

        gen_type, (arch_id, dataset_id) = extract(m[0])
        if gen_type == "pretrain":
            source_model = self.load_pretrained(arch_id, fc=fc, seed=1000)
        elif gen_type == "train":
            source_model = self.load_trained(arch_id, dataset_id=dataset_id, fc=fc, seed=1000)
        else:
            raise NotImplementedError(f"-> [ERROR] method:{gen_type} not found!")

        target_model = source_model
        for item in list(m[1:]):
            gen_type, params = extract(item)
            if gen_type == "transfer":
                target_model = target_model.transfer(dataset_id=params[0], tune_ratio=params[1], seed=seed, **kwargs)
            elif gen_type == "finetune":
                target_model = target_model.finetune(dataset_id=params[0], tune_ratio=params[1], seed=seed, **kwargs)
            elif gen_type == "retraining":
                target_model = target_model.retraining(dataset_id=params[0], tune_ratio=params[1], seed=seed, **kwargs)
            elif gen_type == "distill":
                target_model = target_model.distill(retrain_ratio=params[0], seed=seed, **kwargs)
            elif gen_type == "prune":
                target_model = target_model.prune(params[0], seed=seed, **kwargs)
            elif gen_type == "quantize":
                target_model = target_model.quantize(params[0], seed=seed, **kwargs)
            elif gen_type == "steal":
                target_model = target_model.steal(params[0], seed=seed, **kwargs)
            elif gen_type == "negative":
                target_model = target_model.negative(params[0], seed=seed, **kwargs)
            elif gen_type == "knockoff":
                target_model = target_model.knockoff(params[0], params[1], seed=seed, **kwargs)
            elif gen_type == "removalnet":
                r = float(params[1])
                if round(r, 2) - round(r, 1) != 0:
                    rate = round(r, 2)
                else:
                    rate = round(r, 1)
                target_model = target_model.removalnet(dataset_id=params[0], rate=rate, alpha=params[2], beta=params[3], T=params[4], layer=params[5], seed=seed, **kwargs)
            else:
                raise NotImplementedError(f"-> [ERROR] method:{gen_type} not found!")
        self.logger.info(f"-> load model: {target_model}")
        return target_model

    def list_models(self, cfg, fc=True, seeds=None, methods=["negative", "finetune", "distill", "steal", "prune"]):
        """
        list the models in the benchmark dataset
        :return: a stream of ModelWrapper instances
        """
        source_models = []
        quantization_dtypes = ['qint8', 'float16']
        prune_ratios = [0.5, 0.8]
        finetune_ratios = [0.5, 0.8]
        distill_ratios = [1.0]
        seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] if seeds is None else seeds

        # train source models
        source_models = []
        for arch_id in self.archs:
            for dataset_id in self.datasets:
                source_model = self.load_trained(arch_id, dataset_id, iters=cfg.TRAIN_ITERS, seed=1000, fc=fc)
                source_models.append(source_model)
                yield source_model

        if "negative" in methods:
            # independent training, negative models
            for source_model in source_models:
                for seed in seeds:
                    negative_model = source_model.negative(arch_id=arch_id, iters=cfg.NEGATIVE_ITERS, seed=seed)
                    yield negative_model

        if "retraining" in methods:
            for retrain_model in source_models:
                for ratio in finetune_ratios:
                    for seed in seeds:
                        yield retrain_model.retraining(dataset_id=dataset_id, iters=cfg.FINETUNING_ITERS, tune_ratio=ratio, seed=seed)

        if "finetune" in methods:
            for retrain_model in source_models:
                for ratio in finetune_ratios:
                    for seed in seeds:
                        yield retrain_model.finetune(dataset_id=dataset_id, iters=cfg.FINETUNING_ITERS, tune_ratio=ratio, seed=seed)

        if "prune" in methods:
            # - M_{i,x}/{prune-p} -- Prune M_{i,x} with pruning ratio = p
            for retrain_model in source_models:
                for pr in prune_ratios:
                    for seed in seeds:
                        yield retrain_model.prune(prune_ratio=pr, iters=cfg.PRUNE_ITERS, seed=seed)

        if "quantize" in methods:
            for retrain_model in source_models:
                for quantization_dtype in quantization_dtypes:
                    for seed in seeds:
                        yield retrain_model.quantize(dtype=quantization_dtype, iters=cfg.QUANTIZE_ITERS, seed=seed)

        if "distill" in methods:
            # - M_{i,x}/{distill} -- Distill M_{i,x}
            for retrain_model in source_models:
                for ratio in distill_ratios:
                    for seed in seeds:
                        yield retrain_model.distill(retrain_ratio=ratio, iters=cfg.DISTILL_ITERS, seed=seed)

        if "steal" in methods:
            # - M_{i,x}/{steal-j} -- Steal M_{i,x} to A_j
            for retrain_model in source_models:
                for arch_id in self.archs:
                    for seed in seeds:
                        yield retrain_model.steal(arch_id=arch_id, iters=cfg.STEAL_ITERS, seed=seed)


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", required=False, action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-regenerate", action="store_true", dest="regenerate", default=False,
                        help="Whether to regenerate the models.")
    parser.add_argument("-model1", action="store", dest="model1", default="pretrain(resnet18,ImageNet)-",
                        required=False, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", default="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-",
                        required=False, help="model 2.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-dataset", required=False, type=str, default="CIFAR10", help="model archtecture")
    parser.add_argument("-subset", required=False, type=str, default=None, help="surrogate dataset")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(helper.ROOT, "output")
    args.logs_root = osp.join(helper.ROOT, "logs")

    # support datasets: CIFAR10, CINIC10, CelebA, LFW, VGGFace2, SkinCancer, HAM10000, BCN20000, ImageNet
    # support architectures: resnet50, vgg16_bn, vgg19_bn, densenet121, mobilenet_v2
    args.subset = args.dataset if args.subset is None else args.subset
    args.archs = {
        "CIFAR10": ["vgg19_bn"],
        "CINIC10": ["resnet50"],
        "GTSRB": ["inception_v3"],
        "GTSRB+1": ["inception_v3"],
        "SkinCancer": ["resnet50"],
        "HAM10000": ["inception_v3"],
        "BCN20000": ["resnet50"],
        "ImageNet": ["vit_base_patch32_224"]
    }
    arch_for_celeba = ["inception_v3"]
    for attr_idx in range(40):
        args.archs[f"CelebA32+{attr_idx}"] = arch_for_celeba
        args.archs[f"CelebA+{attr_idx}"] = arch_for_celeba

    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def gen_model():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    logger = logging.getLogger("Benchmark")
    args = get_args()
    print(f"-> Running with config:{args}")

    dataset = args.dataset
    cfg = dloader.load_cfg(dataset_id=dataset, arch_id="")
    benchmk = ImageBenchmark(
        archs=args.archs[dataset], datasets=[dataset],
        datasets_dir=args.datasets_dir, models_dir=args.models_dir
    )

    seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #seeds += [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    seeds1 = [100, 200, 300, 400]
    seeds2 = [400, 500, 600]
    seeds3 = [700, 800, 900, 1000]
    seeds4 = [1000, 600, 300]
    # seeds = seeds2

    models = benchmk.list_models(cfg=cfg, methods=["distill", "finetune", "prune", "negative"], seeds=seeds)
    for idx, model in enumerate(models):
        logger.info(f"-> idx:{idx} runing for model:{model} seed:{model.seed}")
        model.gen_model(seed=model.seed)
        print()


def eval_model():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    logger = logging.getLogger("Benchmark")
    args = get_args()
    print(f"-> Running with config:{args}")

    dataset = args.dataset
    cfg = dloader.load_cfg(dataset_id=dataset, arch_id="")
    benchmk = ImageBenchmark(
        archs=args.archs[dataset], datasets=[dataset],
        datasets_dir=args.datasets_dir, models_dir=args.models_dir
    )
    model = benchmk.load_wrapper(args.model1, seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", batch_size=1000)
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224))

    #metric.topk_test(model, test_loader=test_loader, device=args.device, epoch=0, debug=True)


if __name__ == "__main__":
    gen_model()
























