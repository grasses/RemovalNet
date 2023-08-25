#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pathlib
import math
import torch
import numpy as np
from scipy import spatial
import torch.nn as nn
import os.path as osp
from utils import helper
from defense import Fingerprinting
from utils import ops


class ModelDiff(Fingerprinting):
    N_INPUT_PAIRS = 300
    MAX_VAL = 256

    def __init__(self, model1, model2, test_loader, out_root, device, seed, epsilon=1.0):
        super().__init__(model1, model2, device=device, out_root=out_root)
        self.logger = logging.getLogger('ModelDiff')

        self.seed = seed
        ops.set_default_seed(seed)
        self.arch1 = str(model1.task)
        self.arch2 = str(model2.task)

        self.epsilon = epsilon
        self.gen_inputs = self._gen_profiling_inputs_search
        self.input_metrics = self.metrics_output_diversity
        self.compute_decision_dist = self._compute_decision_dist_output_cos
        self.compare_ddv = self._compare_ddv_cos

        data, _ = next(iter(test_loader))
        self.input_shape = data.shape
        self.batch_size = 50 if self.input_shape[-1] == 224 else 400

        self.test_size = test_loader.batch_size
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.test_loader = test_loader

        model1_root = osp.join(self.fingerprint_root, f"{self.arch1}")
        if not osp.exists(model1_root):
            os.makedirs(model1_root)
        self.fp_path = osp.join(model1_root, f"{self.arch2}_e{epsilon}_s{seed}.pt")

    def get_seed_inputs(self, rand=False):
        seed_inputs = np.concatenate([
            self.model1.get_seed_inputs(self.N_INPUT_PAIRS, rand=rand),
            self.model2.get_seed_inputs(self.N_INPUT_PAIRS, rand=rand)
        ])
        return seed_inputs

    def extract(self, cache=False):
        if cache and osp.exists(self.fp_path):
            self.logger.info(f'-> load fingerprint from:{self.fp_path}')
            fingerprints = torch.load(self.fp_path, map_location="cpu")["fingerprints"]
        else:
            seed_x, seed_y = next(iter(self.test_loader))
            seed_inputs = seed_x.to('cpu').numpy()
            np.random.shuffle(seed_inputs)
            seed_inputs = np.array(seed_inputs)
            seed_inputs = torch.from_numpy(seed_inputs)
            fingerprints = self.gen_inputs(seed_inputs, epsilon=self.epsilon)
            torch.save({"fingerprints": fingerprints}, self.fp_path)
        return fingerprints

    def verify(self, fingerprints, **kwargs):
        fingerprints = fingerprints.to(self.device)
        self.logger.info(f'-> fingerprints inputs generated with shape {fingerprints.shape}')
        self.logger.info(f'-> computing metrics, DDM')
        input_metrics_1 = self.input_metrics(self.model1, fingerprints)
        input_metrics_2 = self.input_metrics(self.model2, fingerprints)
        self.logger.info(f'  input metrics: model1={input_metrics_1} model2={input_metrics_2}')
        ddm = self.compute_similarity_with_ddm(fingerprints)
        ddv = self.compute_similarity_with_ddv(fingerprints)
        ws_abs = self.compute_similarity_with_abs_weight()
        ws = self.compute_similarity_with_fc()
        mr = self.compute_similarity_with_IPGuard(fingerprints)
        dist = {
            "DDM": ddm,
            "DDV": ddv,
            "MR": mr,
            "WS": ws,
            "WS_abs": ws_abs
        }
        print(f"-> {self.arch1} vs {self.arch2} seed:{self.seed} dist:{dist}")
        return dist

    def compute_similarity_with_IPGuard(self, profiling_inputs):
        n_pairs = int(len(list(profiling_inputs)) / 2)
        normal_input = profiling_inputs[:n_pairs]
        adv_input = profiling_inputs[n_pairs:]

        out = self.model1(adv_input).detach().to("cpu").numpy()
        normal_pred = out.argmax(axis=1)
        out = self.model2(adv_input).detach().to("cpu").numpy()
        adv_pred = out.argmax(axis=1)

        consist = int((normal_pred == adv_pred).sum())
        sim = consist / n_pairs
        return sim

    def compute_similarity_with_fc(self):
        name_to_modules = {}
        for name, module in self.model1.named_modules():
            if isinstance(module, nn.Linear):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.named_modules():
            if isinstance(module, nn.Linear):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            weight1 = weight1.view(-1)
            weight2 = weight2.view(-1)
            dist = nn.CosineSimilarity(dim=0)(weight1, weight2)
            layer_dist.append(dist.item())
        sim = np.mean(layer_dist)
        self.logger.info(f'-> weight similarity: {sim}')
        return sim

    def compute_similarity_with_weight(self):
        name_to_modules = {}
        for name, module in self.model1.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            # print(name, float((weight1==0).sum() / weight1.numel()))
            weight1 = weight1.view(-1)
            weight2 = weight2.view(-1)
            dist = nn.MSELoss(reduction="mean")(weight1, weight2)
            layer_dist.append(dist.item())
        sim = np.mean(layer_dist)
        self.logger.info(f'-> weight similarity: {sim}')
        return sim

    def compute_similarity_with_abs_weight(self):
        name_to_modules = {}
        for name, module in self.model1.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            dist = 1 - ((weight1 - weight2)).abs().mean()
            layer_dist.append(dist.item())
        sim = np.mean(layer_dist)
        self.logger.info(f'  model similarity: {sim}')
        return sim

    def compute_similarity_with_bn_weight(self):
        name_to_modules = {}
        for name, module in self.model1.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            weight1 = weight1.view(-1)
            weight2 = weight2.view(-1)
            dist = nn.CosineSimilarity(dim=0)(weight1, weight2)
            layer_dist.append(dist.item())
        sim = np.mean(layer_dist)

        self.logger.info(f'  model similarity: {sim}')
        return sim

    def compute_similarity_with_conv_bn_weight(self):
        name_to_modules = {}
        for name, module in self.model1.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            weight1 = weight1.view(-1)
            weight2 = weight2.view(-1)
            dist = nn.CosineSimilarity(dim=0)(weight1, weight2)
            layer_dist.append(dist.item())
        sim = np.mean(layer_dist)

        self.logger.info(f'  model similarity: {sim}')
        return sim

    def compute_similarity_with_identical_weight(self):
        name_to_modules = {}
        for name, module in self.model1.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            identical = (weight1 == weight2).sum()
            dist = float(identical / weight1.numel())
            layer_dist.append(dist)

        sim = np.mean(layer_dist)

        self.logger.info(f'  model similarity: {sim}')
        return sim

    def compute_similarity_with_whole_weight(self):
        name_to_modules = {}
        for name, module in self.model1.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        model1_weight, model2_weight = [], []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            if (weight1 == weight2).all():
                continue
            weight1 = weight1.view(-1)
            weight2 = weight2.view(-1)
            model1_weight.append(weight1)
            model2_weight.append(weight2)
        model1_weight = torch.cat(model1_weight)
        model2_weight = torch.cat(model2_weight)
        sim = nn.CosineSimilarity(dim=0)(model1_weight, model2_weight).item()

        self.logger.info(f'  model similarity: {sim}')
        return sim

    def compute_similarity_with_feature(self, profiling_inputs):
        # Used to matching features
        def record_act(self, input, output):
            self.out = output

        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module]
                module.register_forward_hook(record_act)
        for name, module in self.model2.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module)
                module.register_forward_hook(record_act)
        # print(name_to_modules.keys())
        self.model1(profiling_inputs)
        self.model2(profiling_inputs)

        feature_dists = []
        b = profiling_inputs.shape[0]
        for name, pack in name_to_modules.items():
            module1, module2 = pack
            feature1 = module1.out.view(-1)
            feature2 = module2.out.view(-1)
            dist = nn.CosineSimilarity(dim=0)(feature1, feature2).item()
            feature_dists.append(dist)
            del module1.out, module2.out, feature1, feature2
        sim = np.mean(feature_dists)

        self.logger.info(f'  model similarity: {sim}')
        return sim

    def compute_similarity_with_last_feature(self, profiling_inputs):
        # Used to matching features
        def record_act(self, input, output):
            self.out = output

        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                module1 = module
        for name, module in self.model2.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                module2 = module
        module1.register_forward_hook(record_act)
        module2.register_forward_hook(record_act)
        # print(name_to_modules.keys())
        self.model1(profiling_inputs)
        self.model2(profiling_inputs)

        feature1 = module1.out.view(-1)
        feature2 = module2.out.view(-1)
        dist = nn.CosineSimilarity(dim=0)(feature1, feature2).item()
        del module1.out, module2.out, feature1, feature2
        sim = dist

        self.logger.info(f'  model similarity: {sim}')
        return sim

    def compute_similarity_with_last_feature_svd(self, profiling_inputs):
        # Used to matching features
        def record_act(self, input, output):
            self.out = output

        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                module1 = module
        for name, module in self.model2.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                module2 = module
        module1.register_forward_hook(record_act)
        module2.register_forward_hook(record_act)
        # print(name_to_modules.keys())
        self.model1(profiling_inputs)
        self.model2(profiling_inputs)

        feature1 = module1.out
        feature2 = module2.out
        b, c, _, _ = feature1.shape
        feature1 = feature1.view(b, c, -1)
        feature2 = feature2.view(b, c, -1)
        for i in range(b):
            u1, s1, v1 = torch.svd(feature1[i])
            u2, s2, v2 = torch.svd(feature2[i])
            #st()
        dist = nn.CosineSimilarity(dim=0)(feature1, feature2).item()
        del module1.out, module2.out, feature1, feature2
        sim = dist

        self.logger.info(f'  model similarity: {sim}')
        return sim

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def compute_similarity_with_ddv(self, profiling_inputs):
        ddv1 = self.compute_ddv(self.model1, profiling_inputs)
        ddv2 = self.compute_ddv(self.model2, profiling_inputs)

        ddv1 = self.normalize(np.array(ddv1))
        ddv2 = self.normalize(np.array(ddv2))

        #ddv_distance = spatial.distance.euclidean(ddv1, ddv2)
        ddv_distance = self.compare_ddv(ddv1, ddv2)
        model_similarity = 1 - ddv_distance
        return model_similarity

    def compute_ddv(self, model, inputs):
        dists = []
        outputs = self.batch_forward(model, inputs, batch_size=self.batch_size).detach().to('cpu').numpy()
        #outputs = model(inputs).detach().to('cpu').numpy()
        self.logger.debug(f'{model}: \n profiling_outputs={outputs.shape}\n{outputs}\n')
        n_pairs = int(len(list(inputs)) / 2)
        for i in range(n_pairs):
            ya = outputs[i]
            yb = outputs[i + n_pairs]
            dist = spatial.distance.cosine(ya, yb)
            dists.append(dist)
        return np.array(dists)

    def compute_similarity_with_ddm(self, profiling_inputs):
        ddm1 = self.compute_ddm(self.model1, profiling_inputs)
        ddm2 = self.compute_ddm(self.model2, profiling_inputs)
        ddm_distance = ModelDiff.mtx_similar1(ddm1, ddm2)
        model_similarity = 1 - ddm_distance

        self.logger.info(f'  model similarity: {model_similarity}')
        return model_similarity

    def compute_ddm(self, model, inputs):
        outputs = self.batch_forward(model, inputs, batch_size=self.batch_size).detach().to('cpu').numpy()
        # outputs = outputs[:, :10]
        outputs_list = list(outputs)
        ddm = spatial.distance.cdist(outputs_list, outputs_list)
        return ddm

    @staticmethod
    def batch_forward(model, x, batch_size=100, argmax=False):
        """
        split x into batch_size, torch.cat result to return
        :param model:
        :param x:
        :param batch_size:
        :param argmax:
        :return:
        """
        device = next(model.parameters()).device
        steps = math.ceil(len(x) / batch_size)
        pred = []
        with torch.no_grad():
            for step in range(steps):
                off = int(step * batch_size)
                batch_x = x[off: off + batch_size].to(device)
                pred.append(model(batch_x).cpu().detach())
        pred = torch.cat(pred)
        return pred.argmax(dim=1) if argmax else pred

    def metrics_output_diversity(self, model, inputs, use_torch=False):
        outputs = self.batch_forward(model, inputs, batch_size=self.batch_size).detach().cpu()
        #outputs = model(inputs)
        output_dists = torch.cdist(outputs, outputs, p=2.0).numpy()
        diversity = np.mean(output_dists)
        return diversity

    def metrics_output_variance(self, model, inputs, use_torch=False):
        batch_output = self.batch_forward(model, inputs, batch_size=self.batch_size).detach().cpu().numpy()
        #batch_output = model(inputs).to('cpu').numpy()
        mean_axis = tuple(list(range(len(batch_output.shape)))[2:])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        # print(batch_output_mean.shape)
        output_variances = np.var(batch_output_mean, axis=0)
        # print(output_variances)
        return np.mean(output_variances)

    @staticmethod
    def metrics_output_range(model, inputs, use_torch=False):
        batch_output = model(inputs).to('cpu').numpy()
        mean_axis = tuple(list(range(len(batch_output.shape)))[2:])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        output_ranges = np.max(batch_output_mean, axis=0) - np.min(batch_output_mean, axis=0)
        return np.mean(output_ranges)

    @staticmethod
    def metrics_neuron_coverage(model, inputs, use_torch=False):
        module_irs = model(inputs)
        neurons = []
        neurons_covered = []
        for module in module_irs:
            ir = module_irs[module]
            # print(f'{tensor["name"]} {batch_tensor_value.shape}')
            # if 'relu' not in tensor["name"].lower():
            #     continue
            squeeze_axis = tuple(list(range(len(ir.shape)))[:-1])
            squeeze_ir = np.max(ir, axis=squeeze_axis)
            for i in range(squeeze_ir.shape[-1]):
                neuron_name = f'{module}-{i}'
                neurons.append(neuron_name)
                neuron_value = squeeze_ir[i]
                covered = neuron_value > 0.1
                if covered:
                    neurons_covered.append(neuron_name)
        neurons_not_covered = [neuron for neuron in neurons if neuron not in neurons_covered]
        print(f'{len(neurons_not_covered)} neurons not covered: {neurons_not_covered}')
        return float(len(neurons_covered)) / len(neurons)

    @staticmethod
    def _compute_decision_dist_output_cos(model, xa, xb):
        ya = model(xa)
        yb = model(xb)
        return spatial.distance.cosine(ya, yb)

    @staticmethod
    def _gen_profiling_inputs_none(comparator, seed_inputs, use_torch=False):
        return seed_inputs

    @staticmethod
    def _gen_profiling_inputs_random(comparator, seed_inputs, use_torch=False):
        return np.random.normal(size=seed_inputs.shape).astype(np.float32)

    # @staticmethod
    # def _gen_profiling_inputs_1pixel(comparator, seed_inputs):
    #     input_shape = seed_inputs[0].shape
    #     for i in range(len(seed_inputs)):
    #         x = np.zeros(input_shape, dtype=np.float32)
    #         random_index = np.unravel_index(np.argmax(np.random.normal(size=input_shape)), input_shape)
    #         x[random_index] = 1
    #         yield x

    def _gen_profiling_inputs_search(self, seed_inputs, epsilon=0.2, max_iterations=1000):
        input_shape = seed_inputs[0].shape
        n_inputs = seed_inputs.shape[0]

        seed_inputs = seed_inputs.to(self.device)
        model1 = self.model1.to(self.device)
        model2 = self.model2.to(self.device)
        ndims = np.prod(input_shape)

        initial_outputs1 = self.batch_forward(model1, seed_inputs, batch_size=self.batch_size).detach().to('cpu')
        initial_outputs2 = self.batch_forward(model2, seed_inputs, batch_size=self.batch_size).detach().to('cpu')
        #initial_outputs1 = model1(seed_inputs).detach().to('cpu')
        #initial_outputs2 = model2(seed_inputs).detach().to('cpu')

        def evaluate_inputs(inputs):
            inputs = inputs.to(self.device)
            metrics1 = self.input_metrics(self.model1, inputs)
            metrics2 = self.input_metrics(self.model2, inputs)
            outputs1 = self.batch_forward(model1, seed_inputs, batch_size=self.batch_size).detach().to('cpu')
            outputs2 = self.batch_forward(model2, seed_inputs, batch_size=self.batch_size).detach().to('cpu')
            output_dist1 = torch.mean(torch.diagonal(torch.cdist(outputs1, initial_outputs1, p=2))).numpy()
            output_dist2 = torch.mean(torch.diagonal(torch.cdist(outputs2, initial_outputs2, p=2))).numpy()
            return output_dist1 * output_dist2 * metrics1 * metrics2

        inputs = seed_inputs
        score = evaluate_inputs(inputs)

        for step in range(max_iterations):
            #comparator._compute_distance(inputs)
            # mutation_idx = random.randint(0, len(inputs))
            # mutation = np.random.random_sample(size=input_shape).astype(np.float32)

            mutation_pos = np.random.randint(0, ndims)
            mutation = np.zeros(ndims).astype(np.float32)
            mutation[mutation_pos] = epsilon
            mutation = np.reshape(mutation, input_shape)

            mutation_batch = np.zeros(shape=inputs.shape).astype(np.float32)
            mutation_idx = np.random.randint(0, n_inputs)
            mutation_batch[mutation_idx] = mutation
            mutation_batch = torch.from_numpy(mutation_batch).to(self.device)

            mutate_right_inputs = inputs + mutation_batch
            mutate_right_score = evaluate_inputs(mutate_right_inputs)
            mutate_left_inputs = inputs - mutation_batch
            mutate_left_score = evaluate_inputs(mutate_left_inputs)

            if mutate_right_score <= score and mutate_left_score <= score:
                if step % 50 == 0:
                    print(f"-> bypass:{step} epsilon:{epsilon}  left_score:{mutate_left_score} score:{score} right_score:{mutate_right_score}")
                continue

            if mutate_right_score > mutate_left_score:
                #print(f'-> gen_inputs [{i}/{max_iterations}] mutate right: {score}->{mutate_right_score}')
                inputs = mutate_right_inputs.detach().to(self.device)
                score = mutate_right_score
            else:
                #print(f'-> gen_inputs [{i}/{max_iterations}] mutate left: {score}->{mutate_left_score}')
                inputs = mutate_left_inputs.detach().to(self.device)
                score = mutate_left_score

            if step % 50 == 0:
                print(f"-> gen_inputs [{step}/{max_iterations}] epsilon:{epsilon} score:{score}")
        return inputs

    @staticmethod
    def _compare_ddv_cos(ddv1, ddv2):
        return spatial.distance.cosine(ddv1, ddv2)

    @staticmethod
    def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
        '''
        计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
        注意有展平操作。
        :param arr1:矩阵1
        :param arr2:矩阵2
        :return:实际是夹角的余弦值，ret = (cos+1)/2
        '''
        farr1 = arr1.ravel()
        farr2 = arr2.ravel()
        len1 = len(farr1)
        len2 = len(farr2)
        if len1 > len2:
            farr1 = farr1[:len2]
        else:
            farr2 = farr2[:len1]

        numer = np.sum(farr1 * farr2)
        denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
        similar = numer / denom  # 这实际是夹角的余弦值
        return (similar + 1) / 2  # 姑且把余弦函数当线性

    def mtx_similar2(arr1: np.ndarray, arr2: np.ndarray) -> float:
        '''
        计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
        如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
        :param arr1:矩阵1
        :param arr2:矩阵2
        :return:相似度（0~1之间）
        '''
        if arr1.shape != arr2.shape:
            minx = min(arr1.shape[0], arr2.shape[0])
            miny = min(arr1.shape[1], arr2.shape[1])
            differ = arr1[:minx, :miny] - arr2[:minx, :miny]
        else:
            differ = arr1 - arr2
        numera = np.sum(differ ** 2)
        denom = np.sum(arr1 ** 2)
        similar = 1 - (numera / denom)
        return similar

    def mtx_similar3(arr1: np.ndarray, arr2: np.ndarray) -> float:
        '''
        From CS231n: There are many ways to decide whether
        two matrices are similar; one of the simplest is the Frobenius norm. In case
        you haven't seen it before, the Frobenius norm of two matrices is the square
        root of the squared sum of differences of all elements; in other words, reshape
        the matrices into vectors and compute the Euclidean distance between them.
        difference = np.linalg.norm(dists - dists_one, ord='fro')
        :param arr1:矩阵1
        :param arr2:矩阵2
        :return:相似度（0~1之间）
        '''
        if arr1.shape != arr2.shape:
            minx = min(arr1.shape[0], arr2.shape[0])
            miny = min(arr1.shape[1], arr2.shape[1])
            differ = arr1[:minx, :miny] - arr2[:minx, :miny]
        else:
            differ = arr1 - arr2
        dist = np.linalg.norm(differ, ord='fro')
        len1 = np.linalg.norm(arr1)
        len2 = np.linalg.norm(arr2)  # 普通模长
        denom = (len1 + len2) / 2
        similar = 1 - (dist / denom)
        return similar


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Compare similarity between two models.")

    parser.add_argument("-benchmark_dir", action="store", dest="benchmark_dir",
                        required=False, default=".", help="Path to the benchmark.")
    parser.add_argument("-model1", action="store", dest="model1",
                        required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2",
                        required=True, help="model 2.")
    args, unknown = parser.parse_known_args()
    return args


def evaluate_micro_benchmark():
    lines = pathlib.Path('benchmark_models/model_pairs.txt').read_text().splitlines()
    eval_lines = []
    for line in lines:
        model1_str = line.split()[0]
        model2_str = line.split()[2]
        model1_path = os.path.join('benchmark_models', f'{model1_str}.h5')
        model2_path = os.path.join('benchmark_models', f'{model2_str}.h5')
        model1 = Model(model1_path)
        model2 = Model(model2_path)
        comparison = ModelDiff(model1, model2)
        similarity = comparison.compare()
        eval_line = f'{model1_str} {model2_str} {similarity}'
        eval_lines.append(eval_line)
        print(eval_line)
    pathlib.Path('benchmark_models/model_pairs_eval.txt').write_text('\n'.join(eval_lines))


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="pretrain(resnet18,ImageNet)-",
                        required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2",
                        default="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-", required=True, help="model 2.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed_method", action="store", default="PGD", type=str, choices=["FGSM", "PGD", "CW"], help="Type of blackbox generation")
    parser.add_argument("-batch_size", action="store", default=100, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.modeldiff_root = osp.join(args.out_root, "ModelDiff")
    return args


def main(args):
    from benchmark import ImageBenchmark
    from dataset import loader

    #filename = str(osp.basename(__file__)).split(".")[0]
    #logging.basicConfig(level=logging.INFO,
    #                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    benchmark = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)

    model1 = benchmark.load_wrapper(args.model1, seed=args.seed).torch_model(seed=args.seed)
    model2 = benchmark.load_wrapper(args.model2, seed=args.seed).torch_model(seed=args.seed)
    test_loader = loader.get_dataloader(model1.dataset_id, batch_size=args.batch_size)

    if "quantize" in str(args.model1) or "quantize" in str(args.model2):
        args.device = torch.device("cpu")

    out_root = osp.join(args.out_root, "ModelDiff")
    modeldiff = ModelDiff(model1, model2, test_loader=test_loader,
                          device=args.device, seed=args.seed,
                          out_root=out_root)
    dist = modeldiff.verify(modeldiff.extract())


if __name__ == '__main__':
    args = get_args()
    for seed in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        args.seed = seed
        main(args)
