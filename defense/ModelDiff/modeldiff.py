#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, datetime, pytz
import argparse
import logging
import os.path as osp
import torch
import numpy as np
from scipy import spatial
import torch.nn as nn
from utils import helper, ops
from defense import Fingerprinting
format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S"))
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class ModelDiff(Fingerprinting):
    MAX_VAL = 256
    def __init__(self, model1, model2, out_root, device,
                 gen_inputs=None, input_metrics=None, compute_decision_dist=None,
                 compare_ddv=None, test_size=128):
        super().__init__(model1, model2, device=device, out_root=out_root)
        if list(model1.input_shape) != list(model2.input_shape):
            self.logger.warning('input shapes do not match')
            exit(1)

        # init logger
        self.logger = logging.getLogger('ModelDiff')
        self.logger.info(f'-> comparing {model1} vs {model2}')

        self.test_size = test_size # =N in the paper
        self.input_shape = model1.input_shape

        # init model
        self.arch1 = str(model1)
        self.arch2 = str(model2)
        self.gen_profiling_inputs = gen_inputs if gen_inputs else self.gen_profiling_inputs_search
        self.input_metrics = input_metrics if input_metrics else ModelDiff.metrics_output_diversity
        self.compute_decision_dist = compute_decision_dist if compute_decision_dist else ModelDiff._compute_decision_dist_output_cos
        self.compare_ddv = compare_ddv if compare_ddv else ModelDiff._compare_ddv_cos
        self.fp_path = osp.join(self.fingerprint_root, f"{self.arch1}_{self.arch2}.pt")


    def extract(self, rand=False, **kwargs):
        """
        extract fingerprint samples.
        :return:
        """
        self.logger.info("-> extract fingerprint...")
        if osp.exists(self.fp_path):
            self.logger.info(f"-> load from cache:{self.fp_path}")
            return torch.load(self.fp_path, map_location="cpu")["inputs"]

        self.logger.info(f'-> generating seed inputs')
        seed_inputs = np.concatenate([
            self.model1.get_seed_inputs(self.test_size, rand=rand),
            self.model2.get_seed_inputs(self.test_size, rand=rand)
        ])
        np.random.shuffle(seed_inputs)
        seed_inputs = np.array(seed_inputs)
        seed_inputs = torch.from_numpy(seed_inputs)

        self.logger.info(f'-> generating fingerprint using seed_inputs:{seed_inputs.size()}')
        fingerprint = self.gen_profiling_inputs(seed_inputs=seed_inputs)
        torch.save(fingerprint, self.fp_path)
        return fingerprint["inputs"]

    def verify(self, profiling_inputs, use_torch=True, **kwargs):
        self.logger.info("-> verify ownership...")
        input_metrics_1 = self.input_metrics(self.model1, profiling_inputs, use_torch=use_torch)
        input_metrics_2 = self.input_metrics(self.model2, profiling_inputs, use_torch=use_torch)
        self.logger.info(f'-> input metrics: model1={input_metrics_1} model2={input_metrics_2}')

        self.logger.info("")
        ddm_similarity = self.compute_similarity_with_ddm(profiling_inputs)
        ddv_similarity = self.compute_similarity_with_ddv(profiling_inputs)
        self.logger.info(
            f'-> {self.model1} vs {self.model2} ddm_similarity={ddm_similarity} ddv_similarity={ddv_similarity}')
        return ddm_similarity, ddv_similarity

    def compare(self, **kwargs):
        return self.verify(self.extract(**kwargs))

    def none_optimized_compare(self, profiling_inputs, use_torch=True):
        self.logger.info(f'generating seed inputs')
        seed_inputs = list(self.get_seed_inputs())
        np.random.shuffle(seed_inputs)
        input_metrics_1 = self.input_metrics(self.model1, profiling_inputs, use_torch=use_torch)
        input_metrics_2 = self.input_metrics(self.model2, profiling_inputs, use_torch=use_torch)
        self.logger.info(f'-> input metrics: model1={input_metrics_1} model2={input_metrics_2}')

        self.compute_similarity_with_IPGuard(profiling_inputs)
        self.compute_similarity_with_weight()
        self.compute_similarity_with_abs_weight()
        self.compute_similarity_with_bn_weight()
        self.compute_similarity_with_conv_bn_weight()
        self.compute_similarity_with_identical_weight()
        self.compute_similarity_with_whole_weight()
        self.compute_similarity_with_feature(profiling_inputs)
        self.compute_similarity_with_last_feature(profiling_inputs)
        self.compute_similarity_with_last_feature_svd(profiling_inputs)
        self.compute_similarity_with_ddv(profiling_inputs)
        self.compute_similarity_with_ddm(profiling_inputs)

    def compute_similarity_with_IPGuard(self, profiling_inputs):
        n_pairs = int(len(list(profiling_inputs)) / 2)
        normal_input = profiling_inputs[:n_pairs]
        adv_input = profiling_inputs[n_pairs:]

        out = self.model1.batch_forward(adv_input).to("cpu").numpy()
        normal_pred = out.argmax(axis=1)
        out = self.model2.batch_forward(adv_input).to("cpu").numpy()
        adv_pred = out.argmax(axis=1)

        consist = int((normal_pred == adv_pred).sum())
        sim = consist / n_pairs
        self.logger.info(f'-> model similarity(IPGuard): {sim}')
        return sim

    def compute_similarity_with_weight(self):
        """
        直接计算所有Conv层的CosineSimilarity
        :return:
        """
        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            weight1 = weight1.view(-1)
            weight2 = weight2.view(-1)
            dist = nn.CosineSimilarity(dim=0)(weight1, weight2)
            layer_dist.append(dist.item())
        sim = np.mean(layer_dist)
        self.logger.info(f'-> model similarity(CosineSimilarity weight): {sim}')
        return sim

    def compute_similarity_with_abs_weight(self):
        """
        直接计算所有Conv层的CosineSimilarity
        :return:
        """
        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            dist = 1 - ((weight1 - weight2)).abs().mean()
            layer_dist.append(dist.item())
        sim = np.mean(layer_dist)
        self.logger.info(f'-> model similarity(abs weights): {sim}')
        return sim

    def compute_similarity_with_bn_weight(self):
        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.torch_model.named_modules():
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
        self.logger.info(f'-> model similarity(bn_weight): {sim}')
        return sim

    def compute_similarity_with_conv_bn_weight(self):
        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.torch_model.named_modules():
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
        self.logger.info(f'-> model similarity(conv_bn_weight): {sim}')
        return sim

    def compute_similarity_with_identical_weight(self):
        """
        计算所有Conv层相同数值的数量
        :return:
        """
        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        layer_dist = []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            identical = (weight1 == weight2).sum()
            dist = float(identical / weight1.numel())
            layer_dist.append(dist)
        sim = np.mean(layer_dist)
        self.logger.info(f'-> model similarity(identical_weight): {sim}')
        return sim

    def compute_similarity_with_whole_weight(self):
        print("-> compute_similarity_with_whole_weight")
        name_to_modules = {}
        for name, module in self.model1.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name] = [module.weight]
        for name, module in self.model2.torch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name_to_modules[name].append(module.weight)
        model1_weight, model2_weight = [], []
        for name, pack in name_to_modules.items():
            weight1, weight2 = pack
            if (weight1 == weight2).all():
                continue
            weight1 = weight1.cpu().view(-1)
            weight2 = weight2.cpu().view(-1)
            model1_weight.append(weight1)
            model2_weight.append(weight2)
        model1_weight = torch.cat(model1_weight)
        model2_weight = torch.cat(model2_weight)
        sim = nn.CosineSimilarity(dim=0)(model1_weight, model2_weight).item()
        self.logger.info(f'-> model similarity(whole_weight): {sim}')
        return sim

    def compute_similarity_with_feature(self, profiling_inputs):
        print("-> compute_similarity_with_feature")
        # Used to matching features
        # same to DeepJudge "Layer Outputs Distance, LOD"
        def record_act(self, input, output):
            self.out = output

        try:
            handle1_list = []
            handle2_list = []
            name_to_modules = {}
            for name, module in self.model1.torch_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    name_to_modules[name] = [module]
                    handle2_list.append(module.register_forward_hook(record_act))
            for name, module in self.model2.torch_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    name_to_modules[name].append(module)
                    handle2_list.append(module.register_forward_hook(record_act))
            self.model1.batch_forward(profiling_inputs)
            self.model2.batch_forward(profiling_inputs)

            feature_dists = []
            b = profiling_inputs.shape[0]
            for idx, (name, pack) in enumerate(name_to_modules.items()):
                module1, module2 = pack
                feature1 = module1.out.view(-1)
                feature2 = module2.out.view(-1)
                dist = nn.CosineSimilarity(dim=0)(feature1, feature2).item()
                feature_dists.append(dist)
                del module1.out, module2.out, feature1, feature2
                handle1_list[idx].remove()
                handle2_list[idx].remove()
            sim = np.mean(feature_dists)
            self.logger.info(f'-> model similarity (feature): {sim}')
            return sim
        except Exception as e:
            print(f"-> func:compute_similarity_with_feature error:{e} ")

    def compute_similarity_with_last_feature(self, profiling_inputs):
        print("-> compute_similarity_with_last_feature")
        def record_act(self, input, output):
            self.out = output

        try:
            for name, module in self.model1.torch_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module1 = module
            for name, module in self.model2.torch_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module2 = module
            handle1 = module1.register_forward_hook(record_act)
            handle2 = module2.register_forward_hook(record_act)
            self.model1.batch_forward(profiling_inputs)
            self.model2.batch_forward(profiling_inputs)
            feature1 = module1.out.view(-1)
            feature2 = module2.out.view(-1)
            sim = nn.CosineSimilarity(dim=0)(feature1, feature2).item()

            handle1.remove()
            handle2.remove()
            del module1.out, module2.out, feature1, feature2
            self.logger.info(f'-> model similarity (last_feature): {sim}')
            return sim
        except Exception as e:
            print(f"-> func:compute_similarity_with_last_feature error:{e} ")

    def compute_similarity_with_last_feature_svd(self, profiling_inputs):
        # Used to matching features
        def record_act(self, input, output):
            self.out = output

        try:
            for name, module in self.model1.torch_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module1 = module
            for name, module in self.model2.torch_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module2 = module
            handle1 = module1.register_forward_hook(record_act)
            handle2 = module2.register_forward_hook(record_act)

            self.model1.batch_forward(profiling_inputs)
            self.model2.batch_forward(profiling_inputs)
            feature1 = module1.out
            feature2 = module2.out
            b, c, _, _ = feature1.shape
            feature1 = feature1.clone().cpu().view(b, c, -1)
            feature2 = feature2.clone().cpu().view(b, c, -1)
            # for i in range(b):
            #    u1,s1,v1 = torch.svd(feature1[i])
            #    u2,s2,v2 = torch.svd(feature2[i])
            sim = nn.CosineSimilarity(dim=0)(feature1.view(-1), feature2.view(-1)).item()

            handle1.remove()
            handle2.remove()
            del module1.out, module2.out, feature1, feature2
            self.logger.info(f'-> model similarity (last_feature_svd): {sim}')
            return sim
        except Exception as e:
            print(f"-> func:compute_similarity_with_last_feature_svd error:{e} ")

    def compute_similarity_with_ddv(self, profiling_inputs):
        ddv1 = self.compute_ddv(self.model1, profiling_inputs)
        ddv2 = self.compute_ddv(self.model2, profiling_inputs)
        ddv1 = ops.normalize(np.array(ddv1))
        ddv2 = ops.normalize(np.array(ddv2))
        self.logger.debug(f'-> ddv1={ddv1}\n ddv2={ddv2}')
        ddv_distance = self.compare_ddv(ddv1, ddv2)
        model_similarity = 1 - ddv_distance
        self.logger.info(f'-> model similarity(ddv): {model_similarity}')
        return model_similarity

    def compute_ddv(self, model, inputs):
        dists = []
        outputs = model.batch_forward(inputs).to('cpu').numpy()
        n_pairs = int(len(list(inputs)) / 2)
        for i in range(n_pairs):
            ya = outputs[i]
            yb = outputs[i + n_pairs]
            #           dist = spatial.distance.euclidean(ya, yb)
            dist = spatial.distance.cosine(ya, yb)
            dists.append(dist)
        return np.array(dists)

    def compute_similarity_with_ddm(self, profiling_inputs):
        """
        ddm 为各个样本之间输出相似性矩阵，用这个矩阵再算两个模型相似性矩阵
        :param profiling_inputs:
        :return:
        """
        ddm1 = self.compute_ddm(self.model1, profiling_inputs)
        ddm2 = self.compute_ddm(self.model2, profiling_inputs)
        ddm_distance = ModelDiff.mtx_similar1(ddm1, ddm2)
        model_similarity = 1 - ddm_distance
        self.logger.info(f'-> model similarity(ddm): {model_similarity}')
        return model_similarity

    def compute_ddm(self, model, inputs):
        outputs = model.batch_forward(inputs).to('cpu').numpy()
        # outputs = outputs[:, :10]
        outputs_list = list(outputs)
        ddm = spatial.distance.cdist(outputs_list, outputs_list)
        return ddm

    def compute_actative(self, profiling_inputs, theta=0.5):
        pass

    @staticmethod
    def metrics_output_diversity(model, inputs, use_torch=False):
        outputs = model.batch_forward(inputs).to('cpu').numpy()
        #         output_dists = []
        #         for i in range(0, len(outputs) - 1):
        #             for j in range(i + 1, len(outputs)):
        #                 output_dist = spatial.distance.euclidean(outputs[i], outputs[j])
        #                 output_dists.append(output_dist)
        #         diversity = sum(output_dists) / len(output_dists)
        output_dists = spatial.distance.cdist(list(outputs), list(outputs), p=2.0)
        diversity = np.mean(output_dists)
        return diversity

    @staticmethod
    def metrics_output_variance(model, inputs, use_torch=False):
        batch_output = model.batch_forward(inputs).to('cpu').numpy()
        mean_axis = tuple(list(range(len(batch_output.shape)))[2:])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        # print(batch_output_mean.shape)
        output_variances = np.var(batch_output_mean, axis=0)
        # print(output_variances)
        return np.mean(output_variances)

    @staticmethod
    def metrics_output_range(model, inputs, use_torch=False):
        batch_output = model.batch_forward(inputs).to('cpu').numpy()
        mean_axis = tuple(list(range(len(batch_output.shape)))[2:])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        output_ranges = np.max(batch_output_mean, axis=0) - np.min(batch_output_mean, axis=0)
        return np.mean(output_ranges)

    @staticmethod
    def metrics_neuron_coverage(model, inputs, use_torch=False):
        module_irs = model.batch_forward_with_ir(inputs)
        neurons = []
        neurons_covered = []
        for module in module_irs:
            ir = module_irs[module]
            # self.logger.info(f'{tensor["name"]} {batch_tensor_value.shape}')
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
        ModelDiff.logger.info(f'{len(neurons_not_covered)} neurons not covered: {neurons_not_covered}')
        return float(len(neurons_covered)) / len(neurons)

    @staticmethod
    def _compute_decision_dist_output_cos(model, xa, xb):
        ya = model.batch_forward(xa)
        yb = model.batch_forward(xb)
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


    def gen_profiling_inputs_search(self, seed_inputs, use_torch=False, epsilon=0.2):
        input_shape = seed_inputs[0].shape
        n_inputs = seed_inputs.shape[0]
        max_iterations = 1000
        model1 = self.model1
        model2 = self.model2
        ndims = np.prod(input_shape)

        initial_outputs1 = model1.batch_forward(seed_inputs).to('cpu').numpy()
        initial_outputs2 = model2.batch_forward(seed_inputs).to('cpu').numpy()
        self.logger.info(f"-> initial_outputs1:{initial_outputs1.shape} ndims:{ndims}")
        device = helper.get_args().device

        def evaluate_inputs(inputs):
            outputs1 = model1.batch_forward(inputs).to('cpu').numpy()
            outputs2 = model2.batch_forward(inputs).to('cpu').numpy()

            # diversity𝑓
            metrics1 = self.input_metrics(self.model1, inputs)
            metrics2 = self.input_metrics(self.model2, inputs)

            # divergence𝑓
            output_dist1 = np.mean(spatial.distance.cdist(
                list(outputs1),
                list(initial_outputs1),
                p=2).diagonal())
            output_dist2 = np.mean(spatial.distance.cdist(
                list(outputs2),
                list(initial_outputs2),
                p=2).diagonal())
            self.logger.info(f'-> output distance: {output_dist1},{output_dist2}')
            self.logger.info(f'-> metrics: {metrics1},{metrics2}')
            return output_dist1 * output_dist2 * metrics1 * metrics2, outputs1, outputs2

        inputs = seed_inputs
        score, outputs1, outputs2 = evaluate_inputs(inputs)
        self.logger.info(f"-> outputs1:{outputs1.shape}, outputs2:{outputs2.shape} score:{score}")
        cache_scores = np.zeros([max_iterations + 1])
        cache_predicts = [
            np.zeros([int(max_iterations / 10) + 1, outputs1.shape[0], outputs1.shape[1]]),
            np.zeros([int(max_iterations / 10) + 1, outputs2.shape[0], outputs2.shape[1]])
        ]
        cache_scores[0] = score
        cache_predicts[0][0] = outputs1
        cache_predicts[1][0] = outputs2
        cache_data = {
            "step": 0,
            "scores": cache_scores,
            "outputs": cache_predicts,
        }
        cache_path = osp.join(self.cache_root, f"out_{self.arch1}_{self.arch2}.pt")
        for i in range(1, 1 + max_iterations):
            self.logger.info(f'-> mutation {i}-th iteration: {model1} vs {model2}')
            # 随机选修改位置
            mutation_pos = np.random.randint(0, ndims)
            mutation = np.zeros(ndims).astype(np.float32)
            mutation[mutation_pos] = epsilon
            mutation = np.reshape(mutation, input_shape)

            # 随机选修改样本
            mutation_batch = np.zeros(shape=inputs.shape).astype(np.float32)
            mutation_idx = np.random.randint(0, n_inputs)
            mutation_batch[mutation_idx] = mutation

            # 计算边界 & 选择最优
            mutate_right_inputs = inputs + mutation_batch
            mutate_right_score, _, _ = evaluate_inputs(mutate_right_inputs)
            mutate_left_inputs = inputs - mutation_batch
            mutate_left_score, _, _ = evaluate_inputs(mutate_left_inputs)
            if mutate_right_score <= score and mutate_left_score <= score:
                continue
            if mutate_right_score > mutate_left_score:
                self.logger.info(f'-> mutate right: {score}->{mutate_right_score}')
                inputs = mutate_right_inputs
                score = mutate_right_score
            else:
                self.logger.info(f'-> mutate left: {score}->{mutate_left_score}')
                inputs = mutate_left_inputs
                score = mutate_left_score
            cache_scores[i] = score

            # save fingerprint
            if i % 20 == 0:
                step = int(i / 10)
                score, outputs1, outputs2 = evaluate_inputs(inputs)
                cache_predicts[0][step] = outputs1
                cache_predicts[1][step] = outputs2
                cache_data = {
                    "step": i,
                    "scores": cache_scores,
                    "outputs": cache_predicts.clone().detach().cpu(),
                }
                fingerprint = {
                    "step": i,
                    "scores": cache_scores,
                    "inputs": inputs.clone().detach().cpu()
                }
                torch.save(fingerprint, self.fp_path)
                torch.save(cache_data, cache_path)
            print(f"-> step:{i}: {model1} vs {model2} on {device}\n\n")
        return fingerprint

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
    args.fingerprint_root = osp.join(args.out_root, "IPGuard")
    return args



def multiple_pairs(args, benchmark, logger):
    model1 = None
    model_strs = []
    for model_wrapper in benchmark.list_models():
        if not model_wrapper.torch_model_exists():
            continue
        if model_wrapper.__str__() == args.model1:
            model1 = model_wrapper
        model_strs.append(model_wrapper.__str__())
    if model1 is None:
        logger.info(f'-> model1 not found: {model1} {args.model1}')
        return

    for model2 in benchmark.list_models():
        device = args.device
        if str(model1) != str(model2):
            if "quantize" in str(model1) or "quantize" in str(model2):
                device = torch.device("cpu")
            modeldiff = ModelDiff(model1, model2, out_root=args.modeldiff_root, device=device)
            ddm, ddv = modeldiff.compare()
            logger.info(f'-> the similarity is: ddm={ddm}, ddv={ddv}')
            del modeldiff, ddm, ddv, model2
        print()


def single_pairs(args, benchmark, logger):
    model1 = benchmark.get_model_wrapper(args.model1)
    model2 = benchmark.get_model_wrapper(args.model2)

    if "quantize" in str(model1) or "quantize" in str(model2):
        args.device = torch.device("cpu")
    modeldiff = ModelDiff(model1, model2, device=args.device, out_root=args.modeldiff_root)
    similarity = modeldiff.compare()
    logger.info(f'-> the similarity is {similarity}')


def main(args):
    filename = str(os.path.basename(__file__)).split(".")[0]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                        )  # filename=f"{args.logs_root}/{filename}_{args.namespace}.txt")
    benchmark = ImageBenchmark(
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir
    )
    logger = logging.getLogger('ModelDiff')
    single_pairs(args, benchmark, logger)


if __name__ == '__main__':
    from benchmark import ImageBenchmark
    args = helper.get_args()
    args.modeldiff_root = osp.join(args.out_root, "ModelDiff")
    main(args)

    """
        Example command:
        <===========================  Flower102-resnet18  ===========================>
        model1="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.modeldiff" -model1 $model1 -model2 "train(resnet18,Flower102)-" -device 0
        python -m "defense.modeldiff" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-distill()-" -device 0
        python -m "defense.modeldiff" -model1 $model1 -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-stealthnet(0.7,20)-" -device 0
        
        <===========================  Flower102-mbnetv2  ===========================>
        model1="pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.modeldiff" -model1 $model1 -model2 "train(mbnetv2,Flower102)-" -device 1
        python -m "defense.modeldiff" -model1 $model1 -model2 "pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-distill()-" -device 1
        
    """

















