#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import os
import copy
import logging
import torch
import os.path as osp
import numpy as np
from tqdm import tqdm
from itertools import repeat
from pathos.pools import ProcessPool
from benchmark import ImageBenchmark
from defense import Fingerprinting
from lime.wrappers.scikit_image import SegmentationAlgorithm
from utils import helper
from . import ops
from utils.ops import set_default_seed


class ZEST(Fingerprinting):
    def __init__(self, model1, model2, test_loader, device, out_root, seed, orders=['1', '2', 'inf', 'cosine']):
        super().__init__(model1, model2, out_root=out_root, device=device)
        self.logger = logging.getLogger("ZEST")
        self.logger.info(f'-> comparing {model1.task} and {model2.task}')

        self.seed = seed
        self.task1 = model1.task
        self.task2 = model2.task
        self.model1 = model1
        self.model2 = model2
        self.seed_size = test_loader.batch_size
        self.test_loader = test_loader
        self.orders = orders
        self.num_classes = test_loader.num_classes
        set_default_seed(seed)

        self.fp_root = osp.join(self.fingerprint_root, f"{self.model1.task}")
        self.fp_path = osp.join(self.fp_root, f"{self.model2.task}_s{self.model2.seed}.pt")
        for path in [self.fp_root, out_root]:
            if not osp.exists(path):
                os.makedirs(path)
        self.out_root = out_root

    def extract(self, cache=True, **kwargs):
        if cache and osp.exists(self.fp_path):
            print(f"-> load fingerprints from: {self.fp_path}")
            fingerprint = torch.load(self.fp_path, map_location="cpu")
        else:
            self.logger.info(f'-> [compare] step1: generating lime_data')
            ref_data, raw_ref_data = self.compute_seed_samples()
            self.logger.info(f'-> [compare] step2: generating lime_segment, ref_data={ref_data.shape}')
            lime_segment = self.get_lime_segment(ref_data=raw_ref_data)
            self.logger.info(f'-> [compare] step3: generating lime_mask, lime_segment={lime_segment.shape}')
            ref_dataset, lime_dataset = self.get_lime_dataset(ref_data, lime_segment)
            self.logger.info(f'-> [compare] step4: training lime model')

            self.out1 = []
            self.out2 = []
            self.out1, lime_mask1 = self.compute_lime_signature(self.model1, ref_dataset, lime_dataset)
            self.out2, lime_mask2 = self.compute_lime_signature(self.model2, ref_dataset, lime_dataset)
            fingerprint = {
                "ref_data": ref_data,
                "lime_mask1": lime_mask1,
                "lime_mask2": lime_mask2
            }
            self.logger.info(f"-> save cache to: {self.fp_path}")
        torch.save(fingerprint, self.fp_path)
        return fingerprint

    def verify(self, fingerprint, **kwargs):
        res = self.compute_parameter_distance(fingerprint["lime_mask1"], fingerprint["lime_mask2"], lime=True)
        dist = {}
        for key, value in res.items():
            if key == "2":
                key = "L2"
            elif key == "1":
                key = "L1"
            elif key == "inf":
                key = "Linf"
            dist[key] = value
        self.logger.info(f"-> Zest dist: {res}")
        return dist

    def compare(self, **kwargs):
        return self.verify(self.extract(**kwargs))

    def compute_seed_samples(self):
        images, labels = next(iter(self.test_loader))
        mean = self.test_loader.mean
        std = self.test_loader.std
        raw_images = self.test_loader.unnormalize(images, mean=mean, std=std, clamp=True).to('cpu').numpy() * 255.0
        images = images.to('cpu').numpy()
        return images, raw_images

    def get_lime_segment(self, ref_data):
        '''
        segment image to subimage using quickshift
        :param ref_data:
        :return:
        '''
        if ref_data.shape[1] == 3:
            ref_data = np.moveaxis(ref_data, 1, -1)
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, ratio=0.2, max_dist=200)

        '''
        temp = []
        for idx, image in enumerate(ref_data):
            temp.append(segmentation_fn(image))
        lime_segment = np.stack(temp)
        return lime_segment
        '''
        # README: 2022/12/09 alter for multi-processing pool
        print(f"-> step2: get_lime_segment...size:{len(ref_data)}")
        pool = ProcessPool(nodes=64)
        map_resuts = pool.map(segmentation_fn, list(ref_data))
        lime_segment = np.stack(map_resuts)
        del pool
        return lime_segment

    def get_lime_dataset(self, ref_data, lime_segment, mean=np.array([0, 0, 0]), num_samples=500):
        if ref_data.shape[1] == 3:
            ref_data = np.moveaxis(ref_data, 1, -1)
        fudged_image = np.zeros(ref_data.shape[1:])
        fudged_image += mean.reshape([1, 1, -1])
        ref_dataset = []
        lime_dataset = []

        '''
        phar = tqdm(range(ref_data.shape[0]))
        for i in phar:
            n_features = np.unique(lime_segment[i]).shape[0]
            lime_data = np.random.randint(0, 2, [num_samples, n_features])
            lime_data[0, :] = 1
            ref_dataset.append(self.get_reference_dataset(lime_data, ref_data[i], lime_segment[i], fudged_image))
            lime_dataset.append(lime_data)
            phar.set_description(f"-> [{i}/{ref_data.shape[0]}] step3: get_lime_dataset...")
        return ref_dataset, lime_dataset
        '''
        # README: 2022/12/09 alter for multi-processing pool
        def reference_dataset(idx, size, _ref_data, _lime_segment, fudged_image):
            n_features = np.unique(_lime_segment).shape[0]
            sub_lime_data = np.random.randint(0, 2, [num_samples, n_features])
            sub_lime_data[0, :] = 1
            sub_dataset = []
            for sample in sub_lime_data:
                temp = copy.deepcopy(_ref_data)
                zeros = np.where(sample == 0)[0]
                mask = np.zeros(_lime_segment.shape).astype(bool)
                for z in zeros:
                    mask[_lime_segment == z] = True
                temp[mask] = fudged_image[mask]
                sub_dataset.append(temp)
            if (idx + 1) % 32 == 0:
                print(f"-> step3: get_lime_dataset...[{idx+1}/{size}]")
            return [idx, np.stack(sub_dataset), sub_lime_data]
        
        # run reference_dataset with multi-process
        size = len(ref_data)
        pool = ProcessPool(nodes=64)
        pool_result = pool.map(reference_dataset,
                range(size),
                list(repeat(size, size)),
                [ref_data[i] for i in range(size)],
                [lime_segment[i] for i in range(size)],
                list(repeat(fudged_image, size))
        )
        # order the output data
        for idx in range(size):
            ref_dataset.append(None)
            lime_dataset.append(None)
        for idx, (idx, sub_dataset, sub_lime_data) in enumerate(pool_result):
            ref_dataset[idx] = sub_dataset
            lime_dataset[idx] = sub_lime_data
        del pool
        return ref_dataset, lime_dataset

    def get_reference_dataset(self, lime_data, ref_data, segment, fudged_image):
        sub_dataset = []
        for sample in lime_data:
            temp = copy.deepcopy(ref_data)
            zeros = np.where(sample == 0)[0]
            mask = np.zeros(segment.shape).astype(bool)
            for z in zeros:
                mask[segment == z] = True
            temp[mask] = fudged_image[mask]
            sub_dataset.append(temp)
        return np.stack(sub_dataset)

    def compute_lime_signature(self, model, ref_dataset, lime_dataset, cat=True):
        self.logger.info(f"-> compute_lime_signature")
        datasets = []
        model.to(self.device)
        with torch.no_grad():
            phar = tqdm(range(len(lime_dataset)))
            for i in phar:
                lime_data = lime_dataset[i]
                data = ref_dataset[i]
                inputs = torch.from_numpy(data).permute(0, 3, 1, 2).float()
                outputs = ops.batch_forward(model, inputs, argmax=False).detach().cpu().numpy()
                datasets.append([lime_data, outputs])
                phar.set_description(f"-> [{i}/{len(lime_dataset)}] step4.1: compute_lime_signature...")
                out = outputs

        weights = []
        with torch.no_grad():
            phar = tqdm(datasets)
            for data, label in phar:
                data = torch.from_numpy(data).float().to(self.device)
                label = torch.from_numpy(label).float().to(self.device)
                w = torch.chain_matmul(torch.pinverse(torch.matmul(data.T, data)), data.T, label).to("cpu")
                weights.append(w)
                phar.set_description(f"-> [{i}/{len(datasets)}] step4.2: compute weights...")
        if cat:
            weights = torch.cat(weights).to("cpu")
            return out, weights
        else:
            return out, weights.to("cpu")

    def compute_parameter_distance(self, lime_mask1, lime_mask2, half=False, linear=False, lime=False):
        self.logger.info(f"-> compute_parameter_distance...")
        weights1 = self.__consistent_type(lime_mask1, architecture=None, half=half, linear=linear, lime=lime)
        weights2 = self.__consistent_type(lime_mask2, architecture=None, half=half, linear=linear, lime=lime)
        if not isinstance(self.orders, list):
            orders = [self.orders]
        else:
            orders = self.orders
        res_list = {}
        if lime:
            temp_w1 = copy.copy(weights1)
            temp_w2 = copy.copy(weights2)

        for o in orders:
            if lime:
                weights1, weights2 = self.__lime_align(temp_w1, temp_w2, o)
            res = self.__compute_distance(weights1, weights2, o)
            if isinstance(res, np.ndarray):
                res = float(res)
            res_list[o] = res
        return res_list

    def __compute_distance(self, a, b, order):
        if order == 'inf':
            order = np.inf
        if order == 'cos' or order == 'cosine':
            return 1.0 - torch.nn.CosineSimilarity(dim=0)(a, b).cpu().numpy()
        else:
            if order != np.inf:
                try:
                    order = int(order)
                except:
                    raise TypeError("input metric for distance is not understandable")
            return torch.norm(a - b, p=order).cpu().numpy()

    def __lime_align(self, w1, w2, order):
        shorter = int(w1.shape[1] <= w2.shape[1])
        w = [w1, w2] if shorter else [w2, w1]
        num_class = w1.shape[1] if shorter else w2.shape[1]
        new_w = [None] * num_class
        dist = np.zeros([w[0].shape[1], w[1].shape[1]])
        for j in range(w[0].shape[1]):
            for k in range(w[1].shape[1]):
                dist[j, k] = self.__compute_distance(w[0][:, j], w[1][:, k], order)
        upper_bound = np.max(dist) + 1e10
        for i in range(w[0].shape[1]):
            ind1, ind2 = np.argmin(dist) // dist.shape[1], np.argmin(dist) % dist.shape[1]
            new_w[ind1] = w[1][:, ind2]
            dist[ind1, :] = upper_bound
            dist[:, ind2] = upper_bound
        new_w = torch.stack(new_w, 1)
        res = [w1, new_w] if shorter else [w2, new_w]
        return res[0].reshape([-1]), res[1].reshape([-1])

    def __consistent_type(self, lime_mask, architecture=None, half=False, linear=False, lime=False):
        if isinstance(lime_mask, str):
            state = torch.load(lime_mask)
            if linear:
                weights = state['linear'].reshape(-1)
            elif lime:
                weights = state['lime']
                if isinstance(weights, np.ndarray):
                    weights = torch.from_numpy(weights).float()
            else:
                assert architecture is not None
                net = architecture()
                net.load_state_dict(state['net'])
                weights = torch.cat([i.data.reshape([-1]) for i in list(net.parameters())])
        elif isinstance(lime_mask, np.ndarray):
            if lime:
                weights = torch.from_numpy(lime_mask).float()
            else:
                weights = torch.from_numpy(lime_mask).reshape(-1).float()
        elif not isinstance(lime_mask, torch.Tensor):
            weights = torch.cat([i.data.reshape([-1]) for i in list(lime_mask.parameters())])
        else:
            if not lime:
                weights = lime_mask.reshape(-1)
            else:
                weights = lime_mask
        if half:
            if half == 2:
                weights = weights.type(torch.IntTensor).type(torch.FloatTensor)
            else:
                weights = weights.half()
        return weights.to(self.device)


def main(args):
    from dataset import loader as dloader
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    benchmark = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = benchmark.load_wrapper(args.model1, seed=1000).load_torch_model()
    model2 = benchmark.load_wrapper(args.model2, seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=model1.dataset_id, split="test", batch_size=128)

    comparison = ZEST(model1, model2, test_loader=test_loader, out_root=args.zest_root, device=args.device)
    dist = comparison.compare()
    print(f"-> ZEST dist: {dist}")


if __name__ == "__main__":
    args = helper.get_args()
    args.zest_root = osp.join(args.out_root, "ZEST")
    main(args)














