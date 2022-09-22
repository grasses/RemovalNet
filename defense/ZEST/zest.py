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
from benchmark import ImageBenchmark
from defense import Fingerprinting
from lime.wrappers.scikit_image import SegmentationAlgorithm
from utils import helper
sys_args = helper.get_args()


class ZEST(Fingerprinting):
    def __init__(self, model1, model2, out_root, device,
                 gen_inputs=None, input_metrics=None, compute_decision_dist=None, seed_size=128):
        super().__init__(model1, model2, out_root=out_root, device=device)
        self.logger = logging.getLogger("Zest")
        self.logger.info(f'-> comparing {model1} and {model2}')
        self.logger.debug(f'-> initialize comparison: {self.model1} {self.model2}')
        self.logger.debug(f'-> input shapes: {self.model1.input_shape} {self.model2.input_shape}')
        self.model1 = model1
        self.model2 = model2
        self.seed_size = seed_size
        if not osp.exists(out_root):
            os.makedirs(out_root)
        self.out_root = out_root

    def extract(self, **kwargs):
        pass

    def verify(self, **kwargs):
        pass

    def compare(self, use_torch=True, cache=True):
        path = osp.join(self.out_root, f"lime_{self.model1}_{self.model2}.pt")
        if not osp.exists(path):
            self.logger.info(f'-> [compare] step1: generating lime_data')
            ref_data, unnormalize_ref_data = self.compute_seed_samples()
            self.logger.info(f'-> [compare] step2: generating lime_segment, ref_data={ref_data.shape}')
            lime_segment = self.get_lime_segment(ref_data=unnormalize_ref_data)
            self.logger.info(f'-> [compare] step3: generating lime_mask, lime_segment={lime_segment.shape}')
            ref_dataset, lime_dataset = self.get_lime_dataset(ref_data, lime_segment)
            self.logger.info(f'-> [compare] step4: training lime model')
            lime_mask1 = self.compute_lime_signature(self.model1, ref_dataset, lime_dataset)
            lime_mask2 = self.compute_lime_signature(self.model2, ref_dataset, lime_dataset)
            cache_data = {
                "ref_data": ref_data,
                "lime_mask1": lime_mask1,
                "lime_mask2": lime_mask2
            }
            if cache:
                torch.save(cache_data, path)
                print(f"-> save cache to: {path}")
        else:
            cache_data = torch.load(path)
            print(f"-> load cache from: {path}")
        dist = self.compute_parameter_distance(cache_data["lime_mask1"], cache_data["lime_mask2"], lime=True)
        self.logger.info(f"-> Zest dist: {dist}")
        return dist

    def compute_seed_samples(self, rand=False):
        # Zest 原论文只有一个数据集
        images, unnormalize_images, bounds, labels = self.model1.get_seed_inputs(self.seed_size, rand=rand, unormalize=True)
        return images, unnormalize_images

    def get_lime_segment(self, ref_data):
        '''
        segment image to subimage using quickshift
        :param ref_data:
        :return:
        '''
        temp = []
        if ref_data.shape[1] == 3:
            ref_data = np.moveaxis(ref_data, 1, -1)
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, ratio=0.2, max_dist=200)
        phar = tqdm(enumerate(ref_data))
        for idx, image in phar:
            temp.append(segmentation_fn(image))
            phar.set_description(f"-> [{idx+1}/{len(ref_data)}] step2: get_lime_segment...")
        lime_segment = np.stack(temp)
        return lime_segment

    def get_lime_dataset(self, ref_data, lime_segment, mean=np.array([0, 0, 0]), num_samples=1000):
        if ref_data.shape[1] == 3:
            ref_data = np.moveaxis(ref_data, 1, -1)
        fudged_image = np.zeros(ref_data.shape[1:])
        fudged_image += mean.reshape([1, 1, -1])
        ref_dataset = []
        lime_dataset = []
        phar = tqdm(range(ref_data.shape[0]))
        for i in phar:
            n_features = np.unique(lime_segment[i]).shape[0]
            lime_data = np.random.randint(0, 2, [num_samples, n_features])
            lime_data[0, :] = 1
            ref_dataset.append(self.get_reference_dataset(lime_data, ref_data[i], lime_segment[i], fudged_image))
            lime_dataset.append(lime_data)
            phar.set_description(f"-> [{i}/{ref_data.shape[0]}] step3: get_lime_dataset...")
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
        print(f"-> compute_lime_signature")
        datasets = []
        with torch.no_grad():
            for i in range(len(lime_dataset)):
                lime_data = lime_dataset[i]
                data = ref_dataset[i]
                inputs = torch.from_numpy(data).permute(0, 3, 1, 2).float()
                outputs = model.batch_forward(inputs).detach().cpu().numpy()
                datasets.append([lime_data, outputs])
        weights = []
        with torch.no_grad():
            for data, label in datasets:
                data, label = torch.from_numpy(data).float().to(sys_args.device), torch.from_numpy(label).float().to(sys_args.device)
                w = torch.chain_matmul(torch.pinverse(torch.matmul(data.T, data)), data.T, label)
                weights.append(w)
        if cat:
            return torch.cat(weights)
        else:
            return weights.to("cpu")

    def compute_parameter_distance(self, lime_mask1, lime_mask2, order=['1', '2', 'inf', 'cos'], half=False, linear=False, lime=False):
        print(f"-> compute_parameter_distance...")
        weights1 = self.__consistent_type(lime_mask1, architecture=None, half=half, linear=linear, lime=lime)
        weights2 = self.__consistent_type(lime_mask2, architecture=None, half=half, linear=linear, lime=lime)
        if not isinstance(order, list):
            orders = [order]
        else:
            orders = order
        res_list = []
        if lime:
            temp_w1 = copy.copy(weights1)
            temp_w2 = copy.copy(weights2)
        for o in orders:
            if lime:
                weights1, weights2 = self.__lime_align(temp_w1, temp_w2, o)
            res = self.__compute_distance(weights1, weights2, o)
            if isinstance(res, np.ndarray):
                res = float(res)
            res_list.append(res)
        return np.array(res_list)

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
        return weights.to(sys_args.device)


def main(args):
    filename = str(osp.basename(__file__)).split(".")[0]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                        filename=f"{args.logs_root}/{filename}_{args.namespace}.txt")

    benchmark = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = benchmark.get_model_wrapper(args.model1)
    model2 = benchmark.get_model_wrapper(args.model2)

    comparison = Zest(model1, model2, out_root=args.zest_root, device=args.device)
    dist = comparison.compare()
    print(f"-> ZEST dist: {dist}")


if __name__ == "__main__":
    args = helper.get_args()
    args.zest_root = osp.join(args.out_root, "ZEST")
    main(args)














