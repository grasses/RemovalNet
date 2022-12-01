#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/22, homeway'

"""Basic configure of Dataset"""

import torch
import argparse


def load_cfg(dataset_id):
    parser = argparse.ArgumentParser(description="default model config")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    args, unknown = parser.parse_known_args()
    args.seed = 1000
    args.lr = 1e-2
    args.print_freq = 200
    args.label_smoothing = 0
    args.const_lr = False
    args.backends = False
    args.vgg_output_distill = False
    args.reinit = False
    args.train_all = False
    args.ft_begin_module = None
    args.l2sp_lmda = 0
    args.momentum = 0
    args.weight_decay = 1e-4
    args.beta = 1e-2
    args.feat_lmda = 0
    args.test_interval = 1000
    args.adv_test_interval = -1
    args.feat_layers = '1234'
    args.no_save = False
    args.steal = False
    args.negative = False
    args.mean = (0.4914, 0.4822, 0.4465)
    args.std = (0.2471, 0.2435, 0.2616)
    args.input_size = 32
    args.input_shape = (3, 32, 32)
    args.resize_size = 32
    args.batch_size = 200

    args.TRAIN_ITERS = 50000
    args.STEAL_ITERS = 20000
    args.NEGATIVE_ITERS = 10000
    args.DISTILL_ITERS = 10000
    args.QUANTIZE_ITERS = 1000
    args.PRUNE_ITERS = 1000
    args.FINETUNING_ITERS = 1000
    args.CONTINUE_TRAIN = False
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    return args
