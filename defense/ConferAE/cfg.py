#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/25, homeway'


import torch
import argparse


def model_args():
    parser = argparse.ArgumentParser(description="Build benchmark.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    args, unknown = parser.parse_known_args()
    args.const_lr = False
    args.batch_size = 64
    args.lr = 5e-3
    args.print_freq = 50
    args.label_smoothing = 0
    args.vgg_output_distill = False
    args.reinit = False
    args.l2sp_lmda = 0
    args.train_all = False
    args.ft_begin_module = None
    args.momentum = 0
    args.weight_decay = 1e-4
    args.beta = 1e-2
    args.feat_lmda = 0
    args.iterations = 500
    args.test_interval = 50
    args.adv_test_interval = -1
    args.feat_layers = '1234'
    args.no_save = False
    args.steal = False
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    return args