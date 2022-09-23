#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/22, homeway'

"""Basic configure of Dataset"""

SEED = 1024
INPUT_SHAPE = (3, 32, 32)
INPUT_SIZE = 32
RESIZE_SIZE = 64
BATCH_SIZE = 200
TRAIN_ITERS = 100000
DEFAULT_ITERS = 10000
MEAN = (0.19803014, 0.20101564, 0.19703615)
STD = (0.19803014, 0.20101564, 0.19703615)
TRANSFER_ITERS = DEFAULT_ITERS
QUANTIZE_ITERS = DEFAULT_ITERS  # may be useless
PRUNE_ITERS = DEFAULT_ITERS
DISTILL_ITERS = DEFAULT_ITERS
STEAL_ITERS = DEFAULT_ITERS
CONTINUE_TRAIN = False  # whether to continue previous training