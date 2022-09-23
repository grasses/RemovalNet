#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'

"""system DNN tools"""

import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm