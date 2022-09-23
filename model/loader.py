#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


from .inputx32.vgg import *
from .inputx32.resnet import *
from .inputx224.fe_resnet import *
from .inputx224.fe_vgg16 import *
from .inputx224 import torchvision_models


def load_model(task, arch, **kwargs):
    pass