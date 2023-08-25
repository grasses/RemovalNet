#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


import logging
from dataset import loader as dloader
logger = logging.getLogger("ModelLoader")


def load_model(dataset_id, arch_id, pretrained=False, pretrain=None, **kwargs):
    print("-> dataset_id", dataset_id)
    print("-> arch_id", arch_id)
    if pretrain is not None:
        if "vit" not in arch_id:
            from model.inputx224 import torchvision_models
            torch_model = eval(f"torchvision_models.{arch_id}(num_classes=1000, pretrained='imagenet')")
        else:
            from model.inputx224 import ViT
            torch_model = eval(f"vit.{arch_id}(num_classes=1000, pretrained=True)")
        return torch_model

    num_classes = dloader.get_num_classess(dataset_id)
    if dataset_id in dloader.task_list["CV32"]:
        from model.inputx32 import vgg19_bn, vgg13_bn, vgg11_bn, vgg16_bn, inception_v3
        from model.inputx32 import resnet18, resnet34, resnet50, densenet121, mobilenet_v2

    elif dataset_id in dloader.task_list["CV224"]:
        from model.inputx224 import vgg19_bn, vgg13_bn, vgg11_bn, vgg16_bn
        from model.inputx224 import resnet50, alexnet, densenet121, mobilenet_v2
        from model.inputx224 import vit_base_patch8_224, vit_base_patch16_224, vit_base_patch32_224, \
            vit_base_patch32_224_sam, vit_base_patch32_224_clip_laion2b, vit_base_patch32_224_in21k
    else:
        raise NotImplementedError()

    torch_model = eval(f'{arch_id}')(
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )
    logger.info(f"-> load model arch:{arch_id} dataset:{dataset_id}")
    return torch_model