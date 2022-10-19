#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/07/29, homeway'



import timm
import types
import torch


def vit_base_patch16_224(num_classes=1000, pretrained=True, **kwargs):
    def feature_list(self, x):
        out_list = []
        x = self.forward_features(x)
        y = self.forward_head(x)
        out_list.append(y.contiguous().view(x.size(0), -1))
        return y, out_list

    def bak_forward(self, x, layer_index):
        pass

    def fed_forward(self, x, layer_index):
        pass

    model = timm.create_model('vit_base_patch16_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.bak_forward = types.MethodType(bak_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model