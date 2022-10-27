#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/10/20, homeway'


import types
import torch
import torch.nn.functional as F


def feature_list(self, x):
    """
    Return feature map of each layer
    Args:
        self: Densenet
        x: Tensor
    Returns: Tensor, list
    """
    out_list = []
    x = self.layerx1(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layerx2(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layerx3(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layerx4(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layerx5(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = F.relu(x, inplace=True)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    y = self.classifier(x)
    return y, out_list


def mid_forward(self, x, layer_index):
    """
    Feed x to model from $layer_index layer
    Args:
        self: Densenet
        x: Tensor
        layer_index: Int
    Returns: Tensor
    """
    if layer_index == 1:
        x = self.layerx2(x)
        x = self.layerx3(x)
        x = self.layerx4(x)
        x = self.layerx5(x)
    if layer_index == 2:
        x = self.layerx3(x)
        x = self.layerx4(x)
        x = self.layerx5(x)
    if layer_index == 3:
        x = self.layerx4(x)
        x = self.layerx5(x)
    if layer_index == 4:
        x = self.layerx5(x)
    x = F.relu(x, inplace=True)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x.contiguous()


def fed_forward(self, x, layer_index):
    """
    Feed x to model from head to $layer_index layer
    Args:
        self: Densenet
        x: Tensor
        layer_index: Int
    Returns: Tensor
    """
    x = self.layerx1(x)
    if layer_index == 1: return x.contiguous()
    x = self.layerx2(x)
    if layer_index == 2: return x.contiguous()
    x = self.layerx3(x)
    if layer_index == 3: return x.contiguous()
    x = self.layerx4(x)
    if layer_index == 4: return x.contiguous()
    x = self.layerx5(x)
    if layer_index == 5: return x.contiguous()
    return x.contiguous()


def layerx1(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py"""
    x = self.features.conv0(x)
    x = self.features.norm0(x)
    x = self.features.relu0(x)
    x = self.features.pool0(x)
    return x.contiguous()


def layerx2(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py"""
    x = self.features.denseblock1(x)
    x = self.features.transition1(x)
    return x.contiguous()


def layerx3(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py"""
    x = self.features.denseblock2(x)
    x = self.features.transition2(x)
    return x.contiguous()


def layerx4(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py"""
    x = self.features.denseblock3(x)
    x = self.features.transition3(x)
    return x.contiguous()


def layerx5(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py"""
    x = self.features.denseblock4(x)
    x = self.features.norm5(x)
    return x.contiguous()


def densenet121(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.densenet import densenet121 as torch_densenet121
    model = torch_densenet121(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def densenet161(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.densenet import densenet161 as torch_densenet161
    model = torch_densenet161(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def densenet169(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.densenet import densenet169 as torch_densenet169
    model = torch_densenet169(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def densenet201(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.densenet import densenet201 as torch_densenet201
    model = torch_densenet201(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model



if __name__ == "__main__":
    def record_act(self, input, output):
        self.out = output

    model = densenet121(pretrained=False)
    model.features[2].register_forward_hook(record_act)

    x = torch.randn(1, 3, 224, 224)
    fmap3 = model.fed_forward(x.clone(), layer_index=3)
    logit = model.mid_forward(fmap3, layer_index=3)
    pred1 = logit.argmax(dim=1)

    x = model.fed_forward(x.clone(), layer_index=5)
    x = F.relu(x, inplace=True)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    pred2 = model.classifier(x).argmax(dim=1)
    print(model.features[2].out.shape, pred1, pred2)