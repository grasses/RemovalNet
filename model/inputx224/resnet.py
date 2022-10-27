#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/07/29, homeway'


import types
import torch


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
    x = self.avgpool(x)
    x = x.reshape(x.size(0), -1)
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
    x = self.avgpool(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc(x)
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
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    return x.contiguous()


def layerx2(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
    x = self.layer1(x)
    return x.contiguous()


def layerx3(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
    x = self.layer2(x)
    return x.contiguous()


def layerx4(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
    x = self.layer3(x)
    return x.contiguous()


def layerx5(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
    x = self.layer4(x)
    return x.contiguous()


def classifier(self, x):
    return self.fc(x)


def resnet34(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.resnet import resnet34 as torch_resnet34
    model = torch_resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.classifier = types.MethodType(classifier, model)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def resnet50(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.resnet import resnet50 as torch_resnet50
    model = torch_resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.classifier = types.MethodType(classifier, model)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def resnet101(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.resnet import resnet101 as torch_resnet101
    model = torch_resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.classifier = types.MethodType(classifier, model)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


if __name__ == "__main__":
    model = resnet50(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    fmap3 = model.fed_forward(x, layer_index=3)
    logit = model.mid_forward(fmap3, layer_index=3)
    pred1 = logit.argmax(dim=1)

    fmap5 = model.fed_forward(x, layer_index=5)
    x = model.avgpool(fmap5)
    x = x.reshape(x.size(0), -1)
    pred2 = model.classifier(x).argmax(dim=1)
    print(pred1, pred2)