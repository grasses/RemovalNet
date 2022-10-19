#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/07/29, homeway'


import types
import torch


def feature_list(self, x):
    out_list = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layer1(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layer2(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layer3(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.layer4(x)
    out_list.append(x.contiguous().view(x.size(0), -1))
    x = self.avgpool(x)
    x = x.reshape(x.size(0), -1)
    y = self.fc(x)
    return y, out_list


def bak_forward(self, x, layer_index):
    if layer_index == 1:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x.contiguous()
    if layer_index == 2:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x.contiguous()
    if layer_index == 3:
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x.contiguous()
    if layer_index == 4:
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x.contiguous()
    if layer_index == 5:
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x.contiguous()


def fed_forward(self, x, layer_index):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    if layer_index == 1: return x.contiguous()
    x = self.layer1(x)
    if layer_index == 2: return x.contiguous()
    x = self.layer2(x)
    if layer_index == 3: return x.contiguous()
    x = self.layer3(x)
    if layer_index == 4: return x.contiguous()
    x = self.layer4(x)
    if layer_index == 5: return x.contiguous()
    x = self.avgpool(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc(x)
    return x.contiguous()


def resnet34(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.resnet import resnet34 as torch_resnet34
    model = torch_resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.bak_forward = types.MethodType(bak_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def resnet50(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.resnet import resnet50 as torch_resnet50
    model = torch_resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.bak_forward = types.MethodType(bak_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def resnet101(num_classes=1000, pretrained=True, **kwargs):
    from torchvision.models.resnet import resnet101 as torch_resnet101
    model = torch_resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.bak_forward = types.MethodType(bak_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model