#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/07/29, homeway'



import timm
import types
import torch


def feature_list(self, x):
    out_list = []
    x = self.forward_features(x)
    y = self.forward_head(x)
    out_list.append(y.contiguous().view(x.size(0), -1))
    return y, out_list


def mid_forward(self, z, layer_index=1):
    return self.forward_head(z)


def fed_forward(self, x, layer_index=1):
    return self.forward_features(x)


def vit_base_patch8_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_base_patch8_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_base_patch16_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_base_patch16_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_base_patch32_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_base_patch32_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_base_patch32_224_clip_laion2b(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_base_patch32_224_clip_laion2b', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_base_patch32_224_in21k(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_base_patch32_224_in21k', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_small_patch16_224_in21k(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_small_patch16_224_in21k', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def vit_base_patch32_224_sam(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_base_patch32_224_sam', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def vit_large_patch32_224_in21k(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_large_patch32_224_in21k', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_large_patch32_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_large_patch32_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def xcit_small_12_p8_224_dist(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('xcit_small_12_p8_224_dist', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def swin_s3_base_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('swin_s3_base_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def swin_s3_small_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('swin_s3_small_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def swin_s3_tiny_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('swin_s3_tiny_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def vit_small_patch32_224_in21k(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_small_patch32_224_in21k', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_large_r50_s32_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_large_r50_s32_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def vit_base_r26_s32_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_base_r26_s32_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model


def vit_small_r26_s32_224(num_classes=1000, pretrained=True, **kwargs):
    model = timm.create_model('vit_small_r26_s32_224', num_classes=num_classes, pretrained=pretrained, **kwargs)
    model.feature_list = types.MethodType(feature_list, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    return model

def negative_vit_model(seed):
    seed = int(int(seed-1) / 100)
    models = [
        # vit_small_r26_s32_224
        #vit_small_r26_s32_224,
        #vit_small_patch32_224_in21k,
        #xcit_small_12_p8_224_dist,
        #swin_s3_base_224,
        #swin_s3_small_224,
        #swin_s3_tiny_224,
        #vit_small_patch32_224_in21k,
        vit_base_patch32_224_sam, # ok
        vit_base_patch32_224_clip_laion2b, #ok train(vit_base_patch32_224_clip_laion2b,ImageNet)-
        vit_base_patch32_224_in21k, # ok train(vit_base_patch32_224_in21k,ImageNet)-
    ]
    print(models[seed].__name__)
    return models[seed](num_classes=1000, pretrained=True)


if __name__ == "__main__":
    model = vit_base_patch8_224()
    x = torch.randn(1, 3, 224, 224)
    y, out_list = model.feature_list(x)
    print(f"-> y:{y.argmax(dim=1)}, out_list:{out_list[-1].shape}")

    for seed in range(10):
        negative_vit_model(seed=seed*100)
