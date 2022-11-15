import os
import torch
import torch.nn as nn
import types
import torch.nn.functional as F


__all__ = ["MobileNetV2", "mobilenet_v2"]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # CIFAR10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # END

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # CIFAR10: stride 2 -> 1
        features = [ConvBNReLU(3, input_channel, stride=1)]
        # END

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.features(x)
        z = z.mean([2, 3])
        y = self.classifier(z)
        return y


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
    x = x.mean([2, 3])
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
    x = x.mean([2, 3])
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
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py"""
    x = self.features[0](x)
    x = self.features[1](x)
    x = self.features[2](x)
    x = self.features[3](x)
    return x.contiguous()


def layerx2(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py"""
    x = self.features[4](x)
    x = self.features[5](x)
    x = self.features[6](x)
    return x.contiguous()


def layerx3(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py"""
    x = self.features[7](x)
    x = self.features[8](x)
    x = self.features[9](x)
    x = self.features[10](x)
    return x.contiguous()


def layerx4(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py"""
    x = self.features[11](x)
    x = self.features[12](x)
    x = self.features[13](x)
    x = self.features[14](x)
    x = self.features[15](x)
    x = self.features[16](x)
    return x.contiguous()


def layerx5(self, x):
    """Layer detail see: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py"""
    x = self.features[17](x)
    x = self.features[18](x)
    return x.contiguous()


def mobilenet_v2(pretrained=False, progress=True, device="cpu", **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/mobilenet_v2.pt", map_location=device
        )
        model.load_state_dict(state_dict)

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

    model = mobilenet_v2(pretrained=False)
    model.features[10].register_forward_hook(record_act)

    x = torch.randn(4, 3, 32, 32)
    y = torch.ones([4]).long() * 2

    model.eval()
    pred1 = model(x).argmax(dim=1)

    fmap3 = model.fed_forward(x, layer_index=1)
    logit = model.mid_forward(fmap3, layer_index=1)
    pred2 = logit.argmax(dim=1)

    z = model.fed_forward(x, layer_index=5)
    z = z.mean([2, 3])
    z = torch.flatten(z, 1)
    pred3 = model.classifier(z).argmax(dim=1)

    print(model.features[10].out.shape, pred1.tolist(), pred2.tolist(), pred3.tolist())