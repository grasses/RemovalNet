import os
import os.path as osp
import torch
import torch.nn as nn

__all__ = [
    "VGG",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]
ROOT = os.path.abspath(osp.join(osp.dirname(os.getcwd()), "ckpt"))


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        self.layer_list = features.keys()
        self.num_feats = 25088
        for idx, (name, layer) in enumerate(features.items()):
            setattr(self, name, layer)

        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def layerx1(self, x):
        return self.layer1(x).contiguous()

    def layerx2(self, x):
        return self.layer2(x).contiguous()

    def layerx3(self, x):
        return self.layer3(x).contiguous()

    def layerx4(self, x):
        return self.layer4(x).contiguous()

    def layerx5(self, x):
        return self.layer5(x).contiguous()

    def features(self, x):
        x = self.layerx1(x)
        x = self.layerx2(x)
        x = self.layerx3(x)
        x = self.layerx4(x)
        x = self.layerx5(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
        x = x.view(x.size(0), -1)
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
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y, out_list


def make_layers(cfg, batch_norm=False):
    in_channels = 3
    layers = []
    layer_index = 1
    layers_dict = {}
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers_dict[f"layer{layer_index}"] = nn.Sequential(*layers)
            layers = []
            layer_index += 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers_dict


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, device, task="CIFAR10", **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            f"{ROOT}/train({arch},{task})-/final_ckpt.pth", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def vgg11_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, device, **kwargs)


def vgg13_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, device, **kwargs)


def vgg16_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, device, **kwargs)


def vgg19_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, device, **kwargs)


def convert_from_pytorch_to_removalnet(state_dict_a, state_dict_b):
    """
    This script only to convert pytorch vgg model to removalnet.
    Removalnet uses the layer output, pytorch model only has lost of Conv output.
    Args:
        state_dict_a:
        state_dict_b:

    Returns:

    """
    for k1, k2 in zip(state_dict_a.keys(), state_dict_b.keys()):
        state_dict_a[k1] = state_dict_b[k2].clone().detach().cpu()
    return state_dict_a


if __name__ == "__main__":
    model = vgg19_bn()
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224))
    x = torch.randn(1, 3, 224, 224)
    y, outs = model.feature_list(x)
    for out in outs:
        print(out.shape)