import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision.transforms as transforms
__all__ = ['mobilenet_v2']


def nearby_int(n):
    return int(round(n))


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def weight_decay_config(value=1e-4, log=False):
    def regularize_layer(m):
        non_depthwise_conv = isinstance(m, nn.Conv2d) \
            and m.groups != m.in_channels
        return isinstance(m, nn.Linear) or non_depthwise_conv

    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': regularize_layer}
            }


class ExpandedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=1, kernel_size=3,
                 stride=1, padding=1, residual_block=None):
        expanded = in_channels * expansion
        super(ExpandedConv2d, self).__init__()
        self.add_res = stride == 1 and in_channels == out_channels
        self.residual_block = residual_block
        if expanded == in_channels:
            block = []
        else:
            block = [
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                nn.ReLU6(inplace=True),
            ]

        block += [
            nn.Conv2d(expanded, expanded, kernel_size,
                      stride=stride, padding=padding, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        if self.add_res:
            if self.residual_block is not None:
                x = self.residual_block(x)
            out += x
        return out


def conv(in_channels, out_channels, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel,
                  stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class MobileNet_v2(nn.Module):

    def __init__(self, width=1., regime=None, num_classes=1000, scale_lr=1):
        super(MobileNet_v2, self).__init__()
        in_channels = nearby_int(width * 32)

        layers_config = [
            dict(expansion=1, stride=1, out_channels=nearby_int(width * 16)),
            dict(expansion=6, stride=2, out_channels=nearby_int(width * 24)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 24)),
            dict(expansion=6, stride=2, out_channels=nearby_int(width * 32)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 32)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 32)),
            dict(expansion=6, stride=2, out_channels=nearby_int(width * 64)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 64)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 64)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 64)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 96)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 96)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 96)),
            dict(expansion=6, stride=2, out_channels=nearby_int(width * 160)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 160)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 160)),
            dict(expansion=6, stride=1, out_channels=nearby_int(width * 320)),
        ]

        self.features = nn.Sequential()
        self.features.add_module('conv0', conv(3, in_channels,
                                               kernel=3, stride=2, padding=1))

        for i, layer in enumerate(layers_config):
            layer['in_channels'] = in_channels
            in_channels = layer['out_channels']
            self.features.add_module(
                'bottleneck' + str(i), ExpandedConv2d(**layer))

        out_channels = nearby_int(width * 1280)
        self.features.add_module('conv1', conv(in_channels, out_channels,
                                               kernel=1, stride=1, padding=0))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, True),
            nn.Linear(out_channels, num_classes)
        )
        init_model(self)

        if regime == 'small':
            scale_lr *= 4
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 512},
                {'epoch': 80, 'input_size': 224, 'batch_size': 128},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 128,
                    'scale_size': 160, 'batch_size': 1024},
                {'epoch': 80, 'input_size': 224, 'batch_size': 512},
            ]

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'lr': scale_lr * 1e-1,
             'regularizer': weight_decay_config(1e-4)},
            {'epoch': 30, 'lr': scale_lr * 1e-2},
            {'epoch': 60, 'lr': scale_lr * 1e-3},
            {'epoch': 80, 'lr': scale_lr * 1e-4}
        ]

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenet_v2(**config):
    r"""MobileNet v2 model architecture from the `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_ paper.
    """
    dataset = config.pop('dataset', 'imagenet')
    assert dataset == 'imagenet'
    return MobileNet_v2(**config)
