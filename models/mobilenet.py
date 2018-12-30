import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision.transforms as transforms
__all__ = ['mobilenet']


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


def weight_decay_config(value=1e-4, log=True):
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


class DepthwiseSeparableFusedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super(DepthwiseSeparableFusedConv2d, self).__init__()
        self.components = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.components(x)


class MobileNet(nn.Module):

    def __init__(self, width=1., shallow=False, regime=None, num_classes=1000):
        super(MobileNet, self).__init__()
        num_classes = num_classes or 1000
        width = width or 1.
        layers = [
            nn.Conv2d(3, nearby_int(width * 32),
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nearby_int(width * 32)),
            nn.ReLU(inplace=True),

            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 32), nearby_int(width * 64),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 64), nearby_int(width * 128),
                kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 128), nearby_int(width * 128),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 128), nearby_int(width * 256),
                kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 256), nearby_int(width * 256),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 256), nearby_int(width * 512),
                kernel_size=3, stride=2, padding=1)
        ]
        if not shallow:
            # 5x 512->512 DW-separable convolutions
            layers += [
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
            ]
        layers += [
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 512), nearby_int(width * 1024),
                kernel_size=3, stride=2, padding=1),
            # Paper specifies stride-2, but unchanged size.
            # Assume its a typo and use stride-1 convolution
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 1024), nearby_int(width * 1024),
                kernel_size=3, stride=1, padding=1)
        ]
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(nearby_int(width * 1024), num_classes)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.data_regime = [{
            'transform': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        }]

        if regime == 'small':
            scale_lr = 4
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD',
                 'momentum': 0.9, 'lr': scale_lr * 1e-1, 'regularizer': weight_decay_config(1e-4)},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 512},
                {'epoch': 80, 'input_size': 224, 'batch_size': 128},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 1024},
                {'epoch': 80, 'input_size': 224, 'batch_size': 512},
            ]
        else:
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
                 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4)},
                {'epoch': 30, 'lr': 1e-2},
                {'epoch': 60, 'lr': 1e-3},
                {'epoch': 80, 'lr': 1e-4}
            ]

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet(**config):
    r"""MobileNet model architecture from the `"MobileNets:
    Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    dataset = config.pop('dataset', 'imagenet')
    assert dataset == 'imagenet'
    return MobileNet(**config)
