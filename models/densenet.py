import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .modules.checkpoint import CheckpointModule

__all__ = ['densenet']


def init_model(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)


def weight_decay_config(value=1e-4, log=False):
    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': lambda m: not isinstance(m, nn.BatchNorm2d)}
            }


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    @staticmethod
    def _create_features(num_features):
        # First convolution
        return nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, checkpoint_segments=0):

        super(DenseNet, self).__init__()

        self.features = self._create_features(num_init_features)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        if checkpoint_segments > 0:
            self.features = CheckpointModule(
                self.features, checkpoint_segments)
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        init_model(self)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out


class DenseNet_imagenet(DenseNet):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, regime='normal', scale_lr=1, **kwargs):
        super(DenseNet_imagenet, self).__init__(growth_rate, block_config, num_init_features,
                                                bn_size, drop_rate, num_classes, **kwargs)

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9,
                 'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr),
                 'regularizer': weight_decay_config(1e-4)},
                {'epoch': 5,  'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
        elif regime == 'small':
            scale_lr *= 4
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'lr': scale_lr * 1e-1,
                 'regularizer': weight_decay_config(1e-4)},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256},
                {'epoch': 80, 'input_size': 224, 'batch_size': 64},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 1024},
                {'epoch': 80, 'input_size': 224, 'batch_size': 512},
            ]


class DenseNet_cifar(DenseNet):

    @staticmethod
    def _create_features(num_features):
        # First convolution
        return nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_features,
                                kernel_size=3, stride=1, padding=1, bias=False))
        ]))

    def __init__(self, *kargs, **kwargs):

        super(DenseNet_cifar, self).__init__(*kargs, **kwargs)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
             'regularizer': weight_decay_config(1e-4)},
            {'epoch': 150, 'lr': 1e-2},
            {'epoch': 225, 'lr': 1e-3},
        ]

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet(**config):
    dataset = config.pop('dataset', 'imagenet')
    if config.pop('quantize', False):
        from .modules.quantize import QConv2d, QLinear, RangeBN
        torch.nn.Linear = QLinear
        torch.nn.Conv2d = QConv2d
        torch.nn.BatchNorm2d = RangeBN

    if dataset == 'imagenet':
        config.setdefault('num_classes', 1000)
        num = config.pop('num', 169)
        if num == 121:
            config.update(dict(num_init_features=64,
                               growth_rate=32, block_config=(6, 12, 24, 16)))
        elif num == 169:
            config.update(dict(num_init_features=64,
                               growth_rate=32, block_config=(6, 12, 32, 32)))
        elif num == 201:
            config.update(dict(num_init_features=64,
                               growth_rate=32, block_config=(6, 12, 48, 32)))
        elif num == 161:
            config.update(dict(num_init_features=96,
                               growth_rate=48, block_config=(6, 12, 36, 24)))

        return DenseNet_imagenet(**config)

    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
        config.setdefault('growth_rate', 12)
        config.setdefault('num_init_features', 2 * config['growth_rate'])
        depth = config.pop('depth', 100)
        N = (depth - 4) // 6
        config['block_config'] = (N, N, N)
        return DenseNet_cifar(**config)

    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
        config.setdefault('growth_rate', 12)
        config.setdefault('num_init_features', 2 * config['growth_rate'])
        depth = config.pop('depth', 100)
        N = (depth - 4) // 6
        config['block_config'] = (N, N, N)
        return DenseNet_cifar(**config)
