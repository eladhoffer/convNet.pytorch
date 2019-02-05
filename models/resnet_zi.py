import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .modules.se import SEBlock
from .modules.checkpoint import CheckpointModule
from .modules.fixup import ZIConv2d, ZILinear
__all__ = ['resnet_zi', 'resnet_zi_se']


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False, pre_bias=True, post_bias=True, multiplier=False):
    "3x3 convolution with padding"
    return ZIConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, groups=groups, bias=bias, pre_bias=pre_bias, post_bias=post_bias, multiplier=multiplier)


def init_model(model):
    def zi(layer, m, L):
        layer.weight.data.div_(math.pow(L, -1./(2*m-2)))

    for m in model.modules():
        if isinstance(m, ZIConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
    for m in model.modules():
        if isinstance(m, Bottleneck):
            zi(m.conv1, 3, m.layer_depth)
            zi(m.conv2, 3, m.layer_depth)
            nn.init.constant_(m.conv3.weight, 0)
        elif isinstance(m, BasicBlock):
            zi(m.conv1, 2, m.layer_depth)
            nn.init.constant_(m.conv2.weight, 0)

    model.fc.weight.data.zero_()
    model.fc.bias.data.zero_()


def weight_decay_config(value=1e-4, log=True):
    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: 'bias' not in n and 'multiplier' not in n}
            }


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=1, downsample=None, groups=1, residual_block=None, layer_depth=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, expansion * planes,
                             groups=groups, multiplier=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.layer_depth = layer_depth

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None, layer_depth=1):
        super(Bottleneck, self).__init__()

        self.conv1 = ZIConv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups)
        self.conv3 = ZIConv2d(
            planes, planes * expansion, kernel_size=1, multiplier=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.layer_depth = layer_depth

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNetZI(nn.Module):

    def __init__(self):
        super(ResNetZI, self).__init__()
        self.num_layers = 1

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = ZIConv2d(self.inplanes, out_planes,
                                  kernel_size=1, stride=stride, bias=False)
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block, layer_depth=self.num_layers))
        self.inplanes = planes * expansion
        self.num_layers += 1
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block, layer_depth=self.num_layers))
            self.num_layers += 1

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class ResNetZI_imagenet(ResNetZI):

    def __init__(self, num_classes=1000, inplanes=64,
                 block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3],
                 width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1],
                 regime='normal', scale_lr=1, checkpoint_segments=0):
        super(ResNetZI_imagenet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = ZIConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i])
            if checkpoint_segments > 0:
                layer_checkpoint_segments = min(checkpoint_segments, layers[i])
                layer = CheckpointModule(layer, layer_checkpoint_segments)
            setattr(self, 'layer%s' % str(i + 1), layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = ZILinear(width[-1] * expansion, num_classes,
                           bias=True, post_bias=False)

        init_model(self)

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr)},
                {'epoch': 5,  'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
        elif regime == 'fast':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * 4 * scale_lr, 5004 * 4 / (4 * scale_lr))},
                {'epoch': 4,  'lr': 4 * scale_lr * 1e-1},
                {'epoch': 18, 'lr': scale_lr * 1e-1},
                {'epoch': 21, 'lr': scale_lr * 1e-2},
                {'epoch': 35, 'lr': scale_lr * 1e-3},
                {'epoch': 43, 'lr': scale_lr * 1e-4},
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256},
                {'epoch': 18, 'input_size': 224, 'batch_size': 64},
                {'epoch': 41, 'input_size': 288, 'batch_size': 32},
            ]
        elif regime == 'small':
            scale_lr *= 4
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'regularizer': weight_decay_config(1e-4),
                 'momentum': 0.9, 'lr': scale_lr * 1e-1},
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

        elif regime == 'small_ba':
            scale_lr = 1
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'regularizer': weight_decay_config(1e-4),
                 'momentum': 0.9, 'lr': scale_lr * 1e-1},
                {'epoch': 20, 'lr': scale_lr * 1e-2},
                {'epoch': 25, 'lr': scale_lr * 1e-3},
                {'epoch': 28, 'lr': scale_lr * 1e-4}
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 64, 'duplicates': 4},
                {'epoch': 25, 'input_size': 224, 'batch_size': 64, 'duplicates': 1},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 224, 'batch_size': 128}
            ]


class ResNetZI_cifar(ResNetZI):

    def __init__(self, num_classes=10, inplanes=16,
                 block=BasicBlock, depth=18, width=[16, 32, 64],
                 groups=[1, 1, 1], residual_block=None):
        super(ResNetZI_cifar, self).__init__()
        self.inplanes = inplanes
        n = int((depth - 2) / 6)
        self.conv1 = ZIConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                              bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, width[0], n, groups=groups[
                                       0], residual_block=residual_block)
        self.layer2 = self._make_layer(
            block, width[1], n, stride=2, groups=groups[1], residual_block=residual_block)
        self.layer3 = self._make_layer(
            block, width[2], n, stride=2, groups=groups[2], residual_block=residual_block)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = ZILinear(width[-1], num_classes,
                           bias=True, post_bias=False)
        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
             'regularizer': weight_decay_config(1e-4)},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3},
            {'epoch': 164, 'lr': 1e-4}
        ]


def resnet_zi(**config):
    dataset = config.pop('dataset', 'imagenet')

    if 'imagenet' in dataset:
        config.setdefault('num_classes', 1000)
        depth = config.pop('depth', 50)
        if depth == 18:
            config.update(dict(block=BasicBlock,
                               layers=[2, 2, 2, 2],
                               expansion=1))
        if depth == 34:
            config.update(dict(block=BasicBlock,
                               layers=[3, 4, 6, 3],
                               expansion=1))
        if depth == 50:
            config.update(dict(block=Bottleneck, layers=[3, 4, 6, 3]))
        if depth == 101:
            config.update(dict(block=Bottleneck, layers=[3, 4, 23, 3]))
        if depth == 152:
            config.update(dict(block=Bottleneck, layers=[3, 8, 36, 3]))
        if depth == 200:
            config.update(dict(block=Bottleneck, layers=[3, 24, 36, 3]))

        return ResNetZI_imagenet(**config)

    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
        config.setdefault('depth', 44)
        return ResNetZI_cifar(block=BasicBlock, **config)

    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
        config.setdefault('depth', 44)
        return ResNetZI_cifar(block=BasicBlock, **config)


def resnet_zi_se(**config):
    config['residual_block'] = SEBlock
    return resnet_zi(**config)
