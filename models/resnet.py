import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .modules.se import SEBlock
from .modules.checkpoint import CheckpointModule
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.mixup import MixUp

__all__ = ['resnet', 'resnet_se']


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)

    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()


def weight_decay_config(value=1e-4, log=False):
    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': lambda m: not isinstance(m, nn.BatchNorm2d)}
            }


def mixsize_config(sz, base_size, base_batch, base_duplicates, adapt_batch, adapt_duplicates):
    assert adapt_batch or adapt_duplicates or sz == base_size
    batch_size = base_batch
    duplicates = base_duplicates
    if adapt_batch and adapt_duplicates:
        scale = base_size/sz
    else:
        scale = (base_size/sz)**2

    if scale * duplicates < 0.5:
        adapt_duplicates = False
        adapt_batch = True

    if adapt_batch:
        batch_size = int(round(scale * base_batch))

    if adapt_duplicates:
        duplicates = int(round(scale * duplicates))

    duplicates = max(1, duplicates)
    return {
        'input_size': sz,
        'batch_size': batch_size,
        'duplicates': duplicates
    }


def ramp_up_fn(lr0, lrT, T):
    rate = (lrT - lr0) / T
    return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=1,
                 downsample=None, groups=1, residual_block=None, dropout=0.):
        super(BasicBlock, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, expansion * planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None, dropout=0.):
        super(Bottleneck, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None, dropout=None, mixup=False):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block, dropout=dropout))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block, dropout=dropout))
        if mixup:
            layers.append(MixUp())
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
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


class ResNet_imagenet(ResNet):
    num_train_images = 1281167

    def __init__(self, num_classes=1000, inplanes=64,
                 block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3],
                 width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1],
                 regime='normal', scale_lr=1, ramp_up_lr=True, checkpoint_segments=0, mixup=False,
                 base_devices=4, base_device_batch=64, base_duplicates=1, base_image_size=224, mix_size_regime='D+'):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i],
                                     mixup=mixup)
            if checkpoint_segments > 0:
                layer_checkpoint_segments = min(checkpoint_segments, layers[i])
                layer = CheckpointModule(layer, layer_checkpoint_segments)
            setattr(self, 'layer%s' % str(i + 1), layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[-1] * expansion, num_classes)

        init_model(self)
        batch_size = base_devices * base_device_batch
        num_steps_epoch = math.floor(self.num_train_images / batch_size)

        # base regime
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': scale_lr * 1e-1,
             'momentum': 0.9, 'regularizer': weight_decay_config(1e-4)},
            {'epoch': 30, 'lr': scale_lr * 1e-2},
            {'epoch': 60, 'lr': scale_lr * 1e-3},
            {'epoch': 80, 'lr': scale_lr * 1e-4}
        ]

        if 'cutmix' in regime:
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': scale_lr * 1e-1,
                 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4)},
                {'epoch': 75, 'lr': scale_lr * 1e-2},
                {'epoch': 150, 'lr': scale_lr * 1e-3},
                {'epoch': 225, 'lr': scale_lr * 1e-4}
            ]

        # Sampled regimes from "Mix & Match: training convnets with mixed image sizes for improved accuracy, speed and scale resiliency"
        if 'sampled' in regime:
            # add gradient smoothing
            self.regime[0]['regularizer'] = [{'name': 'GradSmooth', 'momentum': 0.9, 'log': False},
                                             weight_decay_config(1e-4)]
            ramp_up_lr = False
            self.data_regime = None

            def size_config(size): return mixsize_config(size, base_size=base_image_size, base_batch=base_device_batch, base_duplicates=base_duplicates,
                                                         adapt_batch=mix_size_regime == 'B+', adapt_duplicates=mix_size_regime == 'D+')
            increment = int(base_image_size / 7)

            if '144' in regime:
                self.sampled_data_regime = [
                    (0.1, size_config(base_image_size+increment)),
                    (0.1, size_config(base_image_size)),
                    (0.6, size_config(base_image_size - 3*increment)),
                    (0.2, size_config(base_image_size - 4*increment)),
                ]
            else:  # sampled-224
                self.sampled_data_regime = [
                    (0.8/6, size_config(base_image_size - 3*increment)),
                    (0.8/6, size_config(base_image_size - 2*increment)),
                    (0.8/6, size_config(base_image_size - increment)),
                    (0.2, size_config(base_image_size)),
                    (0.8/6, size_config(base_image_size + increment)),
                    (0.8/6, size_config(base_image_size + 2*increment)),
                    (0.8/6, size_config(base_image_size + 3*increment)),
                ]

            self.data_eval_regime = [
                {'epoch': 0, 'input_size': base_image_size}
            ]

        if ramp_up_lr and scale_lr > 1:  # add learning rate ramp-up
            self.regime[0]['step_lambda'] = ramp_up_fn(0.1, 0.1 * scale_lr,
                                                       num_steps_epoch * 5)
            self.regime.insert(1, {'epoch': 5,  'lr': scale_lr * 1e-1})


class ResNet_cifar(ResNet):

    def __init__(self, num_classes=10, inplanes=16,
                 block=BasicBlock, depth=18, width=[16, 32, 64],
                 groups=[1, 1, 1], residual_block=None, regime='normal', dropout=None, mixup=False):
        super(ResNet_cifar, self).__init__()
        self.inplanes = inplanes
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x

        self.layer1 = self._make_layer(block, width[0], n, groups=groups[0],
                                       residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer2 = self._make_layer(block, width[1], n, stride=2, groups=groups[1],
                                       residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer3 = self._make_layer(block, width[2], n, stride=2, groups=groups[2],
                                       residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer4 = lambda x: x
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[-1], num_classes)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
                'regularizer': weight_decay_config(1e-4)},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3},
            {'epoch': 164, 'lr': 1e-4}
        ]

        if 'wide-resnet' in regime:
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
                 'regularizer': weight_decay_config(5e-4)},
                {'epoch': 60, 'lr': 2e-2},
                {'epoch': 120, 'lr': 4e-3},
                {'epoch': 160, 'lr': 8e-4}
            ]

        # Sampled regimes from "Mix & Match: training convnets with mixed image sizes for improved accuracy, speed and scale resiliency"
        if 'sampled' in regime:
            adapt_batch = True if 'B+' in regime else False
            adapt_duplicates = True if ('D+' in regime or not adapt_batch) \
                else False

            def size_config(size): return mixsize_config(size, base_size=32, base_batch=64, base_duplicates=1,
                                                         adapt_batch=adapt_batch, adapt_duplicates=adapt_duplicates)
            # add gradient smoothing
            self.regime[0]['regularizer'] = [{'name': 'GradSmooth', 'momentum': 0.9, 'log': False},
                                             weight_decay_config(1e-4)]
            self.data_regime = None
            self.sampled_data_regime = [
                (0.3, size_config(32)),
                (0.2, size_config(48)),
                (0.3, size_config(24)),
                (0.2, size_config(16)),
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 32, 'scale_size': 32}
            ]


def resnet(**config):
    dataset = config.pop('dataset', 'imagenet')
    if config.pop('quantize', False):
        from .modules.quantize import QConv2d, QLinear, RangeBN
        torch.nn.Linear = QLinear
        torch.nn.Conv2d = QConv2d
        torch.nn.BatchNorm2d = RangeBN

    bn_norm = config.pop('bn_norm', None)
    if bn_norm is not None:
        from .modules.lp_norm import L1BatchNorm2d, TopkBatchNorm2d
        if bn_norm == 'L1':
            torch.nn.BatchNorm2d = L1BatchNorm2d
        if bn_norm == 'TopK':
            torch.nn.BatchNorm2d = TopkBatchNorm2d

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

        return ResNet_imagenet(**config)

    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
        config.setdefault('depth', 44)
        return ResNet_cifar(block=BasicBlock, **config)

    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
        config.setdefault('depth', 44)
        return ResNet_cifar(block=BasicBlock, **config)


def resnet_se(**config):
    config['residual_block'] = SEBlock
    return resnet(**config)
