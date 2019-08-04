import logging
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .modules.se import SESwishBlock
from .modules.activations import Swish, HardSwish

__all__ = ['efficientnet']


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def modify_drop_connect_rate(model, value, log=True):
    for m in model.modules():
        if hasattr(m, 'drop_prob'):
            if log and m.drop_prob != value:
                logging.debug('Modified drop-path rate from %s to %s' %
                              (m.drop_prob, value))
            m.drop_prob = value


def weight_decay_config(value=1e-4, log=False):
    def regularize_layer(m):
        non_depthwise_conv = isinstance(m, nn.Conv2d) \
            and m.groups != m.in_channels
        return not isinstance(m, nn.BatchNorm2d)

    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': regularize_layer}
            }


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, *kargs, **kwargs):
        hard_act = kwargs.pop('hard_act', False)
        kwargs.setdefault('bias', False)

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, *kargs, **kwargs),
            nn.BatchNorm2d(out_channels),
            HardSwish() if hard_act else Swish()
        )


def drop_connect(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = x.new(x.size(0), 1, 1, 1).bernoulli_(keep_prob).float()
        mask.div_(keep_prob)
        x = x.mul(mask)
    return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1, kernel_size=3,
                 stride=1, padding=1, se_ratio=0.25, hard_act=False):
        expanded = in_channels * expansion
        super(MBConv, self).__init__()
        self.add_res = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(in_channels, expanded, 1,
                      hard_act=hard_act) if expanded != in_channels else nn.Identity(),
            ConvBNAct(expanded, expanded, kernel_size,
                      stride=stride, padding=padding, groups=expanded, hard_act=hard_act),
            SESwishBlock(expanded, expanded, int(in_channels*se_ratio),
                         hard_act=hard_act) if se_ratio > 0 else nn.Identity(),
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.drop_prob = 0

    def forward(self, x):
        out = self.block(x)
        if self.add_res:
            if self.training and self.drop_prob > 0.:
                x = drop_connect(x, self.drop_prob)
            out += x
        return out


class MBConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, num, expansion=1, kernel_size=3,
                 stride=1, padding=1, se_ratio=0.25, hard_act=False):
        kwargs = dict(expansion=expansion, kernel_size=kernel_size,
                      stride=stride, padding=padding, se_ratio=se_ratio, hard_act=hard_act)
        first_conv = MBConv(in_channels, out_channels, **kwargs)
        kwargs['stride'] = 1
        super(MBConvBlock, self).__init__(first_conv,
                                          *[MBConv(out_channels, out_channels, **kwargs) for _ in range(num-1)]
                                          )


class EfficientNet(nn.Module):

    def __init__(self, width_coeff=1, depth_coeff=1, resolution=224, se_ratio=0.25, regime='cosine', num_classes=1000,
                 scale_lr=1, dropout_rate=0.2, drop_connect_rate=0.2, num_epochs=200, hard_act=False):
        super(EfficientNet, self).__init__()

        def channels(base_channels, coeff=width_coeff, divisor=8, min_channels=None):
            if coeff == 1:
                return base_channels
            min_channels = min_channels or divisor
            channels = base_channels * coeff
            channels = max(min_channels,
                           int(base_channels + divisor / 2) // divisor * divisor)
            if channels < 0.9 * base_channels:
                channels += divisor
            return int(channels)

        def repeats(repeats, coeff=depth_coeff):
            return int(math.ceil(coeff * repeats))

        def config(out_channels, num, expansion=1, kernel_size=3,
                   stride=1, padding=None, se_ratio=se_ratio, hard_act=hard_act):
            padding = padding or int((kernel_size-1)//2)
            return {'out_channels': channels(out_channels), 'num': repeats(num),
                    'expansion': expansion, 'kernel_size': kernel_size, 'stride': stride,
                    'padding': padding, 'se_ratio': se_ratio, 'hard_act': hard_act}

        stages = [
            config(16, num=1,  expansion=1, kernel_size=3, stride=1),
            config(24, num=2,  expansion=6, kernel_size=3, stride=2),
            config(40, num=2,  expansion=6, kernel_size=5, stride=2),
            config(80, num=3,  expansion=6, kernel_size=3, stride=2),
            config(112, num=3,  expansion=6, kernel_size=5, stride=1),
            config(192, num=4,  expansion=6, kernel_size=5, stride=2),
            config(320, num=1,  expansion=6, kernel_size=3, stride=1),
        ]

        layers = []
        for i in range(len(stages)):
            in_channel = channels(32) if i == 0 \
                else stages[i-1]['out_channels']
            layers.append(MBConvBlock(in_channel, **stages[i]))

        self.features = nn.Sequential(
            ConvBNAct(3, channels(32), 3, 2, 1, hard_act=hard_act),
            *layers,
            ConvBNAct(channels(320), channels(1280), 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate, True)
        )
        self.classifier = nn.Linear(channels(1280), num_classes)

        init_model(self)

        def increase_drop_connect(epoch):
            return lambda: modify_drop_connect_rate(self, min(drop_connect_rate, drop_connect_rate * epoch / float(num_epochs)))

        if regime == 'paper':
            def config_by_epoch(epoch):
                return {'lr': scale_lr * 0.016 * (0.97 ** round(epoch/2.4)),
                        'execute': increase_drop_connect(epoch)}

            """RMSProp optimizer with
            decay 0.9 and momentum 0.9;
            weight decay 1e-5; initial learning rate 0.256 that decays
            by 0.97 every 2.4 epochs"""
            self.regime = [{'optimizer': 'RMSprop', 'alpha': 0.9, 'momentum': 0.9, 'lr': scale_lr * 0.016,
                            'regularizer': weight_decay_config(1e-5),
                            'epoch_lambda': config_by_epoch}]

        elif regime == 'cosine':
            def cosine_anneal_lr(epoch, base_lr=0.025, T_max=num_epochs, eta_min=1e-4):
                return eta_min + (base_lr - eta_min) * \
                    (1 + math.cos(math.pi * epoch / T_max)) / 2

            def config_by_epoch(epoch):
                return {'lr': cosine_anneal_lr(epoch, base_lr=scale_lr * 0.1, T_max=num_epochs),
                        'execute': increase_drop_connect(epoch)}
            self.regime = [{'optimizer': 'SGD', 'momentum': 0.9,
                            'regularizer': weight_decay_config(1e-5),
                            'epoch_lambda': config_by_epoch}]

        self.data_regime = [{'input_size': resolution, 'autoaugment': True}]
        self.data_eval_regime = [{'input_size': resolution,
                                  'scale_size': int(resolution * 8/7)}]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.flatten(1, -1))
        return x


def efficientnet(**config):
    dataset = config.pop('dataset', 'imagenet')
    assert dataset == 'imagenet'

    scale = config.pop('scale', 'b0')

    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'b0': (1.0, 1.0, 224, 0.2),
        'b1': (1.0, 1.1, 240, 0.2),
        'b2': (1.1, 1.2, 260, 0.3),
        'b3': (1.2, 1.4, 300, 0.3),
        'b4': (1.4, 1.8, 380, 0.4),
        'b5': (1.6, 2.2, 456, 0.4),
        'b6': (1.8, 2.6, 528, 0.5),
        'b7': (2.0, 3.1, 600, 0.5),
    }
    assert scale in params_dict.keys()
    config['width_coeff'], config['depth_coeff'], config['resolution'], config['dropout_rate'] = params_dict[scale]
    return EfficientNet(**config)
