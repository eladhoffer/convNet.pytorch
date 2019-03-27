"""
adapted from https://github.com/quark0/darts
"""
import math
import logging
import torch
import torch.nn as nn
from .modules.evolved_modules import NasNetCell, AmoebaNetCell, DARTSCell
import sys
sys.path.append('..')
from utils.cross_entropy import cross_entropy, CrossEntropyLoss

__all__ = ['amoebanet', 'darts', 'nasnet']


def weight_decay_config(value=3e-4, log=False):
    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': lambda m: not isinstance(m, nn.BatchNorm2d)}
            }


def cosine_anneal_lr(epoch, base_lr=0.025, T_max=600., eta_min=0.):
    return eta_min + (base_lr - eta_min) * \
        (1 + math.cos(math.pi * epoch / T_max)) / 2


def modify_drop_path_rate(model, value, log=True):
    if log and model.drop_path != value:
        logging.debug('Modified drop-path rate from %s to %s' %
                      (model.drop_path, value))
    model.drop_path = value


def multi_output_loss(outputs, target, output_weights, loss_fn=cross_entropy, **loss_kwargs):
    """ outputs is a list/tuple of outputs, output_weights is list of same length of output_weights
    """
    assert isinstance(outputs, list) or isinstance(outputs, tuple)
    assert isinstance(output_weights, list) or isinstance(
        output_weights, tuple)
    assert len(outputs) == len(output_weights)

    loss = 0
    for y, w in zip(outputs, output_weights):
        if y is not None and w is not None and w != 0.:
            loss = loss + w * loss_fn(y, target, **loss_kwargs)
    return loss


class MultiOutputLoss(nn.Module):
    def __init__(self, output_weights, loss=CrossEntropyLoss, **loss_kwargs):
        super(MultiOutputLoss, self).__init__()
        self.loss = loss(**loss_kwargs)
        self.output_weights = output_weights

    def forward(self, outputs, target):
        return multi_output_loss(outputs, target, self.output_weights, loss_fn=self.loss)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, channels, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, channels, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class EvolvedNetworkCIFAR(nn.Module):
    def __init__(self, init_channels=36, num_classes=10, layers=20, auxiliary=True, aux_weight=0.4,
                 drop_path=0.2, num_epochs=600, init_lr=0.025, cell_fn=DARTSCell):
        super(EvolvedNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path = drop_path

        stem_multiplier = 3
        channels = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        prev2_channels, prev_channels, channels =\
            channels, channels, init_channels
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                channels *= 2
                reduction = True
            else:
                reduction = False
            cell = cell_fn(prev2_channels, prev_channels,
                           channels, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            prev2_channels, prev_channels = prev_channels, cell.multiplier*channels
            if i == 2*layers//3:
                aux_channels = prev_channels

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(
                aux_channels, num_classes)

            def loss_fn(*kargs, **kwargs):
                return MultiOutputLoss([1., aux_weight], *kargs, **kwargs)
            self.criterion = loss_fn

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev_channels, num_classes)

        def config_by_epoch(epoch):
            return {'lr': cosine_anneal_lr(epoch, base_lr=init_lr, T_max=float(num_epochs)),
                    'execute': lambda: modify_drop_path_rate(self, drop_path * epoch / float(num_epochs))}
        self.regime = [{'optimizer': 'SGD', 'momentum': 0.9,
                        'regularizer': weight_decay_config(3e-4),
                        'epoch_lambda': config_by_epoch}]

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path)
            if i == 2*self._layers//3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary:
            return logits, logits_aux
        else:
            return logits


class EvolvedNetworkImageNet(nn.Module):

    def __init__(self, init_channels=36, num_classes=1000, layers=20, auxiliary=True, aux_weight=0.4, drop_path=0.2, cell_fn=DARTSCell):
        super(EvolvedNetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path = drop_path

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, init_channels // 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels // 2, init_channels,
                      3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels, init_channels, 3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
        )

        prev2_channels, prev_channels, channels = \
            init_channels, init_channels, init_channels

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                channels *= 2
                reduction = True
            else:
                reduction = False
            cell = cell_fn(prev2_channels, prev_channels,
                           channels, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            prev2_channels, prev_channels = prev_channels, cell.multiplier * channels
            if i == 2 * layers // 3:
                aux_channels = prev_channels

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(
                aux_channels, num_classes)

            def loss_fn(*kargs, **kwargs):
                return MultiOutputLoss([1., aux_weight], *kargs, **kwargs)
            self.criterion = loss_fn
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(prev_channels, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary:
            return logits, logits_aux
        else:
            return logits


def evolved_network(**model_config):
    dataset = model_config.pop('dataset')

    if 'cifar' in dataset:
        if dataset == 'cifar10':
            model_config.setdefault('num_classes', 10)
        elif dataset == 'cifar100':
            model_config.setdefault('num_classes', 100)
        network = EvolvedNetworkCIFAR(**model_config)
    elif 'imagenet' in dataset:
        network = EvolvedNetworkImageNet(**model_config)
    return network


def nasnet(**model_config):
    model_config.setdefault('cell_fn', NasNetCell)
    return evolved_network(**model_config)


def darts(**model_config):
    model_config.setdefault('cell_fn', DARTSCell)
    return evolved_network(**model_config)


def amoebanet(**model_config):
    model_config.setdefault('cell_fn', AmoebaNetCell)
    return evolved_network(**model_config)
