import torch
import torch.nn as nn
import math
from .resnet import ResNet_imagenet, ResNet_cifar, BasicBlock, Bottleneck
from .modules.se import SEBlock

__all__ = ['resnext', 'resnext_se']


class ResNeXt_imagenet(ResNet_imagenet):

    def __init__(self, width=[128, 256, 512, 1024], groups=[32, 32, 32, 32], expansion=2, **kwargs):
        kwargs['width'] = width
        kwargs['groups'] = groups
        kwargs['expansion'] = expansion
        super(ResNeXt_imagenet, self).__init__(**kwargs)

class ResNeXt_cifar(ResNet_cifar):

    def __init__(self, width=[64, 128, 256], groups=[4, 8, 16], **kwargs):
        kwargs['width'] = width
        kwargs['groups'] = groups
        super(ResNeXt_cifar, self).__init__(**kwargs)

def resnext(**config):
    dataset = config.pop('dataset', 'imagenet')
    if dataset == 'imagenet':
        config.setdefault('num_classes', 1000)
        depth = config.pop('depth', 50)
        if depth == 18:
            config.update(dict(block=BasicBlock, layers=[2, 2, 2, 2]))
        if depth == 34:
            config.update(dict(block=BasicBlock, layers=[3, 4, 6, 3]))
        if depth == 50:
            config.update(dict(block=Bottleneck, layers=[3, 4, 6, 3]))
        if depth == 101:
            config.update(dict(block=Bottleneck, layers=[3, 4, 23, 3]))
        if depth == 152:
            config.update(dict(block=Bottleneck, layers=[3, 8, 36, 3]))

        return ResNeXt_imagenet(**config)

    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
        config.setdefault('depth', 44)
        return ResNeXt_cifar(block=BasicBlock, **config)

    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
        config.setdefault('depth', 44)
        return ResNeXt_cifar(block=BasicBlock, **config)

def resnext_se(**config):
    config['residual_block'] = SEBlock
    return resnext(**config)
    