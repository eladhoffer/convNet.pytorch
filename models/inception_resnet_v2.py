import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ['inception_resnet_v2']

""" inception_resnet_v2.
References:
    Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi.

Links:
    http://arxiv.org/abs/1602.07261

"""


def conv_bn(in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False):
    "convolution with batchnorm, relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_planes, eps=1e-3),
        nn.ReLU()
    )


class Concat(nn.Sequential):

    def __init__(self, *kargs, **kwargs):
        super(Concat, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        return torch.cat([m(inputs) for m in self._modules.values()], 1)


class block(nn.Module):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block, self).__init__()
        self.scale = scale
        self.activation = activation or (lambda x: x)

    def forward(self, inputs):
        branch0 = self.Branch_0(inputs)
        branch1 = self.Branch_1(inputs)
        if hasattr(self, 'Branch_2'):
            branch2 = self.Branch_2(inputs)
            tower_mixed = torch.cat([branch0, branch1, branch2], 1)
        else:
            tower_mixed = torch.cat([branch0, branch1], 1)
        tower_out = self.Conv2d_1x1(tower_mixed)
        output = self.activation(self.scale * tower_out + inputs)
        return output


class block35(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block35, self).__init__(in_planes, scale, activation)
        self.Branch_0 = nn.Sequential(OrderedDict([
            ('Conv2d_1x1', conv_bn(in_planes, 32, 1))
        ]))
        self.Branch_1 = nn.Sequential(OrderedDict([
            ('Conv2d_0a_1x1', conv_bn(in_planes, 32, 1)),
            ('Conv2d_0b_3x3', conv_bn(32, 32, 3, padding=1))
        ]))
        self.Branch_2 = nn.Sequential(OrderedDict([
            ('Conv2d_0a_1x1', conv_bn(in_planes, 32, 1)),
            ('Conv2d_0b_3x3', conv_bn(32, 48, 3, padding=1)),
            ('Conv2d_0c_3x3', conv_bn(48, 64, 3, padding=1))
        ]))
        self.Conv2d_1x1 = conv_bn(128, in_planes, 1)


class block17(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block17, self).__init__(in_planes, scale, activation)

        self.Branch_0 = nn.Sequential(OrderedDict([
            ('Conv2d_1x1', conv_bn(in_planes, 192, 1))
        ]))
        self.Branch_1 = nn.Sequential(OrderedDict([
            ('Conv2d_0a_1x1', conv_bn(in_planes, 128, 1)),
            ('Conv2d_0b_1x7', conv_bn(128, 160, (1, 7), padding=(0, 3))),
            ('Conv2d_0c_7x1', conv_bn(160, 192, (7, 1), padding=(3, 0)))
        ]))
        self.Conv2d_1x1 = conv_bn(384, in_planes, 1)


class block8(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block8, self).__init__(in_planes, scale, activation)

        self.Branch_0 = nn.Sequential(OrderedDict([
            ('Conv2d_1x1', conv_bn(in_planes, 192, 1))
        ]))
        self.Branch_1 = nn.Sequential(OrderedDict([
            ('Conv2d_0a_1x1', conv_bn(in_planes, 192, 1)),
            ('Conv2d_0b_1x7', conv_bn(192, 224, (1, 3), padding=(0, 1))),
            ('Conv2d_0c_7x1', conv_bn(224, 256, (3, 1), padding=(1, 0)))
        ]))
        self.Conv2d_1x1 = conv_bn(448, in_planes, 1)


class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(InceptionResnetV2, self).__init__()
        self.end_points = {}
        self.num_classes = num_classes

        self.stem = nn.Sequential(OrderedDict([
            ('Conv2d_1a_3x3', conv_bn(3, 32, 3, stride=2, padding=1)),
            ('Conv2d_2a_3x3', conv_bn(32, 32, 3, padding=1)),
            ('Conv2d_2b_3x3', conv_bn(32, 64, 3)),
            ('MaxPool_3a_3x3', nn.MaxPool2d(3, 2)),
            ('Conv2d_3b_1x1', conv_bn(64, 80, 1)),
            ('Conv2d_4a_3x3', conv_bn(80, 192, 3)),
            ('MaxPool_5a_3x3', nn.MaxPool2d(3, 2))
        ]))

        tower_conv = nn.Sequential(OrderedDict([
            ('Conv2d_5b_b0_1x1', conv_bn(192, 96, 1))
        ]))
        tower_conv1 = nn.Sequential(OrderedDict([
            ('Conv2d_5b_b1_0a_1x1', conv_bn(192, 48, 1)),
            ('Conv2d_5b_b1_0b_5x5', conv_bn(48, 64, 5, padding=2))
        ]))
        tower_conv2 = nn.Sequential(OrderedDict([
            ('Conv2d_5b_b2_0a_1x1', conv_bn(192, 64, 1)),
            ('Conv2d_5b_b2_0b_3x3', conv_bn(64, 96, 3, padding=1)),
            ('Conv2d_5b_b2_0c_3x3', conv_bn(96, 96, 3, padding=1))
        ]))
        tower_pool3 = nn.Sequential(OrderedDict([
            ('AvgPool_5b_b3_0a_3x3', nn.AvgPool2d(3, stride=1, padding=1)),
            ('Conv2d_5b_b3_0b_1x1', conv_bn(192, 64, 1))
        ]))

        self.mixed_5b = Concat(OrderedDict([
            ('Branch_0', tower_conv),
            ('Branch_1', tower_conv1),
            ('Branch_2', tower_conv2),
            ('Branch_3', tower_pool3)
        ]))

        self.blocks35 = nn.Sequential()
        for i in range(10):
            self.blocks35.add_module('Block35.%s' %
                                     i, block35(320, scale=0.17))

        tower_conv = nn.Sequential(OrderedDict([
            ('Conv2d_6a_b0_0a_3x3', conv_bn(320, 384, 3, stride=2))
        ]))
        tower_conv1 = nn.Sequential(OrderedDict([
            ('Conv2d_6a_b1_0a_1x1', conv_bn(320, 256, 1)),
            ('Conv2d_6a_b1_0b_3x3', conv_bn(256, 256, 3, padding=1)),
            ('Conv2d_6a_b1_0c_3x3', conv_bn(256, 384, 3, stride=2))
        ]))
        tower_pool = nn.Sequential(OrderedDict([
            ('MaxPool_1a_3x3', nn.MaxPool2d(3, stride=2))
        ]))

        self.mixed_6a = Concat(OrderedDict([
            ('Branch_0', tower_conv),
            ('Branch_1', tower_conv1),
            ('Branch_2', tower_pool)
        ]))

        self.blocks17 = nn.Sequential()
        for i in range(20):
            self.blocks17.add_module('Block17.%s' %
                                     i, block17(1088, scale=0.1))

        tower_conv = nn.Sequential(OrderedDict([
            ('Conv2d_0a_1x1', conv_bn(1088, 256, 1)),
            ('Conv2d_1a_3x3', conv_bn(256, 384, 3, stride=2)),
        ]))
        tower_conv1 = nn.Sequential(OrderedDict([
            ('Conv2d_0a_1x1', conv_bn(1088, 256, 1)),
            ('Conv2d_1a_3x3', conv_bn(256, 64, 3, stride=2))
        ]))
        tower_conv2 = nn.Sequential(OrderedDict([
            ('Conv2d_0a_1x1', conv_bn(1088, 256, 1)),
            ('Conv2d_0b_3x3', conv_bn(256, 288, 3, padding=1)),
            ('Conv2d_1a_3x3', conv_bn(288, 320, 3, stride=2))
        ]))
        tower_pool3 = nn.Sequential(OrderedDict([
            ('MaxPool_1a_3x3', nn.MaxPool2d(3, stride=2))
        ]))

        self.mixed_7a = Concat(OrderedDict([
            ('Branch_0', tower_conv),
            ('Branch_1', tower_conv1),
            ('Branch_2', tower_conv2),
            ('Branch_3', tower_pool3)
        ]))

        self.blocks8 = nn.Sequential()
        for i in range(9):
            self.blocks8.add_module('Block8.%s' %
                                    i, block8(1856, scale=0.2))
        self.blocks8.add_module('Block8.9', block8(
            1856, scale=0.2, activation=None))

        self.conv_pool = nn.Sequential(OrderedDict([
            ('Conv2d_7b_1x1', conv_bn(1856, 1536, 1)),
            ('AvgPool_1a_8x8', nn.AvgPool2d(8, 1)),
            ('Dropout', nn.Dropout(0.2))
        ]))
        self.classifier = nn.Linear(1536, num_classes)

        self.aux_classifier = nn.Sequential(OrderedDict([
            ('Conv2d_1a_3x3', nn.AvgPool2d(5, 3)),
            ('Conv2d_1b_1x1', conv_bn(1088, 128, 1)),
            ('Conv2d_2a_5x5', conv_bn(128, 768, 5)),
            ('Dropout', nn.Dropout(0.2)),
            ('Logits', conv_bn(768, num_classes, 1))
        ]))

        class aux_loss(nn.Module):
            def __init__(self):
                super(aux_loss,self).__init__()
                self.loss = nn.CrossEntropyLoss()

            def forward(self, outputs, target):
                return self.loss(outputs[0], target) +\
                    0.4 * (self.loss(outputs[1], target))
        self.criterion = aux_loss
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 90, 'lr': 1e-4}
        ]

    def forward(self, x):
        x = self.stem(x)  # (B, 192, 35, 35)
        x = self.mixed_5b(x)  # (B, 320, 35, 35)
        x = self.blocks35(x)  # (B, 320, 35, 35)
        x = self.mixed_6a(x)  # (B, 1088, 17, 17)
        branch1 = self.blocks17(x)  # (B, 1088, 17, 17)
        x = self.mixed_7a(branch1)  # (B, 1856, 8, 8)
        x = self.blocks8(x)   # (B, 1856, 8, 8)
        x = self.conv_pool(x) # (B, 1536, 1, 1)
        x = x.view(-1, 1536)  # (B, 1536)
        output = self.classifier(x) # (B, num_classes)
        if hasattr(self, 'aux_classifier'):
            branch1 = self.aux_classifier(branch1).view(-1, self.num_classes)
            output = [output, branch1]
        return output

def inception_resnet_v2(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return InceptionResnetV2(num_classes=num_classes)
