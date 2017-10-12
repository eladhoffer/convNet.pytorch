import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math

__all__ = ['inception_v2']

def conv_bn(in_planes, out_planes, kernel_size, stride=1, padding=0):
    "convolution with batchnorm, relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )


class InceptionModule(nn.Module):

    def __init__(self, in_channels, n1x1_channels, n3x3r_channels,
                 n3x3_channels, dn3x3r_channels, dn3x3_channels,
                 pool_proj_channels=None, type_pool='avg', stride=1):
        super(InceptionModule, self).__init__()
        self.in_channels = in_channels
        self.n1x1_channels = n1x1_channels or 0
        pool_proj_channels = pool_proj_channels or 0
        self.stride = stride

        if n1x1_channels > 0:
            self.conv_1x1 = conv_bn(in_channels, n1x1_channels, 1, stride)
        else:
            self.conv_1x1 = None

        self.conv_3x3 = nn.Sequential(
            conv_bn(in_channels, n3x3r_channels, 1),
            conv_bn(n3x3r_channels, n3x3_channels, 3, stride, padding=1)
        )
        self.conv_d3x3 = nn.Sequential(
            conv_bn(in_channels, dn3x3r_channels, 1),
            conv_bn(dn3x3r_channels, dn3x3_channels, 3, padding=1),
            conv_bn(dn3x3_channels, dn3x3_channels, 3, stride, padding=1)
        )

        if type_pool == 'avg':
            self.pool = nn.AvgPool2d(3, stride, padding=1)
        elif type_pool == 'max':
            self.pool = nn.MaxPool2d(3, stride, padding=1)

        if pool_proj_channels > 0:  # Add pool projection
            self.pool = nn.Sequential(
                self.pool,
                conv_bn(in_channels, pool_proj_channels, 1))

    def forward(self, inputs):
        layer_outputs = []

        if self.conv_1x1 is not None:
            layer_outputs.append(self.conv_1x1(inputs))

        layer_outputs.append(self.conv_3x3(inputs))
        layer_outputs.append(self.conv_d3x3(inputs))
        layer_outputs.append(self.pool(inputs))
        output = torch.cat(layer_outputs, 1)

        return output


class Inception_v2(nn.Module):

    def __init__(self, num_classes=1000, aux_classifiers=True):
        super(inception_v2, self).__init__()
        self.num_classes = num_classes
        self.part1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, 1, 1, bias=False),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            InceptionModule(192, 64, 64, 64, 64, 96, 32, 'avg'),
            InceptionModule(256, 64, 64, 96, 64, 96, 64, 'avg'),
            InceptionModule(320, 0, 128, 160, 64, 96, 0, 'max', 2)
        )

        self.part2 = nn.Sequential(
            InceptionModule(576, 224, 64, 96, 96, 128, 128, 'avg'),
            InceptionModule(576, 192, 96, 128, 96, 128, 128, 'avg'),
            InceptionModule(576, 160, 128, 160, 128, 160, 96, 'avg')
        )
        self.part3 = nn.Sequential(
            InceptionModule(576, 96, 128, 192, 160, 192, 96, 'avg'),
            InceptionModule(576, 0, 128, 192, 192, 256, 0, 'max', 2),
            InceptionModule(1024, 352, 192, 320, 160, 224, 128, 'avg'),
            InceptionModule(1024, 352, 192, 320, 192, 224, 128, 'max')
        )

        self.main_classifier = nn.Sequential(
            nn.AvgPool2d(7, 1),
            nn.Dropout(0.2),
            nn.Conv2d(1024, self.num_classes, 1)
        )
        if aux_classifiers:
            self.aux_classifier1 = nn.Sequential(
                nn.AvgPool2d(5, 3),
                conv_bn(576, 128, 1),
                conv_bn(128, 768, 4),
                nn.Dropout(0.2),
                nn.Conv2d(768, self.num_classes, 1),
            )
            self.aux_classifier2 = nn.Sequential(
                nn.AvgPool2d(5, 3),
                conv_bn(576, 128, 1),
                conv_bn(128, 768, 4),
                nn.Dropout(0.2),
                nn.Conv2d(768, self.num_classes, 1),
            )

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 90, 'lr': 1e-4}
        ]

        class aux_loss(nn.Module):
            def __init__(self):
                super(aux_loss,self).__init__()
                self.loss = nn.CrossEntropyLoss()

            def forward(self, outputs, target):
                return self.loss(outputs[0], target) +\
                    0.4 * (self.loss(outputs[1], target) + self.loss(outputs[2], target))
        self.criterion = aux_loss

    def forward(self, inputs):
        branch1 = self.part1(inputs)
        branch2 = self.part2(branch1)
        branch3 = self.part3(branch1)

        output = self.main_classifier(branch3).view(-1, self.num_classes)
        if hasattr(self, 'aux_classifier1'):
            branch1 = self.aux_classifier1(branch1).view(-1, self.num_classes)
            branch2 = self.aux_classifier2(branch2).view(-1, self.num_classes)
            output = [output, branch1, branch2]
        return output


def inception_v2(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return Inception_v2(num_classes=num_classes)
