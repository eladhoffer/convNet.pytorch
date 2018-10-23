from collections import OrderedDict
import torch
import torch.nn as nn

__all__ = ['googlenet']

class Inception_v1_GoogLeNet(nn.Module):
    input_side = 227
    rescale = 255.0
    rgb_mean = [122.7717, 115.9465, 102.9801]
    rgb_std = [1, 1, 1]

    def __init__(self, num_classes=1000):
        super(Inception_v1_GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Sequential(OrderedDict([
                ('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), bias=False)),
                ('7x7_s2_bn', nn.BatchNorm2d(64, affine=True)),
                ('relu1', nn.ReLU(True)),
                ('pool1', nn.MaxPool2d((3, 3), (2, 2), padding=(1,1)))
            ]))),

            ('conv2', nn.Sequential(OrderedDict([
                ('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0), bias=False)),
                ('3x3_reduce_bn', nn.BatchNorm2d(64, affine=True)),
                ('relu1', nn.ReLU(True)),
                ('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1), bias=False)),
                ('3x3_bn', nn.BatchNorm2d(192, affine=True)),
                ('relu2', nn.ReLU(True)),
                ('pool2', nn.MaxPool2d((3, 3), (2, 2), padding=(1,1)))
            ]))),

            ('inception_3a', InceptionModule(192, 64, 96, 128, 16, 32, 32)),
            ('inception_3b', InceptionModule(256, 128, 128, 192, 32, 96, 64)),

            ('pool3', nn.MaxPool2d((3, 3), (2, 2), padding=(1,1))),

            ('inception_4a', InceptionModule(480, 192, 96, 208, 16, 48, 64)),
            ('inception_4b', InceptionModule(512, 160, 112, 224, 24, 64, 64)),
            ('inception_4c', InceptionModule(512, 128, 128, 256, 24, 64, 64)),
            ('inception_4d', InceptionModule(512, 112, 144, 288, 32, 64, 64)),
            ('inception_4e', InceptionModule(528, 256, 160, 320, 32, 128, 128)),

            ('pool4', nn.MaxPool2d((3, 3), (2, 2), padding=(1,1))),

            ('inception_5a', InceptionModule(832, 256, 160, 320, 32, 128, 128)),
            ('inception_5b', InceptionModule(832, 384, 192, 384, 48, 128, 128)),

            ('pool5', nn.AvgPool2d((7, 7), (1, 1))),

            ('drop5', nn.Dropout(0.2))
        ]))

        self.classifier = nn.Linear(1024, self.num_classes)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 90, 'lr': 1e-3, 'optimizer': 'Adam'}
        ]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5,
                 outplane_pool_proj):
        super(InceptionModule, self).__init__()
        a = nn.Sequential(OrderedDict([
            ('1x1', nn.Conv2d(inplane, outplane_a1x1, (1, 1), (1, 1), (0, 0), bias=False)),
            ('1x1_bn', nn.BatchNorm2d(outplane_a1x1, affine=True)),
            ('1x1_relu', nn.ReLU(True))
        ]))

        b = nn.Sequential(OrderedDict([
            ('3x3_reduce', nn.Conv2d(inplane, outplane_b3x3_reduce, (1, 1), (1, 1), (0, 0), bias=False)),
            ('3x3_reduce_bn', nn.BatchNorm2d(outplane_b3x3_reduce, affine=True)),
            ('3x3_relu1', nn.ReLU(True)),
            ('3x3', nn.Conv2d(outplane_b3x3_reduce, outplane_b3x3, (3, 3), (1, 1), (1, 1), bias=False)),
            ('3x3_bn', nn.BatchNorm2d(outplane_b3x3, affine=True)),
            ('3x3_relu2', nn.ReLU(True))
        ]))

        c = nn.Sequential(OrderedDict([
            ('5x5_reduce', nn.Conv2d(inplane, outplane_c5x5_reduce, (1, 1), (1, 1), (0, 0), bias=False)),
            ('5x5_reduce_bn', nn.BatchNorm2d(outplane_c5x5_reduce, affine=True)),
            ('5x5_relu1', nn.ReLU(True)),
            ('5x5', nn.Conv2d(outplane_c5x5_reduce, outplane_c5x5, (5, 5), (1, 1), (2, 2), bias=False)),
            ('5x5_bn', nn.BatchNorm2d(outplane_c5x5, affine=True)),
            ('5x5_relu2', nn.ReLU(True))
        ]))

        d = nn.Sequential(OrderedDict([
            ('pool_pool', nn.MaxPool2d((3, 3), (1, 1), (1, 1))),
            ('pool_proj', nn.Conv2d(inplane, outplane_pool_proj, (1, 1), (1, 1), (0, 0))),
            ('pool_proj_bn', nn.BatchNorm2d(outplane_pool_proj, affine=True)),
            ('pool_relu', nn.ReLU(True))
        ]))

        for container in [a, b, c, d]:
            for name, module in container.named_children():
                self.add_module(name, module)

        self.branches = [a, b, c, d]

    def forward(self, input):
        return torch.cat([branch(input) for branch in self.branches], 1)


def googlenet(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return Inception_v1_GoogLeNet(num_classes)