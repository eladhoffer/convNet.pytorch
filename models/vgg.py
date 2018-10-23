from torchvision.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
import torch.nn as nn

__all__ = ['vgg']

model_name = {11: 'VGG11', 13: 'VGG13', 16: 'VGG16', 19: 'VGG19'}

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, batch_norm=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], batch_norm)
        self.classifier = self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 60, 'lr': 1e-1 * (0.2**1)},
            {'epoch': 120, 'lr': 1e-1 * (0.2**2)},
            {'epoch': 160, 'lr': 1e-1 * (0.2**3)}
        ]

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg(**config):
    dataset = config.pop('dataset', 'imagenet')
    depth = config.pop('depth', 16)
    bn = config.pop('bn', True)

    if dataset == 'imagenet':
        config.setdefault('num_classes', 1000)
        if depth == 11:
            if bn is False:
                return vgg11(pretrained=False, **config)
            else:
                return vgg11_bn(pretrained=False, **config)
        if depth == 13:
            if bn is False:
                return vgg13(pretrained=False, **config)
            else:
                return vgg13_bn(pretrained=False, **config)
        if depth == 16:
            if bn is False:
                return vgg16(pretrained=False, **config)
            else:
                return vgg16_bn(pretrained=False, **config)
        if depth == 19:
            if bn is False:
                return vgg19(pretrained=False, **config)
            else:
                return vgg19_bn(pretrained=False, **config)
    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
    config.setdefault('batch_norm', bn)
    return VGG(model_name[depth], **config)
