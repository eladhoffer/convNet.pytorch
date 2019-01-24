import torch
import torch.nn as nn
from torch.nn import BatchNorm1d as _BatchNorm1d
from torch.nn import BatchNorm2d as _BatchNorm2d
from torch.nn import BatchNorm3d as _BatchNorm3d

"""
BatchNorm variants that can be disabled by removing all parameters and running stats
"""


def has_running_stats(m):
    return getattr(m, 'running_mean', None) is not None\
        or getattr(m, 'running_var', None) is not None


def has_parameters(m):
    return getattr(m, 'weight', None) is not None\
        or getattr(m, 'bias', None) is not None


class BatchNorm1d(_BatchNorm1d):
    def forward(self, inputs):
        if not (has_parameters(self) or has_running_stats(self)):
            return inputs
        return super(BatchNorm1d, self).forward(inputs)


class BatchNorm2d(_BatchNorm2d):
    def forward(self, inputs):
        if not (has_parameters(self) or has_running_stats(self)):
            return inputs
        return super(BatchNorm2d, self).forward(inputs)


class BatchNorm3d(_BatchNorm3d):
    def forward(self, inputs):
        if not (has_parameters(self) or has_running_stats(self)):
            return inputs
        return super(BatchNorm3d, self).forward(inputs)


class MeanBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm with mean-only normalization"""

    def __init__(self, num_features, momentum=0.1, bias=True):
        nn.Module.__init__(self)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.num_features = num_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if not (has_parameters(self) or has_running_stats(self)):
            return x
        if self.training:
            numel = x.size(0) * x.size(2) * x.size(3)
            mean = x.sum((0, 2, 3)) / numel
            with torch.no_grad():
                self.running_mean.mul_(self.momentum)\
                    .add_(1 - self.momentum, mean)
        else:
            mean = self.running_mean
        if self.bias is not None:
            mean = mean - self.bias
        return x - mean.view(1, -1, 1, 1)

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, bias={has_bias}'.format(
            has_bias=self.bias is not None, **self.__dict__)
