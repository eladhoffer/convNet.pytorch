from torch.nn import BatchNorm1d as _BatchNorm1d
from torch.nn import BatchNorm2d as _BatchNorm2d
from torch.nn import BatchNorm3d as _BatchNorm3d

"""
BatchNorm variants that can be disabled by removing all parameters and running stats
"""


def has_running_stats(m):
    return getattr(m, 'running_mean', None) is not None\
        and getattr(m, 'running_var', None) is not None


def has_parameters(m):
    return getattr(m, 'weight', None) is not None\
        and getattr(m, 'bias', None) is not None


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
