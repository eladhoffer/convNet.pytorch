import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def swish(x):
    # type: (Tensor) -> Tensor
    return x * x.sigmoid()


@torch.jit.script
def hard_sigmoid(x):
    # type: (Tensor) -> Tensor
    return F.relu6(x+3).div_(6)


@torch.jit.script
def hard_swish(x):
    # type: (Tensor) -> Tensor
    return x * hard_sigmoid(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)


class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return hard_sigmoid(x)


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return hard_swish(x)
