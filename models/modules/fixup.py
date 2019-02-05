import torch
import torch.nn as nn


def _sum_tensor_scalar(tensor, scalar, expand_size):
    if scalar is not None:
        scalar = scalar.expand(expand_size).contiguous()
    else:
        return tensor
    if tensor is None:
        return scalar
    return tensor + scalar 


class ZIConv2d(nn.Conv2d):
    def __init__(self,  in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 multiplier=False, pre_bias=True, post_bias=True):
        super(ZIConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        if pre_bias:
            self.pre_bias = nn.Parameter(torch.tensor([0.]))
        else:
            self.register_parameter('pre_bias', None)
        if post_bias:
            self.post_bias = nn.Parameter(torch.tensor([0.]))
        else:
            self.register_parameter('post_bias', None)
        if multiplier:
            self.multiplier = nn.Parameter(torch.tensor([1.]))
        else:
            self.register_parameter('multiplier', None)

    def forward(self, x):
        if self.pre_bias is not None:
            x = x + self.pre_bias
        weight = self.weight if self.multiplier is None\
            else self.weight * self.multiplier
        bias = _sum_tensor_scalar(self.bias, self.post_bias, self.out_channels)
        return nn.functional.conv2d(x, weight, bias, self.stride,
                                    self.padding, self.dilation, self.groups)


class ZILinear(nn.Linear):
    def __init__(self,  in_features, out_features, bias=False,
                 multiplier=False, pre_bias=True, post_bias=True):
        super(ZILinear, self).__init__(in_features, out_features, bias)
        if pre_bias:
            self.pre_bias = nn.Parameter(torch.tensor([0.]))
        else:
            self.register_parameter('pre_bias', None)
        if post_bias:
            self.post_bias = nn.Parameter(torch.tensor([0.]))
        else:
            self.register_parameter('post_bias', None)
        if multiplier:
            self.multiplier = nn.Parameter(torch.tensor([1.]))
        else:
            self.register_parameter('multiplier', None)

    def forward(self, x):
        if self.pre_bias is not None:
            x = x + self.pre_bias
        weight = self.weight if self.multiplier is None\
            else self.weight * self.multiplier
        bias = _sum_tensor_scalar(self.bias, self.post_bias, self.out_features)
        return nn.functional.linear(x, weight, bias)
