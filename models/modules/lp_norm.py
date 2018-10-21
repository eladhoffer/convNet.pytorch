import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torch.nn as nn
import numpy as np


def _norm(x, dim, p=2):
    """Computes the norm over all dimensions except dim"""
    if p == -1:
        func = lambda x, dim: x.max(dim=dim)[0] - x.min(dim=dim)[0]
    elif p == float('inf'):
        func = lambda x, dim: x.max(dim=dim)[0]
    else:
        func = lambda x, dim: torch.norm(x, dim=dim, p=p)
    if dim is None:
        return x.norm(p=p)
    elif dim == 0:
        output_size = (x.size(0),) + (1,) * (x.dim() - 1)
        return func(x.contiguous().view(x.size(0), -1), 1).view(*output_size)
    elif dim == x.dim() - 1:
        output_size = (1,) * (x.dim() - 1) + (x.size(-1),)
        return func(x.contiguous().view(-1, x.size(-1)), 0).view(*output_size)
    else:
        return _norm(x.transpose(0, dim), 0).transpose(0, dim)


def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


def _std(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.std()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).std(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).std(dim=0).view(*output_size)
    else:
        return _std(p.transpose(0, dim), 0).transpose(0, dim)

# L2


class LpBatchNorm2d(nn.Module):
    # This is L2 Baseline

    def __init__(self, num_features, dim=1, p=2, momentum=0.1, bias=True,  eps=1e-5, noise=False):
        super(LpBatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.p = p
        self.eps = eps
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        p = self.p
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            Var = (torch.abs((t.transpose(1, 0) - mean))**p).mean(0)

            scale = (Var + self.eps)**(-1 / p)

            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)

        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)

        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


class TopkBatchNorm2d(nn.Module):
    # this is normalized L_inf

    def __init__(self, num_features, k=10, dim=1, momentum=0.1, bias=True, eps=1e-5, noise=False):
        super(TopkBatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.k = k
        self.eps = eps
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = torch.abs(t.transpose(1, 0) - mean)

            const = 0.5 * (1 + (np.pi * np.log(4)) ** 0.5) / \
                ((2 * np.log(A.size(0))) ** 0.5)

            MeanTOPK = (torch.topk(A, self.k, dim=0)[0].mean(0)) * const
            scale = 1 / (MeanTOPK + self.eps)

            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)

        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)

        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out

# Top10


class GhostTopkBatchNorm2d(nn.Module):
    # This is normalized Top10 batch norm

    def __init__(self, num_features, k=10, dim=1, momentum=0.1, bias=True, eps=1e-5, beta=0.75, noise=False):
        super(GhostTopkBatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        self.register_buffer('meanTOPK', torch.zeros(num_features))
        self.noise = noise
        self.k = k
        self.beta = 0.75
        self.eps = eps
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        # p=5
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = torch.abs(t.transpose(1, 0) - mean)
            beta = 0.75

            MeanTOPK = torch.topk(A, self.k, dim=0)[0].mean(0)
            meanTOPK = beta * \
                torch.autograd.variable.Variable(
                    self.biasTOPK) + (1 - beta) * MeanTOPK

            const = 0.5 * (1 + (np.pi * np.log(4)) ** 0.5) / \
                ((2 * np.log(A.size(0))) ** 0.5)
            meanTOPK = meanTOPK * const

            # print(self.biasTOPK)
            self.biasTOPK.copy_(meanTOPK.data)
            # self.biasTOPK = MeanTOPK.data
            scale = 1 / (meanTOPK + self.eps)

            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)
        # out = (x - mean.view(1, mean.size(0), 1, 1)) * final_scale.view(1, scale.size(0), 1, 1)

        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)

        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


# L1
class L1BatchNorm2d(nn.Module):
    # This is normalized L1 Batch norm; note the normalization term (np.pi / 2) ** 0.5) when multiplying by Var:
    # scale = ((Var * (np.pi / 2) ** 0.5) + self.eps) ** (-1)

    """docstring for L1BatchNorm2d."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, normalized=True, eps=1e-5, noise=False):
        super(L1BatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))
        self.eps = eps
        if normalized:
            self.weight_fix = (np.pi / 2) ** 0.5
        else:
            self.weight_fix = 1

    def forward(self, x):
        p = 1
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            Var = (torch.abs((t.transpose(1, 0) - mean))).mean(0)
            scale = (Var * self.weight_fix + self.eps) ** (-1)
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)

        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)

        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out
