"""
adapted from https://github.com/quark0/darts
"""
from collections import namedtuple
import torch
import torch.nn as nn

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

OPS = {
    'avg_pool_3x3': lambda channels, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda channels, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda channels, stride, affine: Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=affine),
    'sep_conv_3x3': lambda channels, stride, affine: SepConv(channels, channels, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda channels, stride, affine: SepConv(channels, channels, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda channels, stride, affine: SepConv(channels, channels, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda channels, stride, affine: DilConv(channels, channels, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda channels, stride, affine: DilConv(channels, channels, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda channels, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(channels, channels, (1, 7), stride=(1, stride),
                  padding=(0, 3), bias=False),
        nn.Conv2d(channels, channels, (7, 1), stride=(stride, 1),
                  padding=(3, 0), bias=False),
        nn.BatchNorm2d(channels, affine=affine)
    ),
}


# genotypes
GENOTYPES = dict(
    NASNet=Genotype(
        normal=[
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 0),
            ('avg_pool_3x3', 0),
            ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 1),
        ],
        normal_concat=[2, 3, 4, 5, 6],
        reduce=[
            ('sep_conv_5x5', 1),
            ('sep_conv_7x7', 0),
            ('max_pool_3x3', 1),
            ('sep_conv_7x7', 0),
            ('avg_pool_3x3', 1),
            ('sep_conv_5x5', 0),
            ('skip_connect', 3),
            ('avg_pool_3x3', 2),
            ('sep_conv_3x3', 2),
            ('max_pool_3x3', 1),
        ],
        reduce_concat=[4, 5, 6],
    ),

    AmoebaNet=Genotype(
        normal=[
            ('avg_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 3),
            ('sep_conv_3x3', 1),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('avg_pool_3x3', 1),
        ],
        normal_concat=[4, 5, 6],
        reduce=[
            ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 1),
            ('max_pool_3x3', 0),
            ('sep_conv_7x7', 2),
            ('sep_conv_7x7', 0),
            ('avg_pool_3x3', 1),
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('conv_7x1_1x7', 0),
            ('sep_conv_3x3', 5),
        ],
        reduce_concat=[3, 4, 6]
    ),

    DARTS_V1=Genotype(
        normal=[
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 2)],
        normal_concat=[2, 3, 4, 5],
        reduce=[('max_pool_3x3', 0),
                ('max_pool_3x3', 1),
                ('skip_connect', 2),
                ('max_pool_3x3', 0),
                ('max_pool_3x3', 0),
                ('skip_connect', 2),
                ('skip_connect', 2),
                ('avg_pool_3x3', 0)],
        reduce_concat=[2, 3, 4, 5]),
    DARTS=Genotype(normal=[('sep_conv_3x3', 0),
                           ('sep_conv_3x3', 1),
                           ('sep_conv_3x3', 0),
                           ('sep_conv_3x3', 1),
                           ('sep_conv_3x3', 1),
                           ('skip_connect', 0),
                           ('skip_connect', 0),
                           ('dil_conv_3x3', 2)],
                   normal_concat=[2, 3, 4, 5],
                   reduce=[('max_pool_3x3', 0),
                           ('max_pool_3x3', 1),
                           ('skip_connect', 2),
                           ('max_pool_3x3', 1),
                           ('max_pool_3x3', 0),
                           ('skip_connect', 2),
                           ('skip_connect', 2),
                           ('max_pool_3x3', 1)],
                   reduce_concat=[2, 3, 4, 5]),
)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = x.new(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NasNetCell(Cell):
    def __init__(self, *kargs, **kwargs):
        super(NasNetCell, self).__init__(GENOTYPES['NASNet'], *kargs, **kwargs)


class AmoebaNetCell(Cell):
    def __init__(self, *kargs, **kwargs):
        super(AmoebaNetCell, self).__init__(
            GENOTYPES['AmoebaNet'], *kargs, **kwargs)


class DARTSCell(Cell):
    def __init__(self, *kargs, **kwargs):
        super(DARTSCell, self).__init__(GENOTYPES['DARTS'], *kargs, **kwargs)
