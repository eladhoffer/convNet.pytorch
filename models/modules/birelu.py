import torch
from torch.autograd.function import InplaceFunction
import torch.nn as nn


class BiReLUFunction(InplaceFunction):

    @staticmethod
    def forward(ctx, input, inplace=False):
        if input.size(1) % 2 != 0:
            raise RuntimeError("dimension 1 of input must be multiple of 2, "
                               "but got {}".format(input.size(1)))
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        pos, neg = output.chunk(2, dim=1)
        pos.clamp_(min=0)
        neg.clamp_(max=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_variables
        grad_input = grad_output.masked_fill(output.eq(0), 0)
        return grad_input, None


def birelu(x, inplace=False):
    return BiReLUFunction().apply(x, inplace)


class BiReLU(nn.Module):
    """docstring for BiReLU."""

    def __init__(self, inplace=False):
        super(BiReLU, self).__init__()
        self.inplace = inplace

    def forward(self, inputs):
        return birelu(inputs, inplace=self.inplace)

