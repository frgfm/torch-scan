#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Module DMAs
"""

import warnings
from operator import mul
from functools import reduce

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd
from torch.nn import Linear, ReLU, ELU, LeakyReLU, ReLU6, Tanh, Sigmoid, Dropout

__all__ = ['module_dmas']


def module_dmas(module, input, output):
    """Estimate the number of direct memory accesses by the module.
    The implementation overhead is neglected

    Args:
        module (torch.nn.Module): PyTorch module
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module
    Returns:
        int: number of DMAs
    """

    if isinstance(module, Linear):
        return dma_linear(module, input, output)
    elif isinstance(module, (ReLU, ReLU6)):
        return dma_relu(module, input, output)
    elif isinstance(module, (ELU, LeakyReLU)):
        return dma_act_single_param(module, input, output)
    elif isinstance(module, Sigmoid):
        return dma_sigmoid(module, input, output)
    elif isinstance(module, Tanh):
        return dma_tanh(module, input, output)
    elif isinstance(module, _ConvNd):
        return dma_convnd(module, input, output)
    elif isinstance(module, _BatchNorm):
        return dma_bn(module, input, output)
    elif isinstance(module, (_MaxPoolNd, _AvgPoolNd)):
        return dma_pool(module, input, output)
    elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return dma_adaptive_pool(module, input, output)
    elif isinstance(module, Dropout):
        return dma_dropout(module, input, output)
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return 0


def num_params(module):
    """Compute the number of parameters

    Args:
        module (torch.nn.Module): PyTorch module
    Returns:
        int: number of parameter elements
    """

    return sum(p.data.numel() for p in module.parameters())


def dma_linear(module, input, output):
    """DMAs estimation for `torch.nn.Linear`"""

    input_dma = input.numel()
    # Access weight and bias
    ops_dma = num_params(module)
    output_dma = output.numel()

    return input_dma + ops_dma + output_dma


def dma_relu(module, input, output):
    """DMAs estimation for `torch.nn.ReLU`"""

    input_dma = input.numel()
    output_dma = 0 if module.inplace else output.numel()

    return input_dma + output_dma


def dma_act_single_param(module, input, output):
    """DMAs estimation for activations with single parameter"""

    input_dma = input.numel()
    # Access alpha, slope or other
    ops_dma = 1
    output_dma = 0 if module.inplace else output.numel()

    return input_dma + ops_dma + output_dma


def dma_sigmoid(module, input, output):
    """DMAs estimation for `torch.nn.Sigmoid`"""

    # Access for both exp
    input_dma = input.numel()
    output_dma = output.numel()

    return input_dma + output_dma


def dma_tanh(module, input, output):
    """DMAs estimation for `torch.nn.Tanh`"""

    # Access for both exp
    input_dma = input.numel() * 2
    output_dma = output.numel()

    return input_dma + output_dma


def dma_dropout(module, input, output):
    """DMAs estimation for `torch.nn.Dropout`"""

    input_dma = input.numel()

    # Access sampling probability
    ops_dma = 1

    output_dma = 0 if module.inplace else output.numel()

    return input_dma + ops_dma + output_dma


def dma_convnd(module, input, output):
    """DMAs estimation for `torch.nn.modules.conv._ConvNd`"""

    # Each output element required K ** 2 memory access of each input channel
    input_dma = module.in_channels * reduce(mul, module.kernel_size) * output.numel()
    # Correct with groups
    input_dma /= module.groups

    # Access weight & bias
    ops_dma = num_params(module)
    output_dma = output.numel()

    return input_dma + ops_dma + output_dma


def dma_bn(module, input, output):
    """DMAs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""
    input_dma = input.numel()

    # Access running_mean, running_var and eps
    ops_dma = module.running_mean.numel() + module.running_var.numel() + 1
    # Access to weight and bias
    if module.affine:
        ops_dma += module.weight.data.numel() + module.bias.data.numel()
    # Exp avg factor
    if module.momentum:
        ops_dma += 1
    # Update stats
    if module.training and module.track_running_stats:
        # Current mean and std computation only requires access to input, already counted in input_dma
        # Update num of batches and running stats
        ops_dma += module.num_batches_tracked.numel() + module.running_mean.numel() + module.running_var.numel()

    output_dma = output.numel()

    return input_dma + ops_dma + output_dma


def dma_pool(module, input, output):
    """DMAs estimation for spatial pooling modules"""

    # Resolve kernel size and stride size (can be stored as a single integer or a tuple)
    if isinstance(module.kernel_size, tuple):
        kernel_size = module.kernel_size
    else:
        kernel_size = (module.kernel_size,) * (input.ndim - 2)

    # Each output element required K ** 2 memory accesses
    input_dma = reduce(mul, kernel_size) * output.numel()

    output_dma = output.numel()

    return input_dma + output_dma


def dma_adaptive_pool(module, input, output):
    """DMAs estimation for adaptive spatial pooling modules"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(input.shape[2:], module.output_size))
    # Each output element required K ** 2 memory accesses
    input_dma = reduce(mul, kernel_size) * output.numel()

    output_dma = output.numel()

    return input_dma + output_dma
