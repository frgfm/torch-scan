#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Module FLOPs
"""

import warnings
from operator import mul
from functools import reduce

from torch import nn
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin  # renamed to _ConvTransposeNd in next release
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd


__all__ = ['module_flops']


def module_flops(module, input, output):
    """Estimate the number of floating point operations performed by the module

    Args:
        module (torch.nn.Module): PyTorch module
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module
    Returns:
        int: number of FLOPs
    """

    if isinstance(module, nn.Identity):
        return 0
    elif isinstance(module, nn.Linear):
        return flops_linear(module, input, output)
    elif isinstance(module, nn.ReLU):
        return flops_relu(module, input, output)
    elif isinstance(module, nn.ELU):
        return flops_elu(module, input, output)
    elif isinstance(module, nn.LeakyReLU):
        return flops_leakyrelu(module, input, output)
    elif isinstance(module, nn.ReLU6):
        return flops_relu6(module, input, output)
    elif isinstance(module, nn.Tanh):
        return flops_tanh(module, input, output)
    elif isinstance(module, nn.Sigmoid):
        return flops_sigmoid(module, input, output)
    elif isinstance(module, _ConvTransposeMixin):
        return flops_convtransposend(module, input, output)
    elif isinstance(module, _ConvNd):
        return flops_convnd(module, input, output)
    elif isinstance(module, _BatchNorm):
        return flops_bn(module, input, output)
    elif isinstance(module, _MaxPoolNd):
        return flops_maxpool(module, input, output)
    elif isinstance(module, _AvgPoolNd):
        return flops_avgpool(module, input, output)
    elif isinstance(module, _AdaptiveMaxPoolNd):
        return flops_adaptive_maxpool(module, input, output)
    elif isinstance(module, _AdaptiveAvgPoolNd):
        return flops_adaptive_avgpool(module, input, output)
    elif isinstance(module, nn.Dropout):
        return flops_dropout(module, input, output)
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return 0


def flops_linear(module, input, output):
    """FLOPs estimation for `torch.nn.Linear`"""

    # batch size * out_chan * in_chan
    mm_flops = input.shape[0] * output.shape[1] * (2 * input.shape[1] - 1)
    bias_flops = output.numel() if module.bias is not None else 0

    return mm_flops + bias_flops


def flops_sigmoid(module, input, output):
    """FLOPs estimation for `torch.nn.Sigmoid`"""

    # For each element, mul by -1, exp it, add 1, div
    return input.numel() * 4


def flops_relu(module, input, output):
    """FLOPs estimation for `torch.nn.ReLU`"""

    # Each element is compared to 0
    return input.numel()


def flops_elu(module, input, output):
    """FLOPs estimation for `torch.nn.ELU`"""

    # For each element, compare it to 0, exp it, sub 1, mul by alpha, compare it to 0 and sum both
    return input.numel() * 6


def flops_leakyrelu(module, input, output):
    """FLOPs estimation for `torch.nn.LeakyReLU`"""

    # For each element, compare it to 0 (max), compare it to 0 (min), mul by slope and sum both
    return input.numel() * 4


def flops_relu6(module, input, output):
    """FLOPs estimation for `torch.nn.ReLU6`"""

    # For each element, compare it to 0 (max), compare it to 0 (min), mul by slope and sum both
    return input.numel() * 2


def flops_tanh(module, input, output):
    """FLOPs estimation for `torch.nn.Tanh`"""

    # For each element, exp it, mul by -1 and exp it, divide the sub by the add
    return input.numel() * 6


def flops_dropout(module, input, output):
    """FLOPs estimation for `torch.nn.Dropout`"""

    if module.p > 0:
        # Sample a random number for each input element
        return input.numel()
    else:
        return 0


def flops_convtransposend(module, input, output):
    """FLOPs estimation for `torch.nn.modules.conv._ConvTranposeNd`"""

    # Padding (# cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532)
    # Define min and max sizes
    padding_flops = len(module.kernel_size) * 8

    # Once padding is determined, the operations are almost identical to those of a convolution
    conv_flops = flops_convnd(module, input, output)

    return padding_flops + conv_flops


def flops_convnd(module, input, output):
    """FLOPs estimation for `torch.nn.modules.conv._ConvNd`"""

    # For each position, # mult = kernel size, # adds = kernel size - 1
    window_flops_per_chan = 2 * reduce(mul, module.kernel_size) - 1
    # Connections to input channels is controlled by the group parameter
    effective_in_chan = (input.shape[1] // module.groups)
    # N * flops + (N - 1) additions
    window_flops = effective_in_chan * window_flops_per_chan + (effective_in_chan - 1)
    conv_flops = output.numel() * window_flops

    # Each output element gets a bias addition
    bias_flops = output.numel() if module.bias is not None else 0

    return conv_flops + bias_flops


def flops_bn(module, input, output):
    """FLOPs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""

    # for each channel, add eps and running_var, sqrt it
    norm_ops = module.num_features * 2
    # For each element, sub running_mean, div by denom
    norm_ops += input.numel() * 2
    # For each element, mul by gamma, add beta
    scale_ops = input.numel() * 2 if module.affine else 0
    bn_flops = norm_ops + scale_ops

    # Count tracking stats update ops
    # cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L94-L101
    tracking_flops = 0
    if module.track_running_stats and module.training:
        # exponential_average_factor
        if module.momentum is None:
            tracking_flops += 1
        # running_mean: by channel, sum values and div by batch size
        tracking_flops += module.num_features * (input.shape[0] * input.shape[2:].numel())
        # running_var: by channel, sub mean and square values, sum them, divide by batch size
        tracking_flops += 3 * input.numel()
        # Update both runnning stat: rescale previous value (mul by N), add it the new one, then div by (N + 1)
        tracking_flops += 2 * module.num_features * 3

    return bn_flops + tracking_flops


def flops_maxpool(module, input, output):
    """FLOPs estimation for `torch.nn.modules.pooling._MaxPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, check max element in kernel scope
    return output.numel() * (k_size - 1)


def flops_avgpool(module, input, output):
    """FLOPs estimation for `torch.nn.modules.pooling._AvgPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return output.numel() * (k_size - 1 + input.ndim - 2)


def flops_adaptive_maxpool(module, input, output):
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveMaxPoolNd`"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(input.shape[2:], module.output_size))

    # for each spatial output element, check max element in kernel scope
    return output.numel() * (reduce(mul, kernel_size) - 1)


def flops_adaptive_avgpool(module, input, output):
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveAvgPoolNd`"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(input.shape[2:], module.output_size))

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return output.numel() * (reduce(mul, kernel_size) - 1 + len(kernel_size))
