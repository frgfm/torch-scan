#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Module MACs
"""

import warnings
from operator import mul
from functools import reduce

from torch import nn
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin  # renamed to _ConvTransposeNd in next release
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd


__all__ = ['module_macs']


def module_macs(module, input, output):
    """Estimate the number of multiply-accumulation operations performed by the module

    Args:
        module (torch.nn.Module): PyTorch module
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module
    Returns:
        int: number of MACs
    """
    if isinstance(module, nn.Linear):
        return macs_linear(module, input, output)
    elif isinstance(module, (nn.Identity, nn.ReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid)):
        return 0
    elif isinstance(module, _ConvTransposeMixin):
        return macs_convtransposend(module, input, output)
    elif isinstance(module, _ConvNd):
        return macs_convnd(module, input, output)
    elif isinstance(module, _BatchNorm):
        return macs_bn(module, input, output)
    elif isinstance(module, _MaxPoolNd):
        return macs_maxpool(module, input, output)
    elif isinstance(module, _AvgPoolNd):
        return macs_avgpool(module, input, output)
    elif isinstance(module, _AdaptiveMaxPoolNd):
        return macs_adaptive_maxpool(module, input, output)
    elif isinstance(module, _AdaptiveAvgPoolNd):
        return macs_adaptive_avgpool(module, input, output)
    elif isinstance(module, nn.Dropout):
        return 0
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return 0


def macs_linear(module, input, output):
    """MACs estimation for `torch.nn.Linear`"""

    # batch size * out_chan * macs_per_elt (bias already counted in accumulation)
    mm_mac = input.shape[0] * output.shape[1] * input.shape[1]

    return mm_mac


def macs_convtransposend(module, input, output):
    """MACs estimation for `torch.nn.modules.conv._ConvTransposeNd`"""

    # Padding (# cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532)
    # Define min and max sizes, then subtract them
    padding_macs = len(module.kernel_size) * 4

    # Rest of the operations are almost identical to a convolution (given the padding)
    conv_macs = macs_convnd(module, input, output)

    return padding_macs + conv_macs


def macs_convnd(module, input, output):
    """MACs estimation for `torch.nn.modules.conv._ConvNd`"""

    # For each position, # mult = kernel size, # adds = kernel size - 1
    window_macs_per_chan = reduce(mul, module.kernel_size)
    # Connections to input channels is controlled by the group parameter
    effective_in_chan = (input.shape[1] // module.groups)
    # N * mac
    window_mac = effective_in_chan * window_macs_per_chan
    conv_mac = output.numel() * window_mac

    # bias already counted in accumulation
    return conv_mac


def macs_bn(module, input, output):
    """MACs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""

    # sub mean, div by denom
    norm_mac = 1
    # mul by gamma, add beta
    scale_mac = 1 if module.affine else 0

    # Sum everything up
    bn_mac = input.numel() * (norm_mac + scale_mac)

    # Count tracking stats update ops
    # cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L94-L101
    tracking_mac = 0
    if module.track_running_stats and module.training:
        # running_mean: by channel, sum value and div by batch size
        tracking_mac += module.num_features * (input.shape[0] * input.shape[2:].numel() - 1)
        # running_var: by channel, sub mean and square values, sum them, divide by batch size
        active_elts = input.shape[0] * input.shape[2:].numel()
        tracking_mac += module.num_features * (2 * active_elts - 1)
        # Update both runnning stat: rescale previous value (mul by N), add it the new one, then div by (N + 1)
        tracking_mac += 2 * module.num_features * 2

    return bn_mac + tracking_mac


def macs_maxpool(module, input, output):
    """MACs estimation for `torch.nn.modules.pooling._MaxPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, check max element in kernel scope
    return output.numel() * (k_size - 1)


def macs_avgpool(module, input, output):
    """MACs estimation for `torch.nn.modules.pooling._AvgPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return output.numel() * (k_size - 1 + input.ndim - 2)


def macs_adaptive_maxpool(module, input, output):
    """MACs estimation for `torch.nn.modules.pooling._AdaptiveMaxPoolNd`"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(input.shape[2:], module.output_size))

    # for each spatial output element, check max element in kernel scope
    return output.numel() * (reduce(mul, kernel_size) - 1)


def macs_adaptive_avgpool(module, input, output):
    """MACs estimation for `torch.nn.modules.pooling._AdaptiveAvgPoolNd`"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(input.shape[2:], module.output_size))

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return output.numel() * (reduce(mul, kernel_size) - 1 + len(kernel_size))
