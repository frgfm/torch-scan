# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import warnings
from functools import reduce
from operator import mul

from torch import Tensor, nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd

__all__ = ["module_macs"]


def module_macs(module: Module, inp: Tensor, out: Tensor) -> int:
    """Estimate the number of multiply-accumulation operations performed by the module

    Args:
        module (torch.nn.Module): PyTorch module
        inp (torch.Tensor): input to the module
        out (torch.Tensor): output of the module
    Returns:
        int: number of MACs
    """
    if isinstance(module, nn.Linear):
        return macs_linear(module, inp, out)
    elif isinstance(module, (nn.Identity, nn.ReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid, nn.Flatten)):
        return 0
    elif isinstance(module, _ConvTransposeNd):
        return macs_convtransposend(module, inp, out)
    elif isinstance(module, _ConvNd):
        return macs_convnd(module, inp, out)
    elif isinstance(module, _BatchNorm):
        return macs_bn(module, inp, out)
    elif isinstance(module, _MaxPoolNd):
        return macs_maxpool(module, inp, out)
    elif isinstance(module, _AvgPoolNd):
        return macs_avgpool(module, inp, out)
    elif isinstance(module, _AdaptiveMaxPoolNd):
        return macs_adaptive_maxpool(module, inp, out)
    elif isinstance(module, _AdaptiveAvgPoolNd):
        return macs_adaptive_avgpool(module, inp, out)
    elif isinstance(module, nn.Dropout):
        return 0
    else:
        warnings.warn(f"Module type not supported: {module.__class__.__name__}")
        return 0


def macs_linear(module: nn.Linear, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.Linear`"""

    # batch size * out_chan * macs_per_elt (bias already counted in accumulation)
    mm_mac = module.in_features * reduce(mul, out.shape)

    return mm_mac


def macs_convtransposend(module: _ConvTransposeNd, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.modules.conv._ConvTransposeNd`"""

    # Padding (# cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532)
    # Define min and max sizes, then subtract them
    padding_macs = len(module.kernel_size) * 4

    # Rest of the operations are almost identical to a convolution (given the padding)
    conv_macs = macs_convnd(module, inp, out)

    return padding_macs + conv_macs


def macs_convnd(module: _ConvNd, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.modules.conv._ConvNd`"""

    # For each position, # mult = kernel size, # adds = kernel size - 1
    window_macs_per_chan = reduce(mul, module.kernel_size)
    # Connections to input channels is controlled by the group parameter
    effective_in_chan = inp.shape[1] // module.groups
    # N * mac
    window_mac = effective_in_chan * window_macs_per_chan
    conv_mac = out.numel() * window_mac

    # bias already counted in accumulation
    return conv_mac


def macs_bn(module: _BatchNorm, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""

    # sub mean, div by denom
    norm_mac = 1
    # mul by gamma, add beta
    scale_mac = 1 if module.affine else 0

    # Sum everything up
    bn_mac = inp.numel() * (norm_mac + scale_mac)

    # Count tracking stats update ops
    # cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L94-L101
    tracking_mac = 0
    b = inp.shape[0]
    num_spatial_elts = inp.shape[2:].numel()
    if module.track_running_stats and module.training:
        # running_mean: by channel, sum value and div by batch size
        tracking_mac += module.num_features * (b * num_spatial_elts - 1)
        # running_var: by channel, sub mean and square values, sum them, divide by batch size
        active_elts = b * num_spatial_elts
        tracking_mac += module.num_features * (2 * active_elts - 1)
        # Update both runnning stat: rescale previous value (mul by N), add it the new one, then div by (N + 1)
        tracking_mac += 2 * module.num_features * 2

    return bn_mac + tracking_mac


def macs_maxpool(module: _MaxPoolNd, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.modules.pooling._MaxPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, check max element in kernel scope
    return out.numel() * (k_size - 1)


def macs_avgpool(module: _AvgPoolNd, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.modules.pooling._AvgPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return out.numel() * (k_size - 1 + inp.ndim - 2)


def macs_adaptive_maxpool(module: _AdaptiveMaxPoolNd, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.modules.pooling._AdaptiveMaxPoolNd`"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(inp.shape[2:], out.shape[2:])
    )

    # for each spatial output element, check max element in kernel scope
    return out.numel() * (reduce(mul, kernel_size) - 1)


def macs_adaptive_avgpool(module: _AdaptiveAvgPoolNd, inp: Tensor, out: Tensor) -> int:
    """MACs estimation for `torch.nn.modules.pooling._AdaptiveAvgPoolNd`"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(inp.shape[2:], out.shape[2:])
    )

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return out.numel() * (reduce(mul, kernel_size) - 1 + len(kernel_size))
