# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import warnings
from functools import reduce
from operator import mul
from typing import Union

from torch import Tensor, nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd

__all__ = ["module_dmas"]


def module_dmas(module: Module, inp: Tensor, out: Tensor) -> int:
    """Estimate the number of direct memory accesses by the module.
    The implementation overhead is neglected.

    Args:
        module (torch.nn.Module): PyTorch module
        inp (torch.Tensor): input to the module
        out (torch.Tensor): output of the module
    Returns:
        int: number of DMAs
    """

    if isinstance(module, nn.Identity):
        return dmas_identity(module, inp, out)
    elif isinstance(module, nn.Flatten):
        return dmas_flatten(module, inp, out)
    elif isinstance(module, nn.Linear):
        return dmas_linear(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return dmas_relu(module, inp, out)
    elif isinstance(module, (nn.ELU, nn.LeakyReLU)):
        return dmas_act_single_param(module, inp, out)
    elif isinstance(module, nn.Sigmoid):
        return dmas_sigmoid(module, inp, out)
    elif isinstance(module, nn.Tanh):
        return dmas_tanh(module, inp, out)
    elif isinstance(module, _ConvTransposeNd):
        return dmas_convtransposend(module, inp, out)
    elif isinstance(module, _ConvNd):
        return dmas_convnd(module, inp, out)
    elif isinstance(module, _BatchNorm):
        return dmas_bn(module, inp, out)
    elif isinstance(module, (_MaxPoolNd, _AvgPoolNd)):
        return dmas_pool(module, inp, out)
    elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return dmas_adaptive_pool(module, inp, out)
    elif isinstance(module, nn.Dropout):
        return dmas_dropout(module, inp, out)
    else:
        warnings.warn(f"Module type not supported: {module.__class__.__name__}")
        return 0


def num_params(module: Module) -> int:
    """Compute the number of parameters

    Args:
        module (torch.nn.Module): PyTorch module
    Returns:
        int: number of parameter elements
    """

    return sum(p.data.numel() for p in module.parameters())


def dmas_identity(module: nn.Identity, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Identity`"""

    return inp.numel()


def dmas_flatten(module: nn.Flatten, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Flatten`"""

    return 2 * inp.numel()


def dmas_linear(module: nn.Linear, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Linear`"""

    input_dma = inp.numel()
    # Access weight and bias
    ops_dma = num_params(module)
    output_dma = out.numel()

    return input_dma + ops_dma + output_dma


def dmas_relu(module: Union[nn.ReLU, nn.ReLU6], inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.ReLU`"""

    input_dma = inp.numel()
    output_dma = 0 if module.inplace else out.numel()

    return input_dma + output_dma


def dmas_act_single_param(module: Union[nn.ELU, nn.LeakyReLU], inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for activations with single parameter"""

    input_dma = inp.numel()
    # Access alpha, slope or other
    ops_dma = 1
    output_dma = 0 if module.inplace else out.numel()

    return input_dma + ops_dma + output_dma


def dmas_sigmoid(module: nn.Sigmoid, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Sigmoid`"""

    # Access for both exp
    input_dma = inp.numel()
    output_dma = out.numel()

    return input_dma + output_dma


def dmas_tanh(module: nn.Tanh, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Tanh`"""

    # Access for both exp
    input_dma = inp.numel() * 2
    output_dma = out.numel()

    return input_dma + output_dma


def dmas_dropout(module: nn.Dropout, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Dropout`"""

    input_dma = inp.numel()

    # Access sampling probability
    ops_dma = 1

    output_dma = 0 if module.inplace else out.numel()

    return input_dma + ops_dma + output_dma


def dmas_convtransposend(module: _ConvTransposeNd, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.modules.conv._ConvTransposeNd`"""

    # Padding (# cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532)
    # Access stride, padding and kernel_size
    in_padding = len(module.kernel_size) * 4
    out_padding = len(module.kernel_size)

    # The rest is like a classic convolution
    conv_dmas = dmas_convnd(module, inp, out)

    return in_padding + out_padding + conv_dmas


def dmas_convnd(module: _ConvNd, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.modules.conv._ConvNd`"""

    # Each output element required K ** 2 memory access of each input channel
    input_dma = module.in_channels * reduce(mul, module.kernel_size) * out.numel()
    # Correct with groups
    input_dma //= module.groups

    # Access weight & bias
    ops_dma = num_params(module)
    output_dma = out.numel()

    return input_dma + ops_dma + output_dma


def dmas_bn(module: _BatchNorm, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""
    input_dma = inp.numel()

    # Access running_mean, running_var and eps
    ops_dma = module.running_mean.numel() + module.running_var.numel() + 1  # type: ignore[union-attr]
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
        ops_dma += 1 + module.running_mean.numel() + module.running_var.numel()  # type: ignore[union-attr]

    output_dma = out.numel()

    return input_dma + ops_dma + output_dma


def dmas_pool(module: Union[_MaxPoolNd, _AvgPoolNd], inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for spatial pooling modules"""

    # Resolve kernel size and stride size (can be stored as a single integer or a tuple)
    if isinstance(module.kernel_size, tuple):
        kernel_size = module.kernel_size
    elif isinstance(module.kernel_size, int):
        kernel_size = (module.kernel_size,) * (inp.ndim - 2)

    # Each output element required K ** 2 memory accesses
    input_dma = reduce(mul, kernel_size) * out.numel()

    output_dma = out.numel()

    return input_dma + output_dma


def dmas_adaptive_pool(module: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd], inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for adaptive spatial pooling modules"""

    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(inp.shape[2:], out.shape[2:])
    )
    # Each output element required K ** 2 memory accesses
    input_dma = reduce(mul, kernel_size) * out.numel()

    output_dma = out.numel()

    return input_dma + output_dma
