# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
import warnings
from typing import Union

from torch import Tensor, nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd

from ._layout import LAYOUT_TYPES, layout_counts
from ._pooling import adaptive_visits, pool_kernel_volume, pool_rank
from ._primitives import PRIMITIVE_TYPES, primitive_dmas
from ._transformer import _norm_dmas

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
    if isinstance(module, PRIMITIVE_TYPES):
        return primitive_dmas(module, inp, out)
    if isinstance(module, LAYOUT_TYPES):
        return layout_counts(module, inp, out)[2]
    if isinstance(module, nn.Identity):
        return dmas_identity(module, inp, out)
    if isinstance(module, nn.Flatten):
        return dmas_flatten(module, inp, out)
    if isinstance(module, nn.Linear):
        return dmas_linear(module, inp, out)
    if type(module) is nn.LayerNorm:
        if math.prod(module.normalized_shape) == 0:
            raise NotImplementedError("LayerNorm DMAs require a nonempty normalized row.")
        return _norm_dmas(module, inp)
    if isinstance(module, (nn.ReLU, nn.ReLU6)):
        return dmas_relu(module, inp, out)
    if isinstance(module, (nn.ELU, nn.LeakyReLU)):
        return dmas_act_single_param(module, inp, out)
    if isinstance(module, nn.Sigmoid):
        return dmas_sigmoid(module, inp, out)
    if isinstance(module, nn.Tanh):
        return dmas_tanh(module, inp, out)
    if isinstance(module, _ConvTransposeNd):
        return dmas_convtransposend(module, inp, out)
    if isinstance(module, _ConvNd):
        return dmas_convnd(module, inp, out)
    if isinstance(module, _BatchNorm):
        return dmas_bn(module, inp, out)
    if isinstance(module, (_MaxPoolNd, _AvgPoolNd)):
        return dmas_pool(module, inp, out)
    if isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return dmas_adaptive_pool(module, inp, out)
    if isinstance(module, nn.Dropout):
        return dmas_dropout(module, inp, out)
    warnings.warn(f"Module type not supported: {module.__class__.__name__}", stacklevel=1)
    return 0


def num_params(module: Module) -> int:
    """Compute the number of parameters

    Args:
        module (torch.nn.Module): PyTorch module
    Returns:
        int: number of parameter elements
    """
    return sum(p.numel() for p in module.parameters())


def dmas_identity(_: nn.Identity, inp: Tensor, __: Tensor) -> int:
    """DMAs estimation for `torch.nn.Identity`"""
    return inp.numel()


def dmas_flatten(_: nn.Flatten, inp: Tensor, __: Tensor) -> int:
    """DMAs estimation for `torch.nn.Flatten`"""
    return 2 * inp.numel()


def dmas_linear(module: nn.Linear, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Linear`"""
    # Read the inputs, weight and bias; write the output.
    return inp.numel() + num_params(module) + out.numel()


def dmas_relu(module: Union[nn.ReLU, nn.ReLU6], inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.ReLU`"""
    return inp.numel() + (0 if module.inplace else out.numel())


def dmas_act_single_param(module: Union[nn.ELU, nn.LeakyReLU], inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for activations with single parameter"""
    # Include one access to alpha or slope.
    return inp.numel() + 1 + (0 if module.inplace else out.numel())


def dmas_sigmoid(_: nn.Sigmoid, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Sigmoid`"""
    return inp.numel() + out.numel()


def dmas_tanh(_: nn.Tanh, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Tanh`"""
    # Read the input for both exponentials.
    return 2 * inp.numel() + out.numel()


def dmas_dropout(module: nn.Dropout, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.Dropout`"""
    # Include one access to the sampling probability.
    return inp.numel() + 1 + (0 if module.inplace else out.numel())


def dmas_convtransposend(module: _ConvTransposeNd, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.modules.conv._ConvTransposeNd`"""
    # Padding calculation: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532
    # Access stride, padding and kernel size, then count the convolution.
    return 5 * len(module.kernel_size) + dmas_convnd(module, inp, out)


def dmas_convnd(module: _ConvNd, _: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.modules.conv._ConvNd`"""
    # Each output element required K ** 2 memory access of each input channel
    input_dma = module.in_channels * math.prod(module.kernel_size) * out.numel()
    # Correct with groups
    input_dma //= module.groups

    # Access weight & bias
    ops_dma = num_params(module)
    output_dma = out.numel()

    return input_dma + ops_dma + output_dma


def dmas_bn(module: _BatchNorm, inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""
    input_dma = inp.numel()

    # Access eps, running_mean and running_var when tracked
    ops_dma = 1
    if module.running_mean is not None and module.running_var is not None:
        ops_dma += module.running_mean.numel() + module.running_var.numel()
    # Access to weight and bias
    if module.affine and module.weight is not None and module.bias is not None:
        ops_dma += module.weight.numel() + module.bias.numel()
    # Exp avg factor
    if module.momentum is not None:
        ops_dma += 1
    # Update stats
    if (
        module.training
        and module.track_running_stats
        and module.running_mean is not None
        and module.running_var is not None
    ):
        # Current mean and std computation only requires access to input, already counted in input_dma
        # Update num of batches and running stats
        ops_dma += 1 + module.running_mean.numel() + module.running_var.numel()

    output_dma = out.numel()

    return input_dma + ops_dma + output_dma


def dmas_pool(module: Union[_MaxPoolNd, _AvgPoolNd], _inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for spatial pooling modules"""
    return out.numel() * (pool_kernel_volume(module) + 1 + int(getattr(module, "return_indices", False)))


def dmas_adaptive_pool(module: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd], inp: Tensor, out: Tensor) -> int:
    """DMAs estimation for adaptive spatial pooling modules"""
    return adaptive_visits(inp.shape, out.shape, pool_rank(module)) + out.numel() * (
        1 + int(getattr(module, "return_indices", False))
    )
