# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import warnings
from functools import reduce
from operator import mul
from typing import Optional, Tuple

from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd


__all__ = ['module_flops']


def module_flops(module: Module, inputs: Tuple[Tensor, ...], output: Optional[Tensor] = None) -> int:
    """Estimate the number of floating point operations performed by the module

    Args:
        module: PyTorch module
        inputs: input to the module
        output: output of the module
    Returns:
        number of FLOPs
    """

    if isinstance(module, (nn.Identity, nn.Flatten)):
        return 0
    elif isinstance(module, nn.Linear):
        return flops_linear(module, inputs)
    elif isinstance(module, nn.ReLU):
        return flops_relu(module, inputs)
    elif isinstance(module, nn.ELU):
        return flops_elu(module, inputs)
    elif isinstance(module, nn.LeakyReLU):
        return flops_leakyrelu(module, inputs)
    elif isinstance(module, nn.ReLU6):
        return flops_relu6(module, inputs)
    elif isinstance(module, nn.Tanh):
        return flops_tanh(module, inputs)
    elif isinstance(module, nn.Sigmoid):
        return flops_sigmoid(module, inputs)
    elif isinstance(module, _ConvTransposeNd):
        return flops_convtransposend(module, inputs, output)
    elif isinstance(module, _ConvNd):
        return flops_convnd(module, inputs, output)
    elif isinstance(module, _BatchNorm):
        return flops_bn(module, inputs)
    elif isinstance(module, _MaxPoolNd):
        return flops_maxpool(module, inputs, output)
    elif isinstance(module, _AvgPoolNd):
        return flops_avgpool(module, inputs, output)
    elif isinstance(module, _AdaptiveMaxPoolNd):
        return flops_adaptive_maxpool(module, inputs, output)
    elif isinstance(module, _AdaptiveAvgPoolNd):
        return flops_adaptive_avgpool(module, inputs, output)
    elif isinstance(module, nn.Dropout):
        return flops_dropout(module, inputs, output)
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return 0


def flops_linear(module: nn.Linear, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Linear`"""

    # batch size * out_chan * in_chan
    mm_flops = inputs[0].shape[0] * module.out_features * (2 * module.in_features - 1)
    bias_flops = inputs[0].shape[0] * module.out_features if module.bias is not None else 0

    return mm_flops + bias_flops


def flops_sigmoid(module: nn.Sigmoid, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Sigmoid`"""

    # For each element, mul by -1, exp it, add 1, div
    return inputs[0].numel() * 4


def flops_relu(module: nn.ReLU, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.ReLU`"""

    # Each element is compared to 0
    return inputs[0].numel()


def flops_elu(module: nn.ELU, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.ELU`"""

    # For each element, compare it to 0, exp it, sub 1, mul by alpha, compare it to 0 and sum both
    return inputs[0].numel() * 6


def flops_leakyrelu(module: nn.LeakyReLU, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.LeakyReLU`"""

    # For each element, compare it to 0 (max), compare it to 0 (min), mul by slope and sum both
    return inputs[0].numel() * 4


def flops_relu6(module: nn.ReLU6, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.ReLU6`"""

    # For each element, compare it to 0 (max), compare it to 0 (min), mul by slope and sum both
    return inputs[0].numel() * 2


def flops_tanh(module: nn.Tanh, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Tanh`"""

    # For each element, exp it, mul by -1 and exp it, divide the sub by the add
    return inputs[0].numel() * 6


def flops_dropout(module: nn.Dropout, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Dropout`"""

    if module.p > 0:
        # Sample a random number for each input element
        return inputs[0].numel()
    else:
        return 0


def flops_convtransposend(module: _ConvTransposeNd, inputs: Tuple[Tensor, ...], output: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.conv._ConvTranposeNd`"""

    # Padding (# cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532)
    # Define min and max sizes
    padding_flops = len(module.kernel_size) * 8

    # Once padding is determined, the operations are almost identical to those of a convolution
    conv_flops = flops_convnd(module, inputs, output)

    return padding_flops + conv_flops


def flops_convnd(module: _ConvNd, inputs: Tuple[Tensor, ...], output: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.conv._ConvNd`"""

    # For each position, # mult = kernel size, # adds = kernel size - 1
    window_flops_per_chan = 2 * reduce(mul, module.kernel_size) - 1
    # Connections to input channels is controlled by the group parameter
    effective_in_chan = (inputs[0].shape[1] // module.groups)
    # N * flops + (N - 1) additions
    window_flops = effective_in_chan * window_flops_per_chan + (effective_in_chan - 1)
    conv_flops = output.numel() * window_flops

    # Each output element gets a bias addition
    bias_flops = output.numel() if module.bias is not None else 0

    return conv_flops + bias_flops


def flops_bn(module: _BatchNorm, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""

    # for each channel, add eps and running_var, sqrt it
    norm_ops = module.num_features * 2
    # For each element, sub running_mean, div by denom
    norm_ops += inputs[0].numel() * 2
    # For each element, mul by gamma, add beta
    scale_ops = inputs[0].numel() * 2 if module.affine else 0
    bn_flops = norm_ops + scale_ops

    # Count tracking stats update ops
    # cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L94-L101
    tracking_flops = 0
    if module.track_running_stats and module.training:
        # exponential_average_factor
        if module.momentum is None:
            tracking_flops += 1
        # running_mean: by channel, sum values and div by batch size
        tracking_flops += inputs[0].numel()  # type: ignore[attr-defined]
        # running_var: by channel, sub mean and square values, sum them, divide by batch size
        tracking_flops += 3 * inputs[0].numel()  # type: ignore[attr-defined]
        # Update both runnning stat: rescale previous value (mul by N), add it the new one, then div by (N + 1)
        tracking_flops += 2 * module.num_features * 3

    return bn_flops + tracking_flops


def flops_maxpool(module: _MaxPoolNd, inputs: Tuple[Tensor, ...], output: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._MaxPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, check max element in kernel scope
    return output.numel() * (k_size - 1)


def flops_avgpool(module: _AvgPoolNd, inputs: Tuple[Tensor, ...], output: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AvgPoolNd`"""

    k_size = reduce(mul, module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return output.numel() * (k_size - 1 + inputs[0].ndim - 2)  # type: ignore[attr-defined]


def flops_adaptive_maxpool(module: _AdaptiveMaxPoolNd, inputs: Tuple[Tensor, ...], output: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveMaxPoolNd`"""

    if isinstance(module.output_size, tuple):
        o_sizes = module.output_size
    else:
        o_sizes = (module.output_size,) * (inputs[0].ndim - 2)  # type: ignore[attr-defined]
    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(inputs[0].shape[2:], o_sizes))

    # for each spatial output element, check max element in kernel scope
    return output.numel() * (reduce(mul, kernel_size) - 1)


def flops_adaptive_avgpool(module: _AdaptiveAvgPoolNd, inputs: Tuple[Tensor, ...], output: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveAvgPoolNd`"""

    if isinstance(module.output_size, tuple):
        o_sizes = module.output_size
    else:
        o_sizes = (module.output_size,) * (inputs[0].ndim - 2)  # type: ignore[attr-defined]
    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(inputs[0].shape[2:], o_sizes))

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return output.numel() * (reduce(mul, kernel_size) - 1 + len(kernel_size))
