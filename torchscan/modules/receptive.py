# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import warnings
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd
from typing import Tuple, Union


__all__ = ['module_rf']


def module_rf(module: Module, input: Tensor, output: Tensor) -> Tuple[float, float, float]:
    """Estimate the spatial receptive field of the module

    Args:
        module (torch.nn.Module): PyTorch module
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module
    Returns:
        receptive field
        effective stride
        effective padding
    """
    if isinstance(module, (nn.Identity, nn.Flatten, nn.ReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid,
                           _BatchNorm, nn.Dropout, nn.Linear)):
        return 1., 1., 0.
    elif isinstance(module, _ConvTransposeNd):
        return rf_convtransposend(module, input, output)
    elif isinstance(module, (_ConvNd, _MaxPoolNd, _AvgPoolNd)):
        return rf_aggregnd(module, input, output)
    elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return rf_adaptive_poolnd(module, input, output)
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return 1., 1., 0.


def rf_convtransposend(module: _ConvTransposeNd, intput: Tensor, output: Tensor) -> Tuple[float, float, float]:
    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
    s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    return -k, 1. / s, 0.  # type: ignore[operator]


def rf_aggregnd(
    module: Union[_ConvNd, _MaxPoolNd, _AvgPoolNd],
    input: Tensor,
    output: Tensor
) -> Tuple[float, float, float]:
    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
    if hasattr(module, 'dilation'):
        d = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
        k = d * (k - 1) + 1  # type: ignore[operator]
    s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    p = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    return k, s, p  # type: ignore[return-value]


def rf_adaptive_poolnd(
    module: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd],
    input: Tensor,
    output: Tensor
) -> Tuple[int, int, float]:

    stride = math.ceil(input.shape[-1] / output.shape[-1])
    kernel_size = stride
    padding = (input.shape[-1] - kernel_size * stride) / 2

    return kernel_size, stride, padding
