# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
import warnings
from typing import Tuple, Union

from torch import Tensor, nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd

__all__ = ["module_rf"]


def module_rf(module: Module, inp: Tensor, out: Tensor) -> Tuple[float, float, float]:
    """Estimate the spatial receptive field of the module

    Args:
        module (torch.nn.Module): PyTorch module
        inp (torch.Tensor): input to the module
        out (torch.Tensor): output of the module
    Returns:
        receptive field
        effective stride
        effective padding
    """
    if isinstance(
        module,
        (
            nn.Identity,
            nn.Flatten,
            nn.ReLU,
            nn.ELU,
            nn.LeakyReLU,
            nn.ReLU6,
            nn.Tanh,
            nn.Sigmoid,
            _BatchNorm,
            nn.Dropout,
            nn.Linear,
        ),
    ):
        return 1.0, 1.0, 0.0
    elif isinstance(module, _ConvTransposeNd):
        return rf_convtransposend(module, inp, out)
    elif isinstance(module, (_ConvNd, _MaxPoolNd, _AvgPoolNd)):
        return rf_aggregnd(module, inp, out)
    elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return rf_adaptive_poolnd(module, inp, out)
    else:
        warnings.warn(f"Module type not supported: {module.__class__.__name__}")
        return 1.0, 1.0, 0.0


def rf_convtransposend(module: _ConvTransposeNd, intput: Tensor, out: Tensor) -> Tuple[float, float, float]:
    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
    s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    return -k, 1.0 / s, 0.0


def rf_aggregnd(module: Union[_ConvNd, _MaxPoolNd, _AvgPoolNd], inp: Tensor, out: Tensor) -> Tuple[float, float, float]:
    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
    if hasattr(module, "dilation"):
        d = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
        k = d * (k - 1) + 1  # type: ignore[operator]
    s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    p = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    return k, s, p  # type: ignore[return-value]


def rf_adaptive_poolnd(
    module: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd], inp: Tensor, out: Tensor
) -> Tuple[int, int, float]:

    stride = math.ceil(inp.shape[-1] / out.shape[-1])
    kernel_size = stride
    padding = (inp.shape[-1] - kernel_size * stride) / 2

    return kernel_size, stride, padding
