# -*- coding: utf-8 -*-

"""
Module receptive field
"""

import warnings
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd


__all__ = ['module_rf']


def module_rf(module, input, output):
    """Estimate the spatial receptive field of the module

    Args:
        module (torch.nn.Module): PyTorch module
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module
    Returns:
        int: receptive field
        int: effective stride
        int: effective padding
    """
    if isinstance(module, (nn.Identity, nn.ReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid, _BatchNorm)):
        return 1, 1, 0
    elif isinstance(module, _ConvTransposeNd):
        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        return -k, 1 / s, 0
    elif isinstance(module, (_ConvNd, _MaxPoolNd, _AvgPoolNd)):
        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        p = module.padding[0] if isinstance(module.padding, tuple) else module.padding
        return k, s, p
    elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return rf_adaptive_poolnd(module, input, output)
    elif isinstance(module, (nn.Dropout, nn.Linear)):
        return 1, 1, 0
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return 1, 1, 0


def rf_adaptive_poolnd(module, input, output):

    stride = input.shape[-1] // output.shape[-1]
    kernel_size = 2 * (input.shape[-1] % output.shape[-1]) + 1
    padding = (stride * (output.shape[-1] - 1) - input.shape[-1] - kernel_size) / 2

    return kernel_size, stride, padding
