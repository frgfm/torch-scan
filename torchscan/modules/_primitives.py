# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Private native activation and normalization formulas under explicit conventions."""

import math
from typing import cast

from torch import Tensor, nn

from ._transformer import _dense_tensor

PRIMITIVE_TYPES: tuple[type[nn.Module], ...] = (nn.GELU, nn.SiLU, nn.GLU, nn.GroupNorm)
if (rmsnorm := getattr(nn, "RMSNorm", None)) is not None:
    PRIMITIVE_TYPES += (rmsnorm,)
_NATIVE_FORWARDS = {kind: kind.forward for kind in PRIMITIVE_TYPES}


def gelu_flops(elements: int, approximate: str = "none") -> int:
    """Count exact GELU or its tanh expansion; erf costs one operation."""
    if approximate not in ("none", "tanh"):
        raise NotImplementedError("GELU estimates support only 'none' and 'tanh' approximations.")
    return elements * (5 if approximate == "none" else 14)


def rmsnorm_rows(input_shape: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Count nonempty normalized rows, including zero-batch inputs."""
    if (
        not shape
        or any(type(size) is not int or size <= 0 for size in shape)
        or len(shape) > len(input_shape)
        or tuple(input_shape[-len(shape) :]) != shape
    ):
        raise NotImplementedError("RMSNorm estimates require matching nonempty normalized rows.")
    return math.prod(input_shape) // math.prod(shape)


def validate_primitive(module: nn.Module, inp: Tensor) -> tuple[int, int]:
    """Validate the native call boundary and return element and normalization-row counts."""
    kind = type(module)
    if (
        kind not in _NATIVE_FORWARDS
        or getattr(module.forward, "__self__", None) is not module
        or getattr(module.forward, "__func__", None) is not _NATIVE_FORWARDS[kind]
    ):
        raise NotImplementedError("Primitive estimates require exact native types with unchanged forward methods.")
    inp = _dense_tensor(inp)
    elements = inp.numel()
    if kind is nn.GELU:
        gelu_flops(elements, module.approximate)
        return elements, 0
    if kind is nn.SiLU:
        if type(module.inplace) is not bool:
            raise NotImplementedError("SiLU estimates require a native boolean inplace option.")
        return elements, 0
    if kind is nn.GLU:
        if type(module.dim) is not int or not -inp.ndim <= module.dim < inp.ndim or inp.shape[module.dim] % 2:
            raise NotImplementedError("GLU estimates require a valid dimension with an even input size.")
        return elements, 0
    if kind is nn.GroupNorm:
        if (
            type(module.num_groups) is not int
            or type(module.num_channels) is not int
            or module.num_groups <= 0
            or module.num_channels <= 0
            or module.num_channels % module.num_groups
            or inp.ndim < 2
            or inp.shape[1] != module.num_channels
            or math.prod(inp.shape[1:]) == 0
        ):
            raise NotImplementedError("GroupNorm estimates require matching channels and a nonempty normalized group.")
        rows = inp.shape[0] * module.num_groups
        shape = (module.num_channels,)
        parameters = (module.weight, module.bias)
    else:
        shape = tuple(module.normalized_shape)
        rows = rmsnorm_rows(tuple(inp.shape), shape)
        parameters = (module.weight,)
    if (kind is nn.GroupNorm and module.eps is None) or (
        module.eps is not None
        and (isinstance(module.eps, bool) or not isinstance(module.eps, (int, float)) or not math.isfinite(module.eps))
    ):
        raise NotImplementedError("Normalization estimates require a finite scalar epsilon.")
    for parameter in parameters:
        if parameter is not None and tuple(_dense_tensor(parameter).shape) != shape:
            raise NotImplementedError("Normalization parameters must have the native shape.")
    return elements, rows


def primitive_flops(module: nn.Module, inp: Tensor) -> int:
    """Count native primitive arithmetic, independently from operator estimates."""
    elements, rows = validate_primitive(module, inp)
    if type(module) is nn.GELU:
        return gelu_flops(elements, module.approximate)
    if type(module) in (nn.SiLU, nn.GLU):
        return 5 * (elements // 2 if type(module) is nn.GLU else elements)
    if type(module) is nn.GroupNorm:
        return (6 + int(module.weight is not None) + int(module.bias is not None)) * elements + 2 * rows
    return (3 + int(module.weight is not None)) * elements + 2 * rows


def primitive_macs(module: nn.Module, inp: Tensor) -> int:
    """Count norm square-sum and affine terms; activation arithmetic adds no MACs."""
    elements, _ = validate_primitive(module, inp)
    return elements * (1 + int(module.weight is not None)) if type(module) not in (nn.GELU, nn.SiLU, nn.GLU) else 0


def primitive_dmas(module: nn.Module, inp: Tensor, out: Tensor) -> int:
    """Count logical element accesses, independently of kernel fusion and hardware traffic."""
    elements, rows = validate_primitive(module, inp)
    if elements == 0:
        return 0
    if type(module) in (nn.GELU, nn.SiLU, nn.GLU):
        return elements + _dense_tensor(out).numel()
    parameters = (module.weight, module.bias) if type(module) is nn.GroupNorm else (module.weight,)
    parameter_reads = sum(cast(Tensor, parameter).numel() for parameter in parameters if parameter is not None)
    affine = any(parameter is not None for parameter in parameters)
    if type(module) is nn.GroupNorm:
        # The same staged mean/variance/normalize convention as native LayerNorm.
        return 4 * elements + 5 * rows + 1 + parameter_reads + 2 * elements * affine
    # Mean-square: N+R; epsilon/rsqrt: 2R+1; normalize: 2N+R.
    return 3 * elements + 4 * rows + 1 + parameter_reads + 2 * elements * affine
