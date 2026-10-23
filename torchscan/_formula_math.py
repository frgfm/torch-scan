# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from math import gcd, prod
from typing import Any


def adaptive_visits(input_shape: Any, output_shape: Any, rank: int) -> int:
    """Sum exact floor/ceil adaptive windows, including overlap and unbatched inputs."""
    # Sum_i(ceil((i+1)L/O)-floor(iL/O)) = L+O-gcd(L,O).
    return prod(input_shape[:-rank]) * prod(
        length + output - gcd(length, output)
        for length, output in zip(input_shape[-rank:], output_shape[-rank:], strict=True)
    )


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


def softmax_counts(shape: tuple[int, ...], dim: int, *, logarithmic: bool = False) -> tuple[int, int]:
    """Count stable row arithmetic and staged logical accesses."""
    elements = math.prod(shape)
    width = shape[dim] if shape else 1
    if not elements or not width:
        return 0, 0
    rows = elements // width
    return (
        (4 * elements - rows, 8 * elements + 6 * rows)
        if logarithmic
        else (5 * elements - 2 * rows, 8 * elements + 4 * rows)
    )
