# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from math import gcd, prod
from typing import Any, cast

from torch import nn


def pool_rank(module: nn.Module) -> int:
    """Return the spatial rank for batched and unbatched pooling alike."""
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
        return 2
    return 3 if isinstance(module, (nn.MaxPool3d, nn.AvgPool3d, nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d)) else 1


def pool_kernel_volume(module: nn.Module) -> int:
    """Expand scalar and singleton kernels across the module's spatial rank."""
    rank = pool_rank(module)
    kernel = cast(int | tuple[int, ...] | list[int], module.kernel_size)
    return kernel**rank if isinstance(kernel, int) else kernel[0] ** rank if len(kernel) == 1 else prod(kernel)


def adaptive_visits(input_shape: Any, output_shape: Any, rank: int) -> int:
    """Sum exact floor/ceil adaptive windows, including overlap and unbatched inputs."""
    # Sum_i(ceil((i+1)L/O)-floor(iL/O)) = L+O-gcd(L,O).
    return prod(input_shape[:-rank]) * prod(
        length + output - gcd(length, output)
        for length, output in zip(input_shape[-rank:], output_shape[-rank:], strict=True)
    )
