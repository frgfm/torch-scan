# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from math import prod
from typing import cast

from torch import nn

from .._formula_math import adaptive_visits as adaptive_visits


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
