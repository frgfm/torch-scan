# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from torch import Tensor


def adaptive_kernel_size(inp: Tensor, out: Tensor) -> tuple[int, ...]:
    """Return the shared spatial-kernel approximation for FLOPs, MACs, and DMAs."""
    return tuple(
        i_size // o_size if i_size % o_size == 0 else i_size % o_size + 1
        for i_size, o_size in zip(inp.shape[2:], out.shape[2:], strict=False)
    )
