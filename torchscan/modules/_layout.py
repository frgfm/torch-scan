# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Native lookup, reshape, padding, and interpolation estimates."""

from math import prod
from typing import cast

import torch
from torch import Tensor, nn

from ._pooling import adaptive_visits
from ._transformer import _dense_tensor

PADDING_TYPES: tuple[type[nn.Module], ...] = tuple(
    kind
    for prefix in ("Constant", "Zero", "Reflection", "Replication", "Circular")
    for rank in (1, 2, 3)
    if (kind := getattr(nn, f"{prefix}Pad{rank}d", None)) is not None
)
LAYOUT_TYPES = (
    nn.Embedding,
    nn.Unflatten,
    nn.PixelShuffle,
    nn.PixelUnshuffle,
    nn.ChannelShuffle,
    nn.Unfold,
    nn.Upsample,
    nn.UpsamplingNearest2d,
    nn.UpsamplingBilinear2d,
    *PADDING_TYPES,
)
_FORWARDS = {kind: kind.forward for kind in LAYOUT_TYPES}


def layout_counts(module: nn.Module, inp: Tensor, out: Tensor) -> tuple[int, int, int]:
    """Return FLOPs, legacy MACs, and logical element accesses for a native call."""
    kind = type(module)
    if (
        kind not in _FORWARDS
        or getattr(module.forward, "__self__", None) is not module
        or getattr(module.forward, "__func__", None) is not _FORWARDS[kind]
    ):
        raise NotImplementedError("Layout estimates require unchanged native forwards.")
    if inp.is_nested or inp.layout != torch.strided or inp.is_complex():
        raise NotImplementedError("Layout estimates require dense real tensors.")
    if isinstance(module, nn.Embedding):
        if module.max_norm is not None:
            raise NotImplementedError("Embedding estimates do not cover in-place weight renormalization.")
        weight = _dense_tensor(module.weight)
        if tuple(weight.shape) != (module.num_embeddings, module.embedding_dim):
            raise NotImplementedError("Embedding estimates require native weight geometry.")
        return 0, 0, inp.numel() + 2 * out.numel()
    if isinstance(module, nn.Unflatten):
        return 0, 0, 0  # Metadata-only view.
    if isinstance(module, nn.Upsample):
        if module.mode in ("nearest", "nearest-exact"):
            return 0, 0, 2 * out.numel()
        _dense_tensor(inp)
        if module.mode == "area":
            rank = inp.ndim - 2
            visits = adaptive_visits(inp.shape, out.shape, rank)
            return visits, visits + out.numel() * (rank - 1), visits + out.numel()
        neighbors = {"linear": 2, "bilinear": 4, "trilinear": 8, "bicubic": 16}.get(module.mode)
        if neighbors is None:
            raise NotImplementedError("Unsupported native interpolation mode.")
        return (2 * neighbors - 1) * out.numel(), neighbors * out.numel(), (neighbors + 1) * out.numel()
    if kind in PADDING_TYPES and kind.__name__.startswith(("Constant", "Zero")):
        padding = cast(int | tuple[int, ...], module.padding)
        rank = int(kind.__name__[-2])
        pads = (padding,) * (2 * rank) if isinstance(padding, int) else padding
        sizes = list(inp.shape)
        for axis in range(len(pads) // 2):
            sizes[-axis - 1] += min(0, pads[2 * axis]) + min(0, pads[2 * axis + 1])
        return 0, 0, prod(sizes) + out.numel()
    # Permutations, unfold, and reflected/repeated padding gather one operand
    # per output and write it once, under the nominal dense-padding convention.
    return 0, 0, 2 * out.numel()
