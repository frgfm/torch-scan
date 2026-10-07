# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Small shape formulas used only when the installed PyTorch has no formula.

These follow the shape-formula contract shared by PyTorch 2.1 and newer releases.
They never register in PyTorch's process-global mapping.
"""

from functools import partial
from math import prod
from typing import Any

from torch.utils.flop_counter import sdpa_flop_count

from .modules._primitives import gelu_flops, rmsnorm_rows


def _elementwise(*_args: Any, out_shape: Any, cost: int = 1, **_kwargs: Any) -> int:
    return cost * prod(out_shape)


def _gelu(*_args: Any, out_shape: Any, approximate: str = "none", **_kwargs: Any) -> int:
    return gelu_flops(prod(out_shape), approximate)


def _square(_input_shape: Any, exponent: Any, *_args: Any, out_shape: Any, **_kwargs: Any) -> int:
    if not isinstance(exponent, (int, float)) or exponent != 2:
        raise NotImplementedError("Power FLOPs only cover an explicit scalar exponent of two.")
    return prod(out_shape)


def _rms_norm(input_shape: Any, normalized_shape: Any, weight: Any = None, *_args: Any, **_kwargs: Any) -> int:
    rows = rmsnorm_rows(input_shape, tuple(normalized_shape))
    if weight is not None and tuple(weight) != tuple(normalized_shape):
        raise NotImplementedError("RMSNorm FLOPs require a matching weight shape.")
    elements = prod(input_shape)
    return (3 + int(weight is not None)) * elements + 2 * rows


def _add(*_args: Any, out_shape: Any, alpha: float = 1, **_kwargs: Any) -> int:
    if len(_args) > 2:
        alpha = _args[2]
    return prod(out_shape) * (1 + int(alpha != 1))


def _softmax(input_shape: Any, dim: int, *_args: Any, out_shape: Any, **_kwargs: Any) -> int:
    width = input_shape[dim] if input_shape else 1
    return 0 if width == 0 else 5 * prod(out_shape) - 2 * (prod(out_shape) // width)


def _safe_softmax(*args: Any, out_shape: Any, **kwargs: Any) -> int:
    # Stable softmax plus a comparison with -inf and a selection per element.
    # The boolean row reduction is excluded from floating-point arithmetic.
    return _softmax(*args, out_shape=out_shape, **kwargs) + 2 * prod(out_shape)


def _sum(input_shape: Any, *_args: Any, out_shape: Any, **_kwargs: Any) -> int:
    return max(0, prod(input_shape) - prod(out_shape))


def _mean(input_shape: Any, *_args: Any, out_shape: Any, **_kwargs: Any) -> int:
    outputs = prod(out_shape)
    if prod(input_shape) == 0 and outputs:
        raise NotImplementedError("Mean FLOPs require nonempty reduction rows.")
    return _sum(input_shape, out_shape=out_shape) + outputs


def _layer_norm(input_shape: Any, normalized_shape: Any, weight: Any, bias: Any, *_args: Any, **_kwargs: Any) -> int:
    elements = prod(input_shape)
    if prod(normalized_shape) == 0:
        raise NotImplementedError("LayerNorm FLOPs require a nonempty normalized row.")
    rows = elements // prod(normalized_shape)
    return 6 * elements + 2 * rows + elements * (int(weight is not None) + int(bias is not None))


def _group_norm(
    input_shape: Any,
    weight: Any,
    bias: Any,
    batch: int,
    _channels: int,
    _spatial: int,
    groups: int,
    *_args: Any,
    **_kwargs: Any,
) -> int:
    elements = prod(input_shape)
    if prod(input_shape[1:]) == 0:
        raise NotImplementedError("GroupNorm FLOPs require a nonempty normalized group.")
    return 6 * elements + 2 * batch * groups + elements * (int(weight is not None) + int(bias is not None))


def _batch_norm(
    input_shape: Any,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    training: bool,
    *_args: Any,
    **_kwargs: Any,
) -> int:
    elements, channels = prod(input_shape), input_shape[1]
    if elements == 0:
        raise NotImplementedError("Empty native BatchNorm work depends on strides unavailable to the shape formula.")
    result = 2 * elements + 2 * channels + elements * (int(weight is not None) + int(bias is not None))
    if training:
        result += 4 * elements
        result += channels * (3 * int(running_mean is not None) + 5 * int(running_var is not None))
    return result


def _batch_norm_eval(*args: Any, **kwargs: Any) -> int:
    return _batch_norm(*args[:5], False, *args[5:], **kwargs)


def _batch_norm_legit(*args: Any, **kwargs: Any) -> int:
    # The no_stats overload omits both running-stat tensors.
    if len(args) == 6:
        return _batch_norm(*args[:3], None, None, *args[3:], **kwargs)
    return _batch_norm(*args, **kwargs)


def _cpu_attention(
    query: Any,
    key: Any,
    value: Any,
    dropout_p: float = 0,
    is_causal: bool = False,
    *,
    attn_mask: Any = None,
    **_kwargs: Any,
) -> int:
    if (
        len(query) != 4
        or len(key) != 4
        or len(value) != 4
        or query[:2] != key[:2]
        or key[:3] != value[:3]
        or query[3] != key[3]
    ):
        raise NotImplementedError(
            "CPU attention formula requires matching batch/head counts, Q/K widths, and K/V lengths."
        )
    if dropout_p != 0:
        raise NotImplementedError("CPU attention formula does not cover dropout.")
    # Keep PyTorch's two dense matrix products. Count score scaling and stable
    # softmax separately, including one selection/add per mask position.
    rows = query[0] * query[1] * query[2]
    if rows == 0 or key[2] == 0:
        return 0
    scores = rows * key[2]
    return (
        sdpa_flop_count(query, key, value)
        + scores
        + 5 * scores
        - 2 * rows
        + scores * (int(is_causal) + int(attn_mask is not None))
    )


FORMULAS = {
    **dict.fromkeys(["gelu", "gelu_"], _gelu),
    **dict.fromkeys(["silu", "silu_", "glu"], partial(_elementwise, cost=5)),
    **dict.fromkeys(["sigmoid", "sigmoid_"], partial(_elementwise, cost=4)),
    **dict.fromkeys(["tanh", "tanh_"], partial(_elementwise, cost=6)),
    **dict.fromkeys(["pow", "pow_"], _square),
    **dict.fromkeys(["rms_norm", "_fused_rms_norm"], _rms_norm),
    **dict.fromkeys(["add", "add_", "sub", "sub_"], _add),
    **dict.fromkeys(
        [
            "mul",
            "mul_",
            "div",
            "div_",
            "neg",
            "relu",
            "relu_",
            "exp",
            "sqrt",
            "rsqrt",
            "masked_fill",
            "masked_fill_",
            "where",
        ],
        _elementwise,
    ),
    "sum": _sum,
    "mean": _mean,
    "_softmax": _softmax,
    "_safe_softmax": _safe_softmax,
    "native_layer_norm": _layer_norm,
    "native_group_norm": _group_norm,
    "native_batch_norm": _batch_norm,
    "_native_batch_norm_legit": _batch_norm_legit,
    "_native_batch_norm_legit_no_training": _batch_norm_eval,
    "_scaled_dot_product_flash_attention_for_cpu": _cpu_attention,
}
