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

from .modules._pooling import adaptive_visits
from .modules._primitives import gelu_flops, rmsnorm_rows, softmax_counts


def _elementwise(*_args: Any, out_shape: Any, cost: int = 1, **_kwargs: Any) -> int:
    return cost * prod(out_shape)


def _gelu(*_args: Any, out_shape: Any, approximate: str = "none", **_kwargs: Any) -> int:
    return gelu_flops(prod(out_shape), approximate)


def _square(_input_shape: Any, exponent: Any, *_args: Any, out_shape: Any, **_kwargs: Any) -> int:
    if not isinstance(exponent, (int, float)) or exponent != 2:
        raise NotImplementedError("Power FLOPs only cover an explicit scalar exponent of two.")
    return prod(out_shape)


def _pool(
    _input_shape: Any, kernel_size: Any, *_args: Any, out_shape: Any, rank: int, maximum: bool, **_kwargs: Any
) -> int:
    output = out_shape[0] if maximum else out_shape
    kernel = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    volume = kernel[0] ** rank if len(kernel) == 1 else prod(kernel)
    return prod(output) * (volume - int(maximum))


def _adaptive_pool(
    input_shape: Any, output_size: Any, *_args: Any, out_shape: Any, maximum: bool, **_kwargs: Any
) -> int:
    output = out_shape[0] if maximum else out_shape
    return adaptive_visits(input_shape, output, len(output_size)) - prod(output) * int(maximum)


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


def _softmax(input_shape: Any, dim: int, *_args: Any, **_kwargs: Any) -> int:
    return softmax_counts(tuple(input_shape), dim)[0]


def _log_softmax(input_shape: Any, dim: int, *_args: Any, **_kwargs: Any) -> int:
    return softmax_counts(tuple(input_shape), dim, logarithmic=True)[0]


def _activation(input_shape: Any, *_args: Any, cost: int, **_kwargs: Any) -> int:
    # Input geometry also handles kernels returning auxiliary buffers.
    return cost * prod(input_shape)


def _elu(input_shape: Any, _alpha: Any = 1, scale: Any = 1, input_scale: Any = 1, **_kwargs: Any) -> int:
    return (6 + int(scale != 1) + int(input_scale != 1)) * prod(input_shape)


def _rrelu(
    input_shape: Any,
    _noise: Any,
    _lower: Any = 0.125,
    _upper: Any = 1 / 3,
    training: bool = False,
    *_args: Any,
    **_kwargs: Any,
) -> int:
    if training:
        raise NotImplementedError("RReLU FLOPs cover evaluation calls only.")
    return 4 * prod(input_shape)


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
        or query[0] != key[0]
        or key[1] <= 0
        or query[1] % key[1] != 0
        or key[:3] != value[:3]
        or query[3] != key[3]
    ):
        raise NotImplementedError(
            "CPU attention formula requires matching batches, divisible query/KV heads, Q/K widths, and K/V lengths."
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
        2 * scores * (query[3] + value[3])
        + scores
        + 5 * scores
        - 2 * rows
        + scores * (int(is_causal) + int(attn_mask is not None))
    )


def _native_mha(
    query: Any,
    key: Any,
    value: Any,
    embed_dim: int,
    num_head: int,
    _qkv_weight: Any,
    _qkv_bias: Any,
    _proj_weight: Any,
    _proj_bias: Any,
    mask: Any = None,
    need_weights: bool = True,
    average_attn_weights: bool = True,
    *_args: Any,
    **_kwargs: Any,
) -> int:
    if len(query) != 3 or query != key or key != value or query[-1] != embed_dim:
        raise NotImplementedError("Fused MHA counts require dense batched self-attention with matching widths.")
    batch, length, width = query
    rows = batch * num_head * length
    scores = rows * length
    # Four dense projections, Q scaling, two products, stable softmax, masks,
    # and optional returned-weight averaging. Bias follows native 2K dot counts.
    return (
        8 * batch * length * width**2
        + batch * length * width
        + 4 * batch * length**2 * width
        + 5 * scores
        - 2 * rows
        + scores * (int(mask is not None) + int(need_weights and average_attn_weights))
    )


def _native_encoder(
    src: Any,
    embed_dim: int,
    num_head: int,
    qkv_weight: Any,
    qkv_bias: Any,
    proj_weight: Any,
    proj_bias: Any,
    use_gelu: bool,
    _norm_first: bool,
    _eps: float,
    _norm1_weight: Any,
    _norm1_bias: Any,
    _norm2_weight: Any,
    _norm2_bias: Any,
    ff1_weight: Any,
    _ff1_bias: Any,
    _ff2_weight: Any,
    _ff2_bias: Any,
    mask: Any = None,
    *_args: Any,
    **_kwargs: Any,
) -> int:
    attention = _native_mha(
        src, src, src, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, False, False
    )
    rows = prod(src[:-1])
    elements = prod(src)
    hidden = rows * ff1_weight[0]
    return attention + 4 * hidden * embed_dim + (gelu_flops(hidden) if use_gelu else hidden) + 18 * elements + 4 * rows


def _cumsum(input_shape: Any, dim: int, *_args: Any, **_kwargs: Any) -> int:
    width = input_shape[dim] if input_shape else 1
    return prod(input_shape) - prod(input_shape) // width if width else 0


def _extremum(input_shape: Any, *args: Any, out_shape: Any, **_kwargs: Any) -> int:
    if args and isinstance(args[0], (list, tuple)):
        return prod(out_shape)  # Elementwise max/min with a second tensor.
    output = out_shape[0] if out_shape and isinstance(out_shape[0], (list, tuple)) else out_shape
    return max(0, prod(input_shape) - prod(output))


def _norm(input_shape: Any, order: Any = 2, *_args: Any, out_shape: Any, **kwargs: Any) -> int:
    order = kwargs.get("ord", kwargs.get("p", order))
    if order is None or order == 2:
        return 2 * prod(input_shape)  # Squares, sum, square root.
    if order in (1, float("inf"), -float("inf")):
        return max(0, 2 * prod(input_shape) - prod(out_shape))
    raise NotImplementedError("Norm FLOPs cover only orders 1, 2, and +/-infinity.")


def _dot(input_shape: Any, *_args: Any, **_kwargs: Any) -> int:
    return 2 * prod(input_shape)


def _grouped_mm(input_shape: Any, weight: Any, *_args: Any, **_kwargs: Any) -> int:
    if len(input_shape) != 2 or len(weight) != 3 or input_shape[-1] != weight[-2]:
        raise NotImplementedError("Grouped matmul counts cover packed rows and equal-width expert matrices only.")
    return 2 * prod(input_shape) * weight[-1]


def _add_product(*_args: Any, out_shape: Any, value: Any = 1, **_kwargs: Any) -> int:
    if len(_args) > 3:
        value = _args[3]
    return prod(out_shape) * (2 + int(value != 1))


def _clamp(_input_shape: Any, minimum: Any = None, maximum: Any = None, *, out_shape: Any, **kwargs: Any) -> int:
    return prod(out_shape) * (int(kwargs.get("min", minimum) is not None) + int(kwargs.get("max", maximum) is not None))


FORMULAS = {
    "_native_multi_head_attention": _native_mha,
    "_transformer_encoder_layer_fwd": _native_encoder,
    **dict.fromkeys(["cumsum", "cumsum_"], _cumsum),
    **dict.fromkeys(["max", "min"], _extremum),
    **dict.fromkeys(["norm", "linalg_vector_norm"], _norm),
    **dict.fromkeys(["dot", "vdot", "mv"], _dot),
    "_grouped_mm": _grouped_mm,
    **dict.fromkeys(["addcmul", "addcmul_", "addcdiv", "addcdiv_"], _add_product),
    **dict.fromkeys(["clamp", "clamp_"], _clamp),
    **{f"max_pool{rank}d_with_indices": partial(_pool, rank=rank, maximum=True) for rank in (2, 3)},
    **{f"avg_pool{rank}d": partial(_pool, rank=rank, maximum=False) for rank in (2, 3)},
    **{f"adaptive_max_pool{rank}d": partial(_adaptive_pool, maximum=True) for rank in (2, 3)},
    **{f"_adaptive_avg_pool{rank}d": partial(_adaptive_pool, maximum=False) for rank in (2, 3)},
    **{
        name + suffix: partial(_activation, cost=cost)
        for name, cost in {
            "hardtanh": 2,
            "leaky_relu": 4,
            "hardsigmoid": 4,
            "hardswish": 5,
            "mish": 10,
            "softplus": 7,
            "_prelu_kernel": 4,
            "prelu": 4,
            "celu": 7,
            "log_sigmoid_forward": 7,
            "hardshrink": 3,
            "softshrink": 5,
            "threshold": 2,
        }.items()
        for suffix in ("", "_")
    },
    **dict.fromkeys(["elu", "elu_"], _elu),
    **dict.fromkeys(["rrelu_with_noise", "rrelu_with_noise_"], _rrelu),
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
            "abs",
            "abs_",
            "sin",
            "sin_",
            "cos",
            "cos_",
            "log",
            "log_",
            "reciprocal",
            "reciprocal_",
            "clamp_min",
            "clamp_min_",
            "clamp_max",
            "clamp_max_",
            "maximum",
            "minimum",
        ],
        _elementwise,
    ),
    "sum": _sum,
    "mean": _mean,
    "_softmax": _softmax,
    "_log_softmax": _log_softmax,
    "_safe_softmax": _safe_softmax,
    "native_layer_norm": _layer_norm,
    "native_group_norm": _group_norm,
    "native_batch_norm": _batch_norm,
    "_native_batch_norm_legit": _batch_norm_legit,
    "_native_batch_norm_legit_no_training": _batch_norm_eval,
    "_scaled_dot_product_flash_attention_for_cpu": _cpu_attention,
}
