# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
import warnings
from typing import Any, Callable, Tuple, cast

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd

from ._pooling import adaptive_kernel_size

__all__ = ["module_flops"]


def module_flops(module: Module | Callable[..., Tensor], inputs: Tuple[Any, ...], out: Any) -> int:
    """Estimate the number of floating point operations performed by the module

    Args:
        module: PyTorch module
        inputs: input to the module
        out: output of the module
    Returns:
        number of FLOPs
    """
    if isinstance(module, (nn.Identity, nn.Flatten)):
        return 0
    if any(
        isinstance(value, Tensor) and (value.is_complex() or value.is_nested or value.layout != torch.strided)
        for value in (inputs or ())
    ):
        raise NotImplementedError("Module FLOP formulas cover real dense strided tensors only.")
    if isinstance(module, nn.Linear):
        return flops_linear(module, inputs)
    if isinstance(module, nn.ReLU):
        return flops_relu(module, inputs)
    if isinstance(module, nn.ELU):
        return flops_elu(module, inputs)
    if isinstance(module, nn.LeakyReLU):
        return flops_leakyrelu(module, inputs)
    if isinstance(module, nn.ReLU6):
        return flops_relu6(module, inputs)
    if isinstance(module, nn.Tanh):
        return flops_tanh(module, inputs)
    if isinstance(module, nn.Sigmoid):
        return flops_sigmoid(module, inputs)
    if isinstance(module, _ConvTransposeNd):
        return flops_convtransposend(module, inputs, out)
    if isinstance(module, _ConvNd):
        return flops_convnd(module, inputs, out)
    if isinstance(module, _BatchNorm):
        return flops_bn(module, inputs)
    if isinstance(module, _MaxPoolNd):
        return flops_maxpool(module, inputs, out)
    if isinstance(module, _AvgPoolNd):
        return flops_avgpool(module, inputs, out)
    if isinstance(module, _AdaptiveMaxPoolNd):
        return flops_adaptive_maxpool(module, inputs, out)
    if isinstance(module, _AdaptiveAvgPoolNd):
        return flops_adaptive_avgpool(module, inputs, out)
    if isinstance(module, nn.Dropout):
        return flops_dropout(module, inputs)
    if isinstance(module, nn.MultiheadAttention):
        return flops_mha(module, inputs, out)
    if isinstance(module, nn.LayerNorm):
        return flops_layernorm(module, inputs)
    if isinstance(module, nn.GroupNorm):
        return flops_groupnorm(module, inputs)
    if isinstance(module, nn.Transformer):
        return flops_transformer(module, inputs)
    warnings.warn(f"Module type not supported: {module.__class__.__name__}", stacklevel=1)
    return 0


def flops_linear(module: nn.Linear, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Linear`"""
    # batch size * out_chan * in_chan
    num_out_feats = module.out_features * math.prod(inputs[0].shape[:-1])
    mm_flops = num_out_feats * max(0, 2 * module.in_features - 1)
    bias_flops = num_out_feats if module.bias is not None else 0

    return mm_flops + bias_flops


def flops_sigmoid(_: nn.Sigmoid, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Sigmoid`"""
    # For each element, mul by -1, exp it, add 1, div
    return inputs[0].numel() * 4


def flops_relu(_: nn.ReLU, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.ReLU`"""
    # Each element is compared to 0
    return inputs[0].numel()


def flops_elu(_: nn.ELU, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.ELU`"""
    # For each element, compare it to 0, exp it, sub 1, mul by alpha, compare it to 0 and sum both
    return inputs[0].numel() * 6


def flops_leakyrelu(_: nn.LeakyReLU, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.LeakyReLU`"""
    # For each element, compare it to 0 (max), compare it to 0 (min), mul by slope and sum both
    return inputs[0].numel() * 4


def flops_relu6(_: nn.ReLU6, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.ReLU6`"""
    # For each element, compare it to 0 (max), compare it to 0 (min), mul by slope and sum both
    return inputs[0].numel() * 2


def flops_tanh(_: nn.Tanh, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Tanh`"""
    # For each element, exp it, mul by -1 and exp it, divide the sub by the add
    return inputs[0].numel() * 6


def flops_dropout(module: nn.Dropout, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Dropout`"""
    if module.training and 0 < module.p < 1:
        # Apply the mask and rescale. Random number generation is excluded.
        return 2 * inputs[0].numel()
    if module.training and module.p == 1:
        return inputs[0].numel()  # Multiply by a zero mask, without rescaling.
    return 0


def flops_convtransposend(module: _ConvTransposeNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.conv._ConvTransposeNd`"""
    # Dense scatter convention, matching PyTorch's input-based MAC geometry.
    # Shape arithmetic, padding/cropping, and output allocation are excluded.
    products = inputs[0].numel() * (module.out_channels // module.groups) * math.prod(module.kernel_size)
    return 2 * products + out.numel() * int(module.bias is not None)


def flops_convnd(module: _ConvNd, _inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.conv._ConvNd`"""
    # Each grouped dot product uses two operations per term minus one, plus an optional bias.
    terms = math.prod(module.kernel_size) * (module.in_channels // module.groups)
    return out.numel() * (2 * terms - 1 + int(module.bias is not None))


def flops_bn(module: _BatchNorm, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""
    # for each channel, add eps and running_var, sqrt it
    norm_ops = module.num_features * 2
    # For each element, sub running_mean, div by denom
    norm_ops += inputs[0].numel() * 2
    # For each element, mul by gamma, add beta
    scale_ops = inputs[0].numel() * (int(module.weight is not None) + int(module.bias is not None))
    bn_flops = norm_ops + scale_ops

    # Batch statistics are needed in training AND in eval without running stats.
    if module.training or (module.running_mean is None and module.running_var is None):
        # Mean: N ops; biased variance: subtract, square, reduce, divide = 3N.
        bn_flops += 4 * inputs[0].numel()

    # Count floating-point running-stat updates, excluding the integer batch counter.
    tracking_flops = 0
    if (
        module.track_running_stats
        and module.training
        and module.running_mean is not None
        and module.running_var is not None
    ):
        # Convert biased variance to unbiased (multiply/divide), then two
        # exponential averages: two multiplies and one addition each.
        tracking_flops += 8 * module.num_features

    return bn_flops + tracking_flops


def _pool_kernel_volume(module: _MaxPoolNd | _AvgPoolNd) -> int:
    kernel_size = cast(int | Tuple[int, ...] | list[int], module.kernel_size)
    rank = 1
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        rank = 2
    elif isinstance(module, (nn.MaxPool3d, nn.AvgPool3d)):
        rank = 3
    if isinstance(kernel_size, int):
        return kernel_size**rank
    return kernel_size[0] ** rank if len(kernel_size) == 1 else math.prod(kernel_size)


def flops_maxpool(module: _MaxPoolNd, _: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._MaxPoolNd`"""
    # for each spatial output element, check max element in kernel scope
    return out.numel() * (_pool_kernel_volume(module) - 1)


def flops_avgpool(module: _AvgPoolNd, _inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AvgPoolNd`"""
    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return out.numel() * _pool_kernel_volume(module)


def flops_adaptive_maxpool(_: _AdaptiveMaxPoolNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveMaxPoolNd`"""
    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = adaptive_kernel_size(inputs[0], out)

    # for each spatial output element, check max element in kernel scope
    return out.numel() * (math.prod(kernel_size) - 1)


def flops_adaptive_avgpool(_: _AdaptiveAvgPoolNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveAvgPoolNd`"""
    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = adaptive_kernel_size(inputs[0], out)

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return out.numel() * (math.prod(kernel_size) - 1 + len(kernel_size))


def flops_layernorm(module: nn.LayerNorm, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.LayerNorm`"""
    numel = inputs[0].numel()
    if math.prod(module.normalized_shape) == 0:
        raise NotImplementedError("LayerNorm FLOPs require a nonempty normalized row.")
    rows = numel // math.prod(module.normalized_shape)
    return 6 * numel + 2 * rows + numel * int(module.weight is not None) + numel * int(module.bias is not None)


def flops_groupnorm(module: nn.GroupNorm, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.GroupNorm`."""
    numel = inputs[0].numel()
    if math.prod(inputs[0].shape[1:]) == 0:
        raise NotImplementedError("GroupNorm FLOPs require a nonempty normalized group.")
    rows = inputs[0].shape[0] * module.num_groups
    return 6 * numel + 2 * rows + numel * int(module.weight is not None) + numel * int(module.bias is not None)


def flops_mha(module: nn.MultiheadAttention, inputs: Tuple[Any, ...], out: Any = None) -> int:
    """FLOPs estimation for `torch.nn.MultiheadAttention`"""
    q, k, v = inputs[:3]
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise NotImplementedError("MultiheadAttention FLOPs only support batched 3D inputs.")
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise NotImplementedError("MultiheadAttention FLOPs do not support add_bias_kv or add_zero_attn.")

    batch_dim, sequence_dim = (0, 1) if module.batch_first else (1, 0)
    batch_size = q.shape[batch_dim]
    target_length = q.shape[sequence_dim]
    source_length = k.shape[sequence_dim]
    if target_length == 0 or source_length == 0:
        raise NotImplementedError("MultiheadAttention FLOPs require nonempty query and source sequences.")
    projection_bias = int(module.in_proj_bias is not None)

    tot_flops = sum(
        math.prod(tensor.shape[:-1]) * module.embed_dim * (2 * tensor.shape[-1] - 1 + projection_bias)
        for tensor in (q, k, v)
    )
    # One scale multiplication per query element; Python constant arithmetic is excluded.
    tot_flops += batch_size * module.num_heads * target_length * module.head_dim
    tot_flops += batch_size * module.num_heads * target_length * source_length * (2 * module.head_dim - 1)

    # Positional MHA slots 3 and 5 are key_padding_mask and attn_mask.
    visible_masks = int(len(inputs) > 3 and inputs[3] is not None) + int(len(inputs) > 5 and inputs[5] is not None)
    tot_flops += visible_masks * batch_size * module.num_heads * target_length * source_length
    # Stable softmax: max (S-1), subtract S, exp S, sum (S-1), divide S.
    tot_flops += batch_size * module.num_heads * target_length * (5 * source_length - 2)
    if module.training and module.dropout > 0:
        tot_flops += (2 if module.dropout < 1 else 1) * batch_size * module.num_heads * target_length * source_length
    tot_flops += batch_size * module.num_heads * target_length * module.head_dim * (2 * source_length - 1)
    tot_flops += flops_linear(module.out_proj, (q,))

    if isinstance(out, (tuple, list)) and len(out) > 1 and isinstance(out[1], Tensor) and out[1].ndim == 3:
        tot_flops += batch_size * module.num_heads * target_length * source_length

    return tot_flops


def flops_transformer_feedforward(
    module: nn.TransformerEncoderLayer | nn.TransformerDecoderLayer, inputs: Tuple[Tensor, ...]
) -> int:
    """FLOPs estimation for a Transformer layer feed-forward block."""
    if module.activation is not F.relu:
        raise NotImplementedError("Transformer FLOPs only support the default ReLU activation.")

    num_hidden = math.prod(inputs[0].shape[:-1]) * module.linear1.out_features
    dropout_flops = (
        (2 if module.dropout.p < 1 else 1) * num_hidden if module.dropout.training and module.dropout.p > 0 else 0
    )
    return flops_linear(module.linear1, inputs) + num_hidden + dropout_flops + flops_linear(module.linear2, inputs)


def flops_transformer_encoderlayer(module: nn.TransformerEncoderLayer, inputs: Tuple[Any, ...]) -> int:
    """FLOPs estimation for `torch.nn.TransformerEncoderLayer`"""
    input_flops = inputs[0].numel()
    src_mask = inputs[1] if len(inputs) > 1 else None
    src_key_padding_mask = inputs[2] if len(inputs) > 2 else None
    tot_flops = flops_mha(module.self_attn, (inputs[0],) * 3 + (src_key_padding_mask, False, src_mask))

    tot_flops += (flops_dropout(module.dropout1, inputs) if module.dropout1.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm1, inputs)
    tot_flops += flops_transformer_feedforward(module, inputs)
    tot_flops += (flops_dropout(module.dropout2, inputs) if module.dropout2.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm2, inputs)

    return tot_flops


def flops_transformer_decoderlayer(module: nn.TransformerDecoderLayer, inputs: Tuple[Any, ...]) -> int:
    """FLOPs estimation for `torch.nn.TransformerDecoderLayer`"""
    input_flops = inputs[0].numel()
    tgt_mask = inputs[2] if len(inputs) > 2 else None
    memory_mask = inputs[3] if len(inputs) > 3 else None
    tgt_key_padding_mask = inputs[4] if len(inputs) > 4 else None
    memory_key_padding_mask = inputs[5] if len(inputs) > 5 else None
    tot_flops = flops_mha(module.self_attn, (inputs[0],) * 3 + (tgt_key_padding_mask, False, tgt_mask))

    tot_flops += (flops_dropout(module.dropout1, inputs) if module.dropout1.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm1, inputs)

    tot_flops += flops_mha(
        module.multihead_attn,
        (inputs[0], inputs[1], inputs[1], memory_key_padding_mask, False, memory_mask),
    )
    tot_flops += (flops_dropout(module.dropout2, inputs) if module.dropout2.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm2, inputs)

    tot_flops += flops_transformer_feedforward(module, inputs)
    tot_flops += (flops_dropout(module.dropout3, inputs) if module.dropout3.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm3, inputs)

    return tot_flops


def flops_transformer(module: nn.Transformer, inputs: Tuple[Any, ...]) -> int:
    """FLOPs estimation for `torch.nn.Transformer`"""
    if not isinstance(module.encoder, nn.TransformerEncoder) or not isinstance(module.decoder, nn.TransformerDecoder):
        raise NotImplementedError("Transformer FLOPs only support the native encoder and decoder stacks.")
    if any(
        stack.norm is not None and not isinstance(stack.norm, (nn.LayerNorm, nn.Identity))
        for stack in (module.encoder, module.decoder)
    ):
        raise NotImplementedError("Transformer FLOPs only support LayerNorm, Identity, or no final normalization.")

    src_mask = inputs[2] if len(inputs) > 2 else None
    tgt_mask = inputs[3] if len(inputs) > 3 else None
    memory_mask = inputs[4] if len(inputs) > 4 else None
    src_key_padding_mask = inputs[5] if len(inputs) > 5 else None
    tgt_key_padding_mask = inputs[6] if len(inputs) > 6 else None
    memory_key_padding_mask = inputs[7] if len(inputs) > 7 else None
    src_inputs = (inputs[0], src_mask, src_key_padding_mask)
    decoder_inputs = (
        inputs[1],
        inputs[0],
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    )
    encoder_flops = sum(flops_transformer_encoderlayer(layer, src_inputs) for layer in module.encoder.layers)

    if isinstance(module.encoder.norm, nn.LayerNorm):
        encoder_flops += flops_layernorm(module.encoder.norm, (inputs[0],))

    decoder_flops = sum(flops_transformer_decoderlayer(layer, decoder_inputs) for layer in module.decoder.layers)

    if isinstance(module.decoder.norm, nn.LayerNorm):
        decoder_flops += flops_layernorm(module.decoder.norm, (inputs[1],))

    return encoder_flops + decoder_flops
