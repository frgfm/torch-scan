# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
import warnings
from typing import Any, Callable, Tuple, cast

from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd

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
    if isinstance(module, nn.Transformer):
        return flops_transformer(module, inputs)
    warnings.warn(f"Module type not supported: {module.__class__.__name__}", stacklevel=1)
    return 0


def flops_linear(module: nn.Linear, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Linear`"""
    # batch size * out_chan * in_chan
    num_out_feats = module.out_features * math.prod(inputs[0].shape[:-1])
    mm_flops = num_out_feats * (2 * module.in_features - 1)
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
    if module.p > 0:
        # Sample a random number for each input element
        return inputs[0].numel()
    return 0


def flops_convtransposend(module: _ConvTransposeNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.conv._ConvTranposeNd`"""
    # Padding (# cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532)
    # Define min and max sizes
    padding_flops = len(module.kernel_size) * 8

    # Once padding is determined, the operations are almost identical to those of a convolution
    conv_flops = flops_convnd(module, inputs, out)

    return padding_flops + conv_flops


def flops_convnd(module: _ConvNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.conv._ConvNd`"""
    # For each position, # mult = kernel size, # adds = kernel size - 1
    window_flops_per_chan = 2 * math.prod(module.kernel_size) - 1
    # Connections to input channels is controlled by the group parameter
    effective_in_chan = inputs[0].shape[1] // module.groups
    # N * flops + (N - 1) additions
    window_flops = effective_in_chan * window_flops_per_chan + (effective_in_chan - 1)
    conv_flops = out.numel() * window_flops

    # Each output element gets a bias addition
    bias_flops = out.numel() if module.bias is not None else 0

    return conv_flops + bias_flops


def flops_bn(module: _BatchNorm, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""
    # for each channel, add eps and running_var, sqrt it
    norm_ops = module.num_features * 2
    # For each element, sub running_mean, div by denom
    norm_ops += inputs[0].numel() * 2
    # For each element, mul by gamma, add beta
    scale_ops = inputs[0].numel() * 2 if module.affine else 0
    bn_flops = norm_ops + scale_ops

    # Count tracking stats update ops
    # cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L94-L101
    tracking_flops = 0
    if module.track_running_stats and module.training:
        # exponential_average_factor
        if module.momentum is None:
            tracking_flops += 1
        # running_mean: by channel, sum values and div by batch size
        tracking_flops += inputs[0].numel()
        # running_var: by channel, sub mean and square values, sum them, divide by batch size
        tracking_flops += 3 * inputs[0].numel()
        # Update both runnning stat: rescale previous value (mul by N), add it the new one, then div by (N + 1)
        tracking_flops += 2 * module.num_features * 3

    return bn_flops + tracking_flops


def flops_maxpool(module: _MaxPoolNd, _: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._MaxPoolNd`"""
    kernel_size = module.kernel_size
    k_size = math.prod(kernel_size) if isinstance(kernel_size, tuple) else kernel_size

    # for each spatial output element, check max element in kernel scope
    return out.numel() * (k_size - 1)


def flops_avgpool(module: _AvgPoolNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AvgPoolNd`"""
    kernel_size = cast(int | Tuple[int, ...], module.kernel_size)
    k_size = math.prod(kernel_size) if isinstance(kernel_size, tuple) else kernel_size

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return out.numel() * (k_size - 1 + inputs[0].ndim - 2)


def flops_adaptive_maxpool(_: _AdaptiveMaxPoolNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveMaxPoolNd`"""
    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(inputs[0].shape[2:], out.shape[2:], strict=False)
    )

    # for each spatial output element, check max element in kernel scope
    return out.numel() * (math.prod(kernel_size) - 1)


def flops_adaptive_avgpool(_: _AdaptiveAvgPoolNd, inputs: Tuple[Tensor, ...], out: Tensor) -> int:
    """FLOPs estimation for `torch.nn.modules.pooling._AdaptiveAvgPoolNd`"""
    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(inputs[0].shape[2:], out.shape[2:], strict=False)
    )

    # for each spatial output element, sum elements in kernel scope and div by kernel size
    return out.numel() * (math.prod(kernel_size) - 1 + len(kernel_size))


def flops_layernorm(module: nn.LayerNorm, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.LayerNorm`"""
    numel = inputs[0].numel()
    rows = numel // math.prod(module.normalized_shape)
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
    projection_bias = int(module.in_proj_bias is not None)

    tot_flops = sum(
        math.prod(tensor.shape[:-1]) * module.embed_dim * (2 * tensor.shape[-1] - 1 + projection_bias)
        for tensor in (q, k, v)
    )
    tot_flops += 1 + batch_size * module.num_heads * target_length * module.head_dim
    tot_flops += batch_size * module.num_heads * target_length * source_length * (2 * module.head_dim - 1)

    visible_masks = int(len(inputs) > 3 and inputs[3] is not None) + int(len(inputs) > 5 and inputs[5] is not None)
    tot_flops += visible_masks * batch_size * module.num_heads * target_length * source_length
    tot_flops += batch_size * module.num_heads * target_length * (3 * source_length - 1)
    if module.training and module.dropout > 0:
        tot_flops += batch_size * module.num_heads * target_length * source_length
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
    dropout_flops = num_hidden if module.dropout.training and module.dropout.p > 0 else 0
    return flops_linear(module.linear1, inputs) + num_hidden + dropout_flops + flops_linear(module.linear2, inputs)


def flops_transformer_encoderlayer(module: nn.TransformerEncoderLayer, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.TransformerEncoderLayer`"""
    input_flops = inputs[0].numel()
    tot_flops = flops_mha(module.self_attn, (inputs[0],) * 3)

    tot_flops += (flops_dropout(module.dropout1, inputs) if module.dropout1.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm1, inputs)
    tot_flops += flops_transformer_feedforward(module, inputs)
    tot_flops += (flops_dropout(module.dropout2, inputs) if module.dropout2.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm2, inputs)

    return tot_flops


def flops_transformer_decoderlayer(module: nn.TransformerDecoderLayer, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.TransformerDecoderLayer`"""
    input_flops = inputs[0].numel()
    tot_flops = flops_mha(module.self_attn, (inputs[0],) * 3)

    tot_flops += (flops_dropout(module.dropout1, inputs) if module.dropout1.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm1, inputs)

    tot_flops += flops_mha(module.multihead_attn, (inputs[0], inputs[1], inputs[1]))
    tot_flops += (flops_dropout(module.dropout2, inputs) if module.dropout2.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm2, inputs)

    tot_flops += flops_transformer_feedforward(module, inputs)
    tot_flops += (flops_dropout(module.dropout3, inputs) if module.dropout3.training else 0) + input_flops
    tot_flops += flops_layernorm(module.norm3, inputs)

    return tot_flops


def flops_transformer(module: nn.Transformer, inputs: Tuple[Tensor, ...]) -> int:
    """FLOPs estimation for `torch.nn.Transformer`"""
    if not isinstance(module.encoder, nn.TransformerEncoder) or not isinstance(module.decoder, nn.TransformerDecoder):
        raise NotImplementedError("Transformer FLOPs only support the native encoder and decoder stacks.")

    src_inputs = (inputs[0],)
    decoder_inputs = (inputs[1], inputs[0])
    encoder_flops = sum(flops_transformer_encoderlayer(layer, src_inputs) for layer in module.encoder.layers)

    if isinstance(module.encoder.norm, nn.LayerNorm):
        encoder_flops += flops_layernorm(module.encoder.norm, src_inputs)

    decoder_flops = sum(flops_transformer_decoderlayer(layer, decoder_inputs) for layer in module.decoder.layers)

    if isinstance(module.decoder.norm, nn.LayerNorm):
        decoder_flops += flops_layernorm(module.decoder.norm, (inputs[1],))

    return encoder_flops + decoder_flops
