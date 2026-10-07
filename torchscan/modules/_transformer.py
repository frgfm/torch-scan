# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Private dense native-Transformer formulas, independent of FLOP formulas.

Inputs follow the native forward signature, including default-valued argument slots.
MACs count matrix contraction terms, variance square-sum terms, and affine
one-term products. Bias additions, reductions without products, scale/divide,
softmax, residual addition, and activation arithmetic add no MACs.

DMAs count element reads/writes in a logical staged algorithm. Each matrix stage
reads its operands and parameters once and writes its result once. Views,
evaluation dropout, and output aliases add no accesses. This is neither measured
traffic nor a model of any fused kernel. See the methodology for stage derivations.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionSpec:
    """Validated shapes and options for one dense batched attention call."""

    query: Tensor
    key: Tensor
    value: Tensor
    batch: int
    target: int
    source: int
    embed: int
    heads: int
    key_padding_mask: Tensor | None
    attn_mask: Tensor | None
    need_weights: bool
    average_attn_weights: bool
    is_causal: bool


def _slot(inputs: tuple[Any, ...], index: int, default: Any = None) -> Any:
    return inputs[index] if len(inputs) > index else default


def _dense_tensor(value: Any) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.is_nested
        or value.layout != torch.strided
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise NotImplementedError("Module estimates require real floating-point dense strided tensors.")
    return value


def _mask(value: Any, shapes: tuple[tuple[int, ...], ...]) -> Tensor | None:
    if value is None:
        return None
    if (
        not isinstance(value, Tensor)
        or value.is_nested
        or value.layout != torch.strided
        or not (value.dtype == torch.bool or value.is_floating_point())
        or tuple(value.shape) not in shapes
    ):
        raise NotImplementedError(
            "Transformer masks must be dense boolean or floating tensors with native mask shapes."
        )
    return value


def _parameter(value: Any, shape: tuple[int, ...]) -> None:
    if (
        not isinstance(value, Tensor)
        or value.is_nested
        or value.layout != torch.strided
        or not value.is_floating_point()
        or value.is_complex()
        or tuple(value.shape) != shape
    ):
        raise NotImplementedError("Transformer parameters must be real dense floating tensors with native shapes.")


def _validate_native_forward(module: nn.Module) -> None:
    forward = module.forward
    if (
        getattr(forward, "__self__", None) is not module
        or getattr(forward, "__func__", None) is not type(module).forward
    ):
        raise NotImplementedError("Transformer estimates require unchanged native forward implementations.")


def validate_native_attention(module: nn.MultiheadAttention) -> None:
    """Validate native attention operands independently of input-shape boundaries."""
    if type(module) is not nn.MultiheadAttention:
        raise NotImplementedError("Transformer estimates require the exact native MultiheadAttention type.")
    _validate_native_forward(module)
    if module.training:
        raise NotImplementedError("Transformer MAC/DMA estimates describe evaluation calls only.")
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise NotImplementedError("Transformer estimates do not support add_bias_kv or add_zero_attn.")
    if type(module.out_proj) not in (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear):
        raise NotImplementedError("Transformer attention requires a native Linear output projection.")
    embed = module.embed_dim
    if module.num_heads <= 0 or embed % module.num_heads or module.head_dim != embed // module.num_heads:
        raise NotImplementedError("Transformer attention head metadata must match native dimensions.")
    if module.out_proj.in_features != embed or module.out_proj.out_features != embed:
        raise NotImplementedError("Transformer output-projection metadata must match native shapes.")
    _parameter(module.out_proj.weight, (embed, embed))
    if module.out_proj.bias is not None:
        _parameter(module.out_proj.bias, (embed,))
    if module._qkv_same_embed_dim:
        _parameter(module.in_proj_weight, (3 * embed, embed))
    else:
        _parameter(module.q_proj_weight, (embed, embed))
        _parameter(module.k_proj_weight, (embed, module.kdim))
        _parameter(module.v_proj_weight, (embed, module.vdim))
    if module.in_proj_bias is not None:
        _parameter(module.in_proj_bias, (3 * embed,))


def _attention_spec(module: nn.MultiheadAttention, inputs: tuple[Any, ...]) -> AttentionSpec:
    validate_native_attention(module)
    if len(inputs) < 3:
        raise NotImplementedError("MultiheadAttention estimates require complete query, key, and value arguments.")
    query, key, value = (_dense_tensor(item) for item in inputs[:3])
    if any(item.ndim != 3 for item in (query, key, value)):
        raise NotImplementedError("Transformer estimates support batched 3D tensors only.")
    batch_dim, token_dim = (0, 1) if module.batch_first else (1, 0)
    batch, target, source = query.shape[batch_dim], query.shape[token_dim], key.shape[token_dim]
    if batch == 0 or target == 0 or source == 0:
        raise NotImplementedError("Transformer estimates require nonempty batches and token axes.")
    if (
        key.shape[batch_dim] != batch
        or value.shape[batch_dim] != batch
        or value.shape[token_dim] != source
        or query.shape[-1] != module.embed_dim
        or key.shape[-1] != module.kdim
        or value.shape[-1] != module.vdim
    ):
        raise NotImplementedError("Transformer estimates require matching native attention dimensions.")
    need_weights, average_attn_weights, is_causal = (
        _slot(inputs, 4, True),
        _slot(inputs, 6, True),
        _slot(inputs, 7, False),
    )
    if any(type(option) is not bool for option in (need_weights, average_attn_weights, is_causal)):
        raise NotImplementedError("Transformer attention options must be native boolean values.")
    padding = _mask(_slot(inputs, 3), ((batch, source),))
    attn_mask = _mask(_slot(inputs, 5), ((target, source), (batch * module.num_heads, target, source)))
    if is_causal and attn_mask is None:
        raise NotImplementedError(
            "A causal hint without an explicit attention mask has ambiguous native fast-path behavior."
        )
    return AttentionSpec(
        query,
        key,
        value,
        batch,
        target,
        source,
        module.embed_dim,
        module.num_heads,
        padding,
        attn_mask,
        need_weights,
        average_attn_weights,
        is_causal,
    )


def _validate_norm(norm: nn.Module | None, embed: int, *, final: bool = False) -> None:
    if norm is None:
        return
    if type(norm) is nn.Identity:
        _validate_native_forward(norm)
        return
    if type(norm) is not nn.LayerNorm or tuple(norm.normalized_shape) != (embed,):
        context = "final normalization" if final else "normalization"
        raise NotImplementedError(f"Transformer {context} must be token-local native LayerNorm, Identity, or None.")
    _validate_native_forward(norm)
    for parameter in (norm.weight, norm.bias):
        if parameter is not None:
            _parameter(parameter, (embed,))


def _validate_layer(module: nn.TransformerEncoderLayer | nn.TransformerDecoderLayer) -> None:
    if type(module) not in (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer):
        raise NotImplementedError("Transformer estimates require exact native encoder/decoder layer types.")
    _validate_native_forward(module)
    if module.training:
        raise NotImplementedError("Transformer MAC/DMA estimates describe evaluation calls only.")
    if type(module.self_attn) is not nn.MultiheadAttention:
        raise NotImplementedError("Transformer layers require exact native MultiheadAttention children.")
    embed = module.self_attn.embed_dim
    if type(module) is nn.TransformerDecoderLayer and (
        type(module.multihead_attn) is not nn.MultiheadAttention
        or module.multihead_attn.embed_dim != embed
        or module.multihead_attn.batch_first != module.self_attn.batch_first
    ):
        raise NotImplementedError(
            "Transformer decoder attention children must share native embedding widths and layouts."
        )
    if (
        type(module.linear1) is not nn.Linear
        or type(module.linear2) is not nn.Linear
        or module.linear1.in_features != embed
        or module.linear2.out_features != embed
        or module.linear1.out_features != module.linear2.in_features
        or module.activation not in (F.relu, F.gelu)
    ):
        raise NotImplementedError("Transformer estimates require native feed-forward Linear layers and ReLU or GELU.")
    for linear in (module.linear1, module.linear2):
        _validate_native_forward(linear)
        _parameter(linear.weight, (linear.out_features, linear.in_features))
        if linear.bias is not None:
            _parameter(linear.bias, (linear.out_features,))
    norms = [module.norm1, module.norm2]
    if type(module) is nn.TransformerDecoderLayer:
        norms.append(module.norm3)
    for norm in norms:
        _validate_norm(norm, embed)
    dropouts = [module.dropout, module.dropout1, module.dropout2]
    if type(module) is nn.TransformerDecoderLayer:
        dropouts.append(module.dropout3)
    if any(type(dropout) is not nn.Dropout or dropout.training for dropout in dropouts):
        raise NotImplementedError("Transformer estimates require native evaluation-mode dropout stages.")
    for dropout in dropouts:
        _validate_native_forward(dropout)


def _encoder_layer_inputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    src = _slot(inputs, 0)
    return src, src, src, _slot(inputs, 2), False, _slot(inputs, 1), True, _causal_option(_slot(inputs, 3, False))


def _causal_option(value: Any) -> bool:
    if value is None:
        # Native stacks infer the causal hint from the explicit mask. It already
        # contributes one mask stage regardless of the inferred hint's value.
        return False
    if type(value) is not bool:
        raise NotImplementedError("Transformer causal hints must be native boolean values or None for stack inference.")
    return value


def _decoder_layer_inputs(inputs: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    target, memory = _slot(inputs, 0), _slot(inputs, 1)
    return (
        (
            target,
            target,
            target,
            _slot(inputs, 4),
            False,
            _slot(inputs, 2),
            True,
            _causal_option(_slot(inputs, 6, False)),
        ),
        (
            target,
            memory,
            memory,
            _slot(inputs, 5),
            False,
            _slot(inputs, 3),
            True,
            _causal_option(_slot(inputs, 7, False)),
        ),
    )


def _validate_encoder(module: nn.TransformerEncoder, inputs: tuple[Any, ...]) -> None:
    if type(module) is not nn.TransformerEncoder or len(module.layers) == 0:
        raise NotImplementedError("Transformer estimates require a nonempty exact native encoder stack.")
    _validate_native_forward(module)
    if module.training:
        raise NotImplementedError("Transformer MAC/DMA estimates describe evaluation calls only.")
    if getattr(module, "use_nested_tensor", False) and _slot(inputs, 2) is not None:
        raise NotImplementedError(
            "Padding-mask encoder estimates require enable_nested_tensor=False to exclude packing."
        )
    _validate_stack_layers(module.layers, nn.TransformerEncoderLayer)
    _validate_norm(module.norm, module.layers[0].self_attn.embed_dim, final=True)


def _validate_decoder(module: nn.TransformerDecoder) -> None:
    if type(module) is not nn.TransformerDecoder or len(module.layers) == 0:
        raise NotImplementedError("Transformer estimates require a nonempty exact native decoder stack.")
    _validate_native_forward(module)
    if module.training:
        raise NotImplementedError("Transformer MAC/DMA estimates describe evaluation calls only.")
    _validate_stack_layers(module.layers, nn.TransformerDecoderLayer)
    _validate_norm(module.norm, module.layers[0].self_attn.embed_dim, final=True)


def _validate_stack_layers(layers: nn.ModuleList, layer_type: type[nn.Module]) -> None:
    if any(type(layer) is not layer_type for layer in layers):
        raise NotImplementedError("Transformer stacks require exact native layer children.")
    native_layers = cast(list[nn.TransformerEncoderLayer | nn.TransformerDecoderLayer], list(layers))
    for layer in native_layers:
        _validate_layer(layer)
    first = native_layers[0].self_attn
    if any(
        layer.self_attn.batch_first != first.batch_first or layer.self_attn.embed_dim != first.embed_dim
        for layer in native_layers[1:]
    ):
        raise NotImplementedError("Transformer stack layers must share embedding widths and batch layouts.")


def _mha_macs(module: nn.MultiheadAttention, inputs: tuple[Any, ...]) -> int:
    spec = _attention_spec(module, inputs)
    # Projections: each output coordinate contracts the corresponding input width.
    projections = spec.embed * (spec.query.numel() + spec.key.numel() + spec.value.numel())
    # QK^T and AV: H heads x D=E/H terms, or S terms per value output.
    attention_products = 2 * spec.batch * spec.target * spec.source * spec.embed
    output_projection = spec.batch * spec.target * spec.embed * spec.embed
    return projections + attention_products + output_projection


def _mha_dmas(module: nn.MultiheadAttention, inputs: tuple[Any, ...]) -> int:
    spec = _attention_spec(module, inputs)
    query = spec.batch * spec.target * spec.embed
    key_value = spec.batch * spec.source * spec.embed
    scores = spec.batch * spec.heads * spec.target * spec.source
    rows = spec.batch * spec.heads * spec.target
    input_reads = spec.query.numel() + spec.key.numel() + spec.value.numel()
    # Each projection reads its logical parameter operand exactly once.
    parameters = spec.embed * sum(item.shape[-1] for item in (spec.query, spec.key, spec.value))
    parameters += spec.embed * spec.embed
    parameters += 3 * spec.embed if module.in_proj_bias is not None else 0
    parameters += spec.embed if module.out_proj.bias is not None else 0
    # Projection results, Q scale, QK, staged stable softmax, AV, output projection.
    count = input_reads + parameters + 7 * query + 4 * key_value + 10 * scores + 4 * rows
    # A boolean or numeric mask is one stored logical operand; applying it reads
    # and writes the score matrix. No implementation-specific conversion buffer.
    count += sum(mask.numel() + 2 * scores for mask in (spec.key_padding_mask, spec.attn_mask) if mask is not None)
    if spec.need_weights and spec.average_attn_weights:
        count += scores + spec.batch * spec.target * spec.source
    # Unaveraged weights return the already counted softmax result by alias.
    return count


def _norm_macs(norm: nn.Module | None, tensor: Tensor) -> int:
    if type(norm) is nn.LayerNorm:
        return tensor.numel() * (1 + int(norm.weight is not None))
    return 0


def _norm_dmas(norm: nn.Module | None, tensor: Tensor) -> int:
    if type(norm) is not nn.LayerNorm:
        return 0
    elements = tensor.numel()
    rows = elements // math.prod(norm.normalized_shape)
    parameters = sum(parameter.numel() for parameter in (norm.weight, norm.bias) if parameter is not None)
    # Mean: N+R. Variance: N+R+R. Normalize: N+2R+1+N.
    # Affine, when present: N+parameters+N. Statistics are logical intermediates.
    affine = norm.weight is not None or norm.bias is not None
    return 4 * elements + 5 * rows + 1 + parameters + (2 * elements if affine else 0)


def _feedforward_macs(module: nn.TransformerEncoderLayer | nn.TransformerDecoderLayer, tensor: Tensor) -> int:
    rows = math.prod(tensor.shape[:-1])
    return rows * (
        module.linear1.in_features * module.linear1.out_features
        + module.linear2.in_features * module.linear2.out_features
    )


def _feedforward_dmas(module: nn.TransformerEncoderLayer | nn.TransformerDecoderLayer, tensor: Tensor) -> int:
    hidden = math.prod(tensor.shape[:-1]) * module.linear1.out_features
    parameters = sum(
        parameter.numel()
        for linear in (module.linear1, module.linear2)
        for parameter in (linear.weight, linear.bias)
        if parameter is not None
    )
    # Two linears read/write N+hidden each; ReLU/GELU reads and writes hidden.
    return 2 * tensor.numel() + 4 * hidden + parameters


def _layer_count(module: nn.Module, inputs: tuple[Any, ...], *, dmas: bool) -> int:
    module = cast(nn.TransformerEncoderLayer | nn.TransformerDecoderLayer, module)
    _validate_layer(module)
    tensor = _dense_tensor(_slot(inputs, 0))
    attention = _mha_dmas if dmas else _mha_macs
    norm_count = _norm_dmas if dmas else _norm_macs
    feedforward = _feedforward_dmas if dmas else _feedforward_macs
    if type(module) is nn.TransformerEncoderLayer:
        count = attention(module.self_attn, _encoder_layer_inputs(inputs))
        norms = [module.norm1, module.norm2]
    else:
        module = cast(nn.TransformerDecoderLayer, module)
        self_inputs, cross_inputs = _decoder_layer_inputs(inputs)
        count = attention(module.self_attn, self_inputs) + attention(module.multihead_attn, cross_inputs)
        norms = [module.norm1, module.norm2, module.norm3]
    count += feedforward(module, tensor) + sum(norm_count(norm, tensor) for norm in norms)
    # Each residual addition reads both operands and writes its result. The
    # pre/post norm order changes aliases, not this logical stage count.
    if dmas:
        count += 3 * len(norms) * tensor.numel()
    return count


def _stack_count(module: nn.Module, inputs: tuple[Any, ...], *, dmas: bool) -> int:
    module = cast(nn.TransformerEncoder | nn.TransformerDecoder, module)
    if type(module) is nn.TransformerEncoder:
        _validate_encoder(module, inputs)
    else:
        _validate_decoder(cast(nn.TransformerDecoder, module))
    count = sum(_layer_count(layer, inputs, dmas=dmas) for layer in module.layers)
    norm_count = _norm_dmas if dmas else _norm_macs
    return count + norm_count(module.norm, _dense_tensor(_slot(inputs, 0)))


def _transformer_count(module: nn.Module, inputs: tuple[Any, ...], *, dmas: bool) -> int:
    if type(module) is nn.MultiheadAttention:
        return (_mha_dmas if dmas else _mha_macs)(module, inputs)
    if type(module) in (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer):
        return _layer_count(module, inputs, dmas=dmas)
    if type(module) in (nn.TransformerEncoder, nn.TransformerDecoder):
        return _stack_count(module, inputs, dmas=dmas)
    if type(module) is not nn.Transformer:
        raise NotImplementedError("Transformer estimates require exact native PyTorch Transformer module types.")
    _validate_native_forward(module)
    if module.training:
        raise NotImplementedError("Transformer MAC/DMA estimates describe evaluation calls only.")
    src, tgt = _slot(inputs, 0), _slot(inputs, 1)
    encoder_inputs = src, _slot(inputs, 2), _slot(inputs, 5), _slot(inputs, 8)
    decoder_inputs = (
        tgt,
        src,
        _slot(inputs, 3),
        _slot(inputs, 4),
        _slot(inputs, 6),
        _slot(inputs, 7),
        _slot(inputs, 9),
        _slot(inputs, 10, False),
    )
    if type(module.encoder) is not nn.TransformerEncoder or type(module.decoder) is not nn.TransformerDecoder:
        raise NotImplementedError("Transformer estimates support the native encoder and decoder stacks only.")
    _validate_encoder(module.encoder, encoder_inputs)
    _validate_decoder(module.decoder)
    if any(
        stack.layers[0].self_attn.batch_first != module.batch_first
        or stack.layers[0].self_attn.embed_dim != module.d_model
        for stack in (module.encoder, module.decoder)
    ):
        raise NotImplementedError("Transformer native stacks must match the parent embedding width and batch layout.")
    return _stack_count(module.encoder, encoder_inputs, dmas=dmas) + _stack_count(
        module.decoder, decoder_inputs, dmas=dmas
    )


def macs_attention(module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> int:
    """Count contraction MACs for a complete dense native attention call."""
    return _transformer_count(module, inputs, dmas=False)


def dmas_attention(module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> int:
    """Count staged logical accesses for a complete dense native attention call."""
    return _transformer_count(module, inputs, dmas=True)


def validate_native_call(module: nn.Module, inputs: tuple[Any, ...], _output: Any = None) -> None:
    """Validate common dense native boundaries before applying an existing formula."""
    _transformer_count(module, inputs, dmas=False)
