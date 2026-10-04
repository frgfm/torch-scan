# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Module-local token dependencies for native attention and Transformer stacks.

Relations describe potential dependence at generic finite parameters. They do not
describe a model graph, token positions introduced by callers, or spatial fields.
Only the main activation output is described, not returned attention weights.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from torchscan.report import TokenDependency, TokenRelation, TokenSource

from ._transformer import (
    _attention_spec,
    _decoder_layer_inputs,
    _encoder_layer_inputs,
    _validate_decoder,
    _validate_encoder,
    _validate_layer,
    _validate_norm,
    validate_native_call,
)

_KINDS = Literal["all", "same_position", "prefix", "none"]
_NATIVE_TYPES = (
    nn.MultiheadAttention,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
    nn.Transformer,
)
_ASSUMPTIONS = [
    "Module-local potential dependencies at generic finite parameters; not graph-wide effective receptive fields.",
    "Normalization acts on the feature dimension; feed-forward activation and projections are token-local.",
    "Only the main activation output is described; returned attention weights are excluded.",
]


@dataclass(frozen=True)
class _Relation:
    """Compact token-axis relation from output positions to input positions."""

    kind: _KINDS
    output_length: int
    source_length: int
    first_position: int = 0
    limit: int | None = None

    @property
    def cap(self) -> int:
        return self.source_length if self.limit is None else self.limit

    @property
    def span(self) -> int:
        if self.kind == "none":
            return 0
        if self.kind in ("prefix", "same_position"):
            return min(self.output_length, self.cap)
        return self.cap

    def description(self) -> TokenRelation:
        relation: TokenRelation = {"kind": self.kind}
        if self.first_position:
            relation["first_position"] = self.first_position
        if self.limit is not None and self.limit < self.source_length:
            relation["limit"] = self.limit
        return relation


def _slot(inputs: tuple[Any, ...], position: int, default: Any = None) -> Any:
    return inputs[position] if len(inputs) > position else default


def _local(length: int) -> _Relation:
    return _Relation("same_position", length, length)


def _none(output_length: int, source_length: int) -> _Relation:
    return _Relation("none", output_length, source_length)


def _union(left: _Relation, right: _Relation) -> _Relation:
    if (left.output_length, left.source_length) != (right.output_length, right.source_length):
        raise NotImplementedError("Token dependency aliases require matching token axes.")
    if left.kind == "none":
        return right
    if right.kind == "none":
        return left
    first = min(left.first_position, right.first_position)
    if left.kind == right.kind:
        return _Relation(left.kind, left.output_length, left.source_length, first, max(left.cap, right.cap))
    if right.kind == "all":
        left, right = right, left
    if left.kind == "all" and left.cap >= right.span:
        return _Relation("all", left.output_length, left.source_length, first, left.cap)
    if right.kind == "same_position":
        left, right = right, left
    if left.kind == "same_position" and right.kind == "prefix" and left.source_length == left.output_length:
        return _Relation("prefix", left.output_length, left.source_length, first, right.cap)
    raise NotImplementedError("This token dependency composition is outside the compact relation boundary.")


def _compose(outer: _Relation, inner: _Relation) -> _Relation:
    """Compose dependency relations without treating module order as a graph."""
    if outer.source_length != inner.output_length:
        raise NotImplementedError("Native token dependency composition requires matching sequence dimensions.")
    if outer.kind == "none" or inner.kind == "none":
        return _none(outer.output_length, inner.source_length)
    first = max(outer.first_position, inner.first_position)
    if outer.kind == "same_position":
        return _Relation(inner.kind, outer.output_length, inner.source_length, first, inner.limit)
    if inner.kind == "same_position":
        return _Relation(outer.kind, outer.output_length, inner.source_length, first, outer.limit)
    if inner.kind == "all":
        return _Relation("all", outer.output_length, inner.source_length, first, inner.cap)
    if outer.kind == "all" and inner.kind == "prefix":
        return _Relation(
            "all", outer.output_length, inner.source_length, outer.first_position, min(outer.cap, inner.cap)
        )
    if outer.kind == "prefix" and inner.kind == "prefix":
        return _Relation("prefix", outer.output_length, inner.source_length, first, min(outer.cap, inner.cap))
    raise NotImplementedError("This token dependency composition is outside the compact relation boundary.")


def _padding_is_local(mask: Tensor | None) -> None:
    if mask is None:
        return
    if mask.device.type == "meta":
        raise NotImplementedError("Token dependencies require observable mask values; meta masks are unsupported.")
    if mask.dtype == torch.bool:
        excludes_tokens = bool(mask.any())
    else:
        if bool((torch.isnan(mask) | torch.isposinf(mask)).any()):
            raise NotImplementedError("Token dependencies do not support NaN or positive-infinity masks.")
        excludes_tokens = bool(torch.isneginf(mask).any())
    if excludes_tokens:
        raise NotImplementedError("Token dependencies do not support token-excluding key padding masks.")


def _mask_kind(mask: Tensor | None, target: int, source: int, is_causal: bool) -> Literal["all", "prefix"]:
    if mask is None:
        if is_causal:
            raise NotImplementedError("Token dependencies require an explicit canonical mask with a causal hint.")
        return "all"
    if mask.device.type == "meta":
        raise NotImplementedError("Token dependencies require observable mask values; meta masks are unsupported.")
    if mask.dtype == torch.bool:
        blocked = mask
    else:
        if bool((torch.isnan(mask) | torch.isposinf(mask)).any()):
            raise NotImplementedError("Token dependencies do not support NaN or positive-infinity masks.")
        blocked = torch.isneginf(mask)
    expected = torch.arange(source, device=mask.device)[None, :] > torch.arange(target, device=mask.device)[:, None]
    canonical = torch.equal(blocked, expected.expand_as(blocked))
    if is_causal and not canonical:
        raise NotImplementedError("A causal hint must agree with an explicit canonical causal mask.")
    if not bool(blocked.any()):
        return "all"
    if canonical:
        return "prefix"
    raise NotImplementedError("Token dependencies only support finite additive masks or canonical causal exclusions.")


def _attention_relations(
    module: nn.MultiheadAttention, inputs: tuple[Any, ...]
) -> tuple[_Relation, _Relation, _Relation]:
    spec = _attention_spec(module, inputs)
    _padding_is_local(spec.key_padding_mask)
    kind = _mask_kind(spec.attn_mask, spec.target, spec.source, spec.is_causal)
    # A softmax over one visible key is constant, so Q and K cannot influence
    # the activation output. V still contributes, including causal position 0.
    if spec.source == 1:
        query = _none(spec.target, spec.target)
        key = _none(spec.target, spec.source)
    else:
        first = int(kind == "prefix")
        query = _Relation("same_position", spec.target, spec.target, first)
        key = _Relation(kind, spec.target, spec.source, first)
    value = _Relation(kind, spec.target, spec.source)
    return query, key, value


def _encoder_layer(module: nn.TransformerEncoderLayer, inputs: tuple[Any, ...]) -> _Relation:
    _validate_layer(module)
    _validate_dependency_norm(module.norm1, module.self_attn.embed_dim)
    _validate_dependency_norm(module.norm2, module.self_attn.embed_dim)
    query, key, value = _attention_relations(module.self_attn, _encoder_layer_inputs(inputs))
    # Residual paths preserve token-local dependency even at a one-key softmax.
    return _union(_local(query.output_length), _union(query, _union(key, value)))


def _decoder_layer(module: nn.TransformerDecoderLayer, inputs: tuple[Any, ...]) -> tuple[_Relation, _Relation]:
    _validate_layer(module)
    for norm in (module.norm1, module.norm2, module.norm3):
        _validate_dependency_norm(norm, module.self_attn.embed_dim)
    self_inputs, cross_inputs = _decoder_layer_inputs(inputs)
    self_query, self_key, self_value = _attention_relations(module.self_attn, self_inputs)
    target = _union(_local(self_query.output_length), _union(self_query, _union(self_key, self_value)))
    _, memory_key, memory_value = _attention_relations(module.multihead_attn, cross_inputs)
    return target, _union(memory_key, memory_value)


def _encoder(module: nn.TransformerEncoder, inputs: tuple[Any, ...]) -> _Relation:
    _validate_encoder(module, inputs)
    src = inputs[0]
    first_layer = module.layers[0]
    _validate_layer(first_layer)
    axis = 1 if first_layer.self_attn.batch_first else 0
    relation = _local(src.shape[axis])
    for layer in module.layers:
        layer_relation = _encoder_layer(layer, (src, _slot(inputs, 1), _slot(inputs, 2), _slot(inputs, 3, False)))
        relation = _compose(layer_relation, relation)
    _validate_dependency_norm(module.norm, first_layer.self_attn.embed_dim)
    return relation


def _decoder(module: nn.TransformerDecoder, inputs: tuple[Any, ...]) -> tuple[_Relation, _Relation]:
    _validate_decoder(module)
    tgt, memory = inputs[:2]
    first_layer = module.layers[0]
    _validate_layer(first_layer)
    axis = 1 if first_layer.self_attn.batch_first else 0
    target = _local(tgt.shape[axis])
    source = _none(tgt.shape[axis], memory.shape[axis])
    for layer in module.layers:
        layer_target, layer_source = _decoder_layer(layer, inputs)
        target = _compose(layer_target, target)
        source = _union(_compose(layer_target, source), layer_source)
    _validate_dependency_norm(module.norm, first_layer.self_attn.embed_dim)
    return target, source


def _source(arguments: list[str], axis: int, relation: _Relation) -> TokenSource:
    return {
        "arguments": arguments,
        "sequence_axis": axis,
        "length": relation.source_length,
        "relation": relation.description(),
    }


def _validate_dependency_norm(norm: nn.Module | None, width: int) -> None:
    _validate_norm(norm, width)
    if type(norm) is nn.LayerNorm and width == 1:
        raise NotImplementedError("Token dependencies do not support width-one LayerNorm, whose output is constant.")


def module_token_dependencies(module: nn.Module, inputs: tuple[Any, ...], _output: Any = None) -> TokenDependency:
    """Describe dependencies of one native module call's main activation output.

    Args:
        module: Exact native attention or Transformer module.
        inputs: Complete forward arguments in signature order, including defaults.
        _output: Actual module output; reserved for the shared formula call contract.
    """
    if type(module) not in _NATIVE_TYPES:
        raise NotImplementedError("Token dependencies require exact native attention or Transformer types.")
    if isinstance(module, nn.MultiheadAttention):
        axis = 1 if module.batch_first else 0
        relations = _attention_relations(module, inputs)
        groups: dict[int, tuple[list[str], _Relation]] = {}
        for name, tensor, relation in zip(("query", "key", "value"), inputs[:3], relations, strict=True):
            identity = id(tensor)
            if identity in groups:
                arguments, earlier = groups[identity]
                groups[identity] = ([*arguments, name], _union(earlier, relation))
            else:
                groups[identity] = ([name], relation)
        sources = [_source(arguments, axis, relation) for arguments, relation in groups.values()]
        length = relations[0].output_length
    elif isinstance(module, nn.TransformerEncoderLayer):
        axis = 1 if module.self_attn.batch_first else 0
        relation = _encoder_layer(module, inputs)
        sources = [_source(["src"], axis, relation)]
        length = relation.output_length
    elif isinstance(module, nn.TransformerDecoderLayer):
        axis = 1 if module.self_attn.batch_first else 0
        target, source = _decoder_layer(module, inputs)
        sources = [_source(["tgt"], axis, target), _source(["memory"], axis, source)]
        length = target.output_length
    elif isinstance(module, nn.TransformerEncoder):
        relation = _encoder(module, inputs)
        axis = 1 if module.layers[0].self_attn.batch_first else 0
        sources = [_source(["src"], axis, relation)]
        length = relation.output_length
    elif isinstance(module, nn.TransformerDecoder):
        target, source = _decoder(module, inputs)
        axis = 1 if module.layers[0].self_attn.batch_first else 0
        sources = [_source(["tgt"], axis, target), _source(["memory"], axis, source)]
        length = target.output_length
    else:
        validate_native_call(module, inputs, _output)
        if type(module.encoder) is not nn.TransformerEncoder or type(module.decoder) is not nn.TransformerDecoder:
            raise NotImplementedError("Transformer token dependencies require native encoder and decoder stacks.")
        axis = 1 if module.batch_first else 0
        encoder = _encoder(module.encoder, (inputs[0], _slot(inputs, 2), _slot(inputs, 5), _slot(inputs, 8)))
        target, memory = _decoder(
            module.decoder,
            (
                inputs[1],
                inputs[0],
                _slot(inputs, 3),
                _slot(inputs, 4),
                _slot(inputs, 6),
                _slot(inputs, 7),
                _slot(inputs, 9),
                _slot(inputs, 10, False),
            ),
        )
        source = _compose(memory, encoder)
        sources = [_source(["tgt"], axis, target), _source(["src"], axis, source)]
        length = target.output_length
    return {
        "status": "complete",
        "scope": "module_call",
        "method": "torchscan_token_dependency_v1",
        "output": {"sequence_axis": axis, "length": length},
        "sources": sources,
        "assumptions": list(_ASSUMPTIONS),
    }
