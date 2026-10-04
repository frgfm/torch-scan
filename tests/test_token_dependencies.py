import inspect
from contextlib import suppress

import pytest
import torch
from torch import nn

from torchscan import crawl_module
from torchscan.modules._token_dependencies import module_token_dependencies


def _ordered(module, *args, **kwargs):
    bound = inspect.signature(module.forward).bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments.values())


def _layout(tensor, batch_first):
    return tensor if batch_first else tensor.transpose(0, 1)


def _causal(target, source=None, floating=False):
    source = target if source is None else source
    blocked = torch.ones(target, source, dtype=torch.bool).triu(1)
    return torch.zeros(target, source, dtype=torch.double).masked_fill(blocked, float("-inf")) if floating else blocked


def _dependency_blocks(function, tensor):
    # Both the independent variable and the selected output use (B, tokens, E).
    # The largest absolute derivative in each feature-to-feature block reveals
    # token influence without relying on a particular output coordinate.
    jacobian = torch.autograd.functional.jacobian(function, tensor)
    return jacobian[0, :, :, 0, :, :].abs().amax(dim=(1, 3)) > 1e-10


def _relation(report, argument):
    return next(source["relation"] for source in report["sources"] if argument in source["arguments"])


@pytest.mark.parametrize(("batch_first", "floating"), [(False, False), (True, True)])
def test_self_attention_reports_real_causal_and_unmasked_token_dependencies(batch_first, floating):
    torch.manual_seed(101)
    module = nn.MultiheadAttention(4, 2, batch_first=batch_first, dropout=0).double().eval()
    tensor = torch.randn(1, 4, 4, dtype=torch.double)
    inp = _layout(tensor, batch_first)
    unmasked = module_token_dependencies(module, _ordered(module, inp, inp, inp))
    assert unmasked["output"] == {"sequence_axis": int(batch_first), "length": 4}
    assert unmasked["sources"][0]["arguments"] == ["query", "key", "value"]
    assert _relation(unmasked, "query") == {"kind": "all"}

    def attention(variable, mask=None):
        variable = _layout(variable, batch_first)
        output = module(variable, variable, variable, attn_mask=mask, need_weights=False)[0]
        return _layout(output, batch_first)

    assert _dependency_blocks(attention, tensor).all()
    mask = _causal(4, floating=floating)
    causal = module_token_dependencies(module, _ordered(module, inp, inp, inp, attn_mask=mask, is_causal=True))
    assert _relation(causal, "query") == {"kind": "prefix"}
    assert torch.equal(_dependency_blocks(lambda value: attention(value, mask), tensor), ~_causal(4))


def test_cross_attention_distinguishes_query_from_unequal_source_tokens():
    torch.manual_seed(103)
    module = nn.MultiheadAttention(4, 2, kdim=6, vdim=6, batch_first=True).double().eval()
    query = torch.randn(1, 2, 4, dtype=torch.double)
    memory = torch.randn(1, 3, 6, dtype=torch.double)
    report = module_token_dependencies(module, _ordered(module, query, memory, memory))
    assert _relation(report, "query") == {"kind": "same_position"}
    assert _relation(report, "key") == {"kind": "all"}
    assert report["sources"][1]["arguments"] == ["key", "value"]
    assert torch.equal(_dependency_blocks(lambda q: module(q, memory, memory)[0], query), torch.eye(2).bool())
    assert _dependency_blocks(lambda kv: module(query, kv, kv)[0], memory).all()
    mask = _causal(2, 3)
    causal = module_token_dependencies(module, _ordered(module, query, memory, memory, attn_mask=mask))
    assert _relation(causal, "query") == {"kind": "same_position", "first_position": 1}
    assert _relation(causal, "key") == {"kind": "prefix"}
    assert torch.equal(
        _dependency_blocks(lambda q: module(q, memory, memory, attn_mask=mask)[0], query),
        torch.tensor([[False, False], [False, True]]),
    )
    assert torch.equal(_dependency_blocks(lambda kv: module(query, kv, kv, attn_mask=mask)[0], memory), ~mask)


def test_single_key_softmax_has_no_query_or_key_dependency():
    torch.manual_seed(107)
    module = nn.MultiheadAttention(4, 2, batch_first=True).double().eval()
    query = torch.randn(1, 2, 4, dtype=torch.double)
    key = torch.randn(1, 1, 4, dtype=torch.double)
    value = torch.randn(1, 1, 4, dtype=torch.double)
    report = module_token_dependencies(module, _ordered(module, query, key, value))
    assert _relation(report, "query") == {"kind": "none"}
    assert _relation(report, "key") == {"kind": "none"}
    assert _relation(report, "value") == {"kind": "all"}
    assert not _dependency_blocks(lambda tensor: module(tensor, key, value)[0], query).any()
    assert not _dependency_blocks(lambda tensor: module(query, tensor, value)[0], key).any()
    assert _dependency_blocks(lambda tensor: module(query, key, tensor)[0], value).all()


@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("norm_first", [False, True])
def test_native_layers_and_stacks_preserve_causal_dependencies(decoder, norm_first):
    torch.manual_seed(109)
    layer_type = nn.TransformerDecoderLayer if decoder else nn.TransformerEncoderLayer
    stack_type = nn.TransformerDecoder if decoder else nn.TransformerEncoder
    layer = layer_type(4, 2, 7, dropout=0, norm_first=norm_first, batch_first=True).double().eval()
    options = {} if decoder else {"enable_nested_tensor": False}
    stack = stack_type(layer, 2, norm=nn.LayerNorm(4).double(), **options).eval()
    tensor = torch.randn(1, 3, 4, dtype=torch.double)
    memory = torch.randn(1, 4, 4, dtype=torch.double)
    other = (memory,) if decoder else ()
    for module, mask_name in (
        (layer, "tgt_mask" if decoder else "src_mask"),
        (stack, "tgt_mask" if decoder else "mask"),
    ):
        masks = {mask_name: _causal(3, floating=True)}
        report = module_token_dependencies(module, _ordered(module, tensor, *other, **masks))
        assert _relation(report, "tgt" if decoder else "src") == {"kind": "prefix"}
        assert torch.equal(
            _dependency_blocks(lambda value, module=module, masks=masks: module(value, *other, **masks), tensor),
            ~_causal(3),
        )
        if decoder:
            assert _relation(report, "memory") == {"kind": "all"}
            assert _dependency_blocks(
                lambda value, module=module, masks=masks: module(tensor, value, **masks), memory
            ).all()


def test_unmasked_decoder_self_attention_composes_causal_memory_prefixes_across_layers():
    torch.manual_seed(127)
    layer = nn.TransformerDecoderLayer(4, 2, 7, dropout=0, batch_first=True).double()
    stack = nn.TransformerDecoder(layer, 2).eval()
    target = torch.randn(1, 2, 4, dtype=torch.double)
    memory = torch.randn(1, 4, 4, dtype=torch.double)
    mask = _causal(2, 4)
    report = module_token_dependencies(stack, _ordered(stack, target, memory, memory_mask=mask))
    assert _relation(report, "memory") == {"kind": "all", "limit": 2}
    expected = torch.tensor([[True, True, False, False], [True, True, False, False]])
    assert torch.equal(_dependency_blocks(lambda value: stack(target, value, memory_mask=mask), memory), expected)


@pytest.mark.parametrize("causal_encoder", [False, True])
def test_transformer_composes_encoder_mixing_before_causal_cross_attention(causal_encoder):
    torch.manual_seed(131)
    module = nn.Transformer(4, 2, 2, 2, 7, dropout=0, batch_first=True).double().eval()
    module.encoder.enable_nested_tensor = False
    module.encoder.use_nested_tensor = False
    src = torch.randn(1, 4, 4, dtype=torch.double)
    target = torch.randn(1, 3, 4, dtype=torch.double)
    masks = {"tgt_mask": _causal(3), "memory_mask": _causal(3, 4)}
    if causal_encoder:
        masks["src_mask"] = _causal(4)
    report = module_token_dependencies(module, _ordered(module, src, target, **masks))
    assert _relation(report, "tgt") == {"kind": "prefix"}
    assert _relation(report, "src") == {"kind": "prefix" if causal_encoder else "all"}
    dependency = _dependency_blocks(lambda value: module(value, target, **masks), src)
    assert torch.equal(dependency, ~_causal(3, 4) if causal_encoder else torch.ones(3, 4, dtype=torch.bool))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attn_mask": torch.tensor([[False, True], [True, False]])}, "canonical causal"),
        ({"key_padding_mask": torch.tensor([[False, True]])}, "key padding"),
        ({"attn_mask": torch.tensor([[0.0, float("nan")], [0.0, 0.0]])}, "NaN"),
        ({"attn_mask": torch.tensor([[0.0, float("inf")], [0.0, 0.0]])}, "positive-infinity"),
        ({"attn_mask": torch.zeros(2, 2), "is_causal": True}, "causal hint"),
        ({"is_causal": True}, "causal hint"),
        ({"attn_mask": torch.empty(2, 2, device="meta")}, "observable mask"),
    ],
)
def test_unsupported_mask_semantics_are_explicit(kwargs, message):
    module = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    tensor = torch.randn(1, 2, 4)
    with pytest.raises(NotImplementedError, match=message):
        module_token_dependencies(module, _ordered(module, tensor, tensor, tensor, **kwargs))


def test_finite_additive_and_noop_padding_masks_keep_dense_token_dependencies():
    module = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    tensor = torch.randn(2, 3, 4)
    masks = {"attn_mask": torch.randn(4, 3, 3), "key_padding_mask": torch.zeros(2, 3, dtype=torch.bool)}
    report = module_token_dependencies(module, _ordered(module, tensor, tensor, tensor, **masks))
    assert _relation(report, "query") == {"kind": "all"}


def test_token_feature_normalization_and_activation_boundaries_are_explicit():
    tensor = torch.randn(1, 3, 4)
    module = nn.TransformerEncoderLayer(4, 2, 7, batch_first=True, activation="gelu").eval()
    assert _relation(module_token_dependencies(module, _ordered(module, tensor)), "src") == {"kind": "all"}
    module.norm1 = nn.LayerNorm((3, 4))
    with pytest.raises(NotImplementedError, match="token-local"):
        module_token_dependencies(module, _ordered(module, tensor))
    module.norm1 = nn.LayerNorm(4)
    module.activation = torch.sigmoid
    with pytest.raises(NotImplementedError, match="ReLU or GELU"):
        module_token_dependencies(module, _ordered(module, tensor))


def test_width_one_layernorm_is_intrinsically_constant_and_not_reported_as_generic_dependency():
    module = nn.TransformerEncoderLayer(1, 1, 2, dropout=0, batch_first=True).double().eval()
    tensor = torch.randn(1, 3, 1, dtype=torch.double)
    assert not _dependency_blocks(module, tensor).any()
    with pytest.raises(NotImplementedError, match="width-one LayerNorm"):
        module_token_dependencies(module, _ordered(module, tensor))


def test_distinct_views_are_not_assumed_to_share_a_token_argument_identity():
    module = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    tensor = torch.randn(1, 3, 4)
    key, value = tensor.view_as(tensor), tensor.view_as(tensor)
    report = module_token_dependencies(module, _ordered(module, tensor, key, value))
    assert [source["arguments"] for source in report["sources"]] == [["query"], ["key"], ["value"]]
    assert _relation(report, "query") == {"kind": "same_position"}
    assert _relation(report, "key") == {"kind": "all"}


def test_causal_per_head_masks_have_the_same_token_relation_as_broadcast_masks():
    module = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    tensor = torch.randn(2, 3, 4)
    mask = _causal(3).expand(4, 3, 3)
    report = module_token_dependencies(module, _ordered(module, tensor, tensor, tensor, attn_mask=mask))
    assert _relation(report, "query") == {"kind": "prefix"}


def test_caught_native_forward_failure_has_unavailable_tokens_and_no_dense_upper_bound():
    class Recovery(nn.Sequential):
        def forward(self, source):
            with suppress(RuntimeError):
                self[0](source.double())  # Shape-valid input cannot multiply FP32 projection parameters.
            return self[1](source)

    module = Recovery(nn.TransformerEncoderLayer(4, 2, 7, dropout=0), nn.Identity())
    source = torch.randn(3, 1, 4)
    report = crawl_module(module, args=(source,))
    failed = next(layer for layer in report["layers"] if layer["path"] == "0")
    assert failed["output"]["kind"] == "failed"
    assert failed["token_dependencies"]["status"] == "unavailable"
    for metric in ("module_flops", "macs", "dmas"):
        assert failed["metrics"][metric]["status"] == "unavailable"
        assert failed["metrics"][metric]["known_value"] is None
        assert report["totals"][metric]["status"] == "partial"
        assert report["totals"][metric]["known_value"] == (source.numel() if metric == "dmas" else 0)
    child = next(layer for layer in report["layers"] if layer["path"] == "0.self_attn")
    assert "macs" not in child["metrics"]
    assert child["metric_owners"]["macs"] == {"path": "0", "call_index": 0}
