import inspect
import json

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


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("floating", [False, True])
def test_self_attention_reports_real_causal_and_unmasked_token_dependencies(batch_first, floating):
    torch.manual_seed(101)
    module = nn.MultiheadAttention(4, 2, batch_first=batch_first, dropout=0).double().eval()
    tensor = torch.randn(1, 4, 4, dtype=torch.double)
    inp = _layout(tensor, batch_first)
    unmasked = module_token_dependencies(module, _ordered(module, inp, inp, inp))
    assert unmasked["output"] == {"sequence_axis": int(batch_first), "length": 4}
    assert unmasked["sources"] == [
        {
            "arguments": ["query", "key", "value"],
            "sequence_axis": int(batch_first),
            "length": 4,
            "relation": {"kind": "all"},
        },
    ]

    def attention(variable, mask=None):
        variable = _layout(variable, batch_first)
        output = module(variable, variable, variable, attn_mask=mask, need_weights=False)[0]
        return _layout(output, batch_first)

    assert _dependency_blocks(attention, tensor).all()
    mask = _causal(4, floating=floating)
    causal = module_token_dependencies(module, _ordered(module, inp, inp, inp, attn_mask=mask, is_causal=True))
    assert _relation(causal, "query") == {"kind": "prefix"}
    assert torch.equal(_dependency_blocks(lambda value: attention(value, mask), tensor), ~_causal(4))
    assert json.loads(json.dumps(causal)) == causal


@pytest.mark.parametrize("batch_first", [False, True])
def test_cross_attention_distinguishes_query_from_unequal_source_tokens(batch_first):
    torch.manual_seed(103)
    module = nn.MultiheadAttention(4, 2, kdim=6, vdim=6, batch_first=batch_first, dropout=0).double().eval()
    target = torch.randn(1, 2, 4, dtype=torch.double)
    source = torch.randn(1, 3, 6, dtype=torch.double)
    query, memory = _layout(target, batch_first), _layout(source, batch_first)
    report = module_token_dependencies(module, _ordered(module, query, memory, memory, need_weights=False))
    assert _relation(report, "query") == {"kind": "same_position"}
    assert _relation(report, "key") == {"kind": "all"}
    assert report["sources"][1]["arguments"] == ["key", "value"]

    def attention(q, kv, mask=None):
        q, kv = _layout(q, batch_first), _layout(kv, batch_first)
        return _layout(module(q, kv, kv, attn_mask=mask, need_weights=False)[0], batch_first)

    assert torch.equal(_dependency_blocks(lambda value: attention(value, source), target), torch.eye(2).bool())
    assert _dependency_blocks(lambda value: attention(target, value), source).all()
    mask = _causal(2, 3)
    causal = module_token_dependencies(module, _ordered(module, query, memory, memory, attn_mask=mask))
    assert _relation(causal, "query") == {"kind": "same_position", "first_position": 1}
    assert _relation(causal, "key") == {"kind": "prefix"}
    assert torch.equal(
        _dependency_blocks(lambda value: attention(value, source, mask), target),
        torch.tensor([[False, False], [False, True]]),
    )
    assert torch.equal(_dependency_blocks(lambda value: attention(target, value, mask), source), ~mask)


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


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("batch_first", [False, True])
def test_native_encoder_layers_and_stacks_preserve_prefix_semantics(norm_first, batch_first):
    torch.manual_seed(109)
    layer = nn.TransformerEncoderLayer(4, 2, 7, dropout=0, norm_first=norm_first, batch_first=batch_first).double()
    stack = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(4).double(), enable_nested_tensor=False).eval()
    layer.eval()
    tensor = torch.randn(1, 3, 4, dtype=torch.double)
    src = _layout(tensor, batch_first)
    mask = _causal(3, floating=True)
    for module in (layer, stack):
        mask_name = "src_mask" if isinstance(module, nn.TransformerEncoderLayer) else "mask"
        inputs = _ordered(module, src, **{mask_name: mask})
        report = module_token_dependencies(module, inputs)
        assert _relation(report, "src") == {"kind": "prefix"}

        def call(value, module=module, mask_name=mask_name):
            return _layout(module(_layout(value, batch_first), **{mask_name: mask}), batch_first)

        assert torch.equal(_dependency_blocks(call, tensor), ~_causal(3))


@pytest.mark.parametrize("norm_first", [False, True])
def test_decoder_stack_preserves_causal_target_and_all_memory_dependencies(norm_first):
    torch.manual_seed(113)
    layer = nn.TransformerDecoderLayer(4, 2, 7, dropout=0, norm_first=norm_first, batch_first=True).double()
    stack = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(4).double()).eval()
    layer.eval()
    target = torch.randn(1, 3, 4, dtype=torch.double)
    memory = torch.randn(1, 4, 4, dtype=torch.double)
    mask = _causal(3)
    for module in (layer, stack):
        report = module_token_dependencies(module, _ordered(module, target, memory, tgt_mask=mask))
        assert _relation(report, "tgt") == {"kind": "prefix"}
        assert _relation(report, "memory") == {"kind": "all"}
        assert torch.equal(
            _dependency_blocks(lambda value, module=module: module(value, memory, tgt_mask=mask), target), ~_causal(3)
        )
        assert _dependency_blocks(lambda value, module=module: module(target, value, tgt_mask=mask), memory).all()


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
    module = (
        nn
        .Transformer(
            d_model=4,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=7,
            dropout=0,
            batch_first=True,
        )
        .double()
        .eval()
    )
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


def test_caught_native_forward_failure_has_unavailable_metrics_and_preserves_ownership_cleanup():
    class CaughtNativeError(nn.Module):
        def __init__(self):
            super().__init__()
            self.bad = nn.TransformerEncoderLayer(4, 2, 7, batch_first=False, dropout=0)
            self.identity = nn.Identity()
            self.caught = 0

        def forward(self, source):
            try:
                self.bad(source.double())  # Shape-valid inputs cannot multiply FP32 projection parameters.
            except RuntimeError:
                self.caught += 1
            return self.identity(source)

    module = CaughtNativeError()
    source = torch.randn(3, 1, 4)
    flags = [child.training for child in module.modules()]
    report = crawl_module(module, args=(source,))
    assert module.caught == 1
    failed = next(layer for layer in report["layers"] if layer["path"] == "bad")
    assert failed["output"]["kind"] == "failed"
    for metric in ("module_flops", "macs", "dmas"):
        assert failed["metrics"][metric]["status"] == "unavailable"
        assert failed["metrics"][metric]["known_value"] is None
        assert failed["metric_ownership"][metric] == "subtree"
        # The successful Identity contributes its legacy zero compute / input
        # DMA count. The failed branch cannot add a full dense upper bound.
        assert report["totals"][metric]["status"] == "partial"
        assert report["totals"][metric]["known_value"] == (source.numel() if metric == "dmas" else 0)
    assert failed["token_dependencies"]["status"] == "unavailable"
    child = next(layer for layer in report["layers"] if layer["path"] == "bad.self_attn")
    assert "macs" not in child["metrics"]
    assert child["metric_owners"]["macs"] == {"path": "bad", "call_index": 0}
    assert all(
        diagnostic["metric"] == "calls"
        for diagnostic in report["diagnostics"]
        if diagnostic.get("path") == "bad.self_attn"
    )
    sibling = next(layer for layer in report["layers"] if layer["path"] == "identity")
    assert "metric_owners" not in sibling
    assert sibling["metrics"]["macs"]["status"] == "complete"
    assert [child.training for child in module.modules()] == flags
    assert all(not child._forward_hooks and not child._forward_pre_hooks for child in module.modules())
