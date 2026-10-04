import pytest
import torch
from torch import nn

from torchscan import IncompleteAnalysisError, crawl_module
from torchscan.extensions import ModuleHandler

_COMPUTE = ("module_flops", "macs", "dmas")


def _tokens(length, width=4, *, batch_first=True, batch_size=1):
    return torch.randn((batch_size, length, width) if batch_first else (length, batch_size, width))


def _native(kind, *, batch_first=True, depth=1, norm_first=False, **options):
    if kind == "attention":
        return nn.MultiheadAttention(4, 2, dropout=0, batch_first=batch_first, **options)
    config = {
        "d_model": 4,
        "nhead": 2,
        "dim_feedforward": 8,
        "dropout": 0,
        "batch_first": batch_first,
        "norm_first": norm_first,
    }
    if kind == "transformer":
        return nn.Transformer(num_encoder_layers=depth, num_decoder_layers=depth, **config, **options)
    layer_type = nn.TransformerEncoderLayer if kind.startswith("encoder") else nn.TransformerDecoderLayer
    layer = layer_type(**config, **options)
    if kind.endswith("_layer"):
        return layer
    if kind == "encoder":
        return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
    return nn.TransformerDecoder(layer, depth)


def _args(kind, *, batch_first=True):
    source, target = _tokens(3, batch_first=batch_first), _tokens(2, batch_first=batch_first)
    if kind == "attention":
        return source, source, source
    if kind.startswith("encoder"):
        return (source,)
    return (source, target) if kind == "transformer" else (target, source)


def _value(report, metric):
    result = report["totals"][metric]
    assert result["status"] == "complete", report["diagnostics"]
    return result["value"]


def _rows(report, metric="macs"):
    return [layer for layer in report["layers"] if metric in layer["metrics"]]


def _assert_parameters(report, module):
    expected = sum(parameter.numel() for parameter in module.parameters())
    assert _value(report, "parameters") == expected
    assert sum(row["parameters"]["trainable"] + row["parameters"]["frozen"] for row in report["layers"]) == expected


def _assert_counts(report, module, macs, dmas):
    assert (_value(report, "macs"), _value(report, "dmas")) == (macs, dmas)
    _assert_parameters(report, module)


def _assert_unavailable(report, metrics=_COMPUTE):
    for metric in metrics:
        result = report["totals"][metric]
        assert result["status"] == "unavailable"
        assert result["value"] is result["known_value"] is None
        assert any(item["metric"] == metric for item in report["diagnostics"])
    assert report["layers"][0]["token_dependencies"]["status"] == "unavailable"


class _Wrapped(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.block = module

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)


@pytest.mark.parametrize(
    ("cross", "batch_first", "bias", "batch", "macs", "dmas"),
    [
        (False, False, False, 2, 528, 808),
        (True, True, True, 1, 244, 373),
    ],
)
def test_complete_attention_arguments(cross, batch_first, bias, batch, macs, dmas):
    module = _native("attention", batch_first=batch_first, bias=bias, **({"kdim": 6, "vdim": 5} if cross else {}))
    query = _tokens(2 if cross else 3, batch_first=batch_first, batch_size=batch)
    key = _tokens(3, 6, batch_size=batch) if cross else query
    value = _tokens(3, 5, batch_size=batch) if cross else query
    # Exercise mixed positional/keyword cross inputs and a full keyword self call.
    kwargs = {"key": key, "value": value, "need_weights": False}
    if not cross:
        kwargs["query"] = query
    report = crawl_module(module, args=(query,) if cross else (), kwargs=kwargs)
    _assert_counts(report, module, macs, dmas)
    if not cross:
        assert _value(report, "module_flops") == 1080


@pytest.mark.parametrize(
    ("need", "average", "dmas", "shape"),
    [
        (False, True, 452, None),
        (True, False, 452, [1, 2, 3, 3]),
        (True, True, 479, [1, 3, 3]),
    ],
)
def test_returned_attention_weights(need, average, dmas, shape):
    module = _native("attention")
    report = crawl_module(
        module, args=_args("attention"), kwargs={"need_weights": need, "average_attn_weights": average}
    )
    _assert_counts(report, module, 264, dmas)
    weights = report["layers"][0]["output"]["items"][1]
    assert weights.get("shape") == shape
    if shape is None:
        assert weights["kind"] == "none"


@pytest.mark.parametrize(
    ("mask", "padding", "accesses"),
    [
        (torch.tensor([[False, True, False], [False, False, True]]), None, 30),
        (torch.tensor([[0.0, float("-inf"), 0.0], [0.0, 0.0, float("-inf")]]), None, 30),
        (None, torch.tensor([[False, False, True]]), 27),
        (torch.zeros(2, 3, dtype=torch.bool), torch.tensor([[False, False, True]]), 57),
    ],
)
def test_dense_masks_keep_matrix_counts(mask, padding, accesses):
    module = _native("attention")
    query, memory = _tokens(2), _tokens(3)
    report = crawl_module(
        module,
        args=(query, memory, memory),
        kwargs={
            "need_weights": False,
            "attn_mask": mask,
            "key_padding_mask": padding,
        },
    )
    _assert_counts(report, module, 208, 352 + accesses)
    assert report["layers"][0]["token_dependencies"]["status"] == "unavailable"
    assert any(item["code"] == "unsupported_token_dependencies" for item in report["diagnostics"])


@pytest.mark.parametrize(
    ("kind", "macs", "dmas"),
    [
        ("attention", 264, 452),
        ("encoder_layer", 504, 912),
        ("decoder_layer", 544, 1069),
        ("encoder", 504, 912),
        ("decoder", 544, 1069),
        ("transformer", 1088, 2144),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_owners_standalone_and_wrapped(kind, macs, dmas, wrapped):
    # Pair sequence-first/post-norm standalone with batch-first/pre-norm wrapped.
    module = _native(kind, batch_first=wrapped, norm_first=wrapped)
    args = _args(kind, batch_first=wrapped)
    module = _Wrapped(module) if wrapped else module
    report = crawl_module(module, args=args, kwargs={"need_weights": False} if kind == "attention" else {})
    _assert_counts(report, module, macs, dmas)
    owner = _rows(report)
    assert [row["path"] for row in owner] == ["block" if wrapped else ""]
    assert owner[0]["token_dependencies"]["status"] == "complete"
    for metric in ("receptive_field", "effective_stride", "effective_padding"):
        assert owner[0]["metrics"][metric]["status"] == "unavailable"


def test_full_transformer_keyword_masks():
    module = _native("transformer")
    module.encoder.enable_nested_tensor = module.encoder.use_nested_tensor = False
    source, target = _args("transformer")
    report = crawl_module(
        module,
        kwargs={
            "src": source,
            "tgt": target,
            "src_mask": torch.ones(3, 3, dtype=torch.bool).triu(1),
            "tgt_mask": torch.ones(2, 2, dtype=torch.bool).triu(1),
            "memory_mask": torch.zeros(2, 3, dtype=torch.bool),
            "src_key_padding_mask": torch.zeros(1, 3, dtype=torch.bool),
            "tgt_key_padding_mask": torch.zeros(1, 2, dtype=torch.bool),
            "memory_key_padding_mask": torch.zeros(1, 3, dtype=torch.bool),
            "src_is_causal": True,
            "tgt_is_causal": True,
            "memory_is_causal": False,
        },
    )
    # Encoder masks45+39; decoder self20+18 and cross30+27 logical accesses.
    _assert_counts(report, module, 1088, 2323)
    assert _value(report, "module_flops") == 2770


def test_dense_padded_encoder():
    module = _native("encoder")
    report = crawl_module(
        module, kwargs={"src": _tokens(3), "src_key_padding_mask": torch.tensor([[False, False, True]])}
    )
    _assert_counts(report, module, 504, 951)
    assert any(item["code"] == "unsupported_token_dependencies" for item in report["diagnostics"])


@pytest.mark.parametrize("repeat", [False, True])
def test_repeated_owner_and_sibling_prefix(repeat):
    class Pipeline(_Wrapped):
        def __init__(self):
            super().__init__(_native("encoder_layer"))
            if not repeat:
                self.block_extra = nn.Linear(4, 4)

        def forward(self, source):
            output = self.block(source)
            return self.block(output) if repeat else self.block_extra(output)

    module = Pipeline()
    report = crawl_module(module, args=(_tokens(3),))
    _assert_counts(report, module, 1008 if repeat else 552, 1824 if repeat else 956)
    assert [(row["path"], row["call_index"]) for row in _rows(report)] == (
        [("block", 0), ("block", 1)] if repeat else [("block", 0), ("block_extra", 0)]
    )


@pytest.mark.parametrize(
    ("affine", "bias", "macs", "dmas"), [(True, True, 24, 96), (True, False, 24, 92), (False, True, 12, 64)]
)
def test_standalone_normalization(affine, bias, macs, dmas):
    module = nn.LayerNorm(4, elementwise_affine=affine, bias=bias)
    _assert_counts(crawl_module(module, args=(_tokens(3),)), module, macs, dmas)


@pytest.mark.parametrize(
    ("kind", "option", "macs", "dmas"),
    [
        ("encoder_layer", "no_bias", 504, 876),
        ("decoder_layer", "no_bias", 544, 1013),
        ("encoder_layer", "no_affine", 480, 848),
        ("decoder_layer", "no_affine", 520, 997),
        ("encoder_layer", "gelu", 504, 912),
        ("decoder_layer", "gelu", 544, 1069),
    ],
)
def test_layer_options(kind, option, macs, dmas):
    # PyTorch2.1 requires sequence-first to avoid its absent-affine encoder bug.
    batch_first = kind != "encoder_layer" or getattr(torch.backends, "mha", None) is not None
    module = _native(
        kind, batch_first=batch_first, bias=option != "no_bias", activation="gelu" if option == "gelu" else "relu"
    )
    if option == "no_affine":
        for name in ("norm1", "norm2", "norm3"):
            if hasattr(module, name):
                setattr(module, name, nn.LayerNorm(4, elementwise_affine=False))
    report = crawl_module(module, args=_args(kind, batch_first=batch_first))
    _assert_counts(report, module, macs, dmas)
    if option == "gelu":
        assert report["totals"]["module_flops"]["status"] == "unavailable"


@pytest.mark.parametrize("option", ["add_bias_kv", "add_zero_attn", "unbatched", "empty_source"])
def test_unsupported_attention_has_no_upper_bound(option):
    module = _native("attention", **({option: True} if option.startswith("add_") else {}))
    query, memory = _tokens(2), _tokens(0 if option == "empty_source" else 2)
    if option == "unbatched":
        query, memory = query.squeeze(0), memory.squeeze(0)
    _assert_unavailable(crawl_module(module, args=(query, memory, memory), kwargs={"need_weights": False}))
    with pytest.raises(IncompleteAnalysisError):
        crawl_module(module, args=(query, memory, memory), kwargs={"need_weights": False}, strict=True)


@pytest.mark.parametrize("option", ["activation", "final_norm", "nested_padding"])
def test_unsupported_stack_is_diagnostic(option):
    module = nn.TransformerEncoder(
        _native("encoder_layer", activation=torch.sin if option == "activation" else "relu"),
        1,
        norm=nn.Linear(4, 4) if option == "final_norm" else None,
        enable_nested_tensor=option == "nested_padding",
    )
    kwargs = {"src_key_padding_mask": torch.tensor([[False, False, True]])} if option == "nested_padding" else {}
    _assert_unavailable(crawl_module(module, args=(_tokens(3),), kwargs=kwargs))


@pytest.mark.parametrize("kind", ["encoder", "transformer"])
def test_mixed_layouts_are_diagnostic(kind):
    module = _native(kind, depth=2 if kind == "encoder" else 1)
    layer = module.layers[1] if kind == "encoder" else module.decoder.layers[0]
    layer.self_attn.batch_first = False
    if kind == "transformer":
        layer.multihead_attn.batch_first = False
    source = _tokens(2, batch_size=2)  # Equal axes keep the modified forward valid.
    report = crawl_module(module, args=(source,) if kind == "encoder" else (source, source))
    _assert_unavailable(report)
    assert any("layout" in item["message"].lower() for item in report["diagnostics"])


@pytest.mark.parametrize(("kind", "macs", "dmas"), [("attention", 264, 452), ("encoder_layer", 504, 912)])
def test_flops_only_override_retains_native_fallback(kind, macs, dmas):
    module, seen = _native(kind), []
    kwargs = dict(zip(("query", "key", "value") if kind == "attention" else ("src",), _args(kind), strict=True))
    if kind == "attention":
        kwargs["need_weights"] = False

    def estimate(call):
        assert call.module is module
        assert call.args == ()
        assert call.kwargs.keys() == kwargs.keys()
        assert all(call.kwargs[key] is value for key, value in kwargs.items())
        seen.append(True)
        return {"module_flops": 7}

    report = crawl_module(
        module, kwargs=kwargs, custom_modules={type(module): ModuleHandler(estimate, frozenset({"module_flops"}))}
    )
    _assert_counts(report, module, macs, dmas)
    assert seen == [True]
    assert _value(report, "module_flops") == 7
    assert len(_rows(report)) == len(_rows(report, "module_flops")) == 1
    assert _rows(report)[0]["token_dependencies"]["status"] == "complete"


@pytest.mark.parametrize("enabled", [False, True])
def test_fastpath_counts_and_causal_influence(enabled, monkeypatch):
    fastpath = getattr(torch.backends, "mha", None)
    if fastpath is None:
        pytest.skip("This PyTorch release has no attention fastpath switch.")
    native, calls = torch._native_multi_head_attention, []

    def observe(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(torch, "_native_multi_head_attention", observe)
    previous = fastpath.get_fastpath_enabled()
    module = _Wrapped(_native("attention")).eval()
    query = _tokens(3)
    try:
        fastpath.set_fastpath_enabled(enabled)
        report = crawl_module(module, args=(query,) * 3, kwargs={"need_weights": False})
        assert bool(calls) == enabled
        _assert_counts(report, module, 264, 452)
        mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
        changed = query.clone()
        changed[:, 2] += torch.tensor([1.0, -2.0, 3.0, -4.0])
        with torch.no_grad():
            output = module(query, query, query, attn_mask=mask, is_causal=True, need_weights=False)[0]
            altered = module(changed, changed, changed, attn_mask=mask, is_causal=True, need_weights=False)[0]
        report = crawl_module(
            module, args=(query,) * 3, kwargs={"attn_mask": mask, "is_causal": True, "need_weights": False}
        )
        torch.testing.assert_close(altered[:, :2], output[:, :2])
        assert not torch.allclose(altered[:, 2], output[:, 2])
        _assert_counts(report, module, 264, 497)
        assert _rows(report)[0]["token_dependencies"]["sources"][0]["relation"]["kind"] == "prefix"
    finally:
        fastpath.set_fastpath_enabled(previous)


def test_structure_mode_restores_state_and_skips_estimators():
    module = _Wrapped(_native("encoder_layer"))
    module.block.norm1.eval()
    flags = [(child, child.training) for child in module.modules()]
    report = crawl_module(module, args=(_tokens(3),), mode="structure", strict=True)
    assert all(child.training == training for child, training in flags)
    assert all(set(row["metrics"]) == {"calls"} and "token_dependencies" not in row for row in report["layers"])
    assert all(report["totals"][metric]["method"] == "not_requested" for metric in (*_COMPUTE, "operator_flops"))
    _assert_parameters(report, module)


@pytest.mark.parametrize("kind", ["attention", "encoder_layer", "decoder_layer", "encoder", "decoder", "transformer"])
def test_replaced_forward_needs_explicit_estimates(kind):
    module, args = _native(kind), _args(kind)
    module.forward = lambda *inputs, **_kwargs: (inputs[0], None) if kind == "attention" else inputs[-1]
    report = crawl_module(module, args=args)
    _assert_unavailable(report)
    assert _value(report, "operator_flops") == 0
    assert any("native forward" in item["message"] for item in report["diagnostics"])
    estimates = dict.fromkeys(_COMPUTE, 0) | {"receptive_field": 1, "effective_stride": 1, "effective_padding": 0}
    custom = crawl_module(
        module,
        args=args,
        custom_modules={type(module): ModuleHandler(lambda _call: estimates, frozenset(estimates))},
        strict=True,
    )
    assert all(_value(custom, name) == 0 for name in _COMPUTE)
    assert "token_dependencies" not in custom["layers"][0]
    _assert_parameters(custom, module)


@pytest.mark.parametrize("stage", ["self_attn", "linear1", "norm1", "dropout"])
def test_changed_invoked_forward_is_unavailable(stage):
    module = _native("encoder_layer")
    forward = getattr(module, stage).forward
    getattr(module, stage).forward = lambda *args, **kwargs: forward(*args, **kwargs)
    _assert_unavailable(crawl_module(module, args=_args("encoder_layer")))


def test_uninspectable_forward_is_diagnostic():
    module = _native("encoder_layer")
    module.forward = torch.clone
    _assert_unavailable(crawl_module(module, args=_args("encoder_layer")))


@pytest.mark.parametrize("change", ["weight", "in_features", "out_features", "head_dim"])
def test_malformed_attention_metadata_is_unavailable(change):
    module = _native("attention", bias=False)
    if change == "weight":
        module.out_proj.weight = nn.Parameter(torch.randn(6, 4))
    else:
        setattr(module if change == "head_dim" else module.out_proj, change, 1 if change == "head_dim" else 6)
    _assert_unavailable(crawl_module(module, args=(_tokens(2), _tokens(3), _tokens(3)), kwargs={"need_weights": False}))


def test_unused_projection_forward_and_legacy_zero_batch():
    module = _native("attention")
    module.out_proj.forward = lambda _input: (_ for _ in ()).throw(AssertionError("unused stage"))
    _assert_counts(crawl_module(module, args=_args("attention"), kwargs={"need_weights": False}), module, 264, 452)
    query, memory = _tokens(2, batch_size=0), _tokens(3, batch_size=0)
    report = crawl_module(module, args=(query, memory, memory), kwargs={"need_weights": False})
    assert _value(report, "module_flops") == 0
    _assert_unavailable(report, ("macs", "dmas"))
