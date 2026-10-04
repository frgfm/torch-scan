import copy
import json

import pytest
import torch
from torch import nn

from torchscan import IncompleteAnalysisError, compare_reports, crawl_module, render_report
from torchscan.extensions import ModuleHandler
from torchscan.utils import aggregate_info, format_info


def _tokens(length, width=4, *, batch_first=True, batch_size=1):
    shape = (batch_size, length, width) if batch_first else (length, batch_size, width)
    return torch.randn(shape)


def _value(report, metric):
    result = report["totals"][metric]
    assert result["status"] == "complete", report["diagnostics"]
    return result["value"]


def _metric_rows(report, metric="macs"):
    return [layer for layer in report["layers"] if metric in layer["metrics"]]


def _assert_parameter_accounting(report, module):
    expected = sum(parameter.numel() for parameter in module.parameters())
    assert _value(report, "parameters") == expected
    assert (
        sum(layer["parameters"]["trainable"] + layer["parameters"]["frozen"] for layer in report["layers"]) == expected
    )


class _Wrapped(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.block = module

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_attention_self_independent_matrix_and_memory_derivation(batch_first, bias, batch_size):
    module = nn.MultiheadAttention(4, 2, dropout=0, bias=bias, batch_first=batch_first)
    query = _tokens(3, batch_first=batch_first, batch_size=batch_size)
    report = crawl_module(module, kwargs={"query": query, "key": query, "value": query, "need_weights": False})

    # Four 3x4 by 4x4 projections: 192 terms. QK and AV: 2x3x3x4=72 terms.
    # Bias, scaling and softmax do not turn these into FLOPs/2.
    assert _value(report, "macs") == 264 * batch_size
    # Projection reads/parameters/writes132; Q scale24; QK42; softmax168;
    # AV42; output projection44. Removing four length-4 biases removes16 reads.
    assert (
        _value(report, "dmas") == {(1, True): 452, (1, False): 436, (2, True): 824, (2, False): 808}[batch_size, bias]
    )
    assert _value(report, "module_flops") == (588 if bias else 540) * batch_size
    _assert_parameter_accounting(report, module)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_attention_cross_unequal_lengths_and_projection_dimensions(batch_first, bias, batch_size):
    module = nn.MultiheadAttention(4, 2, kdim=6, vdim=5, dropout=0, bias=bias, batch_first=batch_first)
    query = _tokens(2, batch_first=batch_first, batch_size=batch_size)
    key = _tokens(3, 6, batch_first=batch_first, batch_size=batch_size)
    value = _tokens(3, 5, batch_first=batch_first, batch_size=batch_size)
    report = crawl_module(module, args=(query,), kwargs={"key": key, "value": value, "need_weights": False})

    # Q32 + K72 + V60 + QK24 + AV24 + output32 =244 matrix terms.
    assert _value(report, "macs") == 244 * batch_size
    # Inputs41, parameters92, staged intermediates240 =373 logical accesses.
    assert (
        _value(report, "dmas") == {(1, True): 373, (1, False): 357, (2, True): 654, (2, False): 638}[batch_size, bias]
    )
    _assert_parameter_accounting(report, module)


@pytest.mark.parametrize(
    ("need_weights", "average", "expected_dma"), [(False, True, 452), (True, False, 452), (True, True, 479)]
)
def test_attention_returned_weight_storage(need_weights, average, expected_dma):
    module = nn.MultiheadAttention(4, 2, dropout=0, batch_first=True)
    query = _tokens(3)
    report = crawl_module(
        module,
        args=(query, query, query),
        kwargs={"need_weights": need_weights, "average_attn_weights": average},
    )

    assert _value(report, "macs") == 264
    # Per-head weights reuse the probability tensor. Head averaging reads18
    # probabilities and writes9 returned elements; no new matrix MACs occur.
    assert _value(report, "dmas") == expected_dma
    returned_weights = _metric_rows(report)[0]["output"]["items"][1]
    if need_weights:
        assert returned_weights["shape"] == ([1, 3, 3] if average else [1, 2, 3, 3])
    else:
        assert returned_weights["kind"] == "none"


@pytest.mark.parametrize("mask_kind", ["attention_bool", "attention_float", "padding", "both"])
def test_dense_attention_masks_preserve_matrix_work_and_count_logical_accesses(mask_kind):
    module = nn.MultiheadAttention(4, 2, dropout=0, batch_first=True)
    query, memory = _tokens(2), _tokens(3)
    kwargs = {"need_weights": False}
    if mask_kind in {"attention_bool", "both"}:
        kwargs["attn_mask"] = torch.tensor([[False, True, False], [False, False, True]])
    elif mask_kind == "attention_float":
        kwargs["attn_mask"] = torch.tensor([[0.0, float("-inf"), 0.0], [0.0, 0.0, float("-inf")]])
    if mask_kind in {"padding", "both"}:
        kwargs["key_padding_mask"] = torch.tensor([[False, False, True]])

    report = crawl_module(module, args=(query, memory, memory), kwargs=kwargs)

    # Dense Q32+K48+V48+QK24+AV24+out32 =208, even when entries are masked.
    assert _value(report, "macs") == 208
    # A=12 scores. Each mask stage reads its stored entries and reads/writes A.
    mask_accesses = {"attention_bool": 30, "attention_float": 30, "padding": 27, "both": 57}
    assert _value(report, "dmas") == 352 + mask_accesses[mask_kind]
    # Arbitrary exclusions retain exact dense arithmetic/access estimates, but
    # the compact dependency representation cannot claim an exact relation.
    dependencies = _metric_rows(report)[0]["token_dependencies"]
    assert dependencies["status"] == "unavailable"
    assert "known_value" not in dependencies
    assert any(item["code"] == "unsupported_token_dependencies" for item in report["diagnostics"])


def test_causal_attention_preserves_dense_mac_and_memory_counts():
    module = nn.MultiheadAttention(4, 2, dropout=0, batch_first=True)
    query = _tokens(3)
    mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
    report = crawl_module(
        module, args=(query, query, query), kwargs={"attn_mask": mask, "is_causal": True, "need_weights": False}
    )

    assert _value(report, "macs") == 264
    assert _value(report, "dmas") == 497  # Baseline452 + mask9 + 2x18 score accesses.


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_native_layer_hand_derivation(batch_first, norm_first, kind):
    layer_type = nn.TransformerEncoderLayer if kind == "encoder" else nn.TransformerDecoderLayer
    module = layer_type(4, 2, dim_feedforward=8, dropout=0, batch_first=batch_first, norm_first=norm_first)
    source = _tokens(3, batch_first=batch_first)
    target = _tokens(2, batch_first=batch_first)
    args = (source,) if kind == "encoder" else (target, source)
    report = crawl_module(module, args=args)

    # Encoder: attention264 + FF(3x4x8 + 3x8x4)=192 + two affine norms2x24.
    # Decoder: self160 + cross208 + FF128 + three affine norms3x16.
    assert _value(report, "macs") == (504 if kind == "encoder" else 544)
    # Encoder: attention452 + FF196 + residual72 + two norms2x96.
    # Decoder: self288 + cross352 + FF156 + residual72 + three norms3x67.
    assert _value(report, "dmas") == (912 if kind == "encoder" else 1069)
    _assert_parameter_accounting(report, module)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("kind", ["encoder", "decoder", "transformer"])
def test_native_stack_hand_derivation(batch_first, kind):
    source = _tokens(3, batch_first=batch_first)
    target = _tokens(2, batch_first=batch_first)
    if kind == "encoder":
        module = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=batch_first),
            2,
            norm=nn.LayerNorm(4),
            enable_nested_tensor=False,
        )
        args, macs, dmas = (source,), 1032, 1920
    elif kind == "decoder":
        module = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=batch_first),
            2,
            norm=nn.LayerNorm(4),
        )
        args, macs, dmas = (target, source), 1104, 2205
    else:
        module = nn.Transformer(
            d_model=4,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=8,
            dropout=0,
            batch_first=batch_first,
        )
        args, macs, dmas = (source, target), 2136, 4125
    report = crawl_module(module, args=args)

    # Repeated layer work plus a final affine norm: source24 MAC/96 DMA,
    # target16 MAC/67 DMA. Encoder and decoder work never overlap ownership.
    assert _value(report, "macs") == macs
    assert _value(report, "dmas") == dmas
    _assert_parameter_accounting(report, module)


@pytest.mark.parametrize("batch_first", [False, True])
def test_transformer_complete_keyword_call_forwards_every_native_mask(batch_first):
    module = nn.Transformer(
        d_model=4,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
        batch_first=batch_first,
    )
    # Explicitly retain dense padded execution for the encoder estimate.
    module.encoder.enable_nested_tensor = False
    module.encoder.use_nested_tensor = False
    source, target = _tokens(3, batch_first=batch_first), _tokens(2, batch_first=batch_first)
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

    assert _value(report, "macs") == 1088
    # Encoder: masks45+39. Decoder self:20+18; cross:30+27 accesses.
    assert _value(report, "dmas") == 2323
    assert _value(report, "module_flops") == 2770


def test_dense_encoder_padding_with_packing_disabled_is_supported():
    module = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True),
        1,
        enable_nested_tensor=False,
    )
    report = crawl_module(
        module,
        kwargs={
            "src": _tokens(3),
            "src_key_padding_mask": torch.tensor([[False, False, True]]),
        },
    )

    assert _value(report, "macs") == 504
    assert _value(report, "dmas") == 951  # Baseline912 + mask3 + 2x18 scores.
    assert any(item["code"] == "unsupported_token_dependencies" for item in report["diagnostics"])


@pytest.mark.parametrize("kind", ["attention", "encoder_layer", "decoder_layer", "encoder", "decoder", "transformer"])
def test_native_modules_inside_custom_wrapper_own_compute_once(kind):
    source, target = _tokens(3), _tokens(2)
    if kind == "attention":
        module = nn.MultiheadAttention(4, 2, dropout=0, batch_first=True)
        args, kwargs, macs, dmas = (source, source, source), {"need_weights": False}, 264, 452
    elif kind == "encoder_layer":
        module = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True)
        args, kwargs, macs, dmas = (source,), {}, 504, 912
    elif kind == "decoder_layer":
        module = nn.TransformerDecoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True)
        args, kwargs, macs, dmas = (target, source), {}, 544, 1069
    elif kind == "encoder":
        module = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True),
            1,
            enable_nested_tensor=False,
        )
        args, kwargs, macs, dmas = (source,), {}, 504, 912
    elif kind == "decoder":
        module = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True), 1
        )
        args, kwargs, macs, dmas = (target, source), {}, 544, 1069
    else:
        module = nn.Transformer(
            d_model=4,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=8,
            dropout=0,
            batch_first=True,
        )
        args, kwargs, macs, dmas = (source, target), {}, 1088, 2144
    wrapper = _Wrapped(module)
    report = crawl_module(wrapper, args=args, kwargs=kwargs)

    assert _value(report, "macs") == macs
    assert _value(report, "dmas") == dmas
    assert [layer["path"] for layer in _metric_rows(report)] == ["block"]
    _assert_parameter_accounting(report, wrapper)


def test_repeated_native_owner_preserves_calls_and_shared_parameters():
    class RepeatedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True)

        def forward(self, source):
            return self.block(self.block(source))

    module = RepeatedEncoder()
    report = crawl_module(module, args=(_tokens(3),))

    assert _value(report, "macs") == 1008
    assert _value(report, "dmas") == 1824
    assert [(layer["path"], layer["call_index"]) for layer in _metric_rows(report)] == [("block", 0), ("block", 1)]
    _assert_parameter_accounting(report, module)


def test_native_subtree_ownership_preserves_a_sibling_with_matching_path_prefix():
    class NativePipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True)
            self.block_extra = nn.Linear(4, 4)

        def forward(self, source):
            return self.block_extra(self.block(source))

    module = NativePipeline()
    report = crawl_module(module, args=(_tokens(3),))

    assert _value(report, "macs") == 552  # Encoder504 + sibling 3x4x4=48.
    assert _value(report, "dmas") == 956  # Encoder912 + input12/params20/output12.
    assert [layer["path"] for layer in _metric_rows(report)] == ["block", "block_extra"]
    _assert_parameter_accounting(report, module)


@pytest.mark.parametrize(
    ("affine", "bias", "macs", "dmas"), [(True, True, 24, 96), (True, False, 24, 92), (False, True, 12, 64)]
)
def test_normalization_counts_variance_and_affine_mac_terms(affine, bias, macs, dmas):
    module = nn.LayerNorm(4, elementwise_affine=affine, bias=bias)
    report = crawl_module(module, args=(_tokens(3),))

    # Twelve square-accumulation terms for variance; a present scale adds12
    # one-term affine MACs. Epsilon/statistics/intermediates contribute DMA.
    assert _value(report, "macs") == macs
    assert _value(report, "dmas") == dmas


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
@pytest.mark.parametrize("option", ["no_bias", "no_affine", "gelu"])
def test_native_layer_bias_normalization_and_activation_options(kind, option):
    layer_type = nn.TransformerEncoderLayer if kind == "encoder" else nn.TransformerDecoderLayer
    module = layer_type(
        4,
        2,
        dim_feedforward=8,
        dropout=0,
        batch_first=True,
        bias=option != "no_bias",
        activation="gelu" if option == "gelu" else "relu",
    )
    if option == "no_affine":
        module.norm1 = nn.LayerNorm(4, elementwise_affine=False)
        module.norm2 = nn.LayerNorm(4, elementwise_affine=False)
        if kind == "decoder":
            module.norm3 = nn.LayerNorm(4, elementwise_affine=False)
    source, target = _tokens(3), _tokens(2)
    report = crawl_module(module, args=(source,) if kind == "encoder" else (target, source))

    expected = {
        ("encoder", "no_bias"): (504, 876),
        ("decoder", "no_bias"): (544, 1013),
        ("encoder", "no_affine"): (480, 848),
        ("decoder", "no_affine"): (520, 997),
        ("encoder", "gelu"): (504, 912),
        ("decoder", "gelu"): (544, 1069),
    }
    macs, dmas = expected[kind, option]
    assert _value(report, "macs") == macs
    assert _value(report, "dmas") == dmas
    if option == "gelu":
        # The independent matrix/access methods can support token-local GELU
        # without extending the existing ReLU-only module FLOP convention.
        assert report["totals"]["module_flops"]["status"] == "unavailable"
    _assert_parameter_accounting(report, module)


@pytest.mark.parametrize("option", ["add_bias_kv", "add_zero_attn", "unbatched", "empty_source"])
def test_unsupported_attention_configuration_has_no_numeric_upper_bound(option):
    kwargs = {option: True} if option in {"add_bias_kv", "add_zero_attn"} else {}
    module = nn.MultiheadAttention(4, 2, dropout=0, batch_first=True, **kwargs)
    query = _tokens(2)
    memory = _tokens(0) if option == "empty_source" else query
    if option == "unbatched":
        query, memory = query.squeeze(0), memory.squeeze(0)
    report = crawl_module(module, args=(query, memory, memory), kwargs={"need_weights": False})

    for metric in ("macs", "dmas"):
        assert report["totals"][metric]["status"] == "unavailable"
        assert report["totals"][metric]["value"] is None
        assert report["totals"][metric]["known_value"] is None
        assert any(item["metric"] == metric for item in report["diagnostics"])
    with pytest.raises(IncompleteAnalysisError):
        crawl_module(module, args=(query, memory, memory), kwargs={"need_weights": False}, strict=True)


@pytest.mark.parametrize("option", ["activation", "final_norm", "nested_padding"])
def test_unsupported_native_stack_configuration_is_diagnostic(option):
    layer = nn.TransformerEncoderLayer(
        4,
        2,
        dim_feedforward=8,
        dropout=0,
        batch_first=True,
        activation=torch.sin if option == "activation" else "relu",
    )
    module = nn.TransformerEncoder(
        layer,
        1,
        norm=nn.Linear(4, 4) if option == "final_norm" else None,
        enable_nested_tensor=option == "nested_padding",
    )
    kwargs = {"src_key_padding_mask": torch.tensor([[False, False, True]])} if option == "nested_padding" else {}
    report = crawl_module(module, args=(_tokens(3),), kwargs=kwargs)

    for metric in ("macs", "dmas"):
        assert report["totals"][metric]["status"] == "unavailable"
        assert report["totals"][metric]["known_value"] is None
        assert any(item["metric"] == metric for item in report["diagnostics"])


@pytest.mark.parametrize("kind", ["encoder", "transformer"])
def test_mixed_native_batch_layouts_are_diagnostic_even_when_execution_succeeds(kind):
    if kind == "encoder":
        module = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True),
            2,
            enable_nested_tensor=False,
        )
        module.layers[1].self_attn.batch_first = False
        args = (_tokens(3, batch_size=2),)
    else:
        module = nn.Transformer(
            d_model=4,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=8,
            dropout=0,
            batch_first=True,
        )
        module.decoder.layers[0].self_attn.batch_first = False
        module.decoder.layers[0].multihead_attn.batch_first = False
        # Equal batch/token dimensions make the inconsistent interpretation
        # executable, while still failing the documented native boundary.
        args = (_tokens(2, batch_size=2), _tokens(2, batch_size=2))
    report = crawl_module(module, args=args)

    for metric in ("macs", "dmas"):
        assert report["totals"][metric]["status"] == "unavailable"
        assert report["totals"][metric]["known_value"] is None
        assert any(item["metric"] == metric and "layout" in item["message"].lower() for item in report["diagnostics"])
    assert _metric_rows(report)[0]["token_dependencies"]["status"] == "unavailable"


@pytest.mark.parametrize("kind", ["attention", "encoder_layer"])
def test_custom_flop_override_preserves_independent_native_metrics_and_ownership(kind):
    seen = []
    if kind == "attention":
        module = nn.MultiheadAttention(4, 2, kdim=6, vdim=5, dropout=0, batch_first=True)
        kwargs = {
            "query": _tokens(2),
            "key": _tokens(3, 6),
            "value": _tokens(3, 5),
            "need_weights": False,
        }
        macs, dmas = 244, 373
    else:
        module = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0, batch_first=True)
        kwargs = {"src": _tokens(3)}
        macs, dmas = 504, 912

    def custom_flops(call):
        assert call.module is module
        assert call.args == ()
        assert call.kwargs.keys() == kwargs.keys()
        assert all(call.kwargs[name] is value for name, value in kwargs.items())
        seen.append(True)
        return {"module_flops": 7}

    report = crawl_module(
        module,
        kwargs=kwargs,
        custom_modules={
            type(module): ModuleHandler(custom_flops, subtree_metrics=frozenset({"module_flops"})),
        },
    )

    assert seen == [True]
    assert _value(report, "module_flops") == 7
    assert _value(report, "macs") == macs
    assert _value(report, "dmas") == dmas
    assert len(_metric_rows(report)) == len(_metric_rows(report, "module_flops")) == 1
    assert _metric_rows(report)[0]["token_dependencies"]["status"] == "complete"
    _assert_parameter_accounting(report, module)


@pytest.mark.parametrize("enabled", [False, True])
def test_attention_fastpath_preserves_estimates_and_parameter_accounting(enabled, monkeypatch):
    fastpath = getattr(torch.backends, "mha", None)
    if fastpath is None:
        pytest.skip("This PyTorch release exposes no attention fastpath switch.")
    fused_calls = []
    native_attention = getattr(torch, "_native_multi_head_attention", None)
    if native_attention is not None:

        def observe_native(*args, **kwargs):
            fused_calls.append(True)
            return native_attention(*args, **kwargs)

        monkeypatch.setattr(torch, "_native_multi_head_attention", observe_native)
    previous = fastpath.get_fastpath_enabled()
    try:
        fastpath.set_fastpath_enabled(enabled)
        module = _Wrapped(nn.MultiheadAttention(4, 2, dropout=0, batch_first=True))
        query = _tokens(3)
        report = crawl_module(module, args=(query, query, query), kwargs={"need_weights": False})
    finally:
        fastpath.set_fastpath_enabled(previous)

    assert _value(report, "macs") == 264
    assert _value(report, "dmas") == 452
    _assert_parameter_accounting(report, module)
    if native_attention is not None:
        assert bool(fused_calls) == enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_causal_prefix_semantics_under_native_evaluation_fastpaths(enabled):
    fastpath = getattr(torch.backends, "mha", None)
    if fastpath is None:
        pytest.skip("This PyTorch release exposes no attention fastpath switch.")
    previous = fastpath.get_fastpath_enabled()
    module = nn.MultiheadAttention(4, 2, dropout=0, batch_first=True).eval()
    query = _tokens(3)
    changed = query.clone()
    changed[:, 2] += torch.tensor([1.0, -2.0, 3.0, -4.0])
    mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
    try:
        fastpath.set_fastpath_enabled(enabled)
        with torch.no_grad():
            output = module(query, query, query, attn_mask=mask, is_causal=True, need_weights=False)[0]
            changed_output = module(changed, changed, changed, attn_mask=mask, is_causal=True, need_weights=False)[0]
        report = crawl_module(
            module,
            args=(query, query, query),
            kwargs={"attn_mask": mask, "is_causal": True, "need_weights": False},
        )
    finally:
        fastpath.set_fastpath_enabled(previous)

    # Future-token perturbation cannot affect earlier causal output positions.
    torch.testing.assert_close(changed_output[:, :2], output[:, :2])
    assert not torch.allclose(changed_output[:, 2], output[:, 2])
    dependencies = _metric_rows(report)[0]["token_dependencies"]
    assert dependencies["status"] == "complete"
    assert dependencies["sources"][0]["relation"]["kind"] == "prefix"


def test_structure_mode_skips_native_metrics_and_preserves_training_state():
    module = _Wrapped(nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.5, batch_first=True))
    module.train()
    module.block.norm1.eval()
    flags = [(child, child.training) for child in module.modules()]
    report = crawl_module(module, args=(_tokens(3),), mode="structure", strict=True)

    assert all(child.training == training for child, training in flags)
    assert all(set(layer["metrics"]) == {"calls"} for layer in report["layers"])
    assert all("token_dependencies" not in layer for layer in report["layers"])
    assert all(
        report["totals"][metric]["method"] == "not_requested"
        for metric in ("module_flops", "macs", "dmas", "operator_flops")
    )
    _assert_parameter_accounting(report, module)


def test_token_dependency_report_is_additive_and_consumers_keep_it():
    module = _Wrapped(nn.MultiheadAttention(4, 2, dropout=0, batch_first=True))
    query = _tokens(3)
    report = crawl_module(module, args=(query, query, query), kwargs={"need_weights": False})
    owner = _metric_rows(report)[0]

    assert report["schema_version"] == 1
    assert owner["token_dependencies"]["status"] == "complete"
    for field in ("receptive_field", "effective_stride", "effective_padding"):
        assert owner["metrics"][field]["status"] == "unavailable"
        assert owner["metrics"][field]["known_value"] is None
    assert json.loads(json.dumps(report)) == report
    assert compare_reports(report, copy.deepcopy(report))["layers"]["changed"] == []
    assert "MultiheadAttention" in format_info(report, receptive_field=True, effective_rf_stats=True)
    assert "MultiheadAttention" in render_report(report)
    view = aggregate_info(report, 1)
    copied_owner = next(layer for layer in view["layers"] if layer["path"] == "block")
    assert copied_owner["token_dependencies"] == owner["token_dependencies"]
    assert copied_owner["token_dependencies"] is not owner["token_dependencies"]
