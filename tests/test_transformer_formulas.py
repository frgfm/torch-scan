"""Independent tiny matrix and logical-access examples for native Transformers."""

import pytest
import torch
from torch import nn

from torchscan import crawl_module
from torchscan.modules._transformer import dmas_attention, macs_attention, validate_native_call
from torchscan.modules.memory import module_dmas


def _tokens(length, width=4, *, batch_first=True, batch=1):
    return torch.ones((batch, length, width) if batch_first else (length, batch, width))


def _counts(module, inputs):
    return macs_attention(module, inputs, None), dmas_attention(module, inputs, None)


@pytest.mark.parametrize(("batch_first", "bias"), [(False, False), (True, True)])
@pytest.mark.parametrize("cross", [False, True])
def test_independent_attention_contractions_and_stages(batch_first, bias, cross):
    options = {"kdim": 6, "vdim": 5} if cross else {}
    attention = nn.MultiheadAttention(4, 2, batch_first=batch_first, bias=bias, **options).eval()
    query = _tokens(2 if cross else 3, batch_first=batch_first)
    key = _tokens(3, 6, batch_first=batch_first) if cross else query
    value = _tokens(3, 5, batch_first=batch_first) if cross else query
    # Self MACs: projections144 + QK36 + AV36 + output48 =264.
    # Cross MACs: Q32 + K72 + V60 + QK24 + AV24 + output32 =244.
    # Self DMAs: projections132 + Qscale24 + QK42 + softmax168 + AV42 + output44.
    # Cross DMAs: projections145 + Qscale16 + QK32 + softmax112 + AV32 + output36.
    # Bias removal subtracts16 logical reads without changing MACs.
    expected_macs, expected_dma = (244, 373) if cross else (264, 452)
    assert _counts(attention, (query, key, value, None, False)) == (expected_macs, expected_dma - (0 if bias else 16))


@pytest.mark.parametrize(("need_weights", "average", "extra"), [(False, True, 0), (True, False, 0), (True, True, 27)])
def test_returned_weights_alias_probabilities_or_add_head_average(need_weights, average, extra):
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query = _tokens(3)
    # Averaging reads18 probability elements and writes9; no matrix MACs.
    assert _counts(attention, (query, query, query, None, need_weights, None, average)) == (264, 452 + extra)


@pytest.mark.parametrize("floating", [False, True])
def test_dense_masks_keep_matrix_macs_and_add_logical_accesses(floating):
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query = _tokens(3)
    causal, padding = torch.ones(3, 3, dtype=torch.bool).triu(1), torch.tensor([[False, False, True]])
    if floating:
        causal = torch.zeros(3, 3).masked_fill(causal, -torch.inf)
        padding = torch.zeros(1, 3).masked_fill(padding, -torch.inf)
    # Each mask reads its stored9 or3 entries and reads/writes all18 scores.
    assert _counts(attention, (query, query, query, padding, False, causal, True, True)) == (264, 536)


def test_per_head_batch_masks_read_all_stored_entries():
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query, masks = _tokens(3, batch=2), torch.zeros(4, 3, 3, dtype=torch.bool)
    # Activation stages double; parameters are read once. Mask reads36+2*36.
    assert _counts(attention, (query, query, query, None, False, masks)) == (528, 932)


@pytest.mark.parametrize("decoder", [False, True])
def test_independent_layer_counts(decoder):
    layer_type = nn.TransformerDecoderLayer if decoder else nn.TransformerEncoderLayer
    layer = layer_type(4, 2, 8, batch_first=True).eval()
    source, target = _tokens(3), _tokens(2)
    # Encoder MACs264+FF192+norm48; DMA452+FF196+residual72+norm192.
    # Decoder MACs160+208+FF128+norm48; DMA288+352+FF156+residual72+norm201.
    assert _counts(layer, (target, source) if decoder else (source,)) == ((544, 1069) if decoder else (504, 912))


@pytest.mark.parametrize("kind", ["encoder", "decoder", "transformer"])
def test_independent_stacks_and_final_normalization(kind):
    source, target = _tokens(3), _tokens(2)
    if kind == "encoder":
        module = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(4, 2, 8, batch_first=True), 2, nn.LayerNorm(4), enable_nested_tensor=False
        ).eval()
        inputs, expected = (source,), (1032, 1920)
    elif kind == "decoder":
        module = nn.TransformerDecoder(nn.TransformerDecoderLayer(4, 2, 8, batch_first=True), 2, nn.LayerNorm(4)).eval()
        inputs, expected = (target, source), (1104, 2205)
    else:
        module = nn.Transformer(4, 2, 1, 1, 8, batch_first=True).eval()
        inputs, expected = (source, target), (1088, 2144)
    # Final norm adds variance+affine24 MAC/96 DMA(src),16 MAC/67 DMA(tgt).
    # Stacks: 2*504+24, 2*912+96; 2*544+16, 2*1069+67.
    # Full model: 504+24+544+16; 912+96+1069+67.
    assert _counts(module, inputs) == expected


@pytest.mark.parametrize("bias", [False, True])
def test_bias_and_nonaffine_norm_have_independent_effects(bias):
    layer = nn.TransformerEncoderLayer(4, 2, 8, bias=bias, batch_first=True).eval()
    inputs = (_tokens(3),)
    # Bias removal loses16 attention,12 FF, and two4 normalization reads.
    assert _counts(layer, inputs) == (504, 912 if bias else 876)
    layer.norm1 = nn.LayerNorm(4, elementwise_affine=False).eval()
    layer.norm2 = nn.LayerNorm(4, elementwise_affine=False).eval()
    # Each norm loses12 affine MACs and read12/write12/parameter8 (or4) accesses.
    assert _counts(layer, inputs) == (480, 848 if bias else 820)


def test_ambiguous_causal_hint_and_native_subclass_are_rejected():
    class ModifiedAttention(nn.MultiheadAttention):
        pass

    query = _tokens(3)
    for module, inputs, message in (
        (
            nn.MultiheadAttention(4, 2, batch_first=True).eval(),
            (query,) * 3 + (None, False, None, True, True),
            "causal hint",
        ),
        (ModifiedAttention(4, 2, batch_first=True).eval(), (query,) * 3, "exact native"),
    ):
        with pytest.raises(NotImplementedError, match=message):
            validate_native_call(module, inputs)


@pytest.mark.parametrize("mask", [torch.ones(3, 3, dtype=torch.int64), torch.ones(1, 3, 3), torch.ones(2, 2)])
def test_non_native_mask_shapes_and_dtypes_are_unavailable(mask):
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    with pytest.raises(NotImplementedError, match="native mask shapes"):
        validate_native_call(attention, (_tokens(3),) * 3 + (None, False, mask))


def test_non_token_local_normalization_is_unavailable():
    layer = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True).eval()
    layer.norm1 = nn.LayerNorm((3, 4)).eval()
    with pytest.raises(NotImplementedError, match="token-local"):
        validate_native_call(layer, (_tokens(3),))


def test_decoder_attention_layouts_must_match():
    decoder = nn.TransformerDecoderLayer(4, 2, 8, batch_first=True).eval()
    decoder.multihead_attn.batch_first = False
    with pytest.raises(NotImplementedError, match="attention children must share"):
        validate_native_call(decoder, (_tokens(2, batch=2),) * 2)


def test_native_stack_with_custom_children_is_unavailable():
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(4, 2, 8, batch_first=True), 1, enable_nested_tensor=False
    ).eval()
    encoder.layers[0] = nn.Identity()
    with pytest.raises(NotImplementedError, match="exact native layer children"):
        validate_native_call(encoder, (_tokens(3),))


def test_modified_output_width_is_rejected_after_successful_native_execution():
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    attention.out_proj = nn.Linear(4, 6).eval()
    query, memory = _tokens(2), _tokens(3)
    inputs = query, memory, memory, None, False
    with torch.no_grad():
        output = attention(*inputs)
    assert output[0].shape == (1, 2, 6)
    for formula in (macs_attention, dmas_attention):
        with pytest.raises(NotImplementedError, match="native shapes"):
            formula(attention, inputs, output)


@pytest.mark.parametrize("stage", ["self_attn", "linear1", "norm1"])
def test_sparse_parameters_never_receive_dense_counts(stage):
    layer = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True).eval()
    operand = getattr(layer, stage)
    parameter = "in_proj_weight" if stage == "self_attn" else "weight"
    setattr(operand, parameter, nn.Parameter(getattr(operand, parameter).to_sparse()))
    for formula in (macs_attention, dmas_attention):
        with pytest.raises(NotImplementedError, match="real dense floating"):
            formula(layer, (_tokens(3),), None)


@pytest.mark.parametrize("stage", ["linear1", "norm1"])
def test_unused_registered_parameters_do_not_add_native_stage_dma_reads(stage):
    layer = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True).eval()
    source = _tokens(3)
    with torch.no_grad():
        expected_output = layer(source)
    operand = getattr(layer, stage)
    operand.register_parameter("unused", nn.Parameter(torch.ones(100)))
    operand.unused_child = nn.Linear(3, 3).eval()
    with torch.no_grad():
        output = layer(source)
    torch.testing.assert_close(output, expected_output)
    assert dmas_attention(layer, (source,), output) == 912
    report = crawl_module(layer, args=(source,))
    assert report["totals"]["dmas"]["value"] == 912
    assert report["totals"]["parameters"]["value"] == 172 + 100 + 12


@pytest.mark.parametrize(("affine", "expected"), [(True, 96), (False, 64)])
def test_standalone_layernorm_dma_reads_only_affine_operands(affine, expected):
    norm, source = nn.LayerNorm(4, elementwise_affine=affine).eval(), _tokens(3)
    with torch.no_grad():
        expected_output = norm(source)
    norm.register_parameter("unused", nn.Parameter(torch.ones(100)))
    norm.unused_child = nn.Linear(3, 3).eval()
    with torch.no_grad():
        output = norm(source)
    torch.testing.assert_close(output, expected_output)
    # Statistics/normalize4N+5rows+1=64; affine adds2N+weight4+bias4=32.
    assert module_dmas(norm, source, output) == expected
