"""Independent tiny matrix and logical-access examples for native Transformers."""

import pytest
import torch
from torch import nn

from torchscan.modules._transformer import dmas_attention, macs_attention, validate_native_call


def _tokens(length, width=4, *, batch_first=True, batch=1):
    shape = (batch, length, width) if batch_first else (length, batch, width)
    return torch.ones(shape)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_independent_self_attention_contractions_and_stages(batch_first, bias):
    attention = nn.MultiheadAttention(4, 2, batch_first=batch_first, bias=bias).eval()
    query = _tokens(3, batch_first=batch_first)
    inputs = (query, query, query, None, False)
    with torch.no_grad():
        output = attention(*inputs)
    # Three projections: 3x3x4x4=144; QK and AV: each 2x3x3x2=36;
    # output projection: 3x4x4=48. Bias adds no contraction terms.
    assert macs_attention(attention, inputs, output) == 264
    # Logical stages: projections 3*(12+16+[4]+12), Q scale24,
    # QK 12+12+18, stable softmax8*18+4*6, AV18+12+12,
    # output12+16+[4]+12. Bracketed biases add16 accesses in total.
    assert dmas_attention(attention, inputs, output) == (452 if bias else 436)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_independent_unequal_cross_attention(batch_first, bias):
    attention = nn.MultiheadAttention(4, 2, kdim=6, vdim=5, bias=bias, batch_first=batch_first).eval()
    inputs = (
        _tokens(2, batch_first=batch_first),
        _tokens(3, 6, batch_first=batch_first),
        _tokens(3, 5, batch_first=batch_first),
        None,
        False,
    )
    with torch.no_grad():
        output = attention(*inputs)
    # Q projection32, K projection72, V projection60, dense products24+24,
    # output projection32. This is not half the exact arithmetic FLOPs.
    assert macs_attention(attention, inputs, output) == 244
    # Projection stages: Q8+16+[4]+8, K18+24+[4]+12,
    # V15+20+[4]+12; Q scale16; QK8+12+12; softmax96+16;
    # AV12+12+8; output8+16+[4]+8. Bias adds16.
    assert dmas_attention(attention, inputs, output) == (373 if bias else 357)


@pytest.mark.parametrize(("need_weights", "average", "extra"), [(False, True, 0), (True, False, 0), (True, True, 27)])
def test_returned_attention_weights_are_logical_alias_or_head_average(need_weights, average, extra):
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query = _tokens(3)
    inputs = query, query, query, None, need_weights, None, average
    with torch.no_grad():
        output = attention(*inputs)
    assert macs_attention(attention, inputs, output) == 264
    assert dmas_attention(attention, inputs, output) == 452 + extra
    if need_weights:
        assert output[1].shape == ((1, 3, 3) if average else (1, 2, 3, 3))
    else:
        assert output[1] is None


@pytest.mark.parametrize("floating", [False, True])
def test_dense_masks_keep_matrix_macs_and_add_stored_reads_and_score_updates(floating):
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query = _tokens(3)
    causal = torch.ones(3, 3, dtype=torch.bool).triu(1)
    padding = torch.tensor([[False, False, True]])
    if floating:
        causal = torch.zeros(3, 3).masked_fill(causal, -torch.inf)
        padding = torch.zeros(1, 3).masked_fill(padding, -torch.inf)
    inputs = query, query, query, padding, False, causal, True, True
    with torch.no_grad():
        output = attention(*inputs)
    # Each mask updates all18 dense scores. Stored mask reads9+3.
    assert macs_attention(attention, inputs, output) == 264
    assert dmas_attention(attention, inputs, output) == 452 + 9 + 36 + 3 + 36


def test_per_head_batch_masks_scale_logical_stored_operand_reads():
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query = _tokens(3, batch=2)
    masks = torch.zeros(4, 3, 3, dtype=torch.bool)
    inputs = query, query, query, None, False, masks
    with torch.no_grad():
        output = attention(*inputs)
    # Parameters are read once/stage, activation stages double for batch2.
    assert macs_attention(attention, inputs, output) == 528
    assert dmas_attention(attention, inputs, output) == 824 + 36 + 72


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_independent_encoder_and_decoder_layer_counts(batch_first, norm_first, activation):
    kwargs = {
        "d_model": 4,
        "nhead": 2,
        "dim_feedforward": 8,
        "batch_first": batch_first,
        "norm_first": norm_first,
        "activation": activation,
    }
    encoder = nn.TransformerEncoderLayer(**kwargs).eval()
    decoder = nn.TransformerDecoderLayer(**kwargs).eval()
    source, target = _tokens(3, batch_first=batch_first), _tokens(2, batch_first=batch_first)
    with torch.no_grad():
        encoder_output, decoder_output = encoder(source), decoder(target, source)
    # Encoder matrices264+2*3*4*8=456. Two variance12 + affine12
    # normalization blocks add48. Decoder matrices self160+cross208+FF128;
    # three variance8 + affine8 blocks add48.
    assert macs_attention(encoder, (source,), encoder_output) == 504
    assert macs_attention(decoder, (target, source), decoder_output) == 544
    # Encoder attention452+FF196+two residual36+two norm96=912.
    # Decoder self288+cross352+FF156+three residual24+three norm67=1069.
    assert dmas_attention(encoder, (source,), encoder_output) == 912
    assert dmas_attention(decoder, (target, source), decoder_output) == 1069


@pytest.mark.parametrize("bias", [False, True])
def test_bias_and_nonaffine_normalization_have_independent_mac_and_dma_effects(bias):
    layer = nn.TransformerEncoderLayer(4, 2, 8, bias=bias, batch_first=True).eval()
    source = _tokens(3)
    with torch.no_grad():
        output = layer(source)
    assert macs_attention(layer, (source,), output) == 504
    # Bias=False removes16 attention,12 FF, and two4 normalization reads.
    assert dmas_attention(layer, (source,), output) == (912 if bias else 876)
    layer.norm1 = nn.LayerNorm(4, elementwise_affine=False).eval()
    layer.norm2 = nn.LayerNorm(4, elementwise_affine=False).eval()
    # Native encoder fusion dereferences absent affine tensors on some Torch
    # versions. A hook exercises the dense fallback, as crawling does.
    handle = layer.register_forward_hook(lambda *_args: None)
    try:
        with torch.no_grad():
            output = layer(source)
    finally:
        handle.remove()
    assert macs_attention(layer, (source,), output) == 480
    # Each norm loses affine read12/write12 and parameter reads8 (or4).
    assert dmas_attention(layer, (source,), output) == (848 if bias else 820)


@pytest.mark.parametrize("batch_first", [False, True])
def test_independent_native_stacks_and_final_normalization(batch_first):
    kwargs = {"d_model": 4, "nhead": 2, "dim_feedforward": 8, "batch_first": batch_first}
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(**kwargs), 2, nn.LayerNorm(4), enable_nested_tensor=False
    ).eval()
    decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**kwargs), 2, nn.LayerNorm(4)).eval()
    model = nn.Transformer(**kwargs, num_encoder_layers=1, num_decoder_layers=1).eval()
    source, target = _tokens(3, batch_first=batch_first), _tokens(2, batch_first=batch_first)
    with torch.no_grad():
        encoder_output, decoder_output, output = encoder(source), decoder(target, source), model(source, target)
    # Final LayerNorm adds variance+affine24(src),16(tgt) MACs and
    # mean/variance/normalize/affine96(src),67(tgt) logical accesses.
    assert macs_attention(encoder, (source,), encoder_output) == 1032
    assert macs_attention(decoder, (target, source), decoder_output) == 1104
    assert macs_attention(model, (source, target), output) == 1088
    assert dmas_attention(encoder, (source,), encoder_output) == 1920
    assert dmas_attention(decoder, (target, source), decoder_output) == 2205
    assert dmas_attention(model, (source, target), output) == 2144


@pytest.mark.parametrize("option", ["add_bias_kv", "add_zero_attn"])
def test_extra_attention_tokens_are_explicitly_unsupported(option):
    attention = nn.MultiheadAttention(4, 2, batch_first=True, **{option: True}).eval()
    query = _tokens(3)
    for formula in (macs_attention, dmas_attention):
        with pytest.raises(NotImplementedError, match="add_bias_kv or add_zero_attn"):
            formula(attention, (query,) * 3, None)


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


def test_nested_packing_padding_requires_dense_encoder_configuration():
    encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(4, 2, 8, batch_first=True), 1).eval()
    source, padding = _tokens(3), torch.tensor([[False, False, True]])
    with pytest.raises(NotImplementedError, match="enable_nested_tensor=False"):
        macs_attention(encoder, (source, None, padding), None)
    encoder.enable_nested_tensor = encoder.use_nested_tensor = False
    # Attention mask operand3 reads plus two18 score accesses.
    assert macs_attention(encoder, (source, None, padding), None) == 504
    assert dmas_attention(encoder, (source, None, padding), None) == 912 + 3 + 36


@pytest.mark.parametrize("shape", [(3, 4), (1, 0, 4), (0, 3, 4)])
def test_unbatched_and_empty_attention_axes_are_unavailable(shape):
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query = torch.ones(shape)
    with pytest.raises(NotImplementedError, match=r"batched 3D|nonempty"):
        validate_native_call(attention, (query,) * 3)


@pytest.mark.parametrize("mask", [torch.ones(3, 3, dtype=torch.int64), torch.ones(1, 3, 3), torch.ones(2, 2)])
def test_non_native_mask_shapes_and_dtypes_are_unavailable(mask):
    attention = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    query = _tokens(3)
    with pytest.raises(NotImplementedError, match="native mask shapes"):
        validate_native_call(attention, (query,) * 3 + (None, False, mask))


def test_non_token_local_normalization_is_unavailable():
    layer = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True).eval()
    layer.norm1 = nn.LayerNorm((3, 4)).eval()
    with pytest.raises(NotImplementedError, match="token-local"):
        validate_native_call(layer, (_tokens(3),))


def test_native_stack_and_decoder_layout_mismatches_are_unavailable():
    source = _tokens(2, batch=2)
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(4, 2, 8, batch_first=True), 2, enable_nested_tensor=False
    ).eval()
    encoder.layers[1].self_attn.batch_first = False
    with pytest.raises(NotImplementedError, match="stack layers must share"):
        validate_native_call(encoder, (source,))
    decoder = nn.TransformerDecoderLayer(4, 2, 8, batch_first=True).eval()
    decoder.multihead_attn.batch_first = False
    with pytest.raises(NotImplementedError, match="attention children must share"):
        validate_native_call(decoder, (source, source))
    model = nn.Transformer(4, 2, 1, 1, 8, batch_first=True).eval()
    model.batch_first = False
    with pytest.raises(NotImplementedError, match="match the parent"):
        validate_native_call(model, (source, source))


def test_native_stack_with_custom_children_is_explicitly_unavailable():
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(4, 2, 8, batch_first=True), 1, enable_nested_tensor=False
    ).eval()
    encoder.layers[0] = nn.Identity()
    with pytest.raises(NotImplementedError, match="exact native layer children"):
        validate_native_call(encoder, (_tokens(3),))


def test_modified_attention_output_width_is_rejected_after_successful_native_execution():
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


@pytest.mark.parametrize("parameter", ["attention", "feedforward", "normalization"])
def test_native_class_with_sparse_parameters_never_receives_a_dense_count(parameter):
    layer = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True).eval()
    if parameter == "attention":
        layer.self_attn.in_proj_weight = nn.Parameter(layer.self_attn.in_proj_weight.to_sparse())
    elif parameter == "feedforward":
        layer.linear1.weight = nn.Parameter(layer.linear1.weight.to_sparse())
    else:
        layer.norm1.weight = nn.Parameter(layer.norm1.weight.to_sparse())
    for formula in (macs_attention, dmas_attention):
        with pytest.raises(NotImplementedError, match="real dense floating"):
            formula(layer, (_tokens(3),), None)
