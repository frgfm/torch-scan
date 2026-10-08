import pytest
import torch
from torch import nn

from torchscan import modules
from torchscan.modules.flops import flops_transformer_decoderlayer, flops_transformer_encoderlayer


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()


def test_module_flops_warning():
    with pytest.warns(UserWarning, match="Module type not supported"):
        modules.module_flops(MyModule(), None, None)


@pytest.mark.parametrize(
    ("mod", "input_shape", "output_shape", "expected_val"),
    [
        # Check for unknown module that it returns 0 and throws a warning
        (MyModule(), (1,), (1,), 0),
        # Fully-connected
        (nn.Linear(8, 4), (1, 8), (1, 4), 4 * (2 * 8 - 1) + 4),
        (nn.Linear(8, 4, bias=False), (1, 8), (1, 4), 4 * (2 * 8 - 1)),
        (nn.Linear(8, 4), (1, 2, 8), (1, 2, 4), 2 * (4 * (2 * 8 - 1) + 4)),
        # Activations
        (nn.Identity(), (1, 8), (1, 8), 0),
        (nn.Flatten(), (1, 8), (1, 8), 0),
        (nn.ReLU(), (1, 8), (1, 8), 8),
        (nn.ELU(), (1, 8), (1, 8), 48),
        (nn.LeakyReLU(), (1, 8), (1, 8), 32),
        (nn.ReLU6(), (1, 8), (1, 8), 16),
        (nn.Tanh(), (1, 8), (1, 8), 48),
        (nn.Sigmoid(), (1, 8), (1, 8), 32),
        # BN
        (nn.BatchNorm1d(8), (1, 8, 4), (1, 8, 4), 336),
        # Pooling
        (nn.MaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32),
        (nn.AdaptiveMaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveMaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveAvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32),
        (nn.AdaptiveAvgPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32),
        # Dropout
        (nn.Dropout(), (1, 8), (1, 8), 16),
        (nn.Dropout(p=0), (1, 8), (1, 8), 0),
        # Conv
        (nn.Conv2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 30, 30), 388800),
        (nn.ConvTranspose2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 34, 34), 451616),
    ],
)
def test_module_flops(mod, input_shape, output_shape, expected_val):
    assert modules.module_flops(mod, (torch.zeros(input_shape),), torch.zeros(output_shape)) == expected_val


@pytest.mark.parametrize(
    ("elementwise_affine", "bias", "expected"), [(True, True, 204), (True, False, 180), (False, True, 156)]
)
def test_layernorm_flops(elementwise_affine, bias, expected):
    mod = nn.LayerNorm(4, elementwise_affine=elementwise_affine, bias=bias)
    input_t = torch.zeros((2, 3, 4))

    assert modules.module_flops(mod, (input_t,), mod(input_t)) == expected


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(("bias", "self_expected", "cross_expected"), [(False, 1080, 1592), (True, 1176, 1720)])
def test_multihead_attention_flops(batch_first, bias, self_expected, cross_expected):
    mod = nn.MultiheadAttention(4, 2, dropout=0, bias=bias, batch_first=batch_first)
    shape = lambda length: (2, length, 4) if batch_first else (length, 2, 4)
    query, key, value = torch.rand(shape(3)), torch.rand(shape(5)), torch.rand(shape(5))

    self_out = mod(query, query, query, need_weights=False)
    cross_out = mod(query, key, value, need_weights=False)
    assert modules.module_flops(mod, (query, query, query), self_out) == self_expected
    assert modules.module_flops(mod, (query, key, value), cross_out) == cross_expected

    averaged_out = mod(query, key, value)
    assert modules.module_flops(mod, (query, key, value), averaged_out) == cross_expected + 60

    mask = torch.zeros((3, 5))
    masked_out = mod(query, key, value, None, False, mask)
    assert modules.module_flops(mod, (query, key, value, None, False, mask), masked_out) == cross_expected + 60


def test_transformer_flops():
    mod = nn.Transformer(
        d_model=4,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
        batch_first=True,
    )
    src = torch.rand((1, 3, 4))
    tgt = torch.rand((1, 2, 4))

    assert flops_transformer_encoderlayer(mod.encoder.layers[0], (src,)) == 1224
    assert flops_transformer_decoderlayer(mod.decoder.layers[0], (tgt, src)) == 1300
    assert modules.module_flops(mod, (src, tgt), mod(src, tgt)) == 2694

    masks = (
        torch.zeros((3, 3), dtype=torch.bool),
        torch.zeros((2, 2), dtype=torch.bool),
        torch.zeros((2, 3), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros((1, 2), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
    )
    assert modules.module_flops(mod, (src, tgt, *masks), mod(src, tgt, *masks)) == 2770


def test_transformer_flops_rejects_unverified_options():
    query = torch.rand((2, 3, 4))
    mod = nn.MultiheadAttention(4, 2, batch_first=True, add_zero_attn=True)
    with pytest.raises(NotImplementedError, match="add_bias_kv or add_zero_attn"):
        modules.module_flops(mod, (query, query, query), None)

    transformer = nn.Transformer(d_model=4, nhead=2, dim_feedforward=8, activation=torch.sin, batch_first=True)
    with pytest.raises(NotImplementedError, match="ReLU or GELU"):
        modules.module_flops(transformer, (query, query), None)


def test_module_macs_warning():
    with pytest.warns(UserWarning, match="Module type not supported"):
        modules.module_macs(MyModule(), None, None)


@pytest.mark.parametrize(
    ("mod", "input_shape", "output_shape", "expected_val"),
    [
        # Check for unknown module that it returns 0 and throws a warning
        (MyModule(), (1,), (1,), 0),
        # Fully-connected
        (nn.Linear(8, 4), (1, 8), (1, 4), 8 * 4),
        (nn.Linear(8, 4), (1, 2, 8), (1, 2, 4), 8 * 4 * 2),
        # Activations
        (nn.ReLU(), (1, 8), (1, 8), 0),
        # BN
        (nn.BatchNorm1d(8), (1, 8, 4), (1, 8, 4), 64 + 24 + 56 + 32),
        # Pooling
        (nn.MaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        (nn.AdaptiveMaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveMaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveAvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        (nn.AdaptiveAvgPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        # Dropout
        (nn.Dropout(), (1, 8), (1, 8), 0),
        # Conv
        (nn.Conv2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 30, 30), 194400),
        (nn.ConvTranspose2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 34, 34), 221184),
    ],
)
def test_module_macs(mod, input_shape, output_shape, expected_val):
    assert modules.module_macs(mod, torch.zeros(input_shape), torch.zeros(output_shape)) == expected_val


def test_module_dmas_warning():
    with pytest.warns(UserWarning, match="Module type not supported"):
        modules.module_dmas(MyModule(), None, None)


@pytest.mark.parametrize(
    ("mod", "input_shape", "output_shape", "expected_val"),
    [
        # Check for unknown module that it returns 0 and throws a warning
        (MyModule(), (1,), (1,), 0),
        # Fully-connected
        (nn.Linear(8, 4), (1, 8), (1, 4), 4 * (8 + 1) + 8 + 4),
        (nn.Linear(8, 4), (1, 2, 8), (1, 2, 4), 4 * (8 + 1) + 2 * (8 + 4)),
        # Activations
        (nn.Identity(), (1, 8), (1, 8), 8),
        (nn.Flatten(), (1, 8), (1, 8), 16),
        (nn.ReLU(), (1, 8), (1, 8), 8 * 2),
        (nn.ReLU(inplace=True), (1, 8), (1, 8), 8),
        (nn.ELU(), (1, 8), (1, 8), 17),
        (nn.Tanh(), (1, 8), (1, 8), 24),
        (nn.Sigmoid(), (1, 8), (1, 8), 16),
        # BN
        (nn.BatchNorm1d(8), (1, 8, 4), (1, 8, 4), 32 + 17 + 16 + 1 + 17 + 32),
        # Pooling
        (nn.MaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        (nn.MaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        (nn.AdaptiveMaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        (nn.AdaptiveMaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        # Dropout
        (nn.Dropout(), (1, 8), (1, 8), 17),
        # Conv
        (nn.Conv2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 30, 30), 201824),
        (nn.ConvTranspose2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 34, 34), 259178),
    ],
)
def test_module_dmas(mod, input_shape, output_shape, expected_val):
    assert modules.module_dmas(mod, torch.zeros(input_shape), torch.zeros(output_shape)) == expected_val


def test_module_rf_conv_transpose():
    mod = nn.ConvTranspose2d(3, 8, 3)
    input_t = torch.rand((1, 3, 32, 32))
    assert modules.module_rf(mod, input_t, mod(input_t)) == (-3, 1, 0)


@pytest.mark.parametrize("kind", ["Max", "Avg"])
@pytest.mark.parametrize(
    ("dimension", "input_shape", "output_shape", "max_ops", "avg_ops", "dmas"),
    [
        (1, (2, 4, 7), (2, 4, 3), 48, 72, 96),
        (2, (2, 4, 7, 8), (2, 4, 3, 4), 480, 576, 672),
        (3, (2, 4, 7, 8, 5), (2, 4, 3, 4, 2), 3264, 3456, 3648),
    ],
)
def test_adaptive_pool_metrics_for_non_divisible_spatial_shapes(
    kind, dimension, input_shape, output_shape, max_ops, avg_ops, dmas
):
    module = getattr(nn, f"Adaptive{kind}Pool{dimension}d")(output_shape[2:])
    inp, out = torch.zeros(input_shape), torch.zeros(output_shape)
    expected_ops = max_ops if kind == "Max" else avg_ops

    assert modules.module_flops(module, (inp,), out) == expected_ops
    assert modules.module_macs(module, inp, out) == expected_ops + (
        out.numel() * (dimension - 1) if kind == "Avg" else 0
    )
    assert modules.module_dmas(module, inp, out) == dmas
