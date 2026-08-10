import io
import sys
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from torchscan import crawler, modules


def test_apply():
    multi_convs = nn.Sequential(nn.Conv2d(16, 32, 3), nn.Conv2d(32, 64, 3))
    mod = nn.Sequential(nn.Conv2d(3, 16, 3), multi_convs)

    # Tag module attributes
    def tag_name(mod, name):
        mod.__depth__ = len(name.split(".")) - 1
        mod.__name__ = name.rpartition(".")[-1]

    crawler.apply(mod, tag_name)

    assert mod[1][1].__depth__ == 2
    assert mod[1][1].__name__ == "1"


def test_crawl_module(capsys):
    mod = nn.Conv2d(3, 8, 3)

    res = crawler.crawl_module(mod, (3, 32, 32))
    assert isinstance(res, dict)
    assert res["overall"]["grad_params"] == 224
    assert res["layers"][0]["output_shape"] == (-1, 8, 30, 30)

    crawler.summary(mod, (3, 32, 32))
    assert "conv2d    Conv2d    (-1, 8, 30, 30)    224" in capsys.readouterr().out


def test_crawl_module_shared_parameters_and_buffers():
    mod = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    mod[1].weight = mod[0].weight
    mod[0].register_buffer("stats", torch.ones(4))
    mod[1].register_buffer("stats", mod[0].stats)
    res = crawler.crawl_module(mod, (4,))
    layers = {layer["name"]: layer for layer in res["layers"]}
    num_params = mod[0].weight.numel()
    num_buffers = mod[0].stats.numel()

    assert res["overall"]["grad_params"] == num_params
    assert res["overall"]["param_size"] == num_params * mod[0].weight.element_size()
    assert res["overall"]["num_buffers"] == num_buffers
    assert res["overall"]["buffer_size"] == num_buffers * mod[0].stats.element_size()
    assert (layers["0"]["grad_params"], layers["1"]["grad_params"]) == (num_params, 0)
    assert (layers["0"]["num_buffers"], layers["1"]["num_buffers"]) == (num_buffers, 0)
    assert layers["0"]["is_shared"] is False
    assert layers["1"]["is_shared"] is True


def test_crawl_module_two_outputs():
    class TwoOutputs(nn.Conv2d):
        def forward(self, x):
            out = super().forward(x)
            return out, out[..., :1, :1]

    layer = crawler.crawl_module(TwoOutputs(1, 2, 3), (1, 5, 5))["layers"][0]
    plain_layer = crawler.crawl_module(nn.Conv2d(1, 2, 3), (1, 5, 5))["layers"][0]
    assert layer["output_shape"] == ((-1, 2, 3, 3), (-1, 2, 1, 1))
    assert tuple(layer[key] for key in ("flops", "macs", "dmas", "rf", "s", "p")) == tuple(
        plain_layer[key] for key in ("flops", "macs", "dmas", "rf", "s", "p")
    )


def test_crawl_module_nested_outputs(capsys):
    class NestedOutputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            out = self.linear(x)
            return out, [out[:, :2], {"scores": out[:, :1], "optional": None}]

    mod = NestedOutputs()
    output_shape = crawler.crawl_module(mod, (4,))["layers"][0]["output_shape"]
    assert output_shape == ((-1, 4), [(-1, 2), {"scores": (-1, 1), "optional": None}])
    assert list(output_shape[1][1]) == ["scores", "optional"]

    crawler.summary(mod, (4,))
    line = next(line for line in capsys.readouterr().out.splitlines() if "NestedOutputs" in line)
    assert "[...]" in line


def test_crawl_module_maxpool_with_indices():
    mod = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.MaxPool2d(2, return_indices=True))
    plain_mod = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.MaxPool2d(2))

    res = crawler.crawl_module(mod, (1, 4, 4))
    pool = next(layer for layer in res["layers"] if layer["type"] == "MaxPool2d")
    plain_res = crawler.crawl_module(plain_mod, (1, 4, 4))
    plain_pool = next(layer for layer in plain_res["layers"] if layer["type"] == "MaxPool2d")

    assert pool["output_shape"] == ((-1, 1, 2, 2), (-1, 1, 2, 2))
    assert tuple(pool[key] for key in ("flops", "macs", "dmas")) == tuple(
        plain_pool[key] for key in ("flops", "macs", "dmas")
    )


def test_crawl_module_multihead_attention(capsys):
    mod = nn.MultiheadAttention(8, 2, batch_first=True)
    input_shapes = [(4, 8)] * 3

    res = crawler.crawl_module(mod, input_shapes)
    assert res["layers"][0]["output_shape"] == ((-1, 4, 8), (-1, 4, 4))
    query = torch.rand((1, 4, 8))
    assert len(res["layers"]) == 1
    assert res["layers"][0]["flops"] == modules.module_flops(mod, (query, query, query), mod(query, query, query))

    crawler.summary(mod, input_shapes)
    assert "MultiheadAttention" in capsys.readouterr().out


def _tiny_transformer(batch_first):
    return nn.Transformer(
        d_model=4,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
        batch_first=batch_first,
    )


def test_crawl_module_batch_first_transformer(capsys):
    mod = _tiny_transformer(batch_first=True)

    with pytest.warns(UserWarning, match="Module type not supported"):
        result = crawler.crawl_module(mod, [(3, 4), (2, 4)])

    assert len(result["layers"]) == 1
    assert result["layers"][0]["type"] == "Transformer"
    assert result["layers"][0]["flops"] == 2635
    assert sum(layer["flops"] for layer in result["layers"]) == 2635

    crawler.summary(mod, [(3, 4), (2, 4)])
    assert "Transformer" in capsys.readouterr().out


def test_crawl_module_sequence_first_transformer_input_data(capsys):
    mod = _tiny_transformer(batch_first=False)
    src = torch.rand((3, 1, 4))
    tgt = torch.rand((2, 1, 4))

    with pytest.warns(UserWarning, match="Module type not supported"):
        result = crawler.crawl_module(mod, input_data=(src, tgt))

    assert len(result["layers"]) == 1
    assert result["layers"][0]["flops"] == 2635

    src_mask = torch.zeros((3, 3), dtype=torch.bool)
    with pytest.warns(UserWarning, match="Module type not supported"):
        masked_result = crawler.crawl_module(mod, input_data=(src, tgt, src_mask))
    assert masked_result["layers"][0]["flops"] == 2653

    with pytest.warns(UserWarning, match="Module type not supported"):
        crawler.summary(mod, input_data=(src, tgt))
    assert "Transformer" in capsys.readouterr().out


def test_crawl_module_rejects_wrapped_transformer():
    with pytest.raises(NotImplementedError, match="passed directly"):
        crawler.crawl_module(nn.Sequential(_tiny_transformer(batch_first=True)), [(3, 4), (2, 4)])


def test_crawl_module_removes_hooks_after_metric_failure():
    mod = nn.MultiheadAttention(4, 2, batch_first=True, add_zero_attn=True)
    query = torch.rand((1, 3, 4))
    expected_hook_counts = len(mod._forward_pre_hooks), len(mod._forward_hooks)

    for _ in range(2):
        with pytest.raises(NotImplementedError, match="add_bias_kv or add_zero_attn"):
            crawler.crawl_module(mod, input_data=(query, query, query))
        assert (len(mod._forward_pre_hooks), len(mod._forward_hooks)) == expected_hook_counts


def test_crawl_module_rejects_unsupported_output_leaf():
    class UnsupportedOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return {"value": self.linear(x), "label": "unsupported"}

    mod = UnsupportedOutput()
    with pytest.raises(TypeError, match=r"output\['label'\].*str"):
        crawler.crawl_module(mod, (4,))
    assert all(not child._forward_hooks and not child._forward_pre_hooks for child in mod.modules())


def test_crawl_module_rejects_output_without_tensor():
    class NoTensorOutput(nn.Linear):
        def forward(self, _):
            return {"optional": None}

    with pytest.raises(TypeError, match="no tensor found"):
        crawler.crawl_module(NoTensorOutput(4, 4), (4,))


def test_crawl_module_multiple_inputs():
    class TwoInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 2)
            self.right = nn.Linear(6, 2)

        def forward(self, left, right):
            return self.left(left) + self.right(right)

    res = crawler.crawl_module(TwoInputs(), [(4,), (6,)])
    assert res["overall"]["grad_params"] == 24


def test_crawl_module_generated_integer_input():
    class CaptureEmbedding(nn.Embedding):
        def forward(self, tensor):
            self.received = tensor
            return super().forward(tensor)

    mod = CaptureEmbedding(8, 4)
    crawler.crawl_module(mod, (3,), dtype=torch.long)

    assert mod.received.shape == (1, 3)
    assert mod.received.dtype == torch.long
    assert mod.received.device == mod.weight.device


def test_crawl_module_input_data():
    class CaptureLinear(nn.Linear):
        def forward(self, tensor):
            self.received = tensor
            self.grad_enabled = torch.is_grad_enabled()
            return super().forward(tensor)

    mod = CaptureLinear(4, 2)
    input_data = torch.randn(5, 2, 4)
    expected = input_data.clone()
    expected_device = input_data.device
    expected_dtype = input_data.dtype

    crawler.crawl_module(mod, input_data=input_data)

    assert mod.received is input_data
    assert torch.equal(input_data, expected)
    assert input_data.shape == (5, 2, 4)
    assert input_data.device == expected_device
    assert input_data.dtype == expected_dtype
    assert not mod.grad_enabled
    assert mod.training


def test_crawl_module_correlated_input_data():
    class CorrelatedInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)

        def forward(self, xs, xs_len):
            self.received = (xs, xs_len)
            if xs.shape[1] != xs_len.max().item():
                raise ValueError("xs and xs_len are not correlated")
            positions = torch.arange(xs.shape[1], device=xs.device)
            mask = positions < xs_len[:, None]
            return self.embedding(xs) * mask.unsqueeze(-1)

    mod = CorrelatedInputs()
    xs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
    xs_len = torch.tensor([2, 4])
    expected_xs = xs.clone()
    expected_xs_len = xs_len.clone()

    res = crawler.crawl_module(mod, input_data=(xs, xs_len))

    assert mod.received[0] is xs
    assert mod.received[1] is xs_len
    assert torch.equal(xs, expected_xs)
    assert torch.equal(xs_len, expected_xs_len)
    assert res["overall"]["grad_params"] == 32


def test_crawl_module_requires_one_input_source():
    mod = nn.Linear(4, 2)

    with pytest.raises(ValueError, match="Exactly one of input_shape and input_data"):
        crawler.crawl_module(mod)
    with pytest.raises(ValueError, match="Exactly one of input_shape and input_data"):
        crawler.crawl_module(mod, (4,), input_data=torch.randn(2, 4))


def test_crawl_module_rejects_dtype_with_input_data():
    with pytest.raises(ValueError, match="dtype cannot be used with input_data"):
        crawler.crawl_module(nn.Linear(4, 2), dtype=torch.float32, input_data=torch.randn(2, 4))


@pytest.mark.parametrize("input_data", [[], ()])
def test_crawl_module_rejects_empty_input_data(input_data):
    with pytest.raises(ValueError, match="input_data must not be empty"):
        crawler.crawl_module(nn.Linear(4, 2), input_data=input_data)


@pytest.mark.parametrize(
    ("input_data", "message"),
    [
        ({"input": torch.randn(2, 4)}, r"input_data.*dict"),
        ([torch.randn(2, 4), "invalid"], r"input_data\[1\].*str"),
        ((torch.randn(2, 4), 1), r"input_data\[1\].*int"),
    ],
)
def test_crawl_module_rejects_invalid_input_data(input_data, message):
    with pytest.raises(TypeError, match=message):
        crawler.crawl_module(nn.Linear(4, 2), input_data=input_data)


def test_summary_input_data(capsys):
    crawler.summary(nn.Linear(4, 2), input_data=torch.randn(2, 4))
    assert "Total params: 10" in capsys.readouterr().out


def test_summary():
    mod = nn.Conv2d(3, 8, 3)

    # Redirect stdout with StringIO object
    captured_output = io.StringIO()
    sys.stdout = captured_output
    crawler.summary(mod, (3, 32, 32))
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[7] == "Total params: 224"

    # Check receptive field
    captured_output = io.StringIO()
    sys.stdout = captured_output
    crawler.summary(mod, (3, 32, 32), receptive_field=True)
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[1].rpartition("  ")[-1] == "Receptive field"
    assert captured_output.getvalue().split("\n")[3].split()[-1] == "3"
    # Check effective stats
    captured_output = io.StringIO()
    sys.stdout = captured_output
    crawler.summary(mod, (3, 32, 32), receptive_field=True, effective_rf_stats=True)
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[1].rpartition("  ")[-1] == "Effective padding"
    assert captured_output.getvalue().split("\n")[3].split()[-1] == "0"

    # Max depth > model hierarchy
    with pytest.raises(ValueError):
        crawler.summary(mod, (3, 32, 32), max_depth=1)

    mod = nn.Sequential(
        OrderedDict([
            ("features", nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(inplace=True))),
            ("pool", nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))),
            ("classifier", nn.Linear(8, 1)),
        ])
    )

    captured_output = io.StringIO()
    sys.stdout = captured_output
    crawler.summary(mod, (3, 32, 32), max_depth=1)
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[4].startswith("├─features ")


def test_summary_trainable_column(capsys):
    mod = nn.Sequential(
        OrderedDict([
            ("activation", nn.ReLU()),
            ("trainable", nn.Linear(4, 4)),
            ("frozen", nn.Linear(4, 4)),
            ("mixed", nn.Linear(4, 4)),
            ("nested", nn.Sequential(nn.Linear(4, 4))),
        ])
    )
    mod.frozen.requires_grad_(False)
    mod.mixed.weight.requires_grad_(False)
    mod.nested[0].weight.requires_grad_(False)

    crawler.summary(mod, (4,))
    output = capsys.readouterr().out.splitlines()

    assert output[1].rpartition("  ")[-1] == "Trainable"
    assert next(line for line in output if "├─activation" in line).split()[-1] == "-"
    assert next(line for line in output if "├─trainable" in line).split()[-1] == "True"
    assert next(line for line in output if "├─frozen" in line).split()[-1] == "False"
    assert next(line for line in output if "├─mixed" in line).split()[-1] == "True"
    assert "Trainable params: 28" in output
    assert "Non-trainable params: 52" in output
    assert "Total params: 80" in output

    crawler.summary(mod, (4,), max_depth=1)
    aggregated_output = capsys.readouterr().out.splitlines()
    assert next(line for line in aggregated_output if "├─nested" in line).split()[-1] == "True"

    mod.nested[0].bias.requires_grad_(False)
    crawler.summary(mod, (4,), max_depth=1)
    frozen_aggregated_output = capsys.readouterr().out.splitlines()
    assert next(line for line in frozen_aggregated_output if "├─nested" in line).split()[-1] == "False"
