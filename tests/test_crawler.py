import io
import sys
from collections import OrderedDict

import pytest
import torch.nn as nn

from torchscan import crawler


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


def test_crawl_module_two_outputs():
    class TwoOutputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            out = self.linear(x)
            return out, out[:, :2]

    res = crawler.crawl_module(TwoOutputs(), (4,))
    assert res["layers"][0]["output_shape"] == ((-1, 4), (-1, 2))


def test_crawl_module_nested_outputs():
    class NestedOutputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            out = self.linear(x)
            return out, [out[:, :2], {"scores": out[:, :1], "optional": None}]

    output_shape = crawler.crawl_module(NestedOutputs(), (4,))["layers"][0]["output_shape"]
    assert output_shape == ((-1, 4), [(-1, 2), {"scores": (-1, 1), "optional": None}])
    assert list(output_shape[1][1]) == ["scores", "optional"]


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

    crawler.summary(mod, input_shapes)
    assert "MultiheadAttention" in capsys.readouterr().out


def test_crawl_module_rejects_unsupported_output_leaf():
    class UnsupportedOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return {"value": self.linear(x), "label": "unsupported"}

    with pytest.raises(TypeError, match=r"output\['label'\].*str"):
        crawler.crawl_module(UnsupportedOutput(), (4,))


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
