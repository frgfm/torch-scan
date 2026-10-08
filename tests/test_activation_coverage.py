import pytest
import torch
from torch import nn

from torchscan import IncompleteAnalysisError, crawl_module, measure_flops


@pytest.mark.parametrize(
    ("module", "flops", "dmas"),
    [
        (nn.Hardtanh(), 16, 16),
        (nn.Hardsigmoid(), 32, 16),
        (nn.Hardswish(), 40, 16),
        (nn.Mish(), 80, 16),
        (nn.Softplus(), 56, 16),
        (nn.PReLU(), 32, 17),
        (nn.CELU(), 56, 16),
        (nn.SELU(), 56, 16),
        (nn.LogSigmoid(), 56, 16),
        (nn.Hardshrink(), 24, 16),
        (nn.Softshrink(), 40, 16),
        (nn.Softsign(), 24, 16),
        (nn.Tanhshrink(), 56, 16),
        (nn.Threshold(0, 0), 16, 16),
        (nn.RReLU(), 32, 16),
        (nn.Softmax(dim=-1), 36, 72),
        (nn.LogSoftmax(dim=-1), 30, 76),
        (nn.Softmin(dim=-1), 44, 88),
    ],
)
def test_activation_module_function_parity(module, flops, dmas):
    for batch in (2, 0):
        report = crawl_module(module, args=(torch.ones(batch, 4),), strict=True)
        for metric, expected in (("module_flops", flops), ("operator_flops", flops), ("macs", 0), ("dmas", dmas)):
            assert report["totals"][metric]["value"] == (expected if batch else 0)


@pytest.mark.parametrize(("module", "flops"), [(nn.ELU(), 48), (nn.LeakyReLU(), 32), (nn.ReLU6(), 16)])
def test_legacy_functional_activation_counts(module, flops):
    assert measure_flops(lambda: module(torch.ones(2, 4)))["total"]["value"] == flops


def test_softmax_channel_geometry_and_modified_forward():
    report = crawl_module(nn.Softmax2d(), args=(torch.ones(2, 4, 2, 2),), strict=True)
    assert report["totals"]["module_flops"]["value"] == report["totals"]["operator_flops"]["value"] == 144
    module = nn.Softplus()
    module.forward = lambda x: x.square()
    with pytest.raises(IncompleteAnalysisError):
        crawl_module(module, args=(torch.ones(2, 4),), strict=True)
