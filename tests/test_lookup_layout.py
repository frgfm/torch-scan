import pytest
import torch
from torch import nn

from torchscan import IncompleteAnalysisError, crawl_module, measure_flops


@pytest.mark.parametrize(
    ("module", "inputs", "flops", "macs", "dmas"),
    [
        (nn.Embedding(10, 4), torch.tensor([[1, 1], [2, 3]]), 0, 0, 36),
        (nn.Unflatten(1, (2, 2)), torch.ones(2, 4), 0, 0, 0),
        (nn.PixelShuffle(2), torch.ones(1, 4, 2, 2), 0, 0, 32),
        (nn.PixelUnshuffle(2), torch.ones(1, 1, 4, 4), 0, 0, 32),
        (nn.ChannelShuffle(2), torch.ones(1, 4, 2, 2), 0, 0, 32),
        (nn.Unfold(2), torch.ones(1, 1, 3, 3), 0, 0, 32),
        (nn.ConstantPad1d(1, 0), torch.ones(2, 4), 0, 0, 20),
        (nn.ConstantPad1d((-1, 0), 0), torch.ones(2, 4), 0, 0, 12),
        (nn.ConstantPad1d((-1, 0, -1, 0), 0), torch.ones(3, 4), 0, 0, 12),
        (nn.ConstantPad2d((1, 1), 0), torch.ones(3, 4), 0, 0, 30),
        (nn.ReflectionPad1d(1), torch.ones(2, 4), 0, 0, 24),
        (nn.ReplicationPad2d(1), torch.ones(1, 1, 2, 2), 0, 0, 32),
        (nn.CircularPad1d(1), torch.ones(2, 4), 0, 0, 24),
        (nn.Upsample(size=6, mode="linear"), torch.ones(1, 1, 3), 18, 12, 18),
        (nn.UpsamplingBilinear2d(size=4), torch.ones(1, 1, 2, 2), 112, 64, 80),
        (nn.Upsample(size=4, mode="trilinear"), torch.ones(1, 1, 2, 2, 2), 960, 512, 576),
        (nn.UpsamplingNearest2d(scale_factor=2), torch.ones(1, 1, 2, 2), 0, 0, 32),
        (nn.UpsamplingNearest2d(scale_factor=2), torch.ones(1, 1, 2, 2, dtype=torch.uint8), 0, 0, 32),
        (nn.Upsample(size=2, mode="area"), torch.ones(1, 1, 4, 4), 16, 20, 20),
        (nn.Upsample(size=4, mode="bicubic"), torch.ones(1, 1, 2, 2), 496, 256, 272),
    ],
)
def test_native_lookup_layout_counts(module, inputs, flops, macs, dmas):
    report = crawl_module(module, args=(inputs,), strict=True)
    for metric, expected in (("module_flops", flops), ("operator_flops", flops), ("macs", macs), ("dmas", dmas)):
        assert report["totals"][metric]["value"] == expected


def test_lookup_layout_boundaries_and_scalar_control():
    report = crawl_module(nn.Embedding(10, 4), args=(torch.zeros(0, 2, dtype=torch.int64),), strict=True)
    assert all(report["totals"][metric]["value"] == 0 for metric in ("module_flops", "operator_flops", "macs", "dmas"))
    with pytest.raises(IncompleteAnalysisError):
        crawl_module(nn.Embedding(10, 4, max_norm=1), args=(torch.ones(2, dtype=torch.int64),), strict=True)
    report = measure_flops(lambda: torch.arange(4).eq(2).all().item())
    assert report["total"]["value"] == 0
    assert measure_flops(lambda: torch.ones(4).eq(2).all())["total"]["value"] == 4
    for name in ("eq_", "ne_", "ge_", "gt_", "le_", "lt_"):
        assert measure_flops(lambda name=name: getattr(torch.ones(4), name)(2))["total"]["value"] == 4
        assert (
            measure_flops(lambda name=name: getattr(torch.ones(4, dtype=torch.int64), name)(2))["total"]["value"] == 0
        )
    for mode in ("bilinear", "bicubic"):
        with pytest.raises(IncompleteAnalysisError):
            crawl_module(nn.Upsample(size=4, mode=mode), args=(torch.ones(1, 1, 2, 2, dtype=torch.uint8),), strict=True)
