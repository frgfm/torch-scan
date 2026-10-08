import pytest
import torch
from torch import nn

from torchscan import crawl_module


@pytest.mark.parametrize(
    ("rank", "adaptive", "visits", "outputs"),
    [
        (1, False, 16, 8),
        (2, False, 128, 32),
        (3, False, 1024, 128),
        (1, True, 14, 6),
        (2, True, 112, 12),
        (3, True, 672, 36),
    ],
)
@pytest.mark.parametrize("maximum", [False, True])
def test_pooling_module_function_geometry(rank, adaptive, visits, outputs, maximum):
    prefix = "Adaptive" if adaptive else ""
    kind = "Max" if maximum else "Avg"
    shape = (5, 7, 4)[:rank] if adaptive else (8,) * rank
    size = (3, 2, 3)[:rank] if adaptive else 2
    module = getattr(nn, f"{prefix}{kind}Pool{rank}d")(size)
    for batch in (1, 0):
        report = crawl_module(module, args=(torch.ones(batch, 2, *shape),), strict=True)
        flops = visits - outputs if maximum else visits
        assert report["totals"]["module_flops"]["value"] == report["totals"]["operator_flops"]["value"] == flops * batch
        assert report["totals"]["macs"]["value"] == (flops + (0 if maximum else outputs * (rank - 1))) * batch
        assert report["totals"]["dmas"]["value"] == (visits + outputs) * batch


def test_unbatched_pool_indices_and_singleton_kernel():
    for module in (nn.MaxPool2d((2,), return_indices=True), nn.AdaptiveMaxPool2d((2, 2), return_indices=True)):
        report = crawl_module(module, args=(torch.ones(2, 4, 4),), strict=True)
        assert report["totals"]["module_flops"]["value"] == report["totals"]["operator_flops"]["value"] == 24
        assert report["totals"]["macs"]["value"] == 24
        assert report["totals"]["dmas"]["value"] == 48
