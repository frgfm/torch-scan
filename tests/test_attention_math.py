import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torchscan import crawl_module, measure_flops
from torchscan._flop_formulas import FORMULAS


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (lambda x: x.sin(), 8),
        (lambda x: x.sin_(), 8),
        (lambda x: x.cos(), 8),
        (lambda x: x.cos_(), 8),
        (lambda x: x.log_(), 8),
        (lambda x: x.reciprocal(), 8),
        (lambda x: x.reciprocal_(), 8),
        (lambda x: x.abs_(), 8),
        (lambda x: x.ceil_(), 8),
        (lambda x: torch.rsub(x, 2, alpha=3), 16),
        (lambda x: x.pow_(3), 8),
        (lambda x: torch.pow(2, x), 8),
        (lambda x: torch.pow(x.unsqueeze(-1), torch.ones(3)), 24),
        (lambda x: x.long().pow(3), 0),
        (lambda x: x.long().pow(0.5), 8),
        (lambda x: F.dropout(x, training=True), 16),
        (lambda x: (torch.rand(2, 4), torch.rand_like(x), x.bernoulli(), ~x.bool()), 0),
        (lambda x: x.cumsum(-1), 6),
        (lambda x: x.cumsum_(-1), 6),
        (lambda x: x.max(-1).values, 6),
        (lambda x: x.max(), 7),
        (lambda x: torch.linalg.vector_norm(x), 16),
        (lambda x: F.normalize(x), 26),
        (lambda x: torch.addcmul(x, x, x), 16),
        (lambda x: x @ x[0], 16),
        (lambda x: torch.dot(x[0], x[1]), 8),
    ],
)
def test_native_math_counts(function, expected):
    assert measure_flops(lambda: function(torch.ones(2, 4)))["total"]["value"] == expected


@pytest.mark.parametrize("dim", [0, 2])
def test_native_weight_norm(dim):
    module = nn.utils.parametrizations.weight_norm(nn.Conv1d(3, 2, 4, bias=False), dim=dim)
    # 24 squares, 24-R adds, R roots, R divides and 24 multiplies.
    assert measure_flops(lambda: module.weight)["total"]["value"] == {0: 74, 2: 76}[dim]


def test_grouped_query_attention_and_fused_native_mha():
    query, kv = torch.ones(1, 4, 8, 8), torch.ones(1, 2, 8, 8)
    if "enable_gqa" in F.scaled_dot_product_attention.__doc__:
        report = measure_flops(lambda: F.scaled_dot_product_attention(query, kv, kv, enable_gqa=True))
        if "aten._scaled_dot_product_flash_attention_for_cpu" in report["by_operator"]:
            assert report["total"]["value"] == 9664
    else:
        assert FORMULAS["_scaled_dot_product_flash_attention_for_cpu"](query.shape, kv.shape, kv.shape) == 9664
    query = torch.ones(2, 4, 8)
    report = crawl_module(nn.MultiheadAttention(8, 2, batch_first=True), args=(query, query, query), strict=True)
    assert report["totals"]["module_flops"]["value"] == 5408
    if "aten._native_multi_head_attention" in report["operator_flops"]["by_operator"]:
        assert report["totals"]["operator_flops"]["value"] == 5536


def test_math_boundaries_and_reduction_dtype():
    assert measure_flops(lambda: torch.ones(2, 4, dtype=torch.complex64).pow(0.5))["total"]["status"] == "partial"
    assert measure_flops(lambda: torch._weight_norm(torch.ones(3, 0), torch.ones(3, 1)))["total"]["status"] == "partial"
    assert measure_flops(lambda: torch.ones(2, 4, dtype=torch.int64).cumsum(-1))["total"]["value"] == 0
    assert (
        measure_flops(lambda: torch.ones(2, 4, dtype=torch.int64).cumsum(-1, dtype=torch.float32))["total"]["value"]
        == 6
    )
    assert measure_flops(lambda: torch.linalg.vector_norm(torch.ones(2, 4), ord=3))["total"]["status"] == "partial"
    assert FORMULAS["_grouped_mm"]((5, 4), (3, 4, 2)) == 80
    with pytest.raises(NotImplementedError):
        FORMULAS["_grouped_mm"]((5, 4), (4, 2))


def test_arithmetic_operand_dtype():
    integer = torch.ones(2, 4, dtype=torch.int64)
    for value in (1, 1.5):
        assert (
            measure_flops(lambda value=value: torch.addcmul(integer, integer, integer, value=value))["total"]["value"]
            == 0
        )
        assert (
            measure_flops(lambda value=value: integer.clone().addcmul_(integer, integer, value=value))["total"]["value"]
            == 0
        )
    floating = integer.float()
    assert measure_flops(lambda: torch.addcmul(integer, floating, integer, value=1.5))["total"]["value"] == 24
    assert measure_flops(lambda: torch.addcdiv(integer, floating, integer))["total"]["value"] == 16
    if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 14):
        assert measure_flops(lambda: torch.pow(integer, 3, out=floating))["total"]["value"] == 0
        assert (
            measure_flops(lambda: torch.addcmul(integer, integer, integer, value=1.5, out=floating))["total"]["value"]
            == 0
        )
