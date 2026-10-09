import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torchscan import IncompleteAnalysisError, crawl_module, measure_flops, modules
from torchscan._flop_formulas import FORMULAS

_RMS_NORM = getattr(nn, "RMSNorm", None)


@pytest.mark.parametrize(
    ("module", "shape", "flops", "macs", "dmas"),
    [
        (nn.GELU(), (2, 4), 40, 0, 16),
        (nn.GELU(approximate="tanh"), (2, 4), 112, 0, 16),
        (nn.SiLU(), (2, 4), 40, 0, 16),
        (nn.SiLU(inplace=True), (2, 4), 40, 0, 16),
        (nn.GLU(), (2, 4), 20, 0, 12),
        (nn.GLU(dim=0), (2, 4), 20, 0, 12),
        (nn.GroupNorm(2, 4), (2, 4, 2), 136, 32, 125),
        (nn.GroupNorm(2, 4, affine=False), (2, 4, 2), 104, 16, 85),
        (_RMS_NORM((2, 4)) if _RMS_NORM else None, (2, 2, 4), 68, 32, 97),
        (_RMS_NORM((2, 4), elementwise_affine=False) if _RMS_NORM else None, (2, 2, 4), 52, 16, 57),
    ],
)
def test_native_primitive_counts(module, shape, flops, macs, dmas):
    if module is None:
        pytest.skip("Native RMSNorm is absent on older supported PyTorch versions")
    # Eight activation inputs; GLU emits four. GroupNorm has 16 values/four groups.
    # RMSNorm has 16 values/two rows, and eight affine weights when present.
    for batch in (shape[0], 0):
        report = crawl_module(module, args=(torch.ones(batch, *shape[1:]),), strict=True)
        for name, expected in (("module_flops", flops), ("operator_flops", flops), ("macs", macs), ("dmas", dmas)):
            assert report["totals"][name]["value"] == (expected if batch else 0)
        if not isinstance(module, (nn.GELU, nn.SiLU)):
            assert report["layers"][0]["metrics"]["receptive_field"]["method"].endswith("not_applicable")


def test_functional_swiglu_and_rmsnorm():
    values, gates = torch.ones(2, 4), torch.ones(2, 4)
    # Each of eight gates costs five SiLU operations and one final multiply.
    assert measure_flops(lambda: values * F.silu(gates))["total"]["value"] == 48
    # A portable, unweighted RMSNorm: 24 element operations plus four row operations.
    result = measure_flops(lambda: values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert result["total"]["status"] == "complete"
    assert result["total"]["value"] == 28
    empty_rows = torch.ones(2, 0)
    result = measure_flops(lambda: empty_rows * torch.rsqrt(empty_rows.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert result["total"]["status"] == "partial"
    if (rms_norm := getattr(F, "rms_norm", None)) is not None:
        assert measure_flops(lambda: rms_norm(empty_rows, (0,)))["total"]["status"] == "partial"
    if (native := getattr(nn, "RMSNorm", None)) is not None:
        with pytest.raises(NotImplementedError, match="nonempty normalized rows"):
            modules.module_flops(native((0,)), (empty_rows,), empty_rows)


@pytest.mark.parametrize("weighted", [False, True])
def test_fused_rmsnorm_shape_contract(weighted):
    # Exercise the GPU shape contract without executing a device-specific kernel.
    formula = FORMULAS["_fused_rms_norm"]
    assert formula((2, 2, 4), (2, 4), (2, 4) if weighted else None) == (68 if weighted else 52)
    assert formula((0, 4), (4,), (4,) if weighted else None) == 0
    with pytest.raises(NotImplementedError, match="weight shape"):
        formula((2, 4), (4,), (3,))


@pytest.mark.parametrize(
    ("module", "option", "value"),
    [
        (nn.GELU(), "approximate", "unknown"),
        (nn.SiLU(), "inplace", 1),
        (nn.GLU(), "dim", 2),
        (nn.GroupNorm(2, 4), "eps", float("nan")),
        (nn.GroupNorm(2, 4), "weight", nn.Parameter(torch.ones(3))),
    ],
)
def test_invalid_native_options_stay_unknown(module, option, value):
    # Validate estimates without invoking invalid backend kernels.
    setattr(module, option, value)
    inputs = torch.ones(2, 4)
    with pytest.raises(NotImplementedError):
        modules.module_flops(module, (inputs,), inputs)


@pytest.mark.parametrize(("activation", "expected"), [(torch.sigmoid, 32), (torch.tanh, 48)])
def test_integer_gate_promotion(activation, expected):
    inputs = torch.ones(2, 4, dtype=torch.int64)
    report = measure_flops(lambda: activation(inputs))
    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == expected


@pytest.mark.parametrize("inputs", [torch.ones(2, 4, dtype=torch.complex64), torch.ones(2, 4).to_sparse()])
def test_unsupported_primitive_inputs(inputs):
    with pytest.raises(NotImplementedError):
        modules.module_flops(nn.SiLU(), (inputs,), inputs)


def test_modified_forward_stays_unknown():
    module = nn.SiLU()
    module.forward = lambda x: x.square()
    with pytest.raises(IncompleteAnalysisError):
        crawl_module(module, args=(torch.ones(2, 4),), strict=True)


def test_arbitrary_power():
    report = measure_flops(lambda: torch.ones(2, 4).pow(0.5))
    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == 8


def test_complex_rmsnorm_weight_stays_unknown():
    if (rms_norm := getattr(nn, "RMSNorm", None)) is None:
        pytest.skip("Native RMSNorm is absent on older supported PyTorch versions")
    module = rms_norm(4)
    module.weight = nn.Parameter(torch.ones(4, dtype=torch.complex64))
    report = crawl_module(module, args=(torch.ones(2, 4),))
    assert all(report["totals"][name]["status"] == "unavailable" for name in ("module_flops", "macs", "dmas"))
    assert report["totals"]["operator_flops"]["status"] == "partial"
