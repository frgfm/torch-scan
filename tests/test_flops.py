import json
from math import prod

import pytest
import torch
from torch import nn

from torchscan.flops import measure_flops


def test_measure_flops_matmul_and_module_hierarchy():
    left = torch.ones(2, 3)
    right = torch.ones(3, 4)
    linear = nn.Linear(4, 2)

    report = measure_flops(lambda: linear(left @ right), modules=linear)

    assert report["total"] == {
        "status": "complete",
        "value": 80,
        "known_value": 80,
        "unit": "FLOPs",
        "scope": "workload",
        "method": "torch.utils.flop_counter.FlopCounterMode",
    }
    assert report["by_operator"] == {"aten.addmm": 32, "aten.mm": 48}
    assert report["by_module"]["Linear"] == 32
    assert report["diagnostics"] == []
    assert json.loads(json.dumps(report)) == report


def test_measure_flops_custom_mapping():
    inputs = torch.ones(5)

    def sin_flops(input_shape, *, out_shape):
        assert input_shape == out_shape
        return prod(out_shape)

    report = measure_flops(lambda: torch.sin(inputs), custom_mapping={torch.ops.aten.sin: sin_flops})

    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == 5
    assert report["by_operator"] == {"aten.sin": 5}
    assert report["diagnostics"] == []


def test_measure_flops_custom_zero_is_complete():
    inputs = torch.ones(5)

    def zero_flops(input_shape, *, out_shape):
        del input_shape, out_shape
        return 0

    mapping = {torch.ops.aten.sin: zero_flops}

    report = measure_flops(lambda: torch.sin(inputs), custom_mapping=mapping)

    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == 0
    assert report["by_operator"] == {"aten.sin": 0}
    assert mapping.keys() == {torch.ops.aten.sin}


def test_custom_formula_tensor_ops_are_not_recorded_as_workload_ops():
    inputs = torch.ones(5)

    def sin_flops(input_shape, *, out_shape):
        assert input_shape == out_shape
        torch.cos(torch.ones(1))
        return prod(out_shape)

    report = measure_flops(lambda: torch.sin(inputs), custom_mapping={torch.ops.aten.sin: sin_flops})

    assert report["by_operator"] == {"aten.sin": 5}
    assert all(diagnostic.get("operator") != "aten.cos" for diagnostic in report["diagnostics"])


def test_measure_flops_reports_uncounted_operator_as_partial():
    left = torch.ones(2, 3)
    right = torch.ones(3, 4)

    report = measure_flops(lambda: torch.sin(left @ right))

    assert report["total"]["status"] == "partial"
    assert report["total"]["value"] is None
    assert report["total"]["known_value"] == 48
    assert report["by_operator"] == {"aten.mm": 48}
    assert report["diagnostics"] == [
        {
            "code": "uncounted_operator",
            "severity": "warning",
            "metric": "flops",
            "operator": "aten.sin",
            "message": "aten.sin was observed 1 time(s), but no FLOP formula is registered.",
        }
    ]


def test_measure_flops_explicitly_ignores_zero_flop_operator():
    inputs = torch.ones(2, 3)

    report = measure_flops(lambda: inputs.view(3, 2))

    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == 0
    assert report["ignored_operators"] == {"aten.view": {"calls": 1, "reason": "Metadata-only tensor view."}}


def test_measure_flops_invokes_workload_once():
    calls = 0

    def workload():
        nonlocal calls
        calls += 1

    report = measure_flops(workload)

    assert calls == 1
    assert report["total"]["value"] == 0


def test_measure_flops_preserves_workload_exception():
    error = RuntimeError("workload failed")
    calls = 0

    def workload():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(RuntimeError) as exc_info:
        measure_flops(workload)

    assert exc_info.value is error
    assert calls == 1
