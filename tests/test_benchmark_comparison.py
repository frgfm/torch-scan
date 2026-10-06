import json
from copy import deepcopy

import pytest
import torch

from torchscan import compare_benchmarks, measure_latency, profile_workload, render_report


def reports():
    inputs = torch.ones(8)
    before = measure_latency(lambda: inputs + 1, device="cpu", inputs=inputs, min_run_time=0.001, min_repeats=2)
    after = deepcopy(before)
    for report, latency in ((before, 0.02), (after, 0.01)):
        report["totals"]["latency"].update(value=latency, known_value=latency)
        report["totals"]["latency_iqr"].update(value=0.001, known_value=0.001)
    return before, after


def test_checked_comparison_and_safe_html():
    before, after = reports()
    after["context"]["warmup"] = 9
    after["profile"] = profile_workload(lambda: torch.ones(2, 2) @ torch.ones(2, 2), device="cpu")
    result = compare_benchmarks(before, after, check=lambda: None)
    assert result["correctness"] == "passed"
    assert result["totals"]["latency"]["delta"] == pytest.approx(-0.01)
    assert result["latency_change"] == "faster"
    assert result["context_changes"]["warmup"]["after"] == 9
    html = render_report(result, title="<script>alert(1)</script>")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Output check passed" in html
    assert "aten::mm" in html
    assert json.loads(json.dumps(result)) == result
    assert "Single workload" in render_report(after)
    assert "No operator profile" in render_report(before)
    with pytest.raises(ValueError):
        render_report(after, format="svg")
    result["totals"]["latency"]["delta"] = 999
    with pytest.raises(ValueError, match="deltas"):
        render_report(result)


@pytest.mark.parametrize("assertion", [True, False])
def test_failed_output_check_withholds_gains(assertion):
    before, after = reports()

    def check():
        if assertion:
            raise AssertionError("outputs differ")
        return False

    result = compare_benchmarks(before, after, check=check)
    assert result["correctness"] == "failed"
    assert result["latency_change"] == "unavailable"
    assert all(metric["delta"] is None for metric in result["totals"].values())
    assert "Output check failed" in render_report(result)


def test_context_status_and_checker_contract():
    before, after = reports()
    changed = deepcopy(after)
    changed["context"]["device_name"] = "other hardware"
    with pytest.raises(ValueError, match="contexts"):
        compare_benchmarks(before, changed, check=lambda: True)
    changed = deepcopy(after)
    changed["inputs"] = {"kind": "none"}
    with pytest.raises(ValueError, match="input metadata"):
        compare_benchmarks(before, changed, check=lambda: True)
    with pytest.raises(TypeError):
        compare_benchmarks(before, after, check=None)
    with pytest.raises(ValueError, match="check"):
        compare_benchmarks(before, after, check=lambda: "truthy")
    after["totals"]["latency"].update(status="partial", value=None)
    result = compare_benchmarks(before, after, check=lambda: True)
    assert result["totals"]["latency"]["delta"] is None
    assert "lower bound" in render_report(result)


@pytest.mark.parametrize(("latency", "label"), [(0.0205, "within_variability"), (0.03, "slower")])
def test_latency_variability_labels(latency, label):
    before, after = reports()
    after["totals"]["latency"].update(value=latency, known_value=latency)
    result = compare_benchmarks(before, after, check=lambda: True)
    assert result["latency_change"] == label
    result["latency_change"] = "faster"
    with pytest.raises(ValueError, match="Latency change"):
        render_report(result)
