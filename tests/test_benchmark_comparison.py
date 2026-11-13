import json
from copy import deepcopy
from unittest.mock import Mock

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
    assert result["output_check"] == "passed"
    assert result["totals"]["latency"]["delta"] == pytest.approx(-0.01)
    assert result["latency_change"] == "faster"
    assert result["context_changes"]["warmup"]["after"] == 9
    html = render_report(result, title="<script>alert(1)</script>")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Output check passed" in html
    assert "aten::mm" in html
    assert "Recorded context changed: warmup" in html
    assert json.loads(json.dumps(result)) == result
    changed = deepcopy(result)
    changed["context_changes"].clear()
    with pytest.raises(ValueError, match="Context changes"):
        render_report(changed)
    assert "No operator profile" in render_report(before)
    before["profile"] = after["profile"]
    assert "Baseline bottleneck evidence" in render_report(compare_benchmarks(before, after, check=lambda: True))
    assert "Single workload" in render_report(after)
    for field, value in (
        ("schema_version", 2),
        ("operators", None),
        ("context", {"device": "cuda:999", "torch_version": str(torch.__version__)}),
    ):
        changed = deepcopy(after)
        changed["profile"][field] = value
        with pytest.raises(ValueError):
            render_report(changed)
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
    assert result["output_check"] == "failed"
    assert result["latency_change"] == "unavailable"
    assert all(metric["delta"] is None for metric in result["totals"].values())
    assert "Output check failed" in render_report(result)
    result["output_check"] = "unknown"
    with pytest.raises(ValueError, match="comparison"):
        render_report(result)


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
    changed = deepcopy(after)
    changed["totals"]["latency_iqr"]["unit"] = "milliseconds"
    with pytest.raises(ValueError, match="timing unit"):
        compare_benchmarks(changed, changed, check=lambda: True)
    for field, value in (("schema_version", 2), ("totals", {})):
        changed = deepcopy(after)
        changed[field] = value
        with pytest.raises(ValueError):
            render_report(changed)
    changed = deepcopy(after)
    changed["totals"]["latency"].update(value=0, known_value=0)
    with pytest.raises(ValueError, match="timing value"):
        render_report(changed)
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


@pytest.mark.parametrize(
    ("before_value", "after_value", "compatible"),
    [
        (None, None, True),
        ("0", "0", True),
        ("1", "1", True),
        (None, "0", False),
        ("0", "1", False),
        ("unsupported", "unsupported", False),
        (False, False, False),
    ],
)
def test_mps_fallback_comparison(before_value, after_value, compatible):
    before, after = reports()
    # Synthetic saved contexts exercise compatibility without claiming MPS execution.
    for report, value in ((before, before_value), (after, after_value)):
        report["context"].update(device="mps", mps_fallback_env=value)
    check = Mock(return_value=True)
    if compatible:
        result = compare_benchmarks(before, after, check=check)
        assert result["totals"]["latency"]["delta"] == pytest.approx(-0.01)
        assert "Output check passed" in render_report(result)
        check.assert_called_once_with()
    else:
        with pytest.raises(ValueError, match="mps_fallback_env"):
            compare_benchmarks(before, after, check=check)
        check.assert_not_called()


@pytest.mark.parametrize("missing", ["before", "after", "both"])
def test_missing_mps_fallback_requires_new_measurements(missing):
    before, after = reports()
    archived = compare_benchmarks(before, after, check=lambda: True)
    for name, report in (("before", before), ("after", after)):
        report["context"]["device"] = "mps"
        if missing not in (name, "both"):
            report["context"]["mps_fallback_env"] = None
    check = Mock(return_value=True)
    with pytest.raises(ValueError, match="mps_fallback_env"):
        compare_benchmarks(before, after, check=check)
    check.assert_not_called()
    for report in (archived["before"], archived["after"]):
        report["context"]["device"] = "mps"
    # Previously saved comparisons remain readable with their original evidence.
    assert "Output check passed" in render_report(archived)
