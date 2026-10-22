import json
import math
from copy import deepcopy

import pytest
import torch

from torchscan import check_performance, compare_benchmarks, measure_latency, profile_workload, render_report
from torchscan.report import metric_result


def reports(after_latency=0.01):
    inputs = torch.ones(8)
    before = measure_latency(lambda: inputs + 1, device="cpu", inputs=inputs, min_run_time=0.001, min_repeats=2)
    after = deepcopy(before)
    for report, latency in ((before, 0.02), (after, after_latency)):
        report["totals"]["latency"].update(value=latency, known_value=latency)
        report["totals"]["latency_iqr"].update(value=0.001, known_value=0.001)
        report["measurement"].update(number_per_run=1, raw_times_seconds=[latency, latency], measured_calls=2)
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
    assert check_performance(result, max_latency_regression=0)["status"] == "failed"
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


def test_check_real_saved_timing():
    inputs = torch.ones(8)
    report = measure_latency(lambda: inputs + 1, device="cpu", inputs=inputs, min_run_time=0.001, min_repeats=2)
    comparison = compare_benchmarks(report, report, check=lambda: True)
    saved = json.loads(json.dumps(comparison))
    assert check_performance(saved, max_latency_regression=0)["status"] == "passed"


@pytest.mark.parametrize(
    ("factor", "limit", "status"),
    [(1, 0, "passed"), (1.05, 0.05, "passed"), (math.nextafter(1.05, math.inf), 0.05, "failed")],
)
def test_performance_latency_limit(factor, limit, status):
    before, after = reports(0.02 * factor)
    result = check_performance(compare_benchmarks(before, after, check=lambda: True), max_latency_regression=limit)
    assert result["status"] == status
    assert result["reasons"]


@pytest.mark.parametrize(
    ("status", "value", "scope", "expected"),
    [
        ("complete", 100, "pytorch_tensor_bytes", "passed"),
        ("complete", 101, "pytorch_tensor_bytes", "failed"),
        ("missing", None, "pytorch_tensor_bytes", "inconclusive"),
        ("unavailable", None, "pytorch_tensor_bytes", "inconclusive"),
        ("partial", 90, "pytorch_tensor_bytes", "inconclusive"),
        ("complete", 100, "child_process_lifetime", "inconclusive"),
        ("complete", 100, "pytorch_reserved_bytes", "inconclusive"),
    ],
)
def test_performance_memory_limit(status, value, scope, expected):
    before, after = reports()
    if status != "missing":
        after["totals"]["peak_memory"] = metric_result(
            status=status,
            value=value if status == "complete" else None,
            known_value=value if status == "partial" else None,
            unit="bytes",
            scope=scope,
            method="test_memory",
        )
    result = check_performance(
        compare_benchmarks(before, after, check=lambda: True),
        max_latency_regression=0,
        memory_limit={"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": 100},
    )
    assert result["status"] == expected
    assert "peak_memory" in result["reasons"][-1]


@pytest.mark.parametrize("problem", ["partial", "unavailable", "missing_blocks", "scope", "schema", "delta"])
def test_performance_inconclusive_evidence(problem):
    before, after = reports()
    if problem in ("partial", "unavailable"):
        after["totals"]["latency"].update(
            status=problem,
            value=None,
            known_value=0.01 if problem == "partial" else None,
        )
    comparison = compare_benchmarks(before, after, check=lambda: True)
    if problem == "missing_blocks":
        comparison["after"]["measurement"] = {}
    elif problem == "scope":
        comparison["after"]["totals"]["latency"]["scope"] = "other"
    elif problem == "schema":
        comparison["schema_version"] = 99
    elif problem == "delta":
        comparison["totals"]["latency"]["delta"] = False
    assert check_performance(comparison, max_latency_regression=0)["status"] == "inconclusive"


def test_performance_failure_with_incomplete_timing():
    before, after = reports()
    after["measurement"] = {}
    after["totals"]["process_peak_rss"] = metric_result(
        status="complete",
        value=101,
        unit="bytes",
        scope="child_process_lifetime",
        method="test_rss",
    )
    result = check_performance(
        compare_benchmarks(before, after, check=lambda: True),
        max_latency_regression=0,
        memory_limit={"metric": "process_peak_rss", "scope": "child_process_lifetime", "max_bytes": 100},
    )
    assert result["status"] == "failed"


@pytest.mark.parametrize("limit", [-1, math.inf, True])
def test_performance_invalid_limit(limit):
    before, after = reports()
    with pytest.raises(ValueError):
        check_performance(compare_benchmarks(before, after, check=lambda: True), max_latency_regression=limit)
