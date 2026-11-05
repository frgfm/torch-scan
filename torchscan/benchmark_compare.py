# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from copy import deepcopy
from math import isclose
from statistics import median
from typing import Any, Literal, TypedDict, cast

from ._validation import _TIMING_METHOD, _json_tree, _metric, _metrics, _number, _object, _positive_finite
from .compare import _diff_metrics, _MetricDiff
from .report import BenchmarkReport

__all__ = ["BenchmarkComparison", "MemoryLimit", "PerformanceCheck", "check_performance", "compare_benchmarks"]

LatencyChange = Literal["faster", "slower", "within_variability", "unavailable"]


class BenchmarkComparison(TypedDict):
    """Checked before/after evidence, without storing workload outputs."""

    schema_version: int
    report_type: Literal["benchmark_comparison"]
    before: BenchmarkReport
    after: BenchmarkReport
    totals: dict[str, _MetricDiff]
    output_check: Literal["passed", "failed"]
    latency_change: LatencyChange
    context_changes: dict[str, Any]


class MemoryLimit(TypedDict):
    """Candidate byte ceiling for one named metric and exact measurement scope."""

    metric: str
    scope: str
    max_bytes: int


class PerformanceCheck(TypedDict):
    """JSON-serializable decision with reasons for agents and other callers."""

    status: Literal["passed", "failed", "inconclusive"]
    reasons: list[str]


def _validate_benchmark(report: object) -> BenchmarkReport:
    report = _object(report, "benchmark")
    _json_tree(report, "benchmark")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Unsupported benchmark schema_version.")
    _object(report.get("context"), "context")
    _object(report.get("inputs"), "inputs")
    _object(report.get("measurement"), "measurement")
    _metrics(report.get("totals"), "totals")
    if not {"first_call_latency", "latency", "latency_iqr", "throughput"} <= report["totals"].keys():
        raise ValueError("Benchmark timing metrics are missing.")
    for name in ("first_call_latency", "latency", "throughput", "latency_iqr"):
        metric = report["totals"][name]
        unit = f"{report['context'].get('work_unit')}/s" if name == "throughput" else "seconds"
        if metric["unit"] != unit:
            raise ValueError(f"Invalid benchmark timing unit: {name}.")
        if metric["status"] == "complete" and (metric["value"] < 0 or (name != "latency_iqr" and metric["value"] == 0)):
            raise ValueError(f"Invalid benchmark timing value: {name}.")
    return cast(BenchmarkReport, report)


def _matching_context(before: BenchmarkReport, after: BenchmarkReport) -> None:
    required = (
        "device",
        "device_name",
        "processor",
        "machine",
        "platform",
        "torch_version",
        "torchscan_version",
        "python_version",
        "cuda_version",
        "work_units",
        "work_unit",
        "inputs_source",
    )
    for key in required:
        if (
            key not in before["context"]
            or key not in after["context"]
            or before["context"][key] != after["context"][key]
        ):
            raise ValueError(f"Benchmark contexts differ or are missing: {key}.")
    if before["context"]["inputs_source"] != "caller_supplied" or before["inputs"] != after["inputs"]:
        raise ValueError("Comparisons require matching caller-supplied input metadata.")


def _latency_change(
    before: BenchmarkReport, after: BenchmarkReport, differences: dict[str, _MetricDiff]
) -> LatencyChange:
    delta = differences["latency"]["delta"]
    iqrs = (before["totals"]["latency_iqr"], after["totals"]["latency_iqr"])
    if delta is None or any(iqr["status"] != "complete" for iqr in iqrs):
        return "unavailable"
    # ponytail: the IQR rule is descriptive; retain samples for formal statistical analysis.
    variability = max(cast(float, iqr["value"]) for iqr in iqrs)
    return "within_variability" if abs(delta) <= variability else "faster" if delta < 0 else "slower"


def _context_changes(before: BenchmarkReport, after: BenchmarkReport) -> dict[str, Any]:
    return {
        key: {"before": before["context"].get(key), "after": after["context"].get(key)}
        for key in sorted(before["context"].keys() | after["context"].keys())
        if before["context"].get(key) != after["context"].get(key)
    }


def _validate_comparison(report: object) -> BenchmarkComparison:
    report = _object(report, "comparison")
    _json_tree(report, "comparison")
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("report_type") != "benchmark_comparison"
        or report.get("output_check") not in ("passed", "failed")
    ):
        raise ValueError("Invalid benchmark comparison.")
    for name, difference in _object(report.get("totals"), "comparison.totals").items():
        difference = _object(difference, f"comparison.totals.{name}")
        if difference.get("delta") is not None:
            _number(difference["delta"], f"comparison.totals.{name}.delta")
        for side in ("before", "after"):
            if difference.get(side) is not None:
                _metric(difference[side], f"comparison.totals.{name}.{side}")
    expected = compare_benchmarks(
        cast(BenchmarkReport, report.get("before")),
        cast(BenchmarkReport, report.get("after")),
        check=lambda: report["output_check"] == "passed",
    )
    for field, label in (
        ("totals", "Comparison deltas"),
        ("latency_change", "Latency change"),
        ("context_changes", "Context changes"),
    ):
        if report.get(field) != expected[field]:
            raise ValueError(f"{label} do not match the stored measurements/check status.")
    return cast(BenchmarkComparison, report)


def compare_benchmarks(
    before: BenchmarkReport,
    after: BenchmarkReport,
    *,
    check: Callable[[], object],
) -> BenchmarkComparison:
    """Compare matching benchmark reports and run an owner-supplied output check.

    Args:
        before: Baseline report from measure_latency, optionally including memory/profile evidence.
        after: Candidate report for matching hardware, software, inputs, and work units.
        check: Caller-owned output check, executed outside timing. Exactly None or Python True means
            success; Python False or AssertionError means failure. Other exceptions propagate.

    Returns:
        Measured deltas only when the output check passes. Input/hardware/software
        mismatch raises before checking. Changed thread/warmup settings remain visible.
        An IQR-based change label is descriptive, not a significance or accuracy test.

    Raises:
        ValueError: If reports, contexts, methods, or check results are incompatible.
        TypeError: If check is not callable.
    """
    if not callable(check):
        raise TypeError("check must be callable.")
    before, after = _validate_benchmark(before), _validate_benchmark(after)
    _matching_context(before, after)
    differences = _diff_metrics(before["totals"], after["totals"])
    try:
        outcome = check()
    except AssertionError:
        outcome = False
    if outcome is not None and type(outcome) is not bool:
        raise ValueError("check must return None/True on success or False on failure.")
    passed = outcome is not False
    if not passed:
        for difference in differences.values():
            difference["status"], difference["delta"] = "unavailable", None
    return {
        "schema_version": 1,
        "report_type": "benchmark_comparison",
        "before": deepcopy(before),
        "after": deepcopy(after),
        "totals": differences,
        "output_check": "passed" if passed else "failed",
        "latency_change": _latency_change(before, after, differences),
        "context_changes": _context_changes(before, after),
    }


def _has_timing_evidence(report: BenchmarkReport) -> bool:
    latency, measurement = report["totals"]["latency"], report["measurement"]
    number, samples = measurement.get("number_per_run"), measurement.get("raw_times_seconds")
    repeats = report["context"].get("min_repeats")
    return (
        latency["status"] == "complete"
        and _positive_finite(latency["value"])
        and latency["scope"] == "warmed_block_average"
        and latency["method"] == _TIMING_METHOD
        and type(number) is int
        and _positive_finite(number)
        and isinstance(samples, list)
        and type(repeats) is int
        and repeats >= 2
        and len(samples) >= repeats
        and all(_positive_finite(sample) for sample in samples)
        and type(measurement.get("measured_calls")) is int
        and measurement["measured_calls"] == number * len(samples)
        and isclose(median(sample / number for sample in samples), cast(float, latency["value"]), rel_tol=1e-12)
    )


def check_performance(
    comparison: BenchmarkComparison,
    *,
    max_latency_regression: float,
    memory_limit: MemoryLimit | None = None,
) -> PerformanceCheck:
    """Check saved comparison evidence against caller-supplied budgets.

    Args:
        comparison: In-memory or JSON-loaded BenchmarkComparison, including an autoresearch trial.
        max_latency_regression: Nonnegative finite fraction; 0.05 allows 5% slower median latency.
        memory_limit: Optional candidate ceiling naming the exact totals metric, scope, and max_bytes.

    Returns:
        Passed, failed, or inconclusive with reasons. Failed output checks and exceeded usable
        limits fail; missing or incompatible evidence is inconclusive. Equality passes.
        Timing requires matching raw blocks; memory requires the exact metric and scope.
        No workloads execute.

    Raises:
        ValueError: If caller-supplied limits are invalid.
    """
    _number(max_latency_regression, "max_latency_regression")
    if max_latency_regression != 0 and not _positive_finite(max_latency_regression):
        raise ValueError("max_latency_regression must be finite and nonnegative.")
    if memory_limit is not None:
        limit = _object(memory_limit, "memory_limit")
        for field in ("metric", "scope"):
            if not isinstance(limit.get(field), str) or not limit[field].strip():
                raise ValueError(f"memory_limit.{field} must be a non-empty string.")
        _number(limit.get("max_bytes"), "memory_limit.max_bytes", integer=True)
    try:
        comparison = _validate_comparison(comparison)
    except (ValueError, RecursionError) as error:
        return {"status": "inconclusive", "reasons": [str(error)]}
    if comparison["output_check"] == "failed":
        return {"status": "failed", "reasons": ["Recorded output check failed."]}

    ceiling = None
    if all(_has_timing_evidence(comparison[side]) for side in ("before", "after")):
        before = cast(float, comparison["before"]["totals"]["latency"]["value"])
        ceiling = before * (1 + max_latency_regression)
    limits = [("latency", "seconds", "warmed_block_average", ceiling)]
    if memory_limit is not None:
        limits.append((memory_limit["metric"], "bytes", memory_limit["scope"], memory_limit["max_bytes"]))
    failed, incomplete = False, False
    reasons = []
    for name, unit, scope, ceiling in limits:
        metric = comparison["after"]["totals"].get(name)
        if (
            ceiling is None
            or metric is None
            or metric["status"] != "complete"
            or metric["unit"] != unit
            or metric["scope"] != scope
            or cast(float, metric["value"]) < 0
            or metric["method"].strip() in ("", "not_requested", "unavailable")
        ):
            incomplete = True
            reasons.append(f"{name}: missing, incomplete or incompatible evidence ({unit}, {scope}).")
        else:
            exceeded = cast(float, metric["value"]) > ceiling
            failed |= exceeded
            reasons.append(f"{name} ({scope}): {metric['value']} {'>' if exceeded else '<='} {ceiling} {unit}.")
    return {"status": "failed" if failed else "inconclusive" if incomplete else "passed", "reasons": reasons}
