# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, TypedDict, cast

from .benchmark import BenchmarkReport
from .compare import _diff_metrics, _MetricDiff
from .render import _json_tree, _metrics, _object

__all__ = ["BenchmarkComparison", "compare_benchmarks"]

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
