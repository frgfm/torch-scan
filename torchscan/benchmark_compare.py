# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, TypedDict, cast

import torch

from .benchmark import BenchmarkReport, _positive_finite, _validate_timing_settings, measure_latency
from .compare import _diff_metrics, _MetricDiff
from .render import _json_tree, _metrics, _object

__all__ = ["BenchmarkComparison", "RepeatedBenchmarkComparison", "compare_benchmarks", "measure_comparison"]

LatencyChange = Literal["faster", "slower", "within_variability", "unavailable"]
ComparisonVerdict = Literal["faster", "slower", "inconclusive", "check_failed"]


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


class RepeatedBenchmarkComparison(TypedDict):
    """JSON-serializable trials and a descriptive verdict for one paired experiment."""

    schema_version: int
    report_type: Literal["repeated_benchmark_comparison"]
    trials: list[BenchmarkComparison]
    min_latency_change: float
    verdict: ComparisonVerdict


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


def measure_comparison(
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    *,
    check: Callable[[], object],
    min_latency_change: float,
    device: str | torch.device,
    inputs: Any = None,
    trials: int = 3,
    work_units: float = 1,
    work_unit: str = "calls",
    warmup: int = 5,
    min_run_time: float = 0.2,
    min_repeats: int = 5,
) -> RepeatedBenchmarkComparison:
    """Measure repeatable workloads in alternating order and compare every trial.

    Args:
        baseline: Repeatable zero-argument reference workload, as for measure_latency.
        candidate: Repeatable zero-argument changed workload. Both callables own their
            inputs, state, device placement, precision, and gradient mode.
        check: Zero-argument output check, run by compare_benchmarks after each pair,
            outside timing. None/True passes; False or AssertionError fails.
            Other exceptions and invalid return values propagate.
        min_latency_change: Caller-supplied positive finite minimum meaningful absolute
            change in warmed median latency, in seconds.
        device: Shared synchronization device, as for measure_latency.
        inputs: Required caller-supplied input metadata for compatible comparisons.
            It is recorded by measure_latency, never forwarded to either workload.
        trials: Number of paired trials, at least two. Trial one measures baseline
            first, trial two candidate first, and subsequent trials alternate.
        work_units: Shared positive number of work units completed by each invocation.
        work_unit: Shared unit name for throughput, such as "images" or "tokens".
        warmup: Shared explicit warmup calls after each first call.
        min_run_time: Shared minimum warmed measurement duration in seconds.
        min_repeats: Shared minimum timed blocks per measurement, at least two.

    Returns:
        Every trial as a BenchmarkComparison, always with baseline as before and
        candidate as after. Any failed check yields check_failed; all trials still run.
        Faster/slower requires every trial to have that compare_benchmarks label and
        an absolute latency delta strictly larger than min_latency_change. Thus each
        delta also exceeds both runs' IQRs. Mixed, negligible, within-variability, or
        unavailable changes yield inconclusive. This is a descriptive consistency
        rule, not a statistical significance test. Raw evidence is retained per trial.

    Raises:
        TypeError: If either workload or check is not callable.
        ValueError: If trials, the threshold, shared settings, or comparisons are invalid.

    Notes:
        Measurements and checks run sequentially and share caller state. Workload
        exceptions and side effects are preserved, as are measure_latency device errors.
        Matching input/hardware/software contexts are required within and across trials.
        No model state, outputs, or timing settings are changed by this helper.
    """
    for workload in (baseline, candidate):
        _validate_timing_settings(workload, work_units, work_unit, warmup, min_run_time, min_repeats)
    if not callable(check):
        raise TypeError("check must be callable.")
    if type(trials) is not int or trials < 2:
        raise ValueError("trials must be an integer >= 2.")
    if not _positive_finite(min_latency_change):
        raise ValueError("min_latency_change must be a positive finite number of seconds.")
    if inputs is None:
        raise ValueError("Comparisons require matching caller-supplied input metadata.")
    settings = {
        "device": device,
        "inputs": inputs,
        "work_units": work_units,
        "work_unit": work_unit,
        "warmup": warmup,
        "min_run_time": min_run_time,
        "min_repeats": min_repeats,
    }
    comparisons: list[BenchmarkComparison] = []
    for index in range(trials):
        if index % 2 == 0:
            before = measure_latency(baseline, **settings)
            after = measure_latency(candidate, **settings)
        else:
            after = measure_latency(candidate, **settings)
            before = measure_latency(baseline, **settings)
        if comparisons:
            _matching_context(comparisons[0]["before"], before)
        comparisons.append(compare_benchmarks(before, after, check=check))

    verdict: ComparisonVerdict = "inconclusive"
    if any(comparison["output_check"] == "failed" for comparison in comparisons):
        verdict = "check_failed"
    elif all(abs(comparison["totals"]["latency"]["delta"] or 0) > min_latency_change for comparison in comparisons):
        if all(comparison["latency_change"] == "faster" for comparison in comparisons):
            verdict = "faster"
        elif all(comparison["latency_change"] == "slower" for comparison in comparisons):
            verdict = "slower"
    return {
        "schema_version": 1,
        "report_type": "repeated_benchmark_comparison",
        "trials": comparisons,
        "min_latency_change": min_latency_change,
        "verdict": verdict,
    }
