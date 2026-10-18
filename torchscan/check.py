# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from math import isclose
from statistics import median
from typing import Literal, TypedDict, cast

from .benchmark import _METHOD, BenchmarkReport, _positive_finite
from .benchmark_compare import BenchmarkComparison, _validate_comparison
from .render import _number, _object

__all__ = ["MemoryLimit", "PerformanceCheck", "check_performance"]


class MemoryLimit(TypedDict):
    """Candidate byte ceiling for exactly one named metric and measurement scope."""

    metric: str
    scope: str
    max_bytes: int


class PerformanceCheck(TypedDict):
    """JSON-serializable performance decision and the reasons supporting it."""

    status: Literal["passed", "failed", "inconclusive"]
    reasons: list[str]


def _timing_evidence(report: BenchmarkReport, name: str) -> str | None:
    latency, iqr = (report["totals"][key] for key in ("latency", "latency_iqr"))
    for metric in (latency, iqr):
        if metric["status"] != "complete":
            return f"{name} latency/IQR evidence is {metric['status']}."
        if metric["value"] != 0 and not _positive_finite(metric["value"]):
            return f"{name} latency/IQR evidence must be finite and nonnegative."
        if metric["scope"] != "warmed_block_average" or metric["method"] != _METHOD:
            return f"{name} latency/IQR requires warmed_block_average scope and {_METHOD} method."
    measurement = report["measurement"]
    number = measurement.get("number_per_run")
    samples = measurement.get("raw_times_seconds")
    repeats = report["context"].get("min_repeats")
    if (
        type(number) is not int
        or not _positive_finite(number)
        or not isinstance(samples, list)
        or type(repeats) is not int
        or repeats < 2
        or len(samples) < repeats
        or any(not _positive_finite(sample) for sample in samples)
        or type(measurement.get("measured_calls")) is not int
        or measurement["measured_calls"] != number * len(samples)
    ):
        return f"{name} timing needs at least two valid raw blocks, min_repeats, and matching measured_calls."
    # Match the stored summary to its raw blocks without rerunning the timer.
    if not isclose(median(sample / number for sample in samples), cast(float, latency["value"]), rel_tol=1e-12):
        return f"{name} latency does not match its raw timing blocks."
    return None


def check_performance(
    comparison: BenchmarkComparison,
    *,
    max_latency_regression: float,
    memory_limit: MemoryLimit | None = None,
) -> PerformanceCheck:
    """Check saved comparison evidence against explicit, inclusive owner limits.

    Args:
        comparison: Existing compare_benchmarks result, including JSON-loaded evidence.
            No workload or output callback is executed.
        max_latency_regression: Maximum fractional increase in warmed median latency:
            0.05 allows 5%, and 0 allows no regression. Must be finite and nonnegative.
        memory_limit: Optional candidate ceiling with an exact totals metric name, scope,
            and nonnegative integer max_bytes. Only complete byte metrics are usable.
            RSS, tracked tensors, allocated memory, and reserved memory are never substituted.

    Returns:
        Status and reasons. A recorded failed output check or exceeded usable limit fails.
        Otherwise any invalid, incompatible, missing, partial, or insufficient requested
        evidence is inconclusive. Passing needs complete compatible timing/IQR, at least
        two raw blocks per run meeting recorded min_repeats, and any requested memory.
        Unrequested metrics do not affect the decision. IQR labels do not choose limits
        or establish statistical significance; the latency limit applies to the median.

    Raises:
        ValueError: If the caller's limits are invalid.
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
        return {"status": "inconclusive", "reasons": [f"Invalid or incompatible comparison: {error}"]}
    if comparison["output_check"] == "failed":
        return {"status": "failed", "reasons": ["Recorded output check failed."]}

    reasons: list[str] = []
    incomplete = False
    failed = False
    for name in ("before", "after"):
        reason = _timing_evidence(comparison[name], name)
        if reason is not None:
            incomplete = True
            reasons.append(reason)
    if not incomplete:
        before = cast(float, comparison["before"]["totals"]["latency"]["value"])
        after = cast(float, comparison["after"]["totals"]["latency"]["value"])
        exceeded = after > before * (1 + max_latency_regression)
        failed |= exceeded
        reasons.append(
            f"Candidate latency {after:g} seconds {'exceeds' if exceeded else 'meets'} "
            f"the maximum fractional regression {max_latency_regression:g} from {before:g} seconds."
        )
    if memory_limit is not None:
        metric_name, scope, ceiling = memory_limit["metric"], memory_limit["scope"], memory_limit["max_bytes"]
        memory = comparison["after"]["totals"].get(metric_name)
        label = f"Candidate memory {metric_name!r} (scope {scope!r})"
        if memory is None:
            incomplete = True
            reasons.append(f"{label} is missing.")
        elif memory["unit"] != "bytes" or memory["scope"] != scope:
            incomplete = True
            reasons.append(f"{label} requires matching byte evidence; found {memory['unit']!r}/{memory['scope']!r}.")
        elif (
            memory["status"] != "complete"
            or cast(float, memory["value"]) < 0
            or memory["method"].strip() in ("", "not_requested", "unavailable")
        ):
            incomplete = True
            reasons.append(f"{label} has {memory['status']} or invalid evidence.")
        else:
            exceeded = cast(int, memory["value"]) > ceiling
            failed |= exceeded
            reasons.append(
                f"{label}: {memory['value']} bytes {'exceeds' if exceeded else 'meets'} ceiling {ceiling} bytes."
            )
    return {"status": "failed" if failed else "inconclusive" if incomplete else "passed", "reasons": reasons}
