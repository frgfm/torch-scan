# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Mapping, Sequence
from math import isclose
from statistics import median
from typing import Any, TypeGuard, cast

from .benchmark import _METHOD, _positive_finite
from .benchmark_compare import (
    _context_changes,
    _latency_change,
    _matching_context,
    _validate_benchmark,
)
from .compare import _diff_metrics

_FIXED_SETTINGS = ("num_threads", "num_interop_threads", "warmup", "min_run_time", "min_repeats")
_FIXED_CONTEXT = (
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
    *_FIXED_SETTINGS,
)


def _finite_number(value: object) -> TypeGuard[int | float]:
    return _positive_finite(value) or (type(value) in (int, float) and value == 0)


def _iqr(samples: list[float], number_per_run: int) -> float:
    times = sorted(sample / number_per_run for sample in samples)

    def quantile(probability: float) -> float:
        position = (len(times) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(times) - 1)
        return times[lower] + (times[upper] - times[lower]) * (position - lower)

    return quantile(0.75) - quantile(0.25)


def _timing_evidence(report: Mapping[str, Any]) -> bool:
    context, measurement = report["context"], report["measurement"]
    latency = report["totals"]["latency"]
    samples, number = measurement.get("raw_times_seconds"), measurement.get("number_per_run")
    repeats = context.get("min_repeats")
    return (
        latency["status"] == "complete"
        and latency["unit"] == "seconds"
        and latency["scope"] == "warmed_block_average"
        and latency["method"] == _METHOD
        and _finite_number(latency["value"])
        and type(number) is int
        and number > 0
        and isinstance(samples, list)
        and type(repeats) is int
        and repeats >= 2
        and len(samples) >= repeats
        and all(_finite_number(sample) and sample > 0 for sample in samples)
        and type(measurement.get("measured_calls")) is int
        and measurement["measured_calls"] == number * len(samples)
        and isclose(median(sample / number for sample in samples), latency["value"], rel_tol=1e-12)
        and _finite_number(report["totals"]["latency_iqr"].get("value"))
        and isclose(
            _iqr(samples, number), cast(float, report["totals"]["latency_iqr"]["value"]), rel_tol=1e-9, abs_tol=1e-15
        )
    )


def _memory_evidence(report: Mapping[str, Any], name: str) -> tuple[int | float, tuple[str, str]] | None:
    metric = report["totals"].get(name)
    device = report["context"].get("device", "")
    if not isinstance(metric, Mapping) or metric.get("status") != "complete" or metric.get("unit") != "bytes":
        return None

    expected = {
        "cpu": ("pytorch_tensor_bytes", "torch.profiler.memory_timeline"),
        "cuda": ("pytorch_reserved_bytes", "torch.cuda.max_memory_reserved"),
        "mps": ("pytorch_reserved_bytes", "torch.accelerator.memory.max_memory_reserved"),
    }
    if name == "process_peak_rss":
        expected = dict.fromkeys(("cpu", "cuda"), ("child_process_lifetime", "os.wait4.ru_maxrss"))
    backend = device.partition(":")[0]
    scope, method = expected.get(backend, (None, None))
    if metric.get("scope") != scope or metric.get("method") != method:
        return None
    value = metric.get("value")
    if not _finite_number(value) or value < 0:
        return None
    return value, (cast(str, scope), cast(str, method))


def _validated_trial(trial: object) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(trial, Mapping):
        raise ValueError("trial must be a mapping")
    if trial.get("output_check") != "passed":
        raise ValueError("output check is missing or invalid")

    before, after = _validate_benchmark(trial.get("before")), _validate_benchmark(trial.get("after"))
    _matching_context(before, after)
    for key in _FIXED_SETTINGS:
        if key not in before["context"] or key not in after["context"]:
            raise ValueError(f"fixed context is missing {key!r}")
        if before["context"][key] != after["context"][key]:
            raise ValueError(f"fixed context differs for {key!r}")
    if before["context"]["inputs_source"] != "caller_supplied":
        raise ValueError("caller-supplied input metadata is required")
    if not _timing_evidence(before) or not _timing_evidence(after):
        raise ValueError("complete latency metrics and matching raw timing blocks are required")

    differences = _diff_metrics(before["totals"], after["totals"])
    if trial.get("totals") != differences:
        raise ValueError("stored deltas do not match the measurements")
    if trial.get("latency_change") != _latency_change(before, after, differences):
        raise ValueError("stored latency change does not match the measurements")
    if trial.get("context_changes") != _context_changes(before, after):
        raise ValueError("stored context changes do not match the measurements")
    return before, after


def decide(trials: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return keep, discard or inconclusive from checked repeated measurements.

    One primary objective must improve by the requested fraction in every trial.
    Latency gains must also exceed both runs' IQRs. All other evidence is a gate,
    and incomplete or inconsistent evidence cannot produce ``keep``.
    """
    objective = settings.get("objective")
    improvement = settings.get("min_improvement")
    if objective not in ("latency", "memory"):
        raise ValueError("objective must be 'latency' or 'memory'")
    if not _finite_number(improvement) or not 0 < improvement < 1:
        raise ValueError("min_improvement must be a finite fraction between 0 and 1")
    limit_name = "max_memory_bytes" if objective == "latency" else "max_latency_seconds"
    limit = settings.get(limit_name)
    if limit is not None and (not _finite_number(limit) or limit <= 0):
        raise ValueError(f"{limit_name} must be a positive finite number")
    if len(trials) < 2:
        raise ValueError("at least two trials are required")

    validated: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    uncertainties: list[str] = []
    reference_context: Mapping[str, Any] | None = None
    reference_inputs: Mapping[str, Any] | None = None
    for index, trial in enumerate(trials, start=1):
        if isinstance(trial, Mapping) and trial.get("output_check") == "failed":
            return "discard", [f"Trial {index}: recorded output check failed."]
        try:
            before, after = _validated_trial(trial)
        except (TypeError, ValueError, KeyError, OverflowError, RecursionError) as error:
            uncertainties.append(f"Trial {index} is unusable: {error}.")
            continue
        if reference_context is None:
            reference_context = before["context"]
            reference_inputs = before["inputs"]
        elif (
            any(before["context"].get(key) != reference_context.get(key) for key in _FIXED_CONTEXT)
            or before["inputs"] != reference_inputs
        ):
            uncertainties.append(f"Trial {index} used different hardware or timing settings.")
            continue
        validated.append((index, before, after))

    failures: list[str] = []
    primary_changes: list[float] = []
    latency_labels: list[str] = []
    for index, before, after in validated:
        before_latency = cast(float, before["totals"]["latency"]["value"])
        after_latency = cast(float, after["totals"]["latency"]["value"])
        before_iqr = before["totals"]["latency_iqr"]
        after_iqr = after["totals"]["latency_iqr"]
        valid_iqr = not (
            before_iqr["status"] != "complete"
            or after_iqr["status"] != "complete"
            or before_iqr["unit"] != "seconds"
            or after_iqr["unit"] != "seconds"
            or before_iqr["scope"] != "warmed_block_average"
            or after_iqr["scope"] != "warmed_block_average"
            or before_iqr["method"] != _METHOD
            or after_iqr["method"] != _METHOD
            or not _finite_number(before_iqr["value"])
            or not _finite_number(after_iqr["value"])
            or before_iqr["value"] < 0
            or after_iqr["value"] < 0
        )
        if not valid_iqr:
            uncertainties.append(f"Trial {index} lacks complete latency variability evidence.")

        memory_name = settings.get("memory_metric", "peak_memory")
        before_memory, after_memory = _memory_evidence(before, memory_name), _memory_evidence(after, memory_name)
        if (objective == "memory" or limit is not None) and (
            before_memory is None or after_memory is None or before_memory[1] != after_memory[1]
        ):
            uncertainties.append(f"Trial {index} lacks matching, supported peak-memory evidence.")

        if valid_iqr:
            latency_change = after_latency - before_latency
            variability = max(before_iqr["value"], after_iqr["value"])
            latency_labels.append(
                "faster"
                if latency_change < -variability
                else "slower"
                if latency_change > variability
                else "within_variability"
            )

        if objective == "latency":
            if after_memory is not None and limit is not None:
                candidate_memory = after_memory[0]
                if candidate_memory > limit:
                    failures.append(
                        f"Trial {index} candidate memory {candidate_memory:g} bytes exceeds the {limit:g}-byte ceiling."
                    )
            primary_changes.append((before_latency - after_latency) / before_latency)
        else:
            candidate_latency = after_latency
            if valid_iqr and limit is not None:
                candidate_iqr = after_iqr["value"]
                if candidate_latency - candidate_iqr > limit:
                    failures.append(
                        f"Trial {index} candidate latency {candidate_latency:g}s exceeds the {limit:g}s ceiling."
                    )
                elif candidate_latency + candidate_iqr > limit:
                    uncertainties.append(
                        f"Trial {index} candidate latency overlaps the {limit:g}s ceiling within observed variability."
                    )
            if before_memory is not None and after_memory is not None:
                if before_memory[0] == 0:
                    uncertainties.append(
                        f"Trial {index} baseline peak memory is zero; relative reduction is undefined."
                    )
                else:
                    primary_changes.append((before_memory[0] - after_memory[0]) / before_memory[0])

    if failures:
        return "discard", failures
    if uncertainties:
        return "inconclusive", uncertainties

    if objective == "latency":
        if all(change == "slower" for change in latency_labels):
            return "discard", ["The candidate is consistently slower than the baseline beyond observed IQR."]
        if not all(change == "faster" for change in latency_labels):
            return "inconclusive", ["Latency changes are mixed or within observed IQR."]

    if len(primary_changes) != len(validated):
        return "inconclusive", ["The primary objective is missing from one or more trials."]
    if all(change > improvement and not isclose(change, improvement, rel_tol=1e-12) for change in primary_changes):
        return "keep", [
            f"Every trial improves the {objective} objective by more than the {improvement:.1%} threshold.",
            "All recorded correctness and resource constraints pass.",
        ]
    if objective == "memory" and all(change < 0 for change in primary_changes):
        return "discard", ["The candidate increases peak memory in every trial."]
    return "inconclusive", [
        f"The {objective} improvement does not clear the {improvement:.1%} threshold in every trial."
    ]
