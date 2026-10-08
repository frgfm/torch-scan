# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import torch

from .benchmark import BenchmarkReport, _benchmark_context, _synchronizer, _validate_timing_settings, measure_latency
from .crawler import _describe
from .flops import measure_flops
from .process import measure_peak_memory, measure_peak_rss
from .process.memory import _NoMemoryEventsError
from .profiler import profile_workload
from .report import Diagnostic, MetricResult, metric_result

__all__ = ["measure_workload"]

_WorkloadMetric = Literal["flops", "latency", "throughput", "memory"]
_METRICS: tuple[_WorkloadMetric, ...] = ("flops", "latency", "throughput", "memory")


def _collect(
    collector: Callable[[Callable[[], object]], Any],
    workload: Callable[[], object],
    name: str,
    diagnostics: list[Diagnostic],
) -> Any:
    workload_error: Exception | None = None

    def call() -> object:
        nonlocal workload_error
        try:
            return workload()
        except (RuntimeError, NotImplementedError, ImportError) as error:
            workload_error = error
            raise

    try:
        return collector(call)
    except (_NoMemoryEventsError, NotImplementedError, ImportError) as error:
        if error is workload_error:
            raise
        diagnostics.append({
            "code": "measurement_unavailable",
            "severity": "warning",
            "metric": name,
            "message": str(error),
        })
        return None


def _value(metric: MetricResult) -> str:
    unit = metric["unit"]
    scale, display_unit = (1000, "ms") if unit == "seconds" else (1 / 1024**2, "MiB") if unit == "bytes" else (1, unit)
    if metric["status"] == "unavailable":
        reason = "not requested" if metric["method"] == "not_requested" else "see diagnostics"
        return f"unavailable ({reason})"
    value = metric["value"] if metric["status"] == "complete" else metric["known_value"]
    text = f"{cast(float, value) * scale:,.3f} {display_unit}"
    return f"partial (known lower bound: {text})" if metric["status"] == "partial" else text


def _summary(report: BenchmarkReport) -> str:
    context = report["context"]
    lines = [f"Workload on {context['device']} ({context['num_threads']} PyTorch threads)"]
    for name, label in (
        ("first_call_latency", "First call"),
        ("latency", "Latency (block median)"),
        ("latency_iqr", "Latency IQR"),
        ("throughput", "Throughput"),
        ("operator_flops", "Operator FLOPs"),
        ("peak_memory", "PyTorch peak memory"),
        ("process_peak_rss", "Process peak RSS"),
    ):
        metric = report["totals"][name]
        lines.append(f"  {label}: {_value(metric)}")
        if name in {"peak_memory", "process_peak_rss"} and metric["status"] != "unavailable":
            scope = {
                "pytorch_tensor_bytes": "PyTorch tracked CPU tensors",
                "pytorch_reserved_bytes": "PyTorch device allocator",
                "child_process_lifetime": "fresh child process lifetime",
            }.get(metric["scope"], metric["scope"])
            lines.append(f"    Scope: {scope}")
    lines.append(f"  Profiler: {context['profile_status'].replace('_', ' ')} (separate instrumented pass)")
    lines.extend(f"  {item['metric']}: {item['message']}" for item in report.get("diagnostics", []))
    return "\n".join(lines)


def measure_workload(
    workload: Callable[[], object],
    *,
    device: str | torch.device,
    inputs: Any = None,
    metrics: Sequence[_WorkloadMetric] = _METRICS,
    work_units: float = 1,
    work_unit: str = "samples",
    warmup: int = 5,
    min_run_time: float = 0.2,
    min_repeats: int = 5,
    profile: bool = False,
    rss_command: Sequence[str] | None = None,
    print_summary: bool = True,
) -> BenchmarkReport:
    """Collect selected workload evidence and print ms, work units/s, and MiB.

    Args:
        workload: Repeatable zero-argument callable; owns state, gradients, precision, and placement.
        device: Device on which the workload already executes.
        inputs: Optional input metadata, never forwarded to the workload.
        metrics: Selected measurements; omitted totals stay unavailable with method not_requested.
        work_units: Samples (or other units) per call, such as batch size; never inferred.
        work_unit: Throughput unit name, defaulting to samples.
        warmup: Explicit calls before timing, in addition to timer calibration.
        min_run_time: Minimum warmed timing duration in seconds, not a deadline.
        min_repeats: Minimum timed blocks, at least two.
        profile: Collect a separate operator profile after FLOPs and memory.
        rss_command: Executable and arguments for fresh-process lifetime RSS; must exit.
        print_summary: Print a terminal summary in addition to returning the report.

    Returns:
        A JSON-serializable BenchmarkReport accepted by render_report. Partial and unavailable
        results retain diagnostics. Workload errors and failed RSS commands propagate.

    Notes:
        Timing runs first; each diagnostic pass invokes the same callable once more.
        Caller state and random generators are not reset. Keep threads fixed during all passes.
        First-call time includes all callable work; warmed latency describes block averages, not request
        percentiles. PyTorch peak memory and whole-process RSS have different scopes.
        Use the individual collectors for custom FLOP formulas or trace export.
    """
    _validate_timing_settings(workload, work_units, work_unit, warmup, min_run_time, min_repeats)
    if isinstance(metrics, (str, bytes)) or any(name not in _METRICS for name in metrics):
        raise ValueError(f"metrics must be a sequence of names from {_METRICS}.")
    selected = set(metrics)
    if rss_command is not None and (
        isinstance(rss_command, (str, bytes)) or not rss_command or any(not isinstance(arg, str) for arg in rss_command)
    ):
        raise ValueError("rss_command must be a non-empty sequence of strings.")
    normalized, _ = _synchronizer(torch.device(device))
    diagnostics: list[Diagnostic] = []
    report: BenchmarkReport | None = None
    if selected & {"latency", "throughput"}:
        report = _collect(
            lambda call: measure_latency(
                call,
                device=normalized,
                inputs=inputs,
                work_units=work_units,
                work_unit=work_unit,
                warmup=warmup,
                min_run_time=min_run_time,
                min_repeats=min_repeats,
            ),
            workload,
            "latency",
            diagnostics,
        )
    if report is None:
        report = BenchmarkReport(
            schema_version=1,
            context=_benchmark_context(
                normalized, inputs, work_units, work_unit, warmup, min_run_time, min_repeats, torch.get_num_threads()
            ),
            inputs=_describe(inputs),
            totals={},
            measurement={},
        )
    report["diagnostics"] = diagnostics
    for name, selection, unit, scope in (
        ("first_call_latency", "latency", "seconds", "first_call"),
        ("latency", "latency", "seconds", "warmed_block_average"),
        ("latency_iqr", "latency", "seconds", "warmed_block_average"),
        ("throughput", "throughput", f"{work_unit}/s", "warmed_blocks"),
        ("operator_flops", "flops", "FLOPs", "workload"),
        ("peak_memory", "memory", "bytes", "workload"),
        ("process_peak_rss", "rss", "bytes", "child_process_lifetime"),
    ):
        requested = rss_command is not None if selection == "rss" else selection in selected
        if name not in report["totals"] or not requested:
            report["totals"][name] = metric_result(
                status="unavailable", unit=unit, scope=scope, method="unavailable" if requested else "not_requested"
            )
    if "flops" in selected:
        flops = _collect(measure_flops, workload, "operator_flops", diagnostics)
        if flops is not None:
            report["operator_flops"] = flops
            report["totals"]["operator_flops"] = flops["total"]
            diagnostics.extend(flops["diagnostics"])
    if "memory" in selected:
        memory = _collect(
            lambda call: measure_peak_memory(call, device=normalized), workload, "peak_memory", diagnostics
        )
        if memory is not None:
            report["memory"] = memory
            report["totals"]["peak_memory"] = metric_result(
                status="complete",
                value=memory["peak_bytes"],
                unit="bytes",
                scope=memory["metric"],
                method="torch.profiler.memory_timeline"
                if normalized.type == "cpu"
                else "torch.cuda.max_memory_reserved"
                if normalized.type == "cuda"
                else "torch.accelerator.memory.max_memory_reserved",
            )
    report["context"]["profile_status"] = "not_requested"
    if profile:
        evidence = _collect(
            lambda call: profile_workload(call, device=normalized),
            workload,
            "profile",
            diagnostics,
        )
        report["context"]["profile_status"] = (
            "unavailable" if evidence is None else "partial" if evidence["diagnostics"] else "complete"
        )
        if evidence is not None:
            report["profile"] = evidence
            diagnostics.extend(evidence["diagnostics"])
    if rss_command is not None:
        try:
            report["totals"]["process_peak_rss"] = measure_peak_rss(rss_command)
        except NotImplementedError as error:
            diagnostics.append({
                "code": "measurement_unavailable",
                "severity": "warning",
                "metric": "process_peak_rss",
                "message": str(error),
            })
    if print_summary:
        print(_summary(report))  # ruff: ignore[print]
    return report
