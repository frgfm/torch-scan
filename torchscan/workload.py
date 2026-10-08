# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from .benchmark import BenchmarkReport, _benchmark_context, _synchronizer, _validate_timing_settings, measure_latency
from .crawler import _describe
from .flops import measure_flops
from .process import measure_peak_memory, measure_peak_rss
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
    except (RuntimeError, NotImplementedError, ImportError) as error:
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
    modules: nn.Module | list[nn.Module] | None = None,
    custom_mapping: Mapping[Any, Callable[..., int | float]] | None = None,
    profile: bool = False,
    trace_path: str | Path | None = None,
    profile_limit: int = 20,
    rss_command: Sequence[str] | None = None,
    rss_cwd: str | Path | None = None,
    print_summary: bool = True,
) -> BenchmarkReport:
    """Measure a PyTorch workload and print a compact terminal summary.

    Args:
        workload: Zero-argument callable. It owns model state, gradient mode,
            precision, device placement, and inputs. All workload exceptions propagate.
        device: Device on which the workload already runs; selects synchronization and memory APIs.
        inputs: Optional caller-supplied input metadata, not arguments passed to the workload.
        metrics: Selected measurements. Timing runs if latency or throughput is selected.
            Omitted metrics remain unavailable with method ``not_requested``.
        work_units: Number of samples (or other work units) per call. Set this to your batch size;
            TorchScan does not infer it from inputs. Defaults to one sample per call.
        work_unit: Throughput unit name; defaults to ``samples``.
        warmup: Explicit calls before warmed timing, in addition to timer calibration.
        min_run_time: Minimum warmed timing duration in seconds, not a total runtime limit.
        min_repeats: Minimum number of timed blocks, at least two.
        modules: Optional module attribution passed to measure_flops.
        custom_mapping: Per-call operator FLOP overrides passed to measure_flops.
        profile: Collect a separate operator profile after the other in-process passes.
        trace_path: Optional Chrome trace file. Requires profile=True; existing files are rejected.
        profile_limit: Positive maximum number of grouped profiler rows.
        rss_command: Explicit executable and arguments for a fresh-process RSS measurement.
            The command owns model loading and configuration and must exit. No shell is used.
        rss_cwd: Optional working directory for rss_command. Requires rss_command.
        print_summary: Print times in ms, throughput in work units/s, and memory in MiB.

    Returns:
        A BenchmarkReport accepted by JSON serialization and render_report. Raw timing,
        FLOP, memory, and optional profiler evidence retain their methods and scopes.
        Collector limitations produce unavailable metrics and diagnostics.

    Raises:
        ValueError: If options are invalid or contradict the selected passes.
        TypeError: If workload is not callable.
        Exception: Workload errors and RSS command failures propagate unchanged.

    Notes:
        Timing runs first. FLOPs, memory, and profiling each call the workload once more.
        Passes share caller state; TorchScan does not reset state or random generators.
        Use a repeatable callable and manage gradients and mutable state yourself.
        Configure threads before building the model and keep them fixed during measurement.
        First-call time excludes model loading and fresh-process startup. Warmed latency
        describes block averages, not request percentiles. PyTorch memory is backend-specific;
        RSS covers the separate command's whole lifetime, including imports and loading.
    """
    _validate_timing_settings(workload, work_units, work_unit, warmup, min_run_time, min_repeats)
    if isinstance(metrics, (str, bytes)) or any(name not in _METRICS for name in metrics):
        raise ValueError(f"metrics must be a sequence of names from {_METRICS}.")
    selected = set(metrics)
    if trace_path is not None and not profile:
        raise ValueError("trace_path requires profile=True.")
    if profile and (type(profile_limit) is not int or profile_limit < 1):
        raise ValueError("profile_limit must be a positive integer.")
    if trace_path is not None and Path(trace_path).exists():
        raise FileExistsError(f"Trace destination already exists: {trace_path}")
    if rss_cwd is not None and rss_command is None:
        raise ValueError("rss_cwd requires rss_command.")
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
        flops = _collect(
            lambda call: measure_flops(call, modules=modules, custom_mapping=custom_mapping),
            workload,
            "operator_flops",
            diagnostics,
        )
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
            lambda call: profile_workload(call, device=normalized, trace_path=trace_path, limit=profile_limit),
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
            report["totals"]["process_peak_rss"] = measure_peak_rss(rss_command, cwd=rss_cwd)
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
