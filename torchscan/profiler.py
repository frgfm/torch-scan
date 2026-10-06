# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from operator import itemgetter
from pathlib import Path
from typing import Any, TypedDict

import torch

from .benchmark import _synchronizer
from .process.memory import _PEAK_MEMORY_LOCK
from .report import Diagnostic

__all__ = ["ProfileReport", "profile_workload"]


class ProfileReport(TypedDict):
    """Operator diagnostics from a separate, instrumented workload call."""

    schema_version: int
    context: dict[str, Any]
    operators: list[dict[str, Any]]
    diagnostics: list[Diagnostic]


def profile_workload(
    workload: Callable[[], object],
    *,
    device: str | torch.device,
    trace_path: str | Path | None = None,
    limit: int = 20,
) -> ProfileReport:
    """Find expensive operators without treating profiler timings as benchmark latency.

    Args:
        workload: Zero-argument callable, invoked once with its state unchanged by TorchScan.
        device: Device on which the workload already executes. CPU/CUDA are traced;
            MPS returns CPU dispatch events with an explicit GPU-time limitation.
        trace_path: Optional Chrome trace destination. Traces can contain input shapes
            and execution details; keep them with the experiment that owns the data.
        limit: Positive maximum number of grouped operator rows, ranked by device
            self time on CUDA and CPU self time otherwise.

    Returns:
        Operator calls, input shapes, self times in seconds, and net allocated bytes.
        Allocated bytes are event deltas, not peak RAM. Parent/overlapping time must
        not be summed into a claimed model latency.

    Raises:
        TypeError: If workload is not callable.
        ValueError: If limit is invalid.

    Notes:
        Profiling is a separate instrumented pass. Use measure_latency for clean timing.
        No model evaluation, warmup, device movement, or output retention is added.
    """
    if not callable(workload):
        raise TypeError("workload must be callable.")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer.")
    normalized, synchronize = _synchronizer(torch.device(device))
    activities = [torch.profiler.ProfilerActivity.CPU]
    device_time_supported = (
        normalized.type == "cuda" and torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
    )
    if device_time_supported:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with _PEAK_MEMORY_LOCK:
        if trace_path is not None and Path(trace_path).exists():
            raise FileExistsError(f"Trace destination already exists: {trace_path}")
        synchronize()
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            workload()
            synchronize()
        events = profiler.key_averages(group_by_input_shape=True)
        if trace_path is not None:
            profiler.export_chrome_trace(str(trace_path))
    rows: list[dict[str, Any]] = []
    for event in events:
        device_time = None
        if device_time_supported:
            device_time = getattr(event, "self_device_time_total", None)
            if device_time is None:
                device_time = event.self_cuda_time_total
        rows.append({
            "operator": event.key,
            "calls": event.count,
            "input_shapes": event.input_shapes,
            "cpu_self_seconds": event.self_cpu_time_total / 1e6,
            "device_self_seconds": None if device_time is None else device_time / 1e6,
            "cpu_net_bytes": event.self_cpu_memory_usage,
        })
    ranking = "device_self_seconds" if device_time_supported else "cpu_self_seconds"
    rows.sort(key=itemgetter(ranking), reverse=True)
    diagnostics: list[Diagnostic] = []
    if normalized.type == "mps":
        diagnostics.append({
            "code": "mps_gpu_trace_unavailable",
            "severity": "warning",
            "metric": "device_time",
            "message": "Rows contain CPU dispatch time only. Use torch.mps.profiler and Instruments for MPS GPU traces.",
        })
    elif normalized.type == "cuda" and not device_time_supported:
        diagnostics.append({
            "code": "cuda_gpu_trace_unavailable",
            "severity": "warning",
            "metric": "device_time",
            "message": "This PyTorch profiler exposes CPU events only; CUDA device self time is unavailable.",
        })
    return {
        "schema_version": 1,
        "context": {
            "torch_version": str(torch.__version__),
            "device": str(normalized),
            "method": "torch.profiler.profile",
            "operator_groups": len(rows),
            "limit": limit,
            "device_time_supported": device_time_supported,
        },
        "operators": rows[:limit],
        "diagnostics": diagnostics,
    }
