# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import torch

from .crawler import _describe, _package_version
from .report import MetricResult, metric_result

if TYPE_CHECKING:
    from torch.utils.benchmark import Measurement, Timer

__all__ = ["BenchmarkReport", "measure_latency"]

_METHOD = "torch.utils.benchmark.Timer"
# ponytail: native timing uses process-global thread settings; use processes for parallel benchmarks.
_TIMING_LOCK = threading.Lock()


class BenchmarkReport(TypedDict):
    """JSON-serializable timing report for one caller-controlled workload."""

    schema_version: int
    context: dict[str, Any]
    inputs: dict[str, Any]
    totals: dict[str, MetricResult]
    measurement: dict[str, Any]


def _processor() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith(("model name", "Hardware")):
                    return line.partition(":")[2].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor()


def _synchronizer(device: torch.device) -> tuple[torch.device, Callable[[], None]]:
    if device.type == "cpu":
        return torch.device("cpu"), lambda: None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device '{device}' is unavailable in PyTorch {torch.__version__}.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device 'cuda:{index}' is unavailable.")
        device = torch.device("cuda", index)
        return device, lambda: torch.cuda.synchronize(device)
    if device.type == "mps":
        if device.index not in (None, 0) or not torch.backends.mps.is_available():
            raise RuntimeError(f"Requested MPS device '{device}' is unavailable in PyTorch {torch.__version__}.")
        return torch.device("mps"), torch.mps.synchronize
    raise NotImplementedError(f"Latency measurement is not implemented for '{device}'.")


def _benchmark_tools() -> tuple[type["Timer"], type["Measurement"]]:
    try:
        from torch.utils.benchmark import Measurement, Timer
    except ImportError as error:
        raise ImportError(
            "PyTorch's benchmark utilities could not be imported. Install the benchmark dependencies required by your "
            "PyTorch version, or upgrade PyTorch. PyTorch 2.1 requires setuptools<70 and numpy<2 for these utilities."
        ) from error
    return Timer, Measurement


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _run_timing(
    workload: Callable[[], object],
    synchronize: Callable[[], None],
    warmup: int,
    min_run_time: float,
    min_repeats: int,
) -> tuple[float, "Measurement", int]:
    timer_type, measurement_type = _benchmark_tools()

    def clock() -> float:
        synchronize()
        return time.perf_counter()

    with _TIMING_LOCK:
        threads = torch.get_num_threads()
        start = clock()
        workload()
        first_call = clock() - start
        for _ in range(warmup):
            workload()
        timer = timer_type(stmt="workload()", globals={"workload": workload}, timer=clock, num_threads=threads)
        measurement = timer.blocked_autorange(min_run_time=min_run_time)
        samples = list(measurement.raw_times)
        while len(samples) < min_repeats:
            samples.extend(timer.timeit(measurement.number_per_run).raw_times)
        measurement = measurement_type(
            number_per_run=measurement.number_per_run, raw_times=samples, task_spec=measurement.task_spec
        )
    return first_call, measurement, threads


def measure_latency(
    workload: Callable[[], object],
    *,
    device: str | torch.device,
    inputs: Any = None,
    work_units: float = 1,
    work_unit: str = "calls",
    warmup: int = 5,
    min_run_time: float = 0.2,
    min_repeats: int = 5,
) -> BenchmarkReport:
    """Measure first-call time, warmed block-average latency, and throughput.

    Args:
        workload: Zero-argument callable. Owns model state, inputs, gradient mode,
            precision, transfers, and any other work included in the timing.
            It is called repeatedly, including during PyTorch timer calibration.
        device: CPU, CUDA, or MPS device on which the workload already executes.
            Select and place the workload yourself; this argument only chooses synchronization.
            Work on that device is completed before reading each timer boundary.
        inputs: Optional caller-supplied input description. Only recursive metadata
            is recorded; this argument is not forwarded to the workload.
        work_units: Positive number of work units completed by each invocation.
        work_unit: Unit name, such as "images" or "tokens"; throughput uses this unit per second.
        warmup: Explicit warmup calls after the first call. PyTorch also calibrates its timer.
        min_run_time: Positive minimum duration of warmed block measurements in seconds.
            Calibration, first call, warmup, and minimum repeats can extend total runtime.
        min_repeats: Minimum number of timed blocks, at least two.

    Returns:
        Structured timing metrics, block samples, and execution/input metadata.
        Latency statistics describe block averages, not individual request percentiles.
        First-call time is not fresh-process startup or model-load time.

    Raises:
        TypeError: If workload is not callable.
        ValueError: If timing settings or work units are invalid.
        ImportError: If the installed PyTorch benchmark utilities lack their dependencies.
        RuntimeError: If the device is unavailable or timing produces invalid samples.
        NotImplementedError: If the device backend is unsupported.

    Notes:
        Exceptions and workload side effects are preserved. This function does not
        change gradient mode, move inputs, evaluate the model, or measure FLOPs/memory.
        Configure PyTorch threads before building the model. The active count is recorded;
        keep it unchanged during the workload. Use one device per workload;
        multi-device/distributed synchronization is not provided.
    """
    if not callable(workload):
        raise TypeError("workload must be callable.")
    for name, value in (("work_units", work_units), ("min_run_time", min_run_time)):
        if not _positive_finite(value):
            raise ValueError(f"{name} must be a positive finite number.")
    for name, value, minimum in (("warmup", warmup, 0), ("min_repeats", min_repeats, 2)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not isinstance(work_unit, str) or not work_unit.strip():
        raise ValueError("work_unit must be a non-empty string.")
    input_metadata = _describe(inputs)
    normalized_device, synchronize = _synchronizer(torch.device(device))
    processor = _processor()
    device_name = torch.cuda.get_device_name(normalized_device) if normalized_device.type == "cuda" else processor

    first_call, measurement, threads = _run_timing(workload, synchronize, warmup, min_run_time, min_repeats)
    samples = measurement.raw_times

    if any(not _positive_finite(value) for value in [first_call, *samples]):
        raise RuntimeError("Timing must produce positive finite durations.")
    measured_calls = measurement.number_per_run * len(samples)
    throughput = work_units / (sum(samples) / measured_calls)
    if not _positive_finite(throughput):
        raise RuntimeError("Throughput must be positive and finite; adjust work_units.")
    return {
        "schema_version": 1,
        "context": {
            "torchscan_version": _package_version(),
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": processor,
            "device": str(normalized_device),
            "device_name": device_name,
            "num_threads": threads,
            "num_interop_threads": torch.get_num_interop_threads(),
            "cuda_version": torch.version.cuda,
            "inputs_source": "caller_supplied" if inputs is not None else "not_provided",
            "work_units": work_units,
            "work_unit": work_unit,
            "warmup": warmup,
            "min_run_time": min_run_time,
            "min_repeats": min_repeats,
        },
        "inputs": input_metadata,
        "totals": {
            name: metric_result(status="complete", value=value, unit=unit, scope=scope, method=method)
            for name, value, unit, scope, method in (
                ("first_call_latency", first_call, "seconds", "first_call", "synchronized_perf_counter"),
                ("latency", measurement.median, "seconds", "warmed_block_average", _METHOD),
                ("latency_iqr", measurement.iqr, "seconds", "warmed_block_average", _METHOD),
                ("throughput", throughput, f"{work_unit}/s", "warmed_blocks", _METHOD),
            )
        },
        "measurement": {
            "number_per_run": measurement.number_per_run,
            "raw_times_seconds": samples,
            "measured_calls": measured_calls,
        },
    }
