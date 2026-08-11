# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

__all__ = ["measure_peak_memory"]

_PEAK_MEMORY_LOCK = threading.Lock()


def _measure_cpu(workload: Callable[[], object]) -> dict[str, str | int]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as timeline_file:
        timeline_path = Path(timeline_file.name)

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as profiler:
            workload()

        profiler.export_memory_timeline(str(timeline_path), device="cpu")
        with timeline_path.open(encoding="utf-8") as timeline_file:
            _, category_points = json.load(timeline_file)

        totals = [sum(int(value) for value in point) for point in category_points]
        if not totals:
            raise RuntimeError(
                f"PyTorch {torch.__version__} exported no memory timeline points for 'cpu'; "
                "the workload produced no PyTorch-tracked CPU memory events."
            )
        baseline = totals[0]
        peak = max(totals)
        return {
            "device": "cpu",
            "metric": "pytorch_tensor_bytes",
            "baseline_bytes": baseline,
            "peak_bytes": peak,
            "delta_bytes": peak - baseline,
        }
    finally:
        timeline_path.unlink(missing_ok=True)


def _measure_cuda(workload: Callable[[], object], device: torch.device) -> dict[str, str | int]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device '{device}' is unavailable in PyTorch {torch.__version__}; "
            "verify the CUDA-enabled build, NVIDIA driver, and hardware."
        )

    device_count = torch.cuda.device_count()
    device_index = torch.cuda.current_device() if device.index is None else device.index
    normalized_device = torch.device("cuda", device_index)
    if device_index >= device_count:
        raise RuntimeError(
            f"Requested CUDA device '{normalized_device}' is unavailable in PyTorch {torch.__version__}; "
            f"PyTorch reports {device_count} CUDA device(s)."
        )

    required_apis = (
        "synchronize",
        "memory_reserved",
        "reset_peak_memory_stats",
        "max_memory_reserved",
        "max_memory_allocated",
    )
    missing_apis = [f"torch.cuda.{name}" for name in required_apis if not callable(getattr(torch.cuda, name, None))]
    if missing_apis:
        raise NotImplementedError(
            f"Peak memory measurement for '{normalized_device}' is unavailable in PyTorch {torch.__version__}; "
            f"missing APIs: {', '.join(missing_apis)}."
        )

    torch.cuda.synchronize(normalized_device)
    baseline = int(torch.cuda.memory_reserved(normalized_device))
    torch.cuda.reset_peak_memory_stats(normalized_device)
    workload()
    torch.cuda.synchronize(normalized_device)
    peak = int(torch.cuda.max_memory_reserved(normalized_device))
    allocated_peak = int(torch.cuda.max_memory_allocated(normalized_device))
    return {
        "device": str(normalized_device),
        "metric": "pytorch_reserved_bytes",
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "delta_bytes": peak - baseline,
        "allocated_peak_bytes": allocated_peak,
    }


def _measure_mps(workload: Callable[[], object], device: torch.device) -> dict[str, str | int]:
    if device.index not in (None, 0):
        raise RuntimeError(
            f"Requested MPS device '{device}' is unavailable in PyTorch {torch.__version__}; MPS exposes one device."
        )

    normalized_device = torch.device("mps")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        raise RuntimeError(
            f"Requested MPS device '{normalized_device}' is unavailable in PyTorch {torch.__version__}; "
            "verify an MPS-enabled build on supported Apple hardware."
        )

    mps = getattr(torch, "mps", None)
    accelerator = getattr(torch, "accelerator", None)
    memory = getattr(accelerator, "memory", None)
    required_apis = (
        "memory_reserved",
        "reset_peak_memory_stats",
        "max_memory_reserved",
        "max_memory_allocated",
    )
    missing_apis = [
        f"torch.accelerator.memory.{name}" for name in required_apis if not callable(getattr(memory, name, None))
    ]
    if not callable(getattr(mps, "synchronize", None)):
        missing_apis.append("torch.mps.synchronize")
    if missing_apis:
        raise NotImplementedError(
            f"Peak memory measurement for '{normalized_device}' is unavailable in PyTorch {torch.__version__}; "
            f"missing APIs: {', '.join(missing_apis)}."
        )

    mps = cast(Any, mps)
    memory = cast(Any, memory)
    mps.synchronize()
    try:
        baseline = int(memory.memory_reserved(normalized_device))
        memory.reset_peak_memory_stats(normalized_device)
    except (NotImplementedError, RuntimeError) as error:
        api_names = ", ".join(f"torch.accelerator.memory.{name}" for name in required_apis)
        raise NotImplementedError(
            f"Peak memory measurement for '{normalized_device}' is unavailable in PyTorch {torch.__version__}; "
            f"required APIs are not implemented for MPS: {api_names}."
        ) from error
    workload()
    mps.synchronize()
    peak = int(memory.max_memory_reserved(normalized_device))
    allocated_peak = int(memory.max_memory_allocated(normalized_device))
    return {
        "device": str(normalized_device),
        "metric": "pytorch_reserved_bytes",
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "delta_bytes": peak - baseline,
        "allocated_peak_bytes": allocated_peak,
    }


def measure_peak_memory(
    workload: Callable[[], object],
    *,
    device: str | torch.device,
) -> dict[str, str | int]:
    """Measure peak PyTorch memory used by one workload invocation.

    Args:
        workload: Zero-argument callable to invoke exactly once.
        device: Device on which the workload already operates.

    Returns:
        JSON-serializable backend-specific memory statistics in bytes.

    Raises:
        NotImplementedError: If the backend or its measurement APIs are unsupported.
        RuntimeError: If the requested device is unavailable.
    """
    with _PEAK_MEMORY_LOCK:
        normalized_device = torch.device(device)
        if normalized_device.type == "cpu":
            return _measure_cpu(workload)
        if normalized_device.type == "cuda":
            return _measure_cuda(workload, normalized_device)
        if normalized_device.type == "mps":
            return _measure_mps(workload, normalized_device)
        raise NotImplementedError(
            f"Peak memory measurement is not implemented for '{normalized_device}' in PyTorch {torch.__version__}."
        )
