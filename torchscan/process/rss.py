# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Sequence
from pathlib import Path

from ..report import MetricResult, metric_result

__all__ = ["measure_peak_rss"]


def measure_peak_rss(command: Sequence[str], *, cwd: str | Path | None = None) -> MetricResult:
    """Measure the OS peak resident RAM of one new child command.

    Args:
        command: Executable and arguments, passed without a shell. The child owns
            model loading, inputs, device placement, and the workload.
        cwd: Optional child working directory. Not retained in the result.

    Returns:
        Peak resident bytes for the child's whole lifetime, including imports and
        model loading. This is not an inference-only delta or accelerator memory.

    Raises:
        ValueError: If command is empty or contains non-string arguments.
        NotImplementedError: If per-child RSS accounting is unavailable.
        subprocess.CalledProcessError: If the command exits unsuccessfully.

    Notes:
        Linux/macOS use per-child wait4 accounting, not the parent's cumulative
        child high-water mark. Standard streams are inherited. Child process trees
        are not summed, and container/device memory are outside this metric's scope.
    """
    if isinstance(command, (str, bytes)) or not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError("command must be a non-empty sequence of strings.")
    if sys.platform not in {"darwin", "linux"} or not hasattr(os, "wait4"):
        raise NotImplementedError("Per-child peak RSS is supported on Linux and macOS only.")
    with subprocess.Popen(list(command), cwd=cwd) as child:  # ruff: ignore[subprocess-without-shell-equals-true]
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, list(command))
    scale = 1 if sys.platform == "darwin" else 1024
    return metric_result(
        status="complete",
        value=int(usage.ru_maxrss) * scale,
        unit="bytes",
        scope="child_process_lifetime",
        method="os.wait4.ru_maxrss",
    )
