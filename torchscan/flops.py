# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from importlib import import_module
from typing import Any

from torch import nn

from .replay import analyze_capture
from .report import FlopReport

__all__ = ["FlopReport", "measure_flops"]


@contextmanager
def _suspend_collection() -> Generator[None, None, None]:
    native = import_module("_torchscan_native")
    previous = native.pause(True)
    try:
        yield
    finally:
        native.pause(previous)


def measure_flops(
    workload: Callable[[], Any],
    *,
    modules: nn.Module | list[nn.Module] | None = None,
    custom_mapping: Mapping[Any, Callable[..., int | float]] | None = None,
) -> FlopReport:
    """Collect once in C++ and compute FLOPs from the resulting metadata.

    Args:
        workload: Caller-owned zero-argument workload, executed exactly once.
        modules: Optional module roots for inclusive module counts.
        custom_mapping: Per-call operator packet formulas; only these overrides call Python during collection.

    Returns:
        A FlopReport retaining its replayable capture. Unsupported work stays partial.

    Raises:
        TypeError: If a custom key names an overload instead of an operator packet.
        Exception: Workload exceptions propagate unchanged.
    """
    from .capture import capture_workload

    capture = capture_workload(workload, modules=modules, custom_mapping=custom_mapping)
    report = analyze_capture(capture)
    report["capture"] = capture
    return report
