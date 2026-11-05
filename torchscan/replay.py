# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Replay operator metadata without importing PyTorch or executing a workload."""

from importlib import import_module
from typing import cast

from .report import FlopReport, OperatorCapture

__all__ = ["OperatorCapture", "analyze_capture"]

_METHOD = "ATen.RecordFunction/torchscan_flops_v1"
_SCOPE = "calling_thread_and_autograd_eager_ops"


def analyze_capture(capture: OperatorCapture) -> "FlopReport":
    """Convert saved native events into a FLOP report using metadata only.

    Args:
        capture: JSON-loaded capture_workload result. Unknown operators, layouts, lost
            events, and incomplete arguments retain partial counts and diagnostics.

    Returns:
        A standard FlopReport at the calling-thread and autograd eager dispatcher boundary.
        Module counts are inclusive. Complex arithmetic retains an explicitly partial lower bound.

    Raises:
        ValueError: If the capture schema, method, event structure, or loss counters are invalid.
    """
    return cast(FlopReport, import_module("torchscan._analysis").analyze(capture))
