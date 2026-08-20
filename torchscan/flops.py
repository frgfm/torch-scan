# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, TypedDict

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from .report import Diagnostic, MetricResult, metric_result

__all__ = ["FlopReport", "measure_flops"]

_METHOD = "torch.utils.flop_counter.FlopCounterMode"


class IgnoredOperator(TypedDict):
    """Observed operator that TorchScan explicitly excludes from FLOPs."""

    calls: int
    reason: str


class FlopReport(TypedDict):
    """JSON-serializable operator FLOP report."""

    schema_version: int
    context: dict[str, str]
    total: MetricResult
    by_module: dict[str, int]
    by_operator: dict[str, int]
    ignored_operators: dict[str, IgnoredOperator]
    diagnostics: list[Diagnostic]


_IGNORED_OPERATOR_REASONS = {
    # Metadata-only views.
    "aten._unsafe_view": "Metadata-only tensor view.",
    "aten.alias": "Metadata-only tensor view.",
    "aten.as_strided": "Metadata-only tensor view.",
    "aten.detach": "Metadata-only tensor view.",
    "aten.expand": "Metadata-only tensor view.",
    "aten.narrow": "Metadata-only tensor view.",
    "aten.permute": "Metadata-only tensor view.",
    "aten.select": "Metadata-only tensor view.",
    "aten.slice": "Metadata-only tensor view.",
    "aten.squeeze": "Metadata-only tensor view.",
    "aten.t": "Metadata-only tensor view.",
    "aten.transpose": "Metadata-only tensor view.",
    "aten.unbind": "Metadata-only tensor view.",
    "aten.unsqueeze": "Metadata-only tensor view.",
    "aten.view": "Metadata-only tensor view.",
    # Data movement is outside the FLOP convention used by PyTorch's counter.
    "aten._to_copy": "Data movement is excluded from FLOPs.",
    "aten.cat": "Data movement is excluded from FLOPs.",
    "aten.clone": "Data movement is excluded from FLOPs.",
    "aten.contiguous": "Data movement is excluded from FLOPs.",
    "aten.copy_": "Data movement is excluded from FLOPs.",
    "aten.split": "Data movement is excluded from FLOPs.",
    "aten.to": "Data movement is excluded from FLOPs.",
    # Tensor allocation and workload setup are not model arithmetic.
    "aten.empty": "Tensor allocation is excluded from FLOPs.",
    "aten.empty_strided": "Tensor allocation is excluded from FLOPs.",
    "aten.full": "Tensor creation is excluded from FLOPs.",
    "aten.lift_fresh": "Tensor creation is excluded from FLOPs.",
    "aten.lift_fresh_copy": "Tensor creation is excluded from FLOPs.",
    "aten.ones": "Tensor creation is excluded from FLOPs.",
    "aten.scalar_tensor": "Tensor creation is excluded from FLOPs.",
    "aten.zeros": "Tensor creation is excluded from FLOPs.",
}


def _operator_key(operator: Any) -> str:
    packet = getattr(operator, "_overloadpacket", operator)
    return str(packet)


class _OperatorRecorder(TorchDispatchMode):
    # ponytail: PyTorch has no public uncounted-op callback; remove this adapter when FlopCounterMode exposes one.
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.counts[_operator_key(func)] += 1
        return func(*args, **(kwargs or {}))


def measure_flops(
    workload: Callable[[], Any],
    *,
    modules: nn.Module | list[nn.Module] | None = None,
    custom_mapping: Mapping[Any, Callable[..., int | float]] | None = None,
) -> FlopReport:
    """Measure workload FLOPs with PyTorch's native operator counter.

    Args:
        workload: Zero-argument callable invoked exactly once inside the counter.
        modules: Optional module or modules used for hierarchical counts on older supported PyTorch releases.
        custom_mapping: Per-call PyTorch operator-to-FLOP formula overrides.

    Returns:
        A versioned report with known counts and diagnostics for every observed uncounted operator.

    Raises:
        NotImplementedError: If the installed PyTorch counter cannot expose the mapping needed to find missing formulas.
        Exception: Any exception raised by ``workload`` is propagated unchanged.
    """
    mapping = dict(custom_mapping or {})
    overloads = [operator for operator in mapping if getattr(operator, "_overloadpacket", operator) is not operator]
    if overloads:
        raise TypeError(
            "custom_mapping keys must be operator packets such as torch.ops.aten.sin, "
            "not overloads such as torch.ops.aten.sin.default."
        )
    counter = FlopCounterMode(display=False, custom_mapping=mapping)
    if modules is not None and not hasattr(counter, "mod_tracker"):
        counter = FlopCounterMode(mods=modules, display=False, custom_mapping=mapping)
    recorder = _OperatorRecorder()
    with counter, recorder:
        workload()

    raw_counts = counter.get_flop_counts()
    global_counts = raw_counts.get("Global", {})
    by_operator = dict(sorted((_operator_key(operator), int(count)) for operator, count in global_counts.items()))
    counted_operators = set(by_operator)
    by_module = dict(
        sorted((name, int(sum(counts.values()))) for name, counts in raw_counts.items() if name != "Global")
    )

    ignored_operators: dict[str, IgnoredOperator] = {
        operator: {"calls": calls, "reason": _IGNORED_OPERATOR_REASONS[operator]}
        for operator, calls in sorted(recorder.counts.items())
        if operator not in counted_operators and operator in _IGNORED_OPERATOR_REASONS
    }
    uncounted = {
        operator: calls
        for operator, calls in sorted(recorder.counts.items())
        if operator not in counted_operators and operator not in _IGNORED_OPERATOR_REASONS
    }
    diagnostics: list[Diagnostic] = [
        {
            "code": "uncounted_operator",
            "severity": "warning",
            "metric": "flops",
            "operator": operator,
            "message": f"{operator} was observed {calls} time(s), but no FLOP formula is registered.",
        }
        for operator, calls in uncounted.items()
    ]
    known_total = int(sum(global_counts.values()))
    total = metric_result(
        status="partial" if diagnostics else "complete",
        value=None if diagnostics else known_total,
        known_value=known_total if diagnostics else None,
        unit="FLOPs",
        scope="workload",
        method=_METHOD,
    )
    report: FlopReport = {
        "schema_version": 1,
        "context": {"torch_version": torch.__version__, "method": _METHOD},
        "total": total,
        "by_module": by_module,
        "by_operator": by_operator,
        "ignored_operators": ignored_operators,
        "diagnostics": diagnostics,
    }
    return report
