# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, TypedDict

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import FlopCounterMode

from ._flop_formulas import FORMULAS
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
    **dict.fromkeys(
        [
            "aten._unsafe_view",
            "aten.alias",
            "aten.as_strided",
            "aten.detach",
            "aten.expand",
            "aten.narrow",
            "aten.permute",
            "aten.select",
            "aten.slice",
            "aten.squeeze",
            "aten.squeeze_",
            "aten.t",
            "aten.transpose",
            "aten.unbind",
            "aten.unsqueeze",
            "aten.view",
        ],
        "Metadata-only tensor view.",
    ),
    **dict.fromkeys(
        [
            "aten._to_copy",
            "aten.cat",
            "aten.clone",
            "aten.contiguous",
            "aten.copy_",
            "aten.split",
            "aten.split_with_sizes",
            "aten.to",
            "aten.fill_",
            "aten.zero_",
        ],
        "Data movement is excluded from FLOPs.",
    ),
    **dict.fromkeys(
        ["aten.empty", "aten.empty_strided", "aten.empty_like"], "Tensor allocation is excluded from FLOPs."
    ),
    **dict.fromkeys(
        [
            "aten.full",
            "aten.full_like",
            "aten.lift_fresh",
            "aten.lift_fresh_copy",
            "aten.ones",
            "aten.ones_like",
            "aten.scalar_tensor",
            "aten.zeros",
            "aten.zeros_like",
        ],
        "Tensor creation is excluded from FLOPs.",
    ),
}


def _operator_key(operator: Any) -> str:
    packet = getattr(operator, "_overloadpacket", operator)
    return str(packet)


class _OperatorRecorder(TorchDispatchMode):
    # ponytail: PyTorch has no public uncounted-op callback; remove this adapter when FlopCounterMode exposes one.
    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.counts: Counter[Any] = Counter()
        self.floating: dict[Any, bool] = {}
        self.complex: dict[Any, bool] = {}
        self.complex_operators: set[Any] = set()
        self.registry: dict[Any, Any] = {}
        self.guarded_operators: set[Any] = set()
        self.diagnostics = diagnostics

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        packet = getattr(func, "_overloadpacket", func)
        self.counts[packet] += 1
        leaves = tree_flatten((args, kwargs or {}))[0]
        self.floating[packet] = any(
            (isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()))
            or isinstance(value, (float, complex))
            or (isinstance(value, torch.dtype) and (value.is_floating_point or value.is_complex))
            for value in leaves
        )
        self.complex[packet] = any(
            (isinstance(value, torch.Tensor) and value.is_complex())
            or isinstance(value, complex)
            or (isinstance(value, torch.dtype) and value.is_complex)
            for value in leaves
        )
        if packet in {torch.ops.aten.sum, torch.ops.aten.mean}:
            # Reductions cast before arithmetic; an out buffer also sets the dtype.
            dtype = (kwargs or {}).get("dtype")
            out = (kwargs or {}).get("out")
            dtype = dtype or (out.dtype if isinstance(out, torch.Tensor) else args[0].dtype)
            self.floating[packet] = dtype.is_floating_point or dtype.is_complex
            self.complex[packet] = dtype.is_complex
        if self.complex[packet]:
            self.complex_operators.add(packet)
        if packet in {torch.ops.aten.div, torch.ops.aten.div_} and (kwargs or {}).get("rounding_mode") is None:
            self.floating[packet] = True  # True division promotes integer inputs.
        if packet in {torch.ops.aten.exp, torch.ops.aten.sqrt, torch.ops.aten.rsqrt}:
            self.floating[packet] = True  # Transcendentals also promote integer inputs.
        if packet in {torch.ops.aten.masked_fill, torch.ops.aten.masked_fill_}:
            self.floating[packet] = args[0].is_floating_point() or args[0].is_complex()
        if packet in self.guarded_operators and any(
            isinstance(value, torch.Tensor) and (value.is_nested or value.layout != torch.strided) for value in leaves
        ):
            self.diagnostics.append({
                "code": "unsupported_operator_formula",
                "severity": "warning",
                "metric": "flops",
                "operator": _operator_key(packet),
                "message": "Shape FLOP formulas require dense strided tensors; sparse and nested layouts are unsupported.",
            })
            # PyTorch 2.1 extracts shapes before calling formulas. Bypass this
            # invocation's packet before extraction; restore it for later calls.
            # Modern overload aliases still prevent unwanted decomposition.
            formula = self.registry.pop(packet)
            try:
                return func(*args, **(kwargs or {}))
            finally:
                self.registry[packet] = formula
        return func(*args, **(kwargs or {}))


def _scoped_mapping(
    counter: Any, recorder: _OperatorRecorder, diagnostics: list[Diagnostic]
) -> dict[Any, Callable[..., int]]:
    registry = getattr(counter, "flop_registry", getattr(counter, "flop_mapping", None))
    if registry is None:
        raise NotImplementedError("PyTorch's counter does not expose its formula mapping.")

    def guarded(operator: Any, formula: Callable[..., int]) -> Callable[..., int]:
        def count(*args: Any, **kwargs: Any) -> int:
            try:
                if recorder.complex.get(operator, False):
                    raise NotImplementedError("Supplemental FLOP formulas cover real arithmetic only.")
                if not recorder.floating.get(operator, True):
                    # Integer counters and boolean control arithmetic are excluded.
                    return 0
                if kwargs.get("rounding_mode") is not None:
                    raise NotImplementedError("Rounded division is not covered by the scalar arithmetic formula.")
                return formula(*args, **kwargs)
            except NotImplementedError as error:
                diagnostics.append({
                    "code": "unsupported_operator_formula",
                    "severity": "warning",
                    "metric": "flops",
                    "operator": _operator_key(operator),
                    "message": str(error),
                })
                return 0

        return count

    return {
        operator: guarded(operator, formula)
        for name, formula in FORMULAS.items()
        if (operator := getattr(torch.ops.aten, name, None)) is not None and operator not in registry
    }


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
    diagnostics: list[Diagnostic] = []
    recorder = _OperatorRecorder(diagnostics)
    # Inspect the invocation's upstream mapping, then fill gaps. The caller wins.
    base_counter = FlopCounterMode(display=False)
    mapping = {**_scoped_mapping(base_counter, recorder, diagnostics), **mapping}
    counter = FlopCounterMode(
        mods=modules if not hasattr(base_counter, "mod_tracker") else None, display=False, custom_mapping=mapping
    )
    if hasattr(counter, "flop_registry"):
        # Recent PyTorch checks overload keys before decomposing, then packet keys
        # when counting. Alias scoped formulas in this counter only so one logical
        # operation is counted once and the recorder sees the counted operation.
        for operator in mapping:
            for overload in getattr(operator, "overloads", lambda: ())():
                counter.flop_registry[getattr(operator, overload)] = counter.flop_registry[operator]
    recorder.registry = getattr(counter, "flop_registry", getattr(counter, "flop_mapping", {}))
    recorder.guarded_operators = set(recorder.registry) - set(custom_mapping or {})
    with counter, recorder:
        workload()

    raw_counts = counter.get_flop_counts()
    global_counts = raw_counts.get("Global", {})
    by_operator = dict(sorted((_operator_key(operator), int(count)) for operator, count in global_counts.items()))
    counted_operators = set(by_operator)
    by_module = dict(
        sorted((name, int(sum(counts.values()))) for name, counts in raw_counts.items() if name != "Global")
    )
    observed = sorted((_operator_key(operator), calls) for operator, calls in recorder.counts.items())

    ignored_operators: dict[str, IgnoredOperator] = {
        operator: {"calls": calls, "reason": _IGNORED_OPERATOR_REASONS[operator]}
        for operator, calls in observed
        if operator not in counted_operators and operator in _IGNORED_OPERATOR_REASONS
    }
    diagnosed_operators = {item.get("operator") for item in diagnostics}
    uncounted = {
        operator: calls
        for operator, calls in observed
        if operator not in counted_operators
        and operator not in _IGNORED_OPERATOR_REASONS
        and operator not in diagnosed_operators
    }
    diagnostics.extend([
        {
            "code": "uncounted_operator",
            "severity": "warning",
            "metric": "flops",
            "operator": operator,
            "message": f"{operator} was observed {calls} time(s), but no FLOP formula is registered.",
        }
        for operator, calls in uncounted.items()
    ])
    # Native fused attention and complex shape counts omit real arithmetic.
    complex_operators = {_operator_key(packet) for packet in recorder.complex_operators}
    for operator in sorted(counted_operators - {_operator_key(key) for key in mapping}):
        if operator in complex_operators:
            message = "Native shape formula does not account for real operations inside complex arithmetic."
        elif "attention" in operator:
            message = (
                "Native fused attention formula counts matrix products only; scaling, softmax, masks, "
                "and dropout work are not covered."
            )
        else:
            continue
        diagnostics.append({
            "code": "incomplete_operator_formula",
            "severity": "warning",
            "metric": "flops",
            "operator": operator,
            "message": message,
        })
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
        "context": {"torch_version": torch.__version__, "method": _METHOD, "counting_convention": "torchscan_flops_v1"},
        "total": total,
        "by_module": by_module,
        "by_operator": by_operator,
        "ignored_operators": ignored_operators,
        "diagnostics": diagnostics,
    }
    return report
