# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, TypedDict, cast

from torch.nn import Module

from .report import Diagnostic, MetricResult, metric_result

__all__ = ["ModuleCall", "ModuleEstimates", "ModuleHandler"]

_METRIC_UNITS = {
    "module_flops": "FLOPs",
    "macs": "MACs",
    "dmas": "DMAs",
    "receptive_field": "elements",
    "effective_stride": "elements",
    "effective_padding": "elements",
}


@dataclass(frozen=True)
class ModuleCall:
    """Actual forward-call objects supplied to an estimation callback.

    Args:
        module: Module whose forward call just completed.
        args: Complete positional arguments, including containers and non-tensors.
        kwargs: Complete actual keyword arguments; omitted defaults are not inserted.
        output: Complete output, including containers and non-tensors.

    These objects are borrowed, not copied. Callbacks must treat them as read-only.
    The callback runs immediately after forward in evaluation/no-grad mode, with
    operator dispatch counting suspended. Retaining tensors extends their lifetime.
    """

    module: Module
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    output: Any


class ModuleEstimates(TypedDict, total=False):
    """Independent module estimates returned by a callback.

    Numbers mean complete estimates; ``None`` explicitly means unavailable.
    Structured results preserve partial lower bounds and unavailable states.
    Omitted fields fall back to built-in leaf formulas or, for composites, child
    estimates. An omitted field owned by a subtree handler remains unavailable.
    """

    module_flops: int | float | MetricResult | None
    macs: int | float | MetricResult | None
    dmas: int | float | MetricResult | None
    receptive_field: int | float | MetricResult | None
    effective_stride: int | float | MetricResult | None
    effective_padding: int | float | MetricResult | None


@dataclass(frozen=True)
class ModuleHandler:
    """Scoped callback and explicit ownership of inclusive subtree estimates.

    Args:
        estimate: Callback accepting a complete ``ModuleCall`` and returning
            independent ``ModuleEstimates``. Exceptions become report diagnostics.
        subtree_metrics: Names of metrics for which this module owns all work in
            its executed subtree. Children still produce structural call records,
            but their owned metric fields are omitted to avoid double-counting.
            Ownership persists when the callback fails or omits an owned estimate.

    Other estimates describe only this module's own work, excluding its children.
    Registration is supplied to one analysis through ``custom_modules``. Matching
    uses the closest class in the concrete module's MRO; caller handlers are checked
    before built-in handlers, even when the caller registers a base class.
    """

    estimate: Callable[[ModuleCall], ModuleEstimates]
    subtree_metrics: frozenset[str] = field(default_factory=frozenset)


def _validate_handlers(handlers: Mapping[type[Module], ModuleHandler] | None) -> dict[type[Module], ModuleHandler]:
    if handlers is not None and not isinstance(handlers, Mapping):
        raise TypeError("custom_modules must be a mapping of module types to ModuleHandler values.")
    snapshot = dict(handlers or {})
    for module_type, handler in snapshot.items():
        if not isinstance(module_type, type) or not issubclass(module_type, Module):
            raise TypeError("custom_modules keys must be torch.nn.Module types.")
        if not isinstance(handler, ModuleHandler) or not callable(handler.estimate):
            raise TypeError("custom_modules values must be ModuleHandler instances with callable estimates.")
        if not isinstance(handler.subtree_metrics, frozenset) or not handler.subtree_metrics <= _METRIC_UNITS.keys():
            raise ValueError("subtree_metrics must be a frozenset of supported metric names.")
    return snapshot


def _resolve_handler(module: Module, handlers: Mapping[type[Module], ModuleHandler]) -> ModuleHandler | None:
    return next((handlers[cast(type[Module], base)] for base in type(module).__mro__ if base in handlers), None)


def _number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError("Metric counts must be finite, non-negative real numbers (not booleans).")
    return int(value) if isinstance(value, int) else float(value)


def _custom_result(value: Any, *, unit: str, scope: str, method: str) -> MetricResult:
    if value is None:
        return metric_result(status="unavailable", unit=unit, scope=scope, method=method)
    if not isinstance(value, Mapping):
        return metric_result(status="complete", value=_number(value), unit=unit, scope=scope, method=method)
    if set(value) != {"status", "value", "known_value", "unit", "scope", "method"}:
        raise ValueError("A structured estimate must contain exactly the MetricResult fields.")
    if value["unit"] != unit:
        raise ValueError(f"Expected metric unit {unit!r}.")
    if any(not isinstance(value[key], str) or not value[key] for key in ("scope", "method")):
        raise ValueError("Metric scope and method must be non-empty strings.")
    method = f"{method}:{value['method']}"
    if value["status"] == "complete":
        number = _number(value["value"])
        if _number(value["known_value"]) != number:
            raise ValueError("A complete metric requires known_value == value.")
        return metric_result(status="complete", value=number, unit=unit, scope=scope, method=method)
    if value["status"] == "partial":
        if value["value"] is not None:
            raise ValueError("A partial metric requires value=None and a known_value lower bound.")
        return metric_result(
            status="partial", known_value=_number(value["known_value"]), unit=unit, scope=scope, method=method
        )
    if value["status"] == "unavailable":
        if value["value"] is not None or value["known_value"] is not None:
            raise ValueError("An unavailable metric requires value=None and known_value=None.")
        return metric_result(status="unavailable", unit=unit, scope=scope, method=method)
    raise ValueError("Unknown metric status.")


def _handler_metrics(
    handler: ModuleHandler,
    call: ModuleCall,
    requested: set[str],
    diagnostics: list[Diagnostic],
    path: str,
    *,
    custom: bool,
) -> dict[str, MetricResult]:
    callback = handler.estimate
    callback_type = type(callback)
    identity = (
        f"{getattr(callback, '__module__', callback_type.__module__)}."
        f"{getattr(callback, '__qualname__', callback_type.__qualname__)}"
    )
    method = f"custom_module_handler:{identity}" if custom else "torchscan_module_formula"

    def diagnose(code: str, metric: str, message: str) -> None:
        diagnostics.append({"code": code, "severity": "warning", "metric": metric, "path": path, "message": message})

    def unavailable(name: str) -> MetricResult:
        return metric_result(
            status="unavailable",
            unit=_METRIC_UNITS[name],
            scope="subtree" if name in handler.subtree_metrics else "module_call",
            method=method,
        )

    try:
        estimates = callback(call)
    except Exception as error:  # ruff: ignore[blind-except] BLE001  # User callbacks are an analysis boundary.
        for name in sorted(requested):
            diagnose("custom_handler_error", name, f"{identity}: {type(error).__name__}: {error}")
        return {name: unavailable(name) for name in requested}
    if not isinstance(estimates, Mapping) or any(name not in _METRIC_UNITS for name in estimates):
        for name in sorted(requested):
            diagnose("custom_handler_invalid", name, "The handler must return a mapping of supported metric names.")
        return {name: unavailable(name) for name in requested}

    results: dict[str, MetricResult] = {}
    for name in sorted(requested & (estimates.keys() | handler.subtree_metrics)):
        try:
            result = _custom_result(
                estimates.get(name),
                unit=_METRIC_UNITS[name],
                scope="subtree" if name in handler.subtree_metrics else "module_call",
                method=method,
            )
        except (TypeError, ValueError, OverflowError) as error:
            diagnose("custom_metric_invalid", name, f"{identity}: {error}")
            results[name] = unavailable(name)
            continue
        results[name] = result
        if result["status"] != "complete":
            diagnose(
                f"custom_metric_{result['status']}",
                name,
                f"{identity}: estimate is {result['status']}."
                + (" known_value is a lower bound." if result["status"] == "partial" else " No count is available."),
            )
    return results
