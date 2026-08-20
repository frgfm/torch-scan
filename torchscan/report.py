# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from .flops import FlopReport

__all__ = [
    "AnalysisReport",
    "Diagnostic",
    "IncompleteAnalysisError",
    "LayerReport",
    "MetricResult",
    "metric_result",
]

MetricStatus = Literal["complete", "partial", "unavailable"]


class MetricResult(TypedDict):
    """JSON-serializable result for one measurement."""

    status: MetricStatus
    value: float | None
    known_value: float | None
    unit: str
    scope: str
    method: str


class Diagnostic(TypedDict):
    """Machine-readable explanation of an incomplete measurement."""

    code: str
    severity: Literal["warning", "error"]
    metric: str
    message: str
    path: NotRequired[str]
    operator: NotRequired[str]


class LayerReport(TypedDict):
    """Information collected for one module invocation."""

    path: str
    call_index: int
    name: str
    depth: int
    type: str
    input: dict[str, Any]
    output: dict[str, Any]
    parameters: dict[str, int | bool]
    buffers: dict[str, int | bool]
    metrics: dict[str, MetricResult]


class AnalysisReport(TypedDict):
    """Versioned, JSON-serializable result of a module analysis."""

    schema_version: int
    context: dict[str, Any]
    inputs: dict[str, Any]
    layers: list[LayerReport]
    operator_flops: "FlopReport"
    totals: dict[str, MetricResult]
    diagnostics: list[Diagnostic]


def metric_result(
    *,
    status: MetricStatus,
    unit: str,
    scope: str,
    method: str,
    value: float | None = None,
    known_value: float | None = None,
) -> MetricResult:
    """Build a metric result while enforcing its completeness invariant."""
    if status == "complete":
        if value is None:
            raise ValueError("A complete metric requires a value.")
        known_value = value
    elif status == "partial":
        if known_value is None:
            raise ValueError("A partial metric requires a known_value lower bound.")
        value = None
    elif status == "unavailable":
        value = None
        known_value = None
    else:
        raise ValueError("status must be 'complete', 'partial', or 'unavailable'.")
    return {
        "status": status,
        "value": value,
        "known_value": known_value,
        "unit": unit,
        "scope": scope,
        "method": method,
    }


class IncompleteAnalysisError(RuntimeError):
    """Raised when strict analysis encounters incomplete metrics."""

    def __init__(self, report: AnalysisReport) -> None:
        self.report = report
        incomplete = {name for name, result in report["totals"].items() if result["status"] != "complete"}
        incomplete.update(diagnostic["metric"] for diagnostic in report["diagnostics"])
        metrics = ", ".join(sorted(incomplete))
        super().__init__(f"Analysis is incomplete for: {metrics or 'unknown metrics'}.")
