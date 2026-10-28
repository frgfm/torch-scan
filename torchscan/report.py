# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Literal, NotRequired, TypedDict

__all__ = [
    "AnalysisReport",
    "Diagnostic",
    "FlopReport",
    "IncompleteAnalysisError",
    "LayerReport",
    "MetricResult",
    "OperatorCapture",
    "TokenDependency",
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


class OperatorCapture(TypedDict):
    """Versioned, bounded native operator metadata; contains no tensor contents."""

    schema_version: int
    report_type: str
    context: dict[str, Any]
    inputs: dict[str, Any]
    operators: list[dict[str, Any]]
    dropped_events: int
    errors: int


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
    capture: NotRequired["OperatorCapture"]


class TokenRelation(TypedDict):
    """Compact structural relation between output and input token positions."""

    kind: Literal["all", "same_position", "prefix", "none"]
    first_position: NotRequired[int]
    limit: NotRequired[int]


class TokenSource(TypedDict):
    """A module-call input token axis and its dependency relation."""

    arguments: list[str]
    sequence_axis: int
    length: int
    relation: TokenRelation


class TokenDependency(TypedDict):
    """Module-local token dependencies, separate from scalar spatial metrics."""

    status: Literal["complete", "unavailable"]
    scope: str
    method: str
    assumptions: list[str]
    output: NotRequired[dict[str, int]]
    sources: NotRequired[list[TokenSource]]


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
    token_dependencies: NotRequired[TokenDependency]

    metric_ownership: NotRequired[dict[str, Literal["module_call", "subtree"]]]
    metric_owners: NotRequired[dict[str, dict[str, str | int]]]


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
