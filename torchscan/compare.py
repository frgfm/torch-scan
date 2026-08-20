# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Mapping
from copy import deepcopy
from typing import TypedDict, cast

from .report import AnalysisReport, LayerReport, MetricResult, MetricStatus

__all__ = ["ReportDiff", "compare_reports"]

Number = int | float


class _MetricDiff(TypedDict):
    status: MetricStatus
    delta: Number | None
    before: MetricResult | None
    after: MetricResult | None


class _LayerSnapshot(TypedDict):
    path: str
    call_index: int
    metrics: dict[str, MetricResult]


class _LayerDiff(TypedDict):
    path: str
    call_index: int
    metrics: dict[str, _MetricDiff]


class _LayerChanges(TypedDict):
    added: list[_LayerSnapshot]
    removed: list[_LayerSnapshot]
    changed: list[_LayerDiff]


class ReportDiff(TypedDict):
    """JSON-serializable differences between two compatible analysis reports."""

    schema_version: int
    totals: dict[str, _MetricDiff]
    layers: _LayerChanges


def _diff_metric(before: MetricResult | None, after: MetricResult | None) -> _MetricDiff:
    if before is None or after is None:
        status: MetricStatus = "unavailable"
    elif "unavailable" in (before["status"], after["status"]):
        status = "unavailable"
    elif "partial" in (before["status"], after["status"]):
        status = "partial"
    else:
        status = "complete"

    delta: Number | None = None
    if status == "complete" and before is not None and after is not None:
        delta = cast("Number", after["value"]) - cast("Number", before["value"])
    return {"status": status, "delta": delta, "before": deepcopy(before), "after": deepcopy(after)}


def _diff_metrics(
    before: Mapping[str, MetricResult], after: Mapping[str, MetricResult], *, changed_only: bool = False
) -> dict[str, _MetricDiff]:
    differences: dict[str, _MetricDiff] = {}
    for name in sorted(before.keys() | after.keys()):
        before_metric = before.get(name)
        after_metric = after.get(name)
        if changed_only and before_metric == after_metric:
            continue
        if before_metric is not None and after_metric is not None:
            for field in ("unit", "scope", "method"):
                if before_metric.get(field) != after_metric.get(field):
                    raise ValueError(f"metric {name!r} has incompatible {field}")
        differences[name] = _diff_metric(before_metric, after_metric)
    return differences


def _layer_calls(report: AnalysisReport, name: str) -> dict[tuple[str, int], LayerReport]:
    calls: dict[tuple[str, int], LayerReport] = {}
    for layer in report["layers"]:
        key = (layer["path"], layer["call_index"])
        if key in calls:
            raise ValueError(f"{name} contains duplicate layer call {key!r}")
        calls[key] = layer
    return calls


def _snapshot(layer: LayerReport) -> _LayerSnapshot:
    return {
        "path": layer["path"],
        "call_index": layer["call_index"],
        "metrics": deepcopy(layer["metrics"]),
    }


def compare_reports(before: AnalysisReport, after: AnalysisReport) -> ReportDiff:
    """Compare totals and layer-call metrics from two reports.

    Layers are matched by their full path and call index. Numeric deltas are only
    produced when both metric results are complete; incomplete states propagate.

    Args:
        before: Earlier analysis report.
        after: Later analysis report.

    Returns:
        A deterministic, JSON-serializable report difference.

    Raises:
        ValueError: If schema versions or metric methods differ, or a layer-call identity is duplicated.
    """
    before_version = before["schema_version"]
    after_version = after["schema_version"]
    if before_version != after_version:
        raise ValueError(f"schema versions differ: {before_version} != {after_version}")
    if before_version != 1:
        raise ValueError(f"unsupported schema version: {before_version}")

    before_layers = _layer_calls(before, "before")
    after_layers = _layer_calls(after, "after")
    before_keys = before_layers.keys()
    after_keys = after_layers.keys()
    added = [_snapshot(after_layers[key]) for key in sorted(after_keys - before_keys)]
    removed = [_snapshot(before_layers[key]) for key in sorted(before_keys - after_keys)]
    changed: list[_LayerDiff] = []
    for key in sorted(before_keys & after_keys):
        metric_differences = _diff_metrics(
            before_layers[key]["metrics"], after_layers[key]["metrics"], changed_only=True
        )
        if metric_differences:
            changed.append({"path": key[0], "call_index": key[1], "metrics": metric_differences})

    return {
        "schema_version": before_version,
        "totals": _diff_metrics(before["totals"], after["totals"]),
        "layers": {"added": added, "removed": removed, "changed": changed},
    }
