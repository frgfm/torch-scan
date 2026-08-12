# Copyright (C) 2020-2026, François-Guillaume Fernandez.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Mapping
from typing import Literal, NotRequired, TypedDict, cast

__all__ = ["ReportDiff", "compare_reports"]

MetricStatus = Literal["complete", "partial", "unavailable"]
Number = int | float


class _MetricResult(TypedDict):
    status: MetricStatus
    value: Number | None
    known_value: Number | None
    unit: NotRequired[str]
    scope: NotRequired[str]
    method: NotRequired[str]


class _MetricDiff(TypedDict):
    status: MetricStatus
    delta: Number | None
    before: _MetricResult | None
    after: _MetricResult | None


class _LayerSnapshot(TypedDict):
    path: str
    call_index: int
    metrics: dict[str, _MetricResult]


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


def _expect_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    return cast("Mapping[str, object]", value)


def _normalize_number(value: object, location: str) -> Number | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be a number or None")
    return value


def _normalize_metric(value: object, location: str) -> _MetricResult:
    metric = _expect_mapping(value, location)
    for field in ("status", "value", "known_value"):
        if field not in metric:
            raise ValueError(f"{location} must contain '{field}'")
    status = metric.get("status")
    if status not in ("complete", "partial", "unavailable"):
        raise ValueError(f"{location}['status'] must be 'complete', 'partial', or 'unavailable'")

    normalized: dict[str, object] = {
        "status": status,
        "value": _normalize_number(metric.get("value"), f"{location}['value']"),
        "known_value": _normalize_number(metric.get("known_value"), f"{location}['known_value']"),
    }
    if status == "complete":
        if normalized["value"] is None:
            raise ValueError(f"{location}['value'] must be a number when status is 'complete'")
        if normalized["known_value"] != normalized["value"]:
            raise ValueError(f"{location}['known_value'] must equal 'value' when status is 'complete'")
    elif status == "partial":
        if normalized["value"] is not None or normalized["known_value"] is None:
            raise ValueError(f"{location} must have only a known_value when status is 'partial'")
    elif normalized["value"] is not None or normalized["known_value"] is not None:
        raise ValueError(f"{location} numeric values must be None when status is 'unavailable'")

    for field in ("unit", "scope", "method"):
        if field not in metric:
            continue
        field_value = metric[field]
        if not isinstance(field_value, str):
            raise TypeError(f"{location}['{field}'] must be a string")
        normalized[field] = field_value
    return cast("_MetricResult", normalized)


def _normalize_metrics(value: object, location: str) -> dict[str, _MetricResult]:
    metrics = _expect_mapping(value, location)
    for name in metrics:
        if not isinstance(name, str):
            raise TypeError(f"{location} keys must be strings")
    return {name: _normalize_metric(metrics[name], f"{location}[{name!r}]") for name in sorted(metrics)}


def _normalize_layers(value: object, location: str) -> dict[tuple[str, int], _LayerSnapshot]:
    if not isinstance(value, list):
        raise TypeError(f"{location} must be a list")

    layers: dict[tuple[str, int], _LayerSnapshot] = {}
    for index, value_item in enumerate(value):
        item_location = f"{location}[{index}]"
        item = _expect_mapping(value_item, item_location)
        path = item.get("path")
        call_index = item.get("call_index")
        if not isinstance(path, str):
            raise TypeError(f"{item_location}['path'] must be a string")
        if isinstance(call_index, bool) or not isinstance(call_index, int):
            raise TypeError(f"{item_location}['call_index'] must be an integer")

        key = (path, call_index)
        if key in layers:
            raise ValueError(f"{location} contains duplicate layer call {key!r}")
        layers[key] = {
            "path": path,
            "call_index": call_index,
            "metrics": _normalize_metrics(item.get("metrics"), f"{item_location}['metrics']"),
        }
    return layers


def _normalize_report(
    value: object, location: str
) -> tuple[int, dict[str, _MetricResult], dict[tuple[str, int], _LayerSnapshot]]:
    report = _expect_mapping(value, location)
    schema_version = report.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError(f"{location}['schema_version'] must be an integer")
    return (
        schema_version,
        _normalize_metrics(report.get("totals"), f"{location}['totals']"),
        _normalize_layers(report.get("layers"), f"{location}['layers']"),
    )


def _diff_metric(before: _MetricResult | None, after: _MetricResult | None) -> _MetricDiff:
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
    return {"status": status, "delta": delta, "before": before, "after": after}


def _diff_metrics(
    before: Mapping[str, _MetricResult], after: Mapping[str, _MetricResult], *, changed_only: bool = False
) -> dict[str, _MetricDiff]:
    differences: dict[str, _MetricDiff] = {}
    for name in sorted(before.keys() | after.keys()):
        before_metric = before.get(name)
        after_metric = after.get(name)
        if changed_only and before_metric == after_metric:
            continue
        differences[name] = _diff_metric(before_metric, after_metric)
    return differences


def compare_reports(before: object, after: object) -> ReportDiff:
    """Compare totals and layer-call metrics from two reports.

    Layers are matched by their full path and call index. Numeric deltas are only
    produced when both metric results are complete; incomplete states propagate.

    Args:
        before: Earlier analysis report.
        after: Later analysis report.

    Returns:
        A deterministic, JSON-serializable report difference.

    Raises:
        TypeError: If either report is missing an essential field or contains an invalid field type.
        ValueError: If schema versions differ, a status is invalid, or a layer-call identity is duplicated.
    """
    before_version, before_totals, before_layers = _normalize_report(before, "before")
    after_version, after_totals, after_layers = _normalize_report(after, "after")
    if before_version != after_version:
        raise ValueError(f"schema versions differ: {before_version} != {after_version}")
    if before_version != 1:
        raise ValueError(f"unsupported schema version: {before_version}")

    before_keys = before_layers.keys()
    after_keys = after_layers.keys()
    added = [after_layers[key] for key in sorted(after_keys - before_keys)]
    removed = [before_layers[key] for key in sorted(before_keys - after_keys)]
    changed: list[_LayerDiff] = []
    for key in sorted(before_keys & after_keys):
        metric_differences = _diff_metrics(
            before_layers[key]["metrics"], after_layers[key]["metrics"], changed_only=True
        )
        if metric_differences:
            changed.append({"path": key[0], "call_index": key[1], "metrics": metric_differences})

    return {
        "schema_version": before_version,
        "totals": _diff_metrics(before_totals, after_totals),
        "layers": {"added": added, "removed": removed, "changed": changed},
    }
