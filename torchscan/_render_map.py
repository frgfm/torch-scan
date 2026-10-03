# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Shared evidence and geometry for offline module cost maps.

The rectangles encode additive module-call contributions, never inclusive
operator counts. A node's area is a derived sum of its own recorded costs and
its descendants. Missing container formulas do not become recorded zeros.
"""

from collections import defaultdict
from operator import itemgetter
from typing import Any

from .compare import ReportDiff
from .report import AnalysisReport, LayerReport, MetricResult, metric_result

_VIEWS = {
    "module_flops": ("Module FLOPs", "FLOPs"),
    "macs": ("MACs", "MACs"),
    "dmas": ("DMAs", "DMAs"),
    "parameters": ("Attributed parameters", "elements"),
    "parameter_bytes": ("Attributed parameter bytes", "bytes"),
}
_PARAMETERS = {"parameters", "parameter_bytes"}
# Display metadata only. Synthetic group identity is kept separately so no
# caller-supplied method, unit, or scope string becomes a reserved identifier.
_MISSING = ("no_recorded_measurement", "", "module_call")
GroupKey = tuple[str, str, str]
Comparison = tuple[ReportDiff, list[str]]


def _result(layer: LayerReport, view: str) -> MetricResult | None:
    if view in _PARAMETERS:
        statistics = layer["parameters"]
        # The schema already attributes a shared tensor to its first observed
        # call. `shared=True` can coexist with positive counts of other tensors;
        # discarding the entire row would lose those uniquely attributed counts.
        value = statistics["trainable"] + statistics["frozen"] if view == "parameters" else statistics["bytes"]
        return metric_result(
            status="complete",
            value=value,
            unit=_VIEWS[view][1],
            scope="module_call_attribution",
            method="pytorch_shared_tensor_deduplication",
        )
    return layer["metrics"].get(view)


_key = itemgetter("method", "unit", "scope")


def _summary(results: list[MetricResult], group: GroupKey, *, derived: bool = False) -> MetricResult | None:
    if not results:
        return None
    recorded = [result["known_value"] for result in results if result["known_value"] is not None]
    if not recorded:
        status = "unavailable"
    elif any(result["status"] != "complete" for result in results):
        status = "partial"
    else:
        status = "complete"
    known = sum(recorded) if recorded else None
    return metric_result(
        status=status,
        value=known,
        known_value=known,
        unit=group[1],
        scope="observed_subtree" if derived else group[2],
        method="derived_recorded_subtotal" if derived else group[0],
    )


def _unavailable(group: GroupKey, *, derived: bool = False) -> MetricResult:
    return metric_result(
        status="unavailable",
        unit=group[1],
        scope="observed_subtree" if derived else group[2],
        method="derived_recorded_subtotal" if derived else group[0],
    )


def _known(result: MetricResult | None) -> float | None:
    return result["known_value"] if result is not None else None


def _paths(report: AnalysisReport | None) -> set[str]:
    paths: set[str] = {""} if report is not None else set()
    if report is not None:
        for layer in report["layers"]:
            components = layer["path"].split(".") if layer["path"] else []
            paths.update(".".join(components[:depth]) for depth in range(1, len(components) + 1))
    return paths


def _calls(report: AnalysisReport | None, view: str) -> dict[str, list[dict[str, Any]]]:
    paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if report is not None:
        for index, layer in enumerate(report["layers"]):
            result = _result(layer, view)
            if result is not None and result["known_value"] is not None and result["known_value"] < 0:
                raise ValueError(f"{view} contains a negative additive cost at {layer['path']!r}")
            paths[layer["path"]].append({
                "index": index,
                "call_index": layer["call_index"],
                "result": result,
                "shared": layer["parameters"]["shared"],
            })
    return paths


def _group_results(
    calls: list[dict[str, Any]], group: GroupKey, *, leaf: bool, synthetic: bool = False
) -> list[MetricResult]:
    if synthetic:
        return [_unavailable(group) for call in calls if leaf and call["result"] is None]
    return [call["result"] for call in calls if call["result"] is not None and _key(call["result"]) == group]


def _direct_delta(
    path: str,
    calls: list[dict[str, Any]],
    before_calls: list[dict[str, Any]],
    group: GroupKey,
    changed: dict[tuple[str, int], Any],
) -> tuple[float | None, str | None, bool]:
    """Compare this path's own calls once, independently of its descendants."""
    differences: list[float] = []
    after = {call["call_index"]: call["result"] for call in calls}
    earlier = {call["call_index"]: call["result"] for call in before_calls}
    for index in sorted(after.keys() | earlier.keys()):
        left, right = earlier.get(index), after.get(index)
        left_group = left is not None and _key(left) == group
        right_group = right is not None and _key(right) == group
        if not left_group and not right_group:
            continue
        if not left_group or not right_group:
            return None, "Added, removed, or differently measured calls have no comparable numeric delta", True
        if left["status"] != "complete" or right["status"] != "complete":
            return None, "Two comparable complete measurements are required", True
        difference = changed.get((path, index))
        if difference is None:
            # compare_reports omits unchanged call metrics. Equal complete
            # evidence has a zero delta, subject to the same context gate.
            if left != right:
                return None, "Comparable call evidence is missing from compare_reports", True
            differences.append(0)
        elif difference["status"] == "complete" and difference["delta"] is not None:
            differences.append(difference["delta"])
        else:
            return None, "Two comparable complete measurements are required", True
    if not differences:
        return None, None, False
    return sum(differences), None, True


def _deltas(
    paths: list[str],
    children: dict[str, list[str]],
    calls: dict[str, list[dict[str, Any]]],
    before_calls: dict[str, list[dict[str, Any]]],
    view: str,
    group: GroupKey,
    comparison: Comparison | None,
    changed: dict[tuple[str, int], Any],
    *,
    synthetic: bool = False,
) -> dict[str, tuple[float | None, str]]:
    if comparison is None:
        reason = "No comparable before report"
    elif view in _PARAMETERS:
        reason = "Call attribution can move between modules; use the authoritative model total"
    elif comparison[1]:
        reason = "; ".join(comparison[1])
    elif synthetic:
        reason = "No comparable recorded contributions"
    else:
        reason = None
    if reason is not None:
        return dict.fromkeys(paths, (None, reason))

    # Each call is compared once per measurement group. A reverse traversal
    # aggregates child results without rescanning all paths for every ancestor.
    # `relevant` distinguishes an empty structural branch from incomplete
    # evidence, so only actual measurement failures block a parent's delta.
    subtrees: dict[str, tuple[float | None, str | None, bool]] = {}
    for path in reversed(paths):
        direct, failure, relevant = _direct_delta(path, calls[path], before_calls[path], group, changed)
        known = direct if direct is not None else 0
        for child in children[path]:
            child_delta, child_failure, child_relevant = subtrees[child]
            relevant = relevant or child_relevant
            if failure is None and child_failure is not None:
                failure = child_failure
            if child_delta is not None:
                known += child_delta
        subtrees[path] = known if relevant and failure is None else None, failure, relevant
    return {
        path: (
            delta,
            failure
            or ("Comparable complete recorded contributions" if relevant else "No comparable recorded contributions"),
        )
        for path, (delta, failure, relevant) in subtrees.items()
    }


def _geometry(nodes: list[dict[str, Any]], children: dict[str, list[str]]) -> tuple[str, float]:
    by_path = {node["path"]: node for node in nodes}
    for node in reversed(nodes):
        direct_weight = max(node["direct_known"] or 0, _known(node["before_direct"]) or 0)
        node["direct_weight"] = direct_weight
        node["weight"] = direct_weight + sum(by_path[child]["weight"] for child in children[node["path"]])
    scale = "known"
    total = by_path[""]["weight"]
    if total <= 0:
        scale = "structure"
        for node in reversed(nodes):
            own_slot = node["direct"] is not None or node["before_direct"] is not None
            node["direct_weight"] = 1 if own_slot or not children[node["path"]] else 0
            node["weight"] = node["direct_weight"] + sum(by_path[child]["weight"] for child in children[node["path"]])
        total = by_path[""]["weight"]
    positions = {"": 0.0}
    for node in nodes:
        node["x"] = positions[node["path"]] / total
        node["width"] = node["weight"] / total
        node["direct_x"] = node["x"]
        node["direct_width"] = node["direct_weight"] / total
        cursor = positions[node["path"]] + node["direct_weight"]
        for child in children[node["path"]]:
            positions[child] = cursor
            cursor += by_path[child]["weight"]
    return scale, total


def build_maps(
    report: AnalysisReport, *, before: AnalysisReport | None = None, comparison: Comparison | None = None
) -> dict[str, Any]:
    """Build JSON-safe hierarchy maps from validated schema-v1 report evidence.

    Args:
        report: The validated current report.
        before: Optional validated earlier report.
        comparison: The renderer's compare_reports result and context-gating
            reasons. If absent, local numeric comparison deltas are withheld.

    Returns:
        View data containing isolated measurement groups, derived subtotals,
        fixed before/after geometry, and visible unknown/zero/small-cost rails.

    Raises:
        ValueError: If a recorded additive cost is negative.
    """
    after_paths, before_paths = _paths(report), _paths(before)
    paths = sorted(after_paths | before_paths)
    ids = {path: f"node-{index}" for index, path in enumerate(paths)}
    children: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if path:
            children[path.rpartition(".")[0]].append(path)
    # Lexical full-path order is not preorder when a module component contains
    # punctuation, so explicitly traverse the ancestor tree.
    ordered: list[str] = []
    pending = [""]
    while pending:
        path = pending.pop()
        ordered.append(path)
        pending.extend(reversed(children[path]))
    layers = {layer["path"]: layer for layer in (before["layers"] if before else [])}
    layers.update({layer["path"]: layer for layer in report["layers"]})
    views: dict[str, Any] = {}
    for view, (label, unit) in _VIEWS.items():
        calls, before_calls = _calls(report, view), _calls(before, view)
        changed = (
            {
                (layer["path"], layer["call_index"]): layer["metrics"].get(view)
                for layer in comparison[0]["layers"]["changed"]
            }
            if comparison is not None
            else {}
        )
        keys = {
            _key(call["result"])
            for rows in (calls, before_calls)
            for row in rows.values()
            for call in row
            if call["result"] is not None
        }
        missing_leaves = any(
            call["result"] is None
            for rows in (calls, before_calls)
            for path, row in rows.items()
            if not children[path]
            for call in row
        )
        group_keys = [(key, False) for key in keys]
        if missing_leaves or not keys:
            group_keys.append(((_MISSING[0], unit, _MISSING[2]), True))
        groups: list[dict[str, Any]] = []
        for group_index, (group, synthetic) in enumerate(sorted(group_keys)):
            deltas = _deltas(
                ordered, children, calls, before_calls, view, group, comparison, changed, synthetic=synthetic
            )
            direct_results = {
                path: _group_results(calls[path], group, leaf=not children[path], synthetic=synthetic)
                for path in ordered
            }
            before_results = {
                path: _group_results(before_calls[path], group, leaf=not children[path], synthetic=synthetic)
                for path in ordered
            }
            subtree: dict[str, list[MetricResult]] = {}
            before_subtree: dict[str, list[MetricResult]] = {}
            for path in reversed(ordered):
                subtree[path] = direct_results[path] + [result for child in children[path] for result in subtree[child]]
                before_subtree[path] = before_results[path] + [
                    result for child in children[path] for result in before_subtree[child]
                ]
            nodes: list[dict[str, Any]] = []
            for path in ordered:
                direct = _summary(direct_results[path], group)
                before_direct = _summary(before_results[path], group)
                subtotal = _summary(subtree[path], group, derived=True) or _unavailable(group, derived=True)
                before_subtotal = _summary(before_subtree[path], group, derived=True) or _unavailable(
                    group, derived=True
                )
                delta, reason = deltas[path]
                own_results = direct_results[path] + before_results[path]
                rail = ["unknown"] if any(result["status"] != "complete" for result in own_results) else []
                if (
                    not children[path]
                    and own_results
                    and all(result["status"] == "complete" and result["value"] == 0 for result in own_results)
                ):
                    rail.append("zero")
                nodes.append({
                    "id": ids[path],
                    "path": path,
                    "label": path.rpartition(".")[2] if path else "(root)",
                    "type": layers[path]["type"] if path in layers else "Structural ancestor",
                    "parent": ids[path.rpartition(".")[0]] if path else None,
                    "children": [ids[child] for child in children[path]],
                    "depth": path.count(".") + 1 if path else 0,
                    "calls": calls[path],
                    "before_calls": before_calls[path],
                    "direct": direct,
                    "direct_kind": "structural"
                    if direct is None
                    else "attributed"
                    if view in _PARAMETERS
                    else "recorded",
                    "direct_known": _known(direct),
                    "known": _known(subtotal),
                    "subtotal": subtotal,
                    "status": subtotal["status"],
                    "before_direct": before_direct,
                    "before_subtotal": before_subtotal,
                    "before_known": _known(before_subtotal),
                    "delta": delta,
                    "delta_reason": reason,
                    "rail": rail,
                    "change": "matched"
                    if path in after_paths and path in before_paths
                    else "removed"
                    if path in before_paths
                    else "added"
                    if before is not None
                    else "matched",
                })
            scale, layout_total = _geometry(nodes, children)
            if scale == "known":
                for node in nodes:
                    if node["direct_weight"] > 0 and node["width"] < 0.025:
                        node["rail"].append("small")
            groups.append({
                "id": f"map-{view}-{group_index}",
                "kind": "unrecorded" if synthetic else "recorded",
                "label": f"{group[0]} · {group[1]} · {group[2]}",
                "method": group[0],
                "unit": group[1],
                "scope": group[2],
                "comparison": before is not None,
                "scale": scale,
                "known_total": nodes[0]["known"],
                "layout_total": layout_total,
                "max_depth": max(node["depth"] for node in nodes),
                "nodes": nodes,
                "rails": [node["id"] for node in nodes if node["rail"]],
            })
        views[view] = {
            "view": view,
            "label": label,
            "groups": groups,
            "total": report["totals"].get(view),
            "before_total": before["totals"].get(view) if before is not None else None,
            "comparison_delta": comparison[0]["totals"].get(view) if comparison is not None else None,
        }
    return views
