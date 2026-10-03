# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Offline explanatory views of schema-v1 analysis reports, without remeasurement."""

import base64
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from html import escape
from operator import itemgetter
from typing import Any, Literal, cast

from ._render_assets import SCRIPT, STYLE
from ._render_map import _result as _attribution
from ._render_map import build_maps
from ._render_visual import compact_value, default_selection, map_svg, shape_text, visual_svg
from .compare import ReportDiff, compare_reports
from .report import AnalysisReport, Diagnostic, LayerReport, MetricResult, metric_result

__all__ = ["render_report"]

View = Literal["module_flops", "macs", "dmas", "parameters", "parameter_bytes"]
Comparison = tuple[ReportDiff, list[str]]
_VIEWS = {
    "module_flops": "Module FLOPs",
    "macs": "MACs",
    "dmas": "DMAs",
    "parameters": "Attributed parameters",
    "parameter_bytes": "Attributed parameter bytes",
}
_STORAGE = {
    "parameters",
    "trainable_parameters",
    "frozen_parameters",
    "parameter_bytes",
    "buffer_elements",
    "buffer_bytes",
}
_CONTEXT = ("torch_version", "torchscan_version", "python_version", "execution_mode", "devices", "dtypes")
_BOUNDARIES = (
    "This is an observed module hierarchy, not a computational graph. Only executed calls appear. "
    "Module-call estimates are additive contributions; absent container estimates are unavailable, not zero. "
    "Operator module counts are inclusive and must not be summed or joined to module-call rows. "
    "Model totals are authoritative. Parameter attribution counts shared tensors once in execution order; "
    "uncalled parameters can appear only in model totals. "
    "FLOPs do not establish latency. Static parameter and buffer bytes are not measured peak memory."
)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    # Keep both outputs valid, including XML 1.0. Reject invalid Unicode rather
    # than silently changing a stored report's identities or evidence.
    if any(
        (ord(c) < 32 and c not in "\t\r\n") or 0xD800 <= ord(c) <= 0xDFFF or ord(c) in (0xFFFE, 0xFFFF) for c in value
    ):
        raise ValueError(f"{where} contains invalid text")
    return value


def _number(value: Any, where: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    if integer and (not isinstance(value, int) or value < 0):
        raise ValueError(f"{where} must be a nonnegative integer")
    return value


def _json_tree(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _string(key, f"{where} key")
            _json_tree(item, f"{where}.{key}")
    elif isinstance(value, list):
        for item in value:
            _json_tree(item, where)
    elif isinstance(value, str):
        _string(value, where)
    elif value is not None and not isinstance(value, bool):
        _number(value, where)


def _metric(value: Any, where: str) -> None:
    value = _object(value, where)
    for field in ("unit", "scope", "method"):
        _string(value.get(field), f"{where}.{field}")
    status = value.get("status")
    if status == "complete":
        _number(value.get("value"), f"{where}.value")
        _number(value.get("known_value"), f"{where}.known_value")
        valid = value["value"] == value["known_value"]
    elif status == "partial":
        _number(value.get("known_value"), f"{where}.known_value")
        valid = "value" in value and value["value"] is None
    elif status == "unavailable":
        valid = all(field in value and value[field] is None for field in ("value", "known_value"))
    else:
        raise ValueError(f"{where} has invalid status")
    if not valid:
        raise ValueError(f"{where} violates the {status} measurement invariant")


def _metrics(value: Any, where: str) -> None:
    for name, result in _object(value, where).items():
        _metric(result, f"{where}.{name}")


def _diagnostics(value: Any, where: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    for diagnostic in value:
        diagnostic = _object(diagnostic, where)
        for field in ("code", "metric", "message"):
            _string(diagnostic.get(field), f"{where}.{field}")
        if diagnostic.get("severity") not in ("warning", "error"):
            raise ValueError(f"{where} has invalid severity")
        for field in ("path", "operator"):
            if field in diagnostic:
                _string(diagnostic[field], f"{where}.{field}")


def _validate(report: Any) -> None:
    report = _object(report, "report")
    try:
        _json_tree(report, "report")
    except RecursionError:
        raise ValueError("report must be an acyclic JSON object of supported depth") from None
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported analysis schema_version; expected 1")
    for field in ("context", "inputs"):
        _object(report.get(field), field)
    for field in ("devices", "dtypes"):
        if field in report["context"]:
            values = report["context"][field]
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"context.{field} must be a list of strings")
    _metrics(report.get("totals"), "totals")
    _diagnostics(report.get("diagnostics"), "diagnostics")
    if not isinstance(report.get("layers"), list):
        raise ValueError("layers must be a list")
    identities = set()
    for layer in report["layers"]:
        layer = _object(layer, "layer")
        for field in ("path", "name", "type"):
            _string(layer.get(field), f"layer.{field}")
        if layer["path"] and any(not part for part in layer["path"].split(".")):
            raise ValueError("layer.path contains an empty module component")
        for field in ("depth", "call_index"):
            _number(layer.get(field), f"layer.{field}", integer=True)
        key = (layer["path"], layer["call_index"])
        if key in identities:
            raise ValueError(f"duplicate layer call {key!r}")
        identities.add(key)
        for field in ("input", "output"):
            _object(layer.get(field), f"layer.{field}")
        for field, counts in (("parameters", ("trainable", "frozen", "bytes")), ("buffers", ("elements", "bytes"))):
            statistics = _object(layer.get(field), f"layer.{field}")
            for count in counts:
                _number(statistics.get(count), f"layer.{field}.{count}", integer=True)
            if not isinstance(statistics.get("shared"), bool):
                raise ValueError(f"layer.{field}.shared must be a boolean")
        _metrics(layer.get("metrics"), "layer.metrics")
    operators = _object(report.get("operator_flops"), "operator_flops")
    if type(operators.get("schema_version")) is not int or operators["schema_version"] != 1:
        raise ValueError("unsupported operator schema_version; expected 1")
    _object(operators.get("context"), "operator_flops.context")
    _metric(operators.get("total"), "operator_flops.total")
    if "operator_flops" in report["totals"] and report["totals"]["operator_flops"] != operators["total"]:
        raise ValueError("totals.operator_flops must match operator_flops.total")
    for field in ("by_module", "by_operator"):
        for count in _object(operators.get(field), f"operator_flops.{field}").values():
            _number(count, f"operator_flops.{field}", integer=True)
    for ignored in _object(operators.get("ignored_operators"), "ignored_operators").values():
        ignored = _object(ignored, "ignored operator")
        _number(ignored.get("calls"), "ignored operator.calls", integer=True)
        _string(ignored.get("reason"), "ignored operator.reason")
    _diagnostics(operators.get("diagnostics"), "operator_flops.diagnostics")


def _label(layer: LayerReport) -> str:
    return f"{layer['path'] or '(root)'} · call #{layer['call_index']} · {layer['type']}"


def _num(value: float) -> str:
    return f"{value:,}" if isinstance(value, int) else f"{value:,.6g}"


def _value(result: MetricResult, field: Literal["value", "known_value"] = "value") -> float:
    # Status invariants are checked at the API boundary before rendering.
    return cast("float", result[field])


def _text(result: MetricResult | None) -> str:
    if result is None or result["status"] == "unavailable":
        return "unavailable · unknown"
    if result["status"] == "partial":
        return f"partial · at least {_num(_value(result, 'known_value'))} {result['unit']} (lower bound; full value unknown)"
    return f"complete · {_num(_value(result))} {result['unit']}"


def _result(result: MetricResult | None) -> str:
    status = result["status"] if result else "unavailable"
    text = _text(result).split(" · ", 1)[1]
    return f'<span class="badge {status}">{status}</span> {escape(text)}'


def _metadata(value: Any) -> str:
    return f"<pre>{escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"


def _ranked(report: AnalysisReport, view: str) -> list[tuple[int, LayerReport, MetricResult]]:
    rows = [(i, layer, _attribution(layer, view)) for i, layer in enumerate(report["layers"])]
    # Partial lower bounds cannot be ordered against complete costs. No inference
    # is made from a container's absence of an additive formula.
    return sorted(
        [(i, layer, result) for i, layer, result in rows if result is not None and result["status"] == "complete"],
        key=lambda row: (-_value(row[2]), row[1]["path"], row[1]["call_index"]),
    )


def _rank_groups(report: AnalysisReport, view: str) -> list[list[tuple[int, LayerReport, MetricResult]]]:
    groups: dict[tuple[str, str, str], list[tuple[int, LayerReport, MetricResult]]] = defaultdict(list)
    for row in _ranked(report, view):
        result = row[2]
        groups[result["method"], result["unit"], result["scope"]].append(row)
    return [groups[key] for key in sorted(groups)]


def _context_reasons(before: AnalysisReport, after: AnalysisReport) -> list[str]:
    reasons = []
    if before["inputs"] != after["inputs"]:
        reasons.append("input metadata differs")
    for key in _CONTEXT:
        if key not in before["context"] or key not in after["context"]:
            reasons.append(f"{key} context is missing")
        elif before["context"][key] != after["context"][key]:
            reasons.append(f"{key} differs")
    if before["context"].get("analysis_mode", "full") != after["context"].get("analysis_mode", "full"):
        reasons.append("analysis_mode differs")
    return reasons


def _comparison(before: AnalysisReport, after: AnalysisReport) -> tuple[ReportDiff, list[str]]:
    differences = compare_reports(before, after)
    reasons = _context_reasons(before, after)
    # compare_reports owns matching and method/unit/scope compatibility. This
    # presentation adds conservative execution-context gating without changing it.
    if reasons:
        for name, difference in differences["totals"].items():
            if name not in _STORAGE:
                difference["delta"] = None
                difference["status"] = "unavailable"
        for layer in differences["layers"]["changed"]:
            for difference in layer["metrics"].values():
                difference["delta"] = None
                difference["status"] = "unavailable"
    return differences, reasons


def _delta(difference: Mapping[str, Any], reasons: list[str]) -> str:
    if difference["delta"] is not None:
        return f"complete · {difference['delta']:+,} {difference['after']['unit']}"
    if reasons:
        return "unavailable · not comparable: " + "; ".join(reasons)
    return f"{difference['status']} · delta unknown (requires two complete measurements)"


def _tree(group: dict[str, Any], report: AnalysisReport, before: AnalysisReport | None) -> str:
    # The same hierarchy supplies map selection targets and the native fallback.
    nodes = {node["id"]: node for node in group["nodes"]}

    def branch(identifier: str) -> str:
        node = nodes[identifier]
        annotation = f"{len(node['calls'])} observed call(s)" if node["calls"] else "no current call record"
        links = "".join(
            f'<li><a class="call-link" href="#{prefix}-{call["index"]}">{escape(_label(source["layers"][call["index"]]))} · {label}</a></li>'
            for source, field, prefix, label in (
                (report, "calls", "call", "after"),
                (before, "before_calls", "before-call", "before"),
            )
            if source is not None
            for call in node[field]
        )
        return (
            f'<details id="module-{identifier}" tabindex="-1" open><summary>{escape(node["path"] or "(root)")} '
            f"<small>{escape(node['type'])} · {escape(annotation)}</small></summary>"
            f"<ul>{links}</ul>{''.join(branch(child) for child in node['children'])}</details>"
        )

    return branch(group["nodes"][0]["id"])


def _metric_table(metrics: Mapping[str, MetricResult], *, prefix: str = "") -> str:
    rows = []
    for i, (name, result) in enumerate(metrics.items()):
        attributes = f' id="{prefix}{i}" tabindex="-1"' if prefix else ""
        rows.append(
            f'<tr{attributes}><th scope="row">{escape(name)}</th><td>{_result(result)}</td>'
            f"<td>{escape(result['method'])}<small>scope: {escape(result['scope'])}; unit: {escape(result['unit'])}</small></td></tr>"
        )
    return (
        '<div class="table-scroll"><table><caption>Values retain their original measurement method and scope.</caption>'
        f"<thead><tr><th>Metric</th><th>Measurement</th><th>Method and scope</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _ranking(report: AnalysisReport, view: str) -> str:
    groups = _rank_groups(report, view)
    # Only ratios within one unit/method/scope are meaningful. A bar is relative
    # to the largest complete call, never a coverage or model-cost percentage.
    body = []
    for group in groups:
        maximum = _value(group[0][2])
        for rank, (i, layer, result) in enumerate(group[:10], 1):
            ratio = _value(result) / maximum if maximum > 0 else 0
            shared = (
                " · shared tensors accounted elsewhere" if layer["parameters"]["shared"] and view in _STORAGE else ""
            )
            body.append(
                f'<tr><td>{rank}</td><th scope="row"><a href="#call-{i}">{escape(_label(layer))}</a>{escape(shared)}</th>'
                f'<td>{_result(result)}<progress value="{ratio}" max="1" aria-label="Relative to largest complete call using the same method"></progress></td>'
                f"<td>{escape(result['method'])}<small>{escape(result['scope'])}; {escape(result['unit'])}</small></td></tr>"
            )
    unknown = []
    for i, layer in enumerate(report["layers"]):
        result = _attribution(layer, view)
        if result is None or result["status"] != "complete":
            unknown.append(f'<li><a href="#call-{i}">{escape(_label(layer))}</a>: {_result(result)}</li>')
    return (
        f'<div class="ranking-table"><h3>{_VIEWS[view]}</h3>'
        f"<p>Authoritative model total: {_result(report['totals'].get(view))}</p>"
        '<div class="table-scroll"><table><caption>Top 10 complete additive call contributions per method, unit and scope. Ranks restart for each group. Bars are relative to the largest '
        "complete call with the same method, unit and scope; unknown work has no bar. Parameter rows show first attribution, "
        f"not each module's full ownership.</caption><thead><tr><th>Rank</th><th>Call</th><th>Contribution</th><th>Method</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
        f"{'<p>No complete call measurements are available.</p>' if not body else ''}"
        f"<h4>Unranked incomplete or absent measurements ({len(unknown)})</h4><ul>{''.join(unknown)}</ul></div>"
    )


def _all_diagnostics(report: AnalysisReport) -> list[Diagnostic]:
    diagnostics = list(report["diagnostics"])
    for diagnostic in report["operator_flops"]["diagnostics"]:
        if diagnostic not in diagnostics:
            diagnostics.append(diagnostic)
    return diagnostics


def _suggestions(report: AnalysisReport) -> list[tuple[str, str, str]]:
    suggestions = []
    for view in ("module_flops", "parameters"):
        rows = _ranked(report, view)
        if not rows or _value(rows[0][2]) <= 0:
            continue
        i, layer, result = rows[0]
        fact = (
            f"{_label(layer)} has the largest complete {_VIEWS[view].lower()} contribution using {result['method']} "
            f"({result['scope']}, {result['unit']}): {_text(result)}. Incomplete calls may cost more."
        )
        experiment = (
            "Test smaller channel or feature dimensions where valid, or alternative implementations for this call. "
            "Check output quality and benchmark latency on the target device with warmup and repeated trials."
            if view == "module_flops"
            else "Test smaller dimensions or weight sharing if model quality permits. Recount parameters, then measure "
            "peak memory for the intended workload; attributed storage bytes alone cannot predict that peak."
        )
        suggestions.append((fact, experiment, f"call-{i}"))
    diagnostics = [
        (i, diagnostic)
        for i, diagnostic in enumerate(_all_diagnostics(report))
        if diagnostic["metric"] in ("module_flops", "macs", "dmas", "flops")
    ]
    if diagnostics:
        suggestions.append((
            f"{len(diagnostics)} compute diagnostic(s) explain counting limitations; missing work can change the ranking.",
            (
                "Resolve the linked diagnostics before accepting a compute budget. For uncounted operators, try a verified "
                "formula via measure_flops(custom_mapping=...) and rerun the workload. Treat this as a measurement experiment."
            ),
            f"diagnostic-{diagnostics[0][0]}",
        ))
    return suggestions


def _comparison_html(before: AnalysisReport, after: AnalysisReport, differences: ReportDiff, reasons: list[str]) -> str:
    rows = "".join(
        f'<tr><th scope="row">{escape(name)}</th><td>{_result(difference["before"])}</td>'
        f"<td>{_result(difference['after'])}</td><td>{escape(_delta(difference, reasons if name not in _STORAGE else []))}</td></tr>"
        for name, difference in differences["totals"].items()
    )
    changes = []
    layer_changes = cast("Mapping[str, list[dict[str, Any]]]", differences["layers"])
    for kind in ("added", "removed", "changed"):
        entries = []
        for layer in layer_changes[kind]:
            label = f"{layer['path'] or '(root)'} · call #{layer['call_index']}"
            if kind == "changed":
                metrics = "".join(
                    f"<li>{escape(name)}: before {_result(difference['before'])}; after {_result(difference['after'])}; "
                    f"{escape(_delta(difference, reasons))}</li>"
                    for name, difference in layer["metrics"].items()
                )
                entries.append(f"<li>{escape(label)}<ul>{metrics}</ul></li>")
            else:
                entries.append(f"<li>{escape(label)}{_metric_table(layer['metrics'])}</li>")
        changes.append(f"<h3>{kind.capitalize()} calls ({len(entries)})</h3><ul>{''.join(entries)}</ul>")
    return (
        '<section id="comparison" tabindex="-1"><h2>Before / after</h2><p>Deltas are after minus before. '
        "Call identity is module path plus call index. Numeric deltas require complete results and matching methods, units and scopes. "
        "Execution metrics also require matching input metadata and measurement context. Storage totals can be compared independently of inputs.</p>"
        f"<p>{escape('Execution deltas withheld: ' + '; '.join(reasons) if reasons else 'Execution input and measurement context match.')}</p>"
        f'<div class="table-scroll"><table><caption>Total changes</caption><thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Delta</th></tr></thead><tbody>{rows}</tbody></table></div>'
        f"{''.join(changes)}<details><summary>Before input and measurement context</summary>{_metadata(before['inputs'])}{_metadata(before['context'])}"
        f"{_metric_table(before['totals'])}</details><details><summary>After input and measurement context</summary>{_metadata(after['inputs'])}"
        f"{_metadata(after['context'])}{_metric_table(after['totals'])}</details></section>"
    )


def _explorer_data(
    report: AnalysisReport, before: AnalysisReport | None, comparison: Comparison | None
) -> dict[str, Any]:
    maps = build_maps(report, before=before, comparison=comparison)
    for data in maps.values():
        for group in data["groups"]:
            for node in group["nodes"]:
                node["display"] = _text(node["subtotal"])
                node["before_display"] = _text(node["before_subtotal"])
                node["delta_display"] = (
                    f"complete · {node['delta']:+,} {group['unit']} (recorded contribution delta)"
                    if node["delta"] is not None
                    else "unavailable · delta withheld: " + node["delta_reason"]
                )
                for field, source in (("calls", report), ("before_calls", before)):
                    node[field] = [dict(call) for call in node[field]]
                    if source is None:
                        continue
                    for call in node[field]:
                        layer = source["layers"][call["index"]]
                        statistics = layer["parameters"]
                        count = statistics["trainable"] + statistics["frozen"]
                        call["parameter_text"] = (
                            "Shared tensors · no new parameter attribution"
                            if statistics["shared"] and count == 0
                            else f"{count:,} newly attributed parameters"
                            + (" · some shared tensors were attributed earlier" if statistics["shared"] else "")
                        )
                        call["input_text"] = shape_text(layer["input"])
                        call["output_text"] = shape_text(layer["output"])
                        result = call["result"]
                        call["in_group"] = (
                            group["kind"] == "recorded"
                            and result is not None
                            and all(result[key] == group[key] for key in ("method", "unit", "scope"))
                        )
                        call["display"] = (
                            _text(result) if call["in_group"] else "unavailable · not recorded in this method group"
                        )
                        call["display_status"] = result["status"] if call["in_group"] else "unavailable"
    return maps


def _inspector_html(node: dict[str, Any], group: dict[str, Any], before: AnalysisReport | None) -> str:
    calls = node["calls"] or node["before_calls"]
    prefix = "call" if node["calls"] else "before-call"
    shapes = ""
    if calls:
        shapes = (
            '<h4>Input / output shapes</h4><div class="tensor-shapes">'
            + "".join(
                '<div class="tensor-box"><span class="tensor-icon" aria-hidden="true"></span>'
                f"<strong>{label}</strong><p>{escape(calls[0][field])}</p></div>"
                for label, field in (("Input", "input_text"), ("Output", "output_text"))
            )
            + "</div>"
        )
    maximum = max(
        (_value(call["result"]) for call in calls if call["in_group"] and call["result"]["status"] == "complete"),
        default=0,
    )
    rows = []
    for call in calls:
        result = call["result"] if call["in_group"] else None
        bar = ""
        if result is not None and result["status"] == "complete" and maximum > 0:
            ratio = _value(result) / maximum * 100
            bar = f'<div class="call-track" aria-hidden="true"><span style="width:{ratio}%"></span></div>'
        rows.append(
            f'<div class="call-card"><a href="#{prefix}-{call["index"]}">call #{call["call_index"]}</a>'
            f'<p>{_result(result)}</p>{bar}<p class="muted">{escape(call["parameter_text"])}</p>'
            f'<p class="muted">{escape(call["input_text"])} → {escape(call["output_text"])}</p></div>'
        )
    comparison_html = (
        '<div class="comparison-note">'
        f"<p>Before: {escape(node['before_display'])}</p><p>After: {escape(node['display'])}</p>"
        f"<p>{escape(node['delta_display'])}</p></div>"
        if before is not None
        else ""
    )
    structural_note = (
        '<p class="under-map">No direct estimate was recorded for this container.</p>'
        if node["direct_kind"] == "structural"
        else ""
    )
    return (
        '<p class="eyebrow">SELECTED MODULE</p>'
        f'<h3 tabindex="-1">{escape(node["path"] or "(root)")}</h3>'
        f'<p class="muted">{escape(node["type"])} · {len(node["calls"])} observed call(s)</p>'
        f'<p class="metric-value">{_result(node["subtotal"])}</p>'
        '<p class="under-map">Recorded contribution subtotal; derived from this path and its descendants.</p>'
        f"{structural_note}"
        f"{comparison_html}{shapes}<h4>Call evidence · compute and first attribution</h4>{''.join(rows)}"
        f"{'<p>Structural ancestor; no call record.</p>' if not calls else ''}"
        f'<hr><p class="under-map">Method: {escape(group["method"])}</p>'
        f'<p class="under-map">Scope: {escape(group["scope"])} · unit: {escape(group["unit"])}</p>'
    )


def _explorer_panel(report: AnalysisReport, data: dict[str, Any], before: AnalysisReport | None) -> str:
    view = data["view"]
    initial = cast("dict[str, Any]", default_selection(data["groups"][0]))
    diagrams = []
    rails: list[tuple[int, str]] = []
    for group in data["groups"]:
        selected = cast("dict[str, Any]", default_selection(group))
        diagrams.append(
            f'<h3>{escape(group["label"])}</h3><div class="map-scroll">'
            f"{map_svg(group, selected=selected['id'], prefix=group['id'], interactive=True)}</div>"
        )
        for node in group["nodes"]:
            if not node["rail"]:
                continue
            kind = node["rail"][0]
            explanation = {
                "unknown": "Full cost unknown; this card is deliberately unscaled.",
                "zero": "Complete recorded zero; this card is unscaled.",
                "small": "Small recorded contribution; retained here for selection.",
            }[kind]
            result = node["direct"] if kind == "unknown" else node["subtotal"]
            rails.append((
                {"unknown": 0, "small": 1, "zero": 2}[kind],
                (
                    f'<a class="rail-card {kind}" href="#module-{node["id"]}" data-node-id="{node["id"]}" data-group-id="{group["id"]}">'
                    f"<strong>{escape(node['path'] or '(root)')} · {escape(node['type'])}</strong>"
                    f"{_result(result)}<small>{escape(explanation)}</small>"
                    f"{'<small>Before: ' + escape(node['before_display']) + '</small>' if before is not None else ''}</a>"
                ),
            ))
    explanation = (
        "Widths reserve the greater recorded direct contribution across before/after; paired bars use that common scale."
        if before is not None
        else "Width partitions known additive module-call contributions. Parent widths are derived subtotals."
    )
    attribution = ""
    if view in ("parameters", "parameter_bytes"):
        subtotal = data["groups"][0]["known_total"]
        total = data["total"]
        if total is not None and total["status"] == "complete" and subtotal is not None and _value(total) > subtotal:
            attribution = (
                '<p class="note">Model totals include registered parameters not attributed to an executed call.</p>'
            )
    return (
        f'<div class="metric-panel" data-view="{view}" data-initial-node="{initial["id"]}">'
        '<div class="explorer-grid"><div class="map-column"><section class="panel"><h2>Module cost map</h2>'
        f'<p class="under-map">{escape(explanation)} This is a module hierarchy, not a computational graph.</p>'
        '<div class="map-toolbar"><button type="button" data-action="collapse" disabled>Collapse branch</button>'
        '<button type="button" data-action="zoom" disabled>Zoom into branch</button>'
        '<button type="button" data-action="up" disabled>Parent module</button>'
        '<button type="button" data-action="reset" disabled>Reset map</button></div>'
        f'{"".join(diagrams)}<p class="under-map">Select a rectangle to inspect its calls. Arrow keys navigate module ancestry; Enter or Space selects. '
        f"Neutral maps have no numeric cost scale. Unknown work is never a zero-sized estimate.</p>{attribution}</section>"
        '<section class="panel"><h3>Always visible · unknown, zero and small costs</h3>'
        f'<div class="rail-cards">{"".join(card for _, card in sorted(rails, key=itemgetter(0)))}</div>{"<p>No incomplete or tiny contributions in this view.</p>" if not rails else ""}</section>'
        f'<details class="appendix"><summary>Ranked call evidence and exact values</summary>{_ranking(report, view)}</details></div>'
        f'<aside class="panel inspector" data-inspector aria-label="Selected module evidence">{_inspector_html(initial, data["groups"][0], before)}</aside>'
        "</div></div>"
    )


def _html(
    report: AnalysisReport, title: str, view: str, before: AnalysisReport | None, comparison: Comparison | None
) -> str:
    maps = _explorer_data(report, before, comparison)
    call_details = []
    diagnostics = _all_diagnostics(report)
    for i, layer in enumerate(report["layers"]):
        metrics = dict(layer["metrics"])
        for name, unit in (("module_flops", "FLOPs"), ("macs", "MACs"), ("dmas", "DMAs")):
            if name not in metrics:
                metrics[name] = metric_result(
                    status="unavailable",
                    unit=unit,
                    scope="module_call",
                    method="not_requested" if report["context"].get("analysis_mode") == "structure" else "not_recorded",
                )
        diagnostic_links = "".join(
            f'<li><a href="#diagnostic-{j}">{escape(item["code"])}</a>: {escape(item["message"])}</li>'
            for j, item in enumerate(diagnostics)
            if item.get("path") == layer["path"]
        )
        call_details.append(
            f'<details id="call-{i}" tabindex="-1"><summary>{escape(_label(layer))}</summary>'
            f"<h3>Input shapes and metadata</h3>{_metadata(layer['input'])}<h3>Output shapes and metadata</h3>{_metadata(layer['output'])}"
            f"<h3>Parameter and buffer attribution</h3>{_metadata({'parameters': layer['parameters'], 'buffers': layer['buffers']})}"
            "<p>Shared tensors are assigned once; a shared flag means some tensors were already attributed to an earlier call.</p>"
            f"<h3>Measurement methods</h3>{_metric_table(metrics)}"
            f"<h3>Diagnostics for this module path</h3><ul>{diagnostic_links}</ul>"
            "<p>Path diagnostics apply to the module; the schema does not identify a specific repeated call.</p></details>"
        )
    controls = "".join(
        f'<input type="radio" name="view" id="view-{key}"{" checked" if key == view else ""}>'
        f'<label class="view-label" for="view-{key}">{label}</label>'
        for key, label in _VIEWS.items()
    )
    diagnostic_rows = "".join(
        f'<li id="diagnostic-{i}" tabindex="-1"><strong>{escape(item["severity"])} · {escape(item["code"])}</strong> '
        f"[{escape(item['metric'])}] {escape(item['message'])}{_metadata({key: item[key] for key in ('path', 'operator') if key in item})}</li>"
        for i, item in enumerate(_all_diagnostics(report))
    )
    suggestions = "".join(
        f'<li class="suggestion"><p><strong>Recorded fact:</strong> {escape(fact)} <a href="#{target}">Evidence</a></p>'
        f"<p><strong>Experiment to try:</strong> {escape(experiment)}</p></li>"
        for fact, experiment, target in _suggestions(report)
    )
    operators = report["operator_flops"]
    operator_rows = "".join(
        f"<tr><th scope='row'>{escape(name)}</th><td>{_num(count)} known FLOPs</td></tr>"
        for name, count in sorted(operators["by_operator"].items(), key=lambda item: (-item[1], item[0]))
    )
    module_rows = "".join(
        f"<tr><th scope='row'>{escape(name)}</th><td>{_num(count)} known inclusive FLOPs</td></tr>"
        for name, count in sorted(operators["by_module"].items())
    )
    data = {
        "report": report,
        "before": before,
        "comparison": comparison[0] if comparison else None,
        "maps": maps,
        "diagnostics": [{**item, "index": i} for i, item in enumerate(diagnostics)],
        "suggestions": [
            {"fact": fact, "experiment": experiment, "target": target}
            for fact, experiment, target in _suggestions(report)
        ],
    }
    # A script raw-text element does not honor HTML entities. Escape delimiters
    # as JSON Unicode escapes, including line separators, even for non-executable data.
    embedded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, replacement in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        embedded = embedded.replace(character, replacement)
    digest = hashlib.sha256(SCRIPT.encode()).digest()
    # Hash the constant enhancement instead of permitting arbitrary inline code.
    script_hash = base64.b64encode(digest).decode()
    csp = f"default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'"
    cards = []
    for name, label in (
        ("module_flops", "Module formulas"),
        ("parameters", "Unique parameters"),
        ("operator_flops", "Operator dispatch"),
    ):
        result = report["totals"].get(name)
        status = result["status"] if result is not None else "unavailable"
        number = "unknown"
        if result is not None and result["known_value"] is not None:
            number = (
                ("≥ " if status == "partial" else "")
                + compact_value(_value(result, "known_value"))
                + " "
                + result["unit"]
            )
        cards.append(
            f'<div class="summary-card"><span class="eyebrow">{label}</span>'
            f'<p class="value">{escape(number)}</p><span class="badge {status}">{status}</span>'
            f"<small>{escape(result['method'] if result is not None else 'not recorded')}</small></div>"
        )
    baseline_details = ""
    if before is not None:
        baseline_details = (
            "<section><h2>Before module-call details</h2>"
            + "".join(
                f'<details id="before-call-{i}" tabindex="-1"><summary>{escape(_label(layer))}</summary>'
                f"<h3>Before input shapes and metadata</h3>{_metadata(layer['input'])}"
                f"<h3>Before output shapes and metadata</h3>{_metadata(layer['output'])}"
                f"{_metric_table(layer['metrics'])}</details>"
                for i, layer in enumerate(before["layers"])
            )
            + "</section>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{escape(csp, quote=True)}"><title>{escape(title)}</title><style>{STYLE}</style></head><body>'
        '<a class="skip" href="#main">Skip to report</a><main id="main" tabindex="-1">'
        f"<header><p>TorchScan · schema 1 · offline report</p><h1>{escape(title)}</h1><p>{escape(str(report['context'].get('model_type', 'Model')))}</p></header>"
        f'<p class="muted">Input {escape(shape_text(report["inputs"]))} · {escape(str(report["context"].get("devices", [])))} · '
        f'{escape(str(report["context"].get("dtypes", [])))}</p><div class="summary-cards">{"".join(cards)}</div>'
        "<fieldset><legend>Select compute or parameter view</legend>"
        f"{controls}{''.join(_explorer_panel(report, maps[key], before) for key in _VIEWS)}</fieldset>"
        '<details class="appendix"><summary>Measurement interpretation and exact evidence</summary>'
        '<section aria-labelledby="interpretation"><h2 id="interpretation">How to read this report</h2>'
        f'<p class="note">{_BOUNDARIES}</p><p><span class="badge complete">complete</span> full value for the stated method; '
        '<span class="badge partial">partial</span> lower bound only; <span class="badge unavailable">unavailable</span> unknown or not requested. '
        "A complete zero is valid. A partial lower bound of zero still leaves the full value unknown.</p></section>"
        '<section id="totals" tabindex="-1"><h2>Authoritative model totals</h2>'
        f"{_metric_table(report['totals'], prefix='total-')}</section>"
        '<section class="tree"><h2>Clickable module hierarchy</h2><p>Use Tab to focus summaries and links, Enter or Space to collapse or expand a module. '
        f"Call links open its shape and method details below.</p>{_tree(maps['module_flops']['groups'][0], report, before)}</section>"
        f"<section><h2>Module-call details</h2>{''.join(call_details)}</section>"
        f"{baseline_details}"
        '<section id="operators" tabindex="-1"><h2>Operator evidence</h2>'
        f"<p>Global authoritative count: {_result(operators['total'])}</p><p>Method: {escape(operators['total']['method'])}; scope: {escape(operators['total']['scope'])}.</p>"
        '<div class="table-scroll"><table><caption>Known counts by operator. Uncounted operations are in diagnostics.</caption>'
        f"<thead><tr><th>Operator</th><th>Known count</th></tr></thead><tbody>{operator_rows}</tbody></table></div>"
        "<details><summary>Inclusive operator counts by upstream module label</summary><p>These labels are not stable module-call identities. "
        'Parent counts include children. Do not sum these rows or join them to layer contributions.</p><div class="table-scroll"><table>'
        f"<thead><tr><th>Upstream label</th><th>Inclusive count</th></tr></thead><tbody>{module_rows}</tbody></table></div>"
        f"{'<p>No upstream module attribution was supplied.</p>' if not module_rows else ''}</details>"
        f"<details><summary>Explicitly ignored operators (not uncounted work)</summary>{_metadata(operators['ignored_operators'])}</details></section>"
        '<section id="diagnostics" tabindex="-1"><h2>Diagnostics</h2>'
        f"<ul>{diagnostic_rows}</ul>{'<p>No diagnostics were recorded.</p>' if not diagnostic_rows else ''}</section>"
        f"<section><h2>Evidence-linked optimization suggestions</h2><ul>{suggestions}</ul>"
        f"{'<p>No complete positive contribution supports a specific suggestion.</p>' if not suggestions else ''}</section>"
        f"{_comparison_html(before, report, *comparison) if before is not None and comparison is not None else ''}"
        "<section><h2>Input and measurement context</h2><details open><summary>Analysis inputs</summary>"
        f"{_metadata(report['inputs'])}</details><details open><summary>Execution and software</summary>{_metadata(report['context'])}</details></section>"
        f'</details><script id="torchscan-data" type="application/json">{embedded}</script></main><script>{SCRIPT}</script></body></html>\n'
    )


def _svg(
    report: AnalysisReport, title: str, view: str, before: AnalysisReport | None, comparison: Comparison | None
) -> str:
    maps = _explorer_data(report, before, comparison)
    return visual_svg(report, title, maps[view], before=before, comparison=comparison, suggestions=_suggestions(report))


def render_report(
    report: AnalysisReport,
    *,
    format: Literal["html", "svg"] = "html",  # ruff: ignore[builtin-argument-shadowing]
    before: AnalysisReport | None = None,
    title: str = "TorchScan analysis",
    metric: View = "module_flops",
) -> str:
    """Render structured analysis as a self-contained HTML report or standalone SVG.

    Args:
        report: Schema-v1 report returned by ``crawl_module`` or ``summary``.
        format: ``"html"`` for interactive native controls or ``"svg"`` for a static snapshot.
        before: Optional baseline passed to ``compare_reports(before, report)``.
        title: Plain-text report title. Untrusted strings are escaped.
        metric: Initial HTML view or SVG cost map: module_flops, macs, dmas, parameters, or parameter_bytes.

    Returns:
        UTF-8-compatible document text. Save with ``Path("report.html").write_text(result, encoding="utf-8")``;
        open with ``webbrowser.open(Path("report.html").resolve().as_uri())``. No server or extra dependencies are needed.

    Raises:
        ValueError: For invalid schema, metric, format, non-JSON data, invalid measurement states,
            duplicate call identities, or incompatible comparison methods, units or scopes.

    Notes:
        Inputs are never mutated or remeasured. Execution deltas are withheld when input metadata,
        execution mode, device/dtype metadata or software versions differ or are missing. Storage
        totals need compatible complete metrics but do not require identical execution inputs.
        The SVG exports the module map and selected evidence; HTML adds keyboard selection, zoom, collapse and views.
    """
    if format not in ("html", "svg"):
        raise ValueError("format must be 'html' or 'svg'")
    if not isinstance(metric, str) or metric not in _VIEWS:
        raise ValueError(f"metric must be one of {', '.join(_VIEWS)}")
    _string(title, "title")
    _validate(report)
    comparison = None
    if before is not None:
        _validate(before)
        comparison = _comparison(before, report)
    return (
        _html(report, title, metric, before, comparison)
        if format == "html"
        else _svg(report, title, metric, before, comparison)
    )
