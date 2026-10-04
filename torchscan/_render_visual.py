# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Dependency-free SVG composition shared by interactive and static reports."""

import json
import re
import textwrap
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

from ._render_map import _owner_text

_INK = "#20354e"
_MUTED = "#61758b"
_GREEN = "#227659"
_AMBER = "#85551d"
_PALETTE = ("#234564", "#badde9", "#9ed4dc", "#83cdbd", "#a7d9cc", "#bddfdf")


def compact_value(value: float | None) -> str:
    """Format known numbers compactly without converting unknowns to zero."""
    if value is None:
        return "unknown"
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{value / threshold:,.3g}{suffix}"
    return f"{value:,}" if isinstance(value, int) else f"{value:,.5g}"


def shape_text(metadata: Any) -> str:
    """Describe tensor shapes while preserving nested input/output structure."""
    if isinstance(metadata, list):
        return "(" + ", ".join(shape_text(item) for item in metadata) + ")"
    if not isinstance(metadata, Mapping):
        return str(metadata)
    if metadata.get("kind") == "tensor":
        shape = metadata.get("shape", [])
        if not isinstance(shape, list | tuple):
            return str(shape)
        return "[" + ", ".join(str(dimension) for dimension in shape) + "]"
    if "args" in metadata:
        raw_args, raw_kwargs = metadata.get("args", []), metadata.get("kwargs", {})
        args = [shape_text(item) for item in raw_args] if isinstance(raw_args, list) else [shape_text(raw_args)]
        kwargs = (
            [f"{key}={shape_text(item)}" for key, item in raw_kwargs.items()]
            if isinstance(raw_kwargs, Mapping)
            else [shape_text(raw_kwargs)]
        )
        return "; ".join(args + kwargs) or "no inputs"
    if metadata.get("kind") in ("tuple", "list"):
        items = metadata.get("items", [])
        return (
            "(" + ", ".join(shape_text(item) for item in items) + ")" if isinstance(items, list) else shape_text(items)
        )
    if metadata.get("kind") == "mapping":
        items = metadata.get("items", [])
        if isinstance(items, list):
            return (
                "{"
                + ", ".join(
                    f"{item.get('key', item.get('key_type', '?'))}: {shape_text(item.get('value'))}"
                    for item in items
                    if isinstance(item, Mapping)
                )
                + "}"
            )
        return shape_text(items)
    return str(metadata.get("type", metadata.get("kind", "unknown shape")))


def _contributors(group: dict[str, Any], *, before: bool = False) -> list[dict[str, Any]]:
    field = "before_direct" if before else "direct"
    return [node for node in group["nodes"] if node[field] is not None and (node[field]["known_value"] or 0) > 0]


def default_selection(group: dict[str, Any]) -> dict[str, Any] | None:
    """Select a costly direct contributor, then an incomplete measured module."""
    for before in (False, True):
        contributors = _contributors(group, before=before)
        if contributors:
            field = "before_direct" if before else "direct"
            return max(contributors, key=lambda node: node[field]["known_value"])
    return next(
        (node for node in group["nodes"] if node["calls"] and node["direct_kind"] != "structural"),
        next(iter(group["nodes"]), None),
    )


def _identifier(value: str) -> str:
    # Inputs are internal model IDs, never names. Still enforce attribute-safe IDs
    # here so the shared helper cannot become a second injection boundary.
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
        raise ValueError("SVG namespaces and node IDs must be generated identifiers")
    return value


def _measurement(result: Mapping[str, Any] | None, *, units: bool = True) -> str:
    if result is None or result["status"] == "unavailable":
        return "unavailable · unknown"
    unit = " " + str(result["unit"]) if units else ""
    if result["status"] == "partial":
        return f"partial · ≥ {compact_value(result['known_value'])}{unit} · full value unknown"
    return f"complete · {compact_value(result['value'])}{unit}"


def _short_measurement(result: Mapping[str, Any] | None) -> str:
    if result is None or result["status"] == "unavailable":
        return "unknown · unavailable"
    if result["status"] == "partial":
        return f"≥ {compact_value(result['known_value'])} {result['unit']} · partial"
    return f"{compact_value(result['value'])} {result['unit']} · complete"


def _text(
    x: float, y: float, text: str, *, size: int = 12, color: str = _INK, weight: int = 400, extra: str = ""
) -> str:
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}"{extra}>{escape(text)}</text>'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "#fff",
    stroke: str = "#dbe4ed",
    extra: str = "",
    radius: int = 6,
) -> str:
    return (
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{max(0, width):.3f}" height="{max(0, height):.3f}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}"{extra}/>'
    )


def _paragraph(
    output: list[str],
    x: float,
    y: float,
    text: str,
    *,
    width: int = 52,
    size: int = 12,
    color: str = _MUTED,
    leading: int = 19,
    weight: int = 400,
) -> float:
    lines = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]
    output.extend(
        _text(x, y + index * leading, line, size=size, color=color, weight=weight) for index, line in enumerate(lines)
    )
    return y + len(lines) * leading


def _node_description(node: dict[str, Any], group: dict[str, Any]) -> str:
    kind = (
        "uniquely attributed storage" if node["direct_kind"] == "attributed" else "recorded additive call contribution"
    )
    description = f"{node['path'] or '(root)'} · {node['type']} · {len(node['calls'])} observed calls. "
    if node["has_contributions"] or node.get("coverage") is None:
        description += f"Derived subtree subtotal: {_measurement(node['subtotal'])}. "
    description += (
        f"Own calls covered: {node['coverage']}. No separate additive contribution. "
        if node.get("coverage") is not None
        else f"Own {kind}: {_measurement(node['direct'])}. "
    )
    description += f"Method: {group['method']}; scope: {group['scope']}."
    if group.get("comparison"):
        if node["change"] in ("added", "removed"):
            description += f" Module {node['change']} in after report."
        description += f" Before: {node['before_display']}."
        if node["delta"] is None:
            description += " Delta unknown: " + str(node["delta_reason"])
        else:
            description += f" Complete comparable delta: {node['delta']:+,} {group['unit']}."
    if group["scale"] == "structure":
        description += " Unscaled structural layout; widths do not encode cost."
    return description


def map_svg(group: dict[str, Any], *, selected: str, prefix: str, interactive: bool = False) -> str:
    """Draw one method-compatible icicle; unknowns never acquire measured widths."""
    prefix = _identifier(prefix)
    comparison = bool(group.get("comparison"))
    row_height = 76 if comparison else 58
    height = (group["max_depth"] + 1) * row_height + 8
    title_id, desc_id, hatch_id = f"{prefix}-title", f"{prefix}-description", f"{prefix}-hatch"
    status_desc = (
        "Nested rectangles show the observed module hierarchy. Width encodes recorded additive contributions; "
        "parent values are derived subtotals. Unknown and zero contributions remain in an unscaled rail. "
        "Partial values are lower bounds."
        if group["scale"] == "known"
        else "Unscaled observed module hierarchy. No positive cost is known; widths show structure only, never measured zero."
    )
    output = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="module-map" width="900" height="{height}" '
            f'viewBox="0 0 900 {height}" role="{"group" if interactive else "img"}" '
            f'aria-labelledby="{title_id} {desc_id}" data-group-id="{_identifier(group["id"])}">'
        ),
        f'<title id="{title_id}">{escape(group["label"])} module cost map</title>',
        f'<desc id="{desc_id}">{escape(status_desc)}</desc>',
        (
            f'<defs><pattern id="{hatch_id}" width="8" height="8" patternUnits="userSpaceOnUse">'
            '<path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#bd862d" stroke-opacity=".38" stroke-width="1"/>'
            '</pattern></defs><g font-family="system-ui, -apple-system, sans-serif">'
        ),
    ]
    for node in group["nodes"]:
        node_id = _identifier(node["id"])
        x, y = 4 + node["x"] * 892, 4 + node["depth"] * row_height
        width, tile_height = node["width"] * 892, row_height - 6
        if width <= 0:
            continue
        gap = min(1.5, width / 5)
        painted_width = max(0, width - gap)
        clip_id, element_id = f"{prefix}-clip-{node_id}", f"{prefix}-{node_id}"
        is_selected = node_id == selected
        neutral = group["scale"] == "structure" or node["status"] == "unavailable"
        fill = "#e9eef3" if neutral else _PALETTE[min(node["depth"], len(_PALETTE) - 1)]
        if is_selected and not neutral and not interactive:
            fill = "#237e6b"
        text_color = "#fff" if not neutral and (node["depth"] == 0 or (is_selected and not interactive)) else _INK
        target = f"module-{node_id}" if interactive else element_id
        output.append(
            f'<defs><clipPath id="{clip_id}"><rect x="{x + 5:.3f}" y="{y:.3f}" '
            f'width="{max(0, painted_width - 10):.3f}" height="{tile_height:.3f}"/></clipPath></defs>'
        )
        data = (
            f' data-map-node="" data-node-id="{node_id}" data-group-id="{group["id"]}" '
            f'data-parent="{node["parent"] or ""}" data-depth="{node["depth"]}" data-neutral="{str(neutral).lower()}"'
        )
        description = _node_description(node, group)
        current_attribute = ' aria-current="true"' if is_selected else ""
        output.extend((
            (
                f'<a id="{element_id}" href="#{target}" tabindex="0"{data} '
                f'aria-label="{escape(description, quote=True)}"'
                f"{current_attribute}>"
                f"<title>{escape(description + node.get('_evidence', ''))}</title>"
            ),
            _rect(
                x,
                y,
                painted_width,
                tile_height,
                fill=fill,
                stroke="#0b6453" if is_selected and not interactive else "#d6e5ed",
                extra=' data-tile="" stroke-width="2"' if is_selected and not interactive else ' data-tile=""',
                radius=4,
            ),
        ))
        if node["status"] == "partial" and group["scale"] != "structure":
            output.append(_rect(x, y, painted_width, tile_height, fill=f"url(#{hatch_id})", stroke="none", radius=4))
        calls = len(node["calls"])
        label = node["label"] + (f"  \u00d7{calls} calls" if calls > 1 else "")
        if comparison and node["change"] in ("added", "removed"):
            label += " · " + node["change"]
        output.extend((
            f'<g clip-path="url(#{clip_id})">',
            _text(x + 10, y + 21, label, size=13, color=text_color, weight=650),
        ))
        if comparison:
            for offset, known, result, legend in (
                (31, node["before_known"], node["before_subtotal"], "B"),
                (46, node["known"], node["subtotal"], "A"),
            ):
                reserve = 100 if painted_width > 190 else 0
                available = max(0, painted_width - 28 - reserve)
                bar_width = available * max(0, known or 0) / node["weight"] if node["weight"] else 0
                color = (
                    "#71869a"
                    if legend == "B"
                    else _GREEN
                    if node["delta"] is not None and node["delta"] <= 0
                    else "#bc6e42"
                    if node["delta"] is not None
                    else "#668b93"
                )
                output.extend((
                    _text(x + 10, y + offset + 7, legend, size=9, color=text_color, weight=600),
                    _rect(x + 22, y + offset, bar_width, 7, fill=color, stroke="none", radius=2),
                ))
                if reserve:
                    measured_label = (
                        "unknown"
                        if known is None
                        else ("≥ " if result and result["status"] == "partial" else "") + compact_value(known)
                    )
                    output.append(_text(x + available + 30, y + offset + 7, measured_label, size=9, color=text_color))
                elif known is None and painted_width > 55:
                    output.append(_text(x + 24, y + offset + 7, "?", size=9, color=text_color))
                if result is not None and result["status"] == "partial":
                    output.append(
                        _rect(x + 22, y + offset, bar_width, 7, fill=f"url(#{hatch_id})", stroke="none", radius=2)
                    )
            delta_text = (
                "delta unknown" if node["delta"] is None else f"Δ {node['delta']:+,} {group['unit']} · complete"
            )
            output.append(_text(x + 10, y + 65, delta_text, size=10, color=text_color))
        else:
            detail = "structural · unscaled" if group["scale"] == "structure" else _short_measurement(node["subtotal"])
            output.append(_text(x + 10, y + 41, detail, size=11, color=text_color))
        output.append("</g></a>")
        if node["children"] and node["direct_width"] > 0:
            own_width = node["direct_width"] * 892
            own_x = 4 + node["direct_x"] * 892
            output.extend((
                f'<g data-own-node="{node_id}" data-group-id="{group["id"]}" aria-label="{escape(node["label"] + " own contribution gap", quote=True)}">',
                _rect(
                    own_x,
                    y + row_height,
                    max(0, own_width - 1.5),
                    tile_height,
                    fill="#eef3f5",
                    stroke="#dce4eb",
                    radius=4,
                ),
                f"<title>{escape(node['label'])}: own additive contribution; descendants occupy the remaining width.</title>",
            ))
            if own_width > 60:
                output.append(_text(own_x + 8, y + row_height + 23, "own", size=11, color=_MUTED))
            output.append("</g>")
    output.append("</g></svg>")
    return "".join(output)


def _tensor_glyph(x: int, y: float, label: str, metadata: Any, *, prefix: str) -> str:
    return (
        "".join(
            _rect(x + offset, y - offset, 31, 31, fill="#d9edf0", stroke="#85b4c1", radius=3) for offset in (6, 3, 0)
        )
        + _text(x + 44, y + 7, label, size=10, color=_MUTED, weight=650)
        + f'<g clip-path="url(#{prefix})">'
        + _text(x + 44, y + 27, shape_text(metadata), size=12)
        + f"<title>{escape(shape_text(metadata))}</title></g>"
    )


def _call_result(call: Mapping[str, Any], group: Mapping[str, Any]) -> Mapping[str, Any] | None:
    result = call["result"]
    return (
        result
        if group.get("kind") == "recorded"
        and result is not None
        and all(result[field] == group[field] for field in ("method", "unit", "scope"))
        else None
    )


def _inspector(
    report: Mapping[str, Any],
    node: dict[str, Any] | None,
    group: dict[str, Any],
    *,
    x: int,
    y: float,
    width: int,
    before: Mapping[str, Any] | None = None,
) -> tuple[str, float]:
    left, cursor = x + 20, y + 29
    output = [_text(left, cursor, "SELECTED MODULE", size=11, color=_MUTED, weight=700)]
    if node is None:
        output.append(_text(left, cursor + 33, "No observed calls", size=19, weight=650))
        return "".join(output), cursor + 110
    identifier = f"inspector-{group['id']}-{node['id']}"
    output.append(f'<g id="{identifier}"><title>{escape(_node_description(node, group))}</title>')
    cursor = _paragraph(
        output, left, cursor + 31, node["path"] or "(root)", width=30, size=19, leading=25, color=_INK, weight=650
    )
    output.append(
        _text(
            left,
            cursor + 5,
            f"{node['type']} · {len(node['before_calls'])} before calls · removed"
            if node["change"] == "removed"
            else f"{node['type']} · {len(node['calls'])} observed calls",
            size=12,
            color=_MUTED,
        )
    )
    cursor += 39
    covered_only = node["coverage"] is not None and not node["has_contributions"]
    output.append(
        _text(
            left,
            cursor,
            "Own additive contribution"
            if node["direct_kind"] == "recorded"
            else "Own unique attribution"
            if node["direct_kind"] == "attributed"
            else "Covered by inclusive ancestor estimate"
            if covered_only
            else "Derived descendant subtotal",
            size=12,
            color=_MUTED,
        )
    )
    measured = node["direct"] if node["direct_kind"] in ("recorded", "attributed") else node["subtotal"]
    cursor = _paragraph(
        output,
        left,
        cursor + 29,
        node["coverage"] or node["before_coverage"] if covered_only else _measurement(measured),
        width=30,
        size=19,
        leading=25,
        color=_GREEN if measured and measured["status"] == "complete" else _AMBER,
        weight=650,
    )
    if node["coverage"] is not None and node["has_contributions"]:
        cursor = _paragraph(output, left, cursor + 9, f"Own calls: {node['coverage']}", width=43)
    cursor = _paragraph(
        output,
        left,
        cursor + 12,
        f"Method: {group['method']}; scope: {group['scope']}; unit: {group['unit']}",
        width=43,
    )
    if group.get("comparison"):
        cursor = _paragraph(
            output,
            left,
            cursor + 9,
            f"Before subtree: {node['before_display']}. After subtree: {node['display']}.",
            width=43,
        )
        delta = (
            f"Δ {node['delta']:+,} {group['unit']} · complete"
            if node["delta"] is not None
            else "Delta unknown: " + str(node["delta_reason"])
        )
        cursor = _paragraph(
            output,
            left,
            cursor + 4,
            delta,
            width=43,
            color=_GREEN if node["delta"] is not None and node["delta"] <= 0 else _MUTED,
        )
    baseline = not node["calls"] and before is not None and bool(node["before_calls"])
    calls = node["before_calls"] if baseline else node["calls"]
    source = before if baseline and before is not None else report

    if calls:
        cursor += 24
        output.append(
            _text(
                left,
                cursor,
                "Shapes · before call" if baseline else "Shapes · first observed call",
                size=14,
                weight=650,
            )
        )
        first = source["layers"][calls[0]["index"]]
        input_clip = f"{identifier}-input-clip"
        output_clip = f"{identifier}-output-clip"
        cursor += 21
        output.extend((
            f'<defs><clipPath id="{input_clip}"><rect x="{left}" y="{cursor - 10}" width="{width - 40}" height="45"/></clipPath><clipPath id="{output_clip}"><rect x="{left}" y="{cursor + 37}" width="{width - 40}" height="45"/></clipPath></defs>',
            _tensor_glyph(left, cursor, "INPUT", first["input"], prefix=input_clip),
            _tensor_glyph(left, cursor + 47, "OUTPUT", first["output"], prefix=output_clip),
        ))
        cursor += 100
        output.append(
            _text(
                left, cursor, "Call evidence · compute repeats, storage counts once", size=11, color=_MUTED, weight=600
            )
        )
        cursor += 17
        maximum = max(
            (
                result["value"] or 0 if result and result["status"] == "complete" else 0
                for result in (_call_result(call, group) for call in calls)
            ),
            default=0,
        )
        for call in calls:
            layer = source["layers"][call["index"]]
            call_id = f"{'before-call' if baseline else 'call'}-{call['index']}"
            result = _call_result(call, group)
            measurement = _owner_text(call["owner"]) if call.get("owner") is not None else _measurement(result)
            description = f"{node['path'] or '(root)'} call #{call['call_index']}: {measurement}. Input {shape_text(layer['input'])}; output {shape_text(layer['output'])}."
            output.extend((
                f'<g id="{call_id}"><title>{escape(description)}</title>',
                _rect(left, cursor, width - 40, 64, fill="#f3f7fa"),
                _text(left + 10, cursor + 18, f"call #{call['call_index']}", size=11, weight=650),
                _text(
                    left + 94,
                    cursor + 18,
                    "Covered by inclusive estimate" if call.get("owner") is not None else _short_measurement(result),
                    size=11,
                    color=_GREEN if result and result["status"] == "complete" else _AMBER,
                ),
            ))
            if result and result["status"] == "complete" and result["value"] is not None and maximum > 0:
                ratio = result["value"] / maximum
                output.append(
                    _rect(
                        left + 10,
                        cursor + 28,
                        (width - 60) * ratio,
                        5,
                        fill="#70b39c",
                        stroke="none",
                        radius=2,
                    )
                )
            stats = layer["parameters"]
            count = stats["trainable"] + stats["frozen"]
            storage = (
                "Shared tensors · no new parameter attribution"
                if stats["shared"] and count == 0
                else f"{compact_value(count)} newly attributed parameters"
                + (" · shared tensors" if stats["shared"] else "")
            )
            output.extend((_text(left + 10, cursor + 51, storage, size=11, color=_MUTED), "</g>"))
            cursor += 74
    diagnostics = _diagnostics(source)
    view = group.get("view", "module_flops")
    metrics = {view, "flops", "module_flops"} if view in ("module_flops", "macs", "dmas") else {view}
    relevant = [item for item in diagnostics if item.get("path") in (None, node["path"]) and item["metric"] in metrics]
    if relevant:
        cursor += 16
        output.append(
            _text(
                left,
                cursor,
                "Before measurement diagnostics" if baseline else "Measurement diagnostics",
                size=14,
                weight=650,
            )
        )
        for item in relevant:
            cursor = _paragraph(
                output,
                left,
                cursor + 24,
                ("Global · " if item.get("path") is None else "") + f"{item['code']}: {item['message']}",
                width=43,
                color=_AMBER,
            )
    output.append("</g>")
    return "".join(output), cursor + 22


def _diagnostics(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(report["diagnostics"]) + [
        item for item in report["operator_flops"]["diagnostics"] if item not in report["diagnostics"]
    ]


def _context_lines(report: Mapping[str, Any]) -> list[str]:
    context = report["context"]
    return [
        f"Inputs: {shape_text(report['inputs'])} · source: {report['inputs'].get('source', 'recorded')}",
        "Devices: "
        + ", ".join(str(value) for value in context.get("devices", []))
        + " · dtypes: "
        + ", ".join(str(value) for value in context.get("dtypes", [])),
        f"Mode: {context.get('execution_mode', 'unknown')} · analysis: {context.get('analysis_mode', 'full')}",
        f"PyTorch {context.get('torch_version', 'unknown')} · TorchScan {context.get('torchscan_version', 'unknown')} · Python {context.get('python_version', 'unknown')}",
    ]


def visual_svg(
    report: Mapping[str, Any],
    title: str,
    view_data: dict[str, Any],
    *,
    before: Mapping[str, Any] | None = None,
    comparison: Any = None,
    suggestions: Sequence[tuple[str, str, str]] = (),
) -> str:
    """Compose a standalone cost explorer snapshot from the same HTML map model."""
    # Enrich static tooltips without mutating the shared data embedded in HTML.
    groups: list[dict[str, Any]] = []
    call_targets: dict[str, str] = {}
    for original_group in view_data["groups"]:
        current = {
            **original_group,
            "view": view_data["view"],
            "nodes": [dict(node) for node in original_group["nodes"]],
        }
        for node in current["nodes"]:
            call_layers = [report["layers"][call["index"]] for call in node["calls"]]
            node["_evidence"] = " Call evidence: " + json.dumps(call_layers, ensure_ascii=False)
            if before is not None:
                node["_evidence"] += " Before call evidence: " + json.dumps(
                    [before["layers"][call["index"]] for call in node["before_calls"]], ensure_ascii=False
                )
            target = (
                f"static-{current['id']}-{node['id']}" if node["width"] > 0 else f"rail-{current['id']}-{node['id']}"
            )
            if node["width"] > 0 or node["id"] in current["rails"]:
                for call in node["calls"]:
                    call_id = f"call-{call['index']}"
                    if _call_result(call, current) is not None or (
                        current["kind"] == "unrecorded" and call["result"] is None
                    ):
                        call_targets[call_id] = target
                    else:
                        call_targets.setdefault(call_id, target)
        groups.append(current)
    group: dict[str, Any] = (
        groups[0]
        if groups
        else {
            "id": "empty",
            "nodes": [],
            "unit": "",
            "method": "unavailable",
            "scope": "unavailable",
            "view": view_data["view"],
        }
    )
    # Select a recorded group without comparing magnitudes across methods.
    for baseline in (False, True) if before is not None else (False,):
        group = next((candidate for candidate in groups if _contributors(candidate, before=baseline)), group)
        if _contributors(group, before=baseline):
            break
    selected = default_selection(group)
    fragments = [
        _text(32, 36, "TORCHSCAN · OFFLINE VISUAL REPORT", size=12, color=_MUTED, weight=700),
        _text(32, 77, title, size=30, weight=700),
        _text(
            32,
            105,
            "Observed module hierarchy · widths reveal recorded work · not a computational graph",
            size=13,
            color=_MUTED,
        ),
    ]
    totals = (
        (view_data["label"], view_data["total"]),
        ("Unique parameters", report["totals"].get("parameters")),
        ("Inclusive operator count", report["operator_flops"]["total"]),
    )
    for index, (label, total) in enumerate(totals):
        x = 32 + index * 302
        fragments.extend((
            _rect(x, 128, 288, 85),
            _text(x + 15, 151, label, size=12, color=_MUTED, weight=650),
            _text(
                x + 15,
                181,
                _short_measurement(total),
                size=18,
                color=_GREEN if total and total["status"] == "complete" else _AMBER,
                weight=650,
            ),
            _text(
                x + 15,
                200,
                "Authoritative model total" if index < 2 else "Inclusive counts · never sum module rows",
                size=10,
                color=_MUTED,
            ),
        ))
    map_y = 229.0
    rail_nodes = []
    for current in groups:
        current_selection = selected["id"] if current["id"] == group["id"] and selected else ""
        svg = map_svg(current, selected=current_selection, prefix=f"static-{current['id']}")
        nested_height = (current["max_depth"] + 1) * (76 if current.get("comparison") else 58) + 8
        panel_height = nested_height + 113
        # Nested viewports preserve one common drawing implementation.
        nested = (
            svg
            .replace("<svg xmlns=", f'<svg x="47" y="{map_y + 90:.3f}" xmlns=', 1)
            .replace('width="900"', 'width="860"', 1)
            .replace(f'height="{nested_height}"', f'height="{nested_height * 860 / 900:.3f}"', 1)
        )
        fragments.extend((
            _rect(32, map_y, 890, panel_height),
            _text(52, map_y + 30, "Module cost map · " + current["label"], size=16, weight=650),
            _text(
                52,
                map_y + 54,
                "Width = recorded additive cost; containers show derived subtotals."
                if current["scale"] == "known"
                else "STRUCTURAL LAYOUT · widths are unscaled; no positive cost is known.",
                color=_MUTED,
            ),
            _text(
                52, map_y + 74, "Method: " + current["method"] + " · scope: " + current["scope"], size=11, color=_MUTED
            ),
            nested,
        ))
        by_id = {node["id"]: node for node in current["nodes"]}
        rail_nodes.extend((current, by_id[identifier]) for identifier in current["rails"])
        map_y += panel_height + 15
    if not groups:
        fragments.extend((
            _rect(32, map_y, 890, 135),
            _text(52, map_y + 34, "No observed module contributions", size=20, weight=650),
            _text(52, map_y + 63, "Unavailable measurements remain unknown.", color=_MUTED),
        ))
        map_y += 150
    if rail_nodes:
        rail_nodes.sort(
            key=lambda entry: (
                0 if "unknown" in entry[1]["rail"] else 1 if "small" in entry[1]["rail"] else 2,
                entry[1]["path"],
            )
        )
        rail_height = 76 + ((len(rail_nodes) + 1) // 2) * 83
        fragments.extend((
            _rect(32, map_y, 890, rail_height),
            _text(52, map_y + 30, "Always visible · unscaled contributions", size=16, weight=650),
            _text(
                52,
                map_y + 52,
                "Unknown ≠ zero. Small, zero, and incomplete costs retain their own evidence.",
                color=_MUTED,
            ),
        ))
        for index, (current, node) in enumerate(rail_nodes):
            x, y = 52 + (index % 2) * 425, map_y + 66 + (index // 2) * 83
            rail_id = f"rail-{current['id']}-{node['id']}"
            target = f"static-{current['id']}-{node['id']}" if node["width"] > 0 else rail_id
            fill = "#fff4df" if "unknown" in node["rail"] and node["status"] != "unavailable" else "#f0f4f7"
            description = _node_description(node, current)
            evidence = "" if node["width"] > 0 else node.get("_evidence", "")
            fragments.extend((
                f'<a id="{rail_id}" href="#{target}" tabindex="0" aria-label="{escape(description, quote=True)}"><title>{escape(description + evidence)}</title>',
                _rect(x, y, 409, 72, fill=fill),
            ))
            if node["direct"] and node["direct"]["status"] == "partial":
                fragments.append(_rect(x, y, 409, 72, fill="url(#svg-rail-hatch)", stroke="none"))
            name = node["path"] or "(root)"
            fragments.extend((
                f'<defs><clipPath id="{rail_id}-clip"><rect x="{x + 8}" y="{y}" width="393" height="72"/></clipPath></defs><g clip-path="url(#{rail_id}-clip)">',
                _text(x + 12, y + 21, name, size=13, weight=650),
                _text(
                    x + 12,
                    y + 43,
                    _short_measurement(node["direct"]),
                    color=_AMBER if "unknown" in node["rail"] else _MUTED,
                ),
                _text(x + 12, y + 62, " · ".join(node["rail"]) + " · not sized by cost", size=10, color=_MUTED),
                "</g></a>",
            ))
        map_y += rail_height + 15
    inspector, inspector_bottom = _inspector(report, selected, group, x=942, y=128, width=370, before=before)
    fragments.extend((_rect(942, 128, 370, inspector_bottom - 128), inspector))
    cursor = max(map_y, inspector_bottom + 15)
    if before is not None:
        difference = view_data["comparison_delta"]
        delta = difference.get("delta") if difference else None
        reasons = comparison[1] if comparison and isinstance(comparison, tuple) else []
        reason_text = (
            "Execution deltas withheld: " + "; ".join(reasons)
            if reasons
            else "Input and measurement context match. Numeric deltas require two complete measurements."
        )
        reason_lines = textwrap.wrap(reason_text, width=168)
        compare_height = 174 + len(reason_lines) * 19
        fragments.extend((
            _rect(32, cursor, 1280, compare_height),
            _text(52, cursor + 30, "Before / after · stable module positions", size=18, weight=650),
        ))
        maximum = max(
            (metric["known_value"] or 0 if metric else 0 for metric in (view_data["before_total"], view_data["total"])),
            default=0,
        )
        for offset, label, result, color in (
            (54, "Before", view_data["before_total"], "#a0afbf"),
            (84, "After", view_data["total"], "#58a586" if delta is not None and delta <= 0 else "#7d9fa7"),
        ):
            known = result["known_value"] if result else None
            fragments.extend((
                _text(52, cursor + offset + 16, label, color=_MUTED),
                _rect(
                    109,
                    cursor + offset,
                    610 * (known or 0) / maximum if maximum else 0,
                    20,
                    fill=color,
                    stroke="none",
                    radius=3,
                ),
                _text(742, cursor + offset + 15, _measurement(result), color=_MUTED),
            ))
        delta_text = (
            f"Δ {delta:+,} {difference['after']['unit']} · complete"
            if delta is not None
            else "Delta unavailable · requires comparable complete measurements"
        )
        fragments.append(
            _text(
                52,
                cursor + 136,
                delta_text,
                size=13,
                color=_GREEN if delta is not None and delta <= 0 else _MUTED,
                weight=650,
            )
        )
        _paragraph(fragments, 52, cursor + 164, reason_text, width=168)
        cursor += compare_height + 15
    if suggestions:
        content, card_cursor = [], cursor + 59
        content.append(_text(52, cursor + 29, "Evidence-linked experiments", size=18, weight=650))
        selected_call_ids = (
            {f"call-{call['index']}" for call in selected["calls"] if _call_result(call, group) is not None}
            if selected
            else set()
        )
        # Facts and hypotheses remain separate and fully searchable in the SVG.
        for fact, experiment, target in suggestions:
            evidence_target = target if target in selected_call_ids else call_targets.get(target, "svg-evidence")
            content.append(f'<a href="#{evidence_target}" tabindex="0"><title>{escape(fact)}</title>')
            card_cursor = _paragraph(
                content, 52, card_cursor, "Recorded fact: " + fact, width=168, color=_INK, weight=600
            )
            content.append("</a>")
            card_cursor = _paragraph(content, 52, card_cursor + 5, "Experiment to try: " + experiment, width=168)
            card_cursor += 14
        fragments.append(_rect(32, cursor, 1280, card_cursor - cursor + 6))
        fragments.extend(content)
        cursor = card_cursor + 21
    context_body, context_cursor = [], cursor + 60
    context_body.extend((
        _text(52, cursor + 30, "Input and measurement context", size=18, weight=650),
        '<g id="svg-evidence"><title>'
        + escape(
            json.dumps(
                {
                    "inputs": report["inputs"],
                    "context": report["context"],
                    "diagnostics": report["diagnostics"],
                    "operator_diagnostics": report["operator_flops"]["diagnostics"],
                    "before": {
                        "inputs": before["inputs"],
                        "context": before["context"],
                        "diagnostics": _diagnostics(before),
                    }
                    if before is not None
                    else None,
                },
                ensure_ascii=False,
            )
        )
        + "</title>",
    ))
    sources = (("After · ", report), ("Before · ", before)) if before is not None else (("", report),)
    for label, source in sources:
        for line in _context_lines(source):
            context_cursor = _paragraph(
                context_body, 52, context_cursor + (5 if label == "Before · " else 0), label + line, width=162
            )
    if before is not None and _diagnostics(before):
        context_cursor = _paragraph(
            context_body, 52, context_cursor + 15, "Before diagnostics", size=14, color=_INK, weight=650
        )
        for index, diagnostic in enumerate(_diagnostics(before)):
            context_body.append(
                f'<g id="before-diagnostic-{index}"><title>{escape(json.dumps(diagnostic, ensure_ascii=False))}</title>'
            )
            context_cursor = _paragraph(
                context_body,
                52,
                context_cursor + 8,
                f"Before · {diagnostic.get('path') or 'Global'} · {diagnostic['metric']} · {diagnostic['code']}: {diagnostic['message']}",
                width=168,
                color=_AMBER,
            )
            context_body.append("</g>")
    context_cursor = _paragraph(
        context_body,
        52,
        context_cursor + 14,
        "Inclusive operator counts include descendants and are not additive layer contributions. Only observed calls appear. Shared tensors count once in report attribution; uncalled parameters may appear only in authoritative model totals.",
        width=168,
    )
    context_cursor = _paragraph(
        context_body,
        52,
        context_cursor + 8,
        "Complete = full value for its method. Partial = a lower bound, even when that bound is zero. Unavailable = unknown. FLOPs do not establish latency. Static parameter and buffer sizes are not measured peak memory.",
        width=168,
    )
    context_body.append("</g>")
    fragments.append(_rect(32, cursor, 1280, context_cursor - cursor + 13))
    fragments.extend(context_body)
    height = context_cursor + 42
    description = "Offline module cost explorer. Icicle chart shows module ancestry, additive contribution widths, repeated calls, measurement statuses and stable before/after marks. Inspector provides shapes and call evidence. Unknown measurements never mean zero."
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1344" height="{height:.3f}" viewBox="0 0 1344 {height:.3f}" role="img" aria-labelledby="svg-title svg-description">'
        f'<title id="svg-title">{escape(title)}</title><desc id="svg-description">{escape(description)}</desc>'
        f'<rect width="1344" height="{height:.3f}" fill="#f3f6fa"/>'
        '<defs><pattern id="svg-rail-hatch" width="8" height="8" patternUnits="userSpaceOnUse">'
        '<path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#ddb874" stroke-opacity=".55" stroke-width="1"/>'
        '</pattern></defs><g font-family="system-ui, -apple-system, sans-serif">'
        f"{''.join(fragments)}</g></svg>\n"
    )
