# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from copy import deepcopy
from itertools import starmap
from typing import Any

from .report import AnalysisReport, LayerReport, MetricResult


def format_name(name: str, depth: int = 0) -> str:
    """Format a layer name for nested summary output."""
    if depth == 0:
        return name
    if depth == 1:
        return f"├─{name}"
    return f"{'|    ' * (depth - 1)}└─{name}"


def wrap_string(s: str, max_len: int, delimiter: str = ".", wrap: str = "[...]", mode: str = "end") -> str:
    """Wrap a string into a fixed display width."""
    if len(s) <= max_len or mode is None:
        return s
    if mode == "end":
        return s[: max_len - len(wrap)] + wrap
    if mode == "mid":
        final_part = s.rpartition(delimiter)[-1]
        wrapped_end = f"{wrap}.{final_part}"
        return s[: max_len - len(wrapped_end)] + wrapped_end
    raise ValueError("received an unexpected value of argument `mode`")


def unit_scale(val: float) -> tuple[float, str]:
    """Rescale a value using SI display units."""
    if val // 1e12 > 0:
        return val / 1e12, "T"
    if val // 1e9 > 0:
        return val / 1e9, "G"
    if val // 1e6 > 0:
        return val / 1e6, "M"
    if val // 1e3 > 0:
        return val / 1e3, "k"
    return val, ""


def format_s(f_string: str, min_w: int | None = None, max_w: int | None = None) -> str:
    """Pad and truncate a display string."""
    if isinstance(min_w, int):
        f_string = f"{f_string:<{min_w}}"
    if isinstance(max_w, int):
        f_string = f"{f_string:.{max_w}}"
    return f_string


def _shape_text(metadata: dict[str, Any]) -> str:
    kind = metadata.get("kind")
    if kind == "tensor":
        return str(tuple(metadata["shape"]))
    if kind in {"tuple", "list"}:
        opener, closer = ("(", ")") if kind == "tuple" else ("[", "]")
        return f"{opener}{', '.join(_shape_text(item) for item in metadata['items'])}{closer}"
    if kind == "mapping":
        return "{" + ", ".join(_shape_text(item["value"]) for item in metadata["items"]) + "}"
    return str(kind)


def _metric_value(result: MetricResult | None) -> int | float | None:
    if result is None:
        return None
    return result["value"] if result["status"] == "complete" else result["known_value"]


def format_line_str(
    layer: LayerReport,
    col_w: list[int | None] | None = None,
    wrap_mode: str = "mid",
    receptive_field: bool = False,
    effective_rf_stats: bool = False,
) -> list[str]:
    """Format one report layer into summary columns."""
    if col_w is None:
        col_w = [None] * 8
    max_len = col_w[0] + 3 if isinstance(col_w[0], int) else 100
    trainable = int(layer["parameters"]["trainable"])
    frozen = int(layer["parameters"]["frozen"])
    buffers = int(layer["buffers"]["elements"])
    output_shape = _shape_text(layer["output"])
    line_str = [
        format_s(wrap_string(format_name(layer["name"], layer["depth"]), max_len, mode=wrap_mode), col_w[0], col_w[0]),
        format_s(layer["type"], col_w[1], col_w[1]),
        format_s(
            wrap_string(output_shape, col_w[2] if isinstance(col_w[2], int) else 100, mode="end"), col_w[2], col_w[2]
        ),
        format_s(f"{trainable + frozen + buffers:,}", col_w[3], col_w[3]),
        format_s("-" if trainable + frozen == 0 else str(trainable > 0), col_w[4], col_w[4]),
    ]
    if receptive_field:
        receptive = _metric_value(layer["metrics"].get("receptive_field"))
        line_str.append(format_s("?" if receptive is None else f"{receptive:.0f}", col_w[5], col_w[5]))
        if effective_rf_stats:
            stride = _metric_value(layer["metrics"].get("effective_stride"))
            padding = _metric_value(layer["metrics"].get("effective_padding"))
            line_str.extend((
                format_s("?" if stride is None else f"{stride:.0f}", col_w[6], col_w[6]),
                format_s("?" if padding is None else f"{padding:.0f}", col_w[7], col_w[7]),
            ))
    return line_str


def _format_total(name: str, result: MetricResult) -> str:
    value = result["value"] if result["status"] == "complete" else result["known_value"]
    if value is None:
        return f"{name}: unavailable"
    scaled, prefix = unit_scale(float(value))
    qualifier = "" if result["status"] == "complete" else "known "
    return f"{name}: {qualifier}{scaled:.2f} {prefix}{result['unit']}"


def format_info(
    module_info: AnalysisReport,
    wrap_mode: str = "mid",
    receptive_field: bool = False,
    effective_rf_stats: bool = False,
) -> str:
    """Format an analysis report as a human-readable summary."""
    margin = 4
    headers = [
        "Layer",
        "Type",
        "Output Shape",
        "Param #",
        "Trainable",
        "Receptive field",
        "Effective stride",
        "Effective padding",
    ]
    max_w = [27, 20, 25, 15, 9, 15, 16, 17]
    col_w = [len(header) for header in headers]
    for layer in module_info["layers"]:
        col_w = [
            max(width, len(value))
            for width, value in zip(
                col_w,
                format_line_str(layer, receptive_field=True, effective_rf_stats=True),
                strict=True,
            )
        ]
    col_w = list(starmap(min, zip(col_w, max_w, strict=True)))
    if not receptive_field:
        col_w, headers = col_w[:5], headers[:5]
    elif not effective_rf_stats:
        col_w, headers = col_w[:6], headers[:6]

    line_length = sum(col_w) + (len(col_w) - 1) * margin
    thin_line = "_" * line_length
    thick_line = "=" * line_length
    dot_line = "-" * line_length
    margin_str = " " * margin
    lines = [
        thin_line,
        margin_str.join(f"{header:<{width}}" for header, width in zip(headers, col_w, strict=True)),
        thick_line,
    ]
    lines.extend(
        margin_str.join(format_line_str(layer, col_w, wrap_mode, receptive_field, effective_rf_stats))
        for layer in module_info["layers"]
    )
    totals = module_info["totals"]
    lines.extend((
        thick_line,
        f"Trainable params: {int(totals['trainable_parameters']['value'] or 0):,}",
        f"Non-trainable params: {int(totals['frozen_parameters']['value'] or 0):,}",
        f"Total params: {int(totals['parameters']['value'] or 0):,}",
        dot_line,
        f"Model size (params + buffers): {(float(totals['parameter_bytes']['value'] or 0) + float(totals['buffer_bytes']['value'] or 0)) / 1024**2:.2f} Mb",
        dot_line,
        _format_total("Module-formula forward FLOPs", totals["module_flops"]),
        _format_total("Multiply-Accumulations", totals["macs"]),
        _format_total("Direct memory accesses", totals["dmas"]),
    ))
    if "operator_flops" in totals:
        lines.append(_format_total("Operator forward FLOPs", totals["operator_flops"]))
    if module_info["diagnostics"]:
        lines.append(f"Diagnostics: {len(module_info['diagnostics'])} (inspect report['diagnostics'])")
    lines.append(thin_line)
    return "\n".join(lines)


def aggregate_info(info: AnalysisReport, max_depth: int) -> AnalysisReport:
    """Return a report view limited to a maximum module depth."""
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if info["layers"] and not any(layer["depth"] == max_depth for layer in info["layers"]):
        raise ValueError("The `max_depth` argument cannot be higher than module depth.")
    aggregated = deepcopy(info)
    aggregated["layers"] = [layer for layer in aggregated["layers"] if layer["depth"] <= max_depth]
    return aggregated
