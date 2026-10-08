# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
from collections.abc import Mapping
from html import escape
from typing import Any

from ._render_assets import STYLE
from .benchmark_compare import _context_changes, _latency_change, _matching_context, _validate_benchmark
from .compare import _diff_metrics
from .render import _diagnostics, _json_tree, _metadata, _metric_table, _number, _object, _string
from .utils import _display_unit
from .workload import _LABELS, _summary


def _measurements(report: Mapping[str, Any]) -> str:
    diagnostics = report.get("diagnostics", [])
    _diagnostics(diagnostics, "benchmark.diagnostics")
    return (
        f"<pre>{escape(_summary(report))}</pre><details><summary>Exact values, methods, and diagnostics</summary>"
        + _metric_table(report["totals"])
        + "<ul>"
        + "".join(f"<li>{escape(item['message'])}</li>" for item in diagnostics)
        + "</ul></details>"
    )


def _profile(report: Mapping[str, Any]) -> str:
    profile = report.get("profile")
    if profile is None:
        return "<p>No operator profile was supplied. Run profile_workload in a separate diagnostic pass.</p>"
    profile = _object(profile, "profile")
    _json_tree(profile, "profile")
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError("Unsupported profile schema_version.")
    _object(profile.get("context"), "profile.context")
    _diagnostics(profile.get("diagnostics"), "profile.diagnostics")
    if not isinstance(profile.get("operators"), list):
        raise ValueError("Profile operators must be a list.")
    for key in ("device", "torch_version"):
        if profile["context"].get(key) != report["context"].get(key):
            raise ValueError(f"Profile and timing differ: {key}.")
    rows = []
    for row in profile["operators"]:
        row = _object(row, "operator")
        device = row["device_self_seconds"]
        cpu = _number(row["cpu_self_seconds"], "cpu self time") * 1000
        device_text = "Unavailable" if device is None else f"{_number(device, 'device self time') * 1000:.4g}"
        rows.append(
            f"<tr><th scope='row'>{escape(_string(row['operator'], 'operator'))}</th>"
            f"<td>{escape(json.dumps(row['input_shapes']))}</td>"
            f"<td>{_number(row['calls'], 'calls', integer=True)}</td><td>{cpu:.4g}</td><td>{device_text}</td>"
            f"<td>{_number(row['cpu_net_bytes'], 'net bytes')}</td></tr>"
        )
    notes = "".join(f"<li>{escape(item['message'])}</li>" for item in profile["diagnostics"])
    return (
        "<p>Instrumented operator self time and net allocations. These are not clean latency or peak memory.</p>"
        f"<ul>{notes}</ul><div class='table-scroll'><table><caption>Separate operator diagnostic pass</caption>"
        "<thead><tr><th>Operator / runtime event</th><th>Input shapes</th><th>Calls</th><th>CPU self ms</th><th>Device self ms</th><th>CPU net bytes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f"<details><summary>Profiler settings and row limits</summary>{_metadata(profile['context'])}</details>"
    )


def render_benchmark(report: dict[str, Any], *, title: str) -> str:
    _json_tree(report, "report")
    comparison = report.get("report_type") == "benchmark_comparison"
    before = report.get("before") if comparison else None
    after = _validate_benchmark(report.get("after") if comparison else report)
    changes: dict[str, Any] = {}
    if comparison:
        before = _validate_benchmark(before)
        _matching_context(before, after)
        if (
            type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or report.get("output_check") not in {"passed", "failed"}
        ):
            raise ValueError("Invalid benchmark comparison.")
        expected = _diff_metrics(before["totals"], after["totals"])
        if report["output_check"] == "failed":
            for difference in expected.values():
                difference["status"], difference["delta"] = "unavailable", None
        if report["totals"] != expected:
            raise ValueError("Comparison deltas do not match the stored measurements/check status.")
        if report.get("latency_change") != _latency_change(before, after, expected):
            raise ValueError("Latency change does not match the stored measurements/check status.")
        changes = _context_changes(before, after)
        if report.get("context_changes") != changes:
            raise ValueError("Context changes do not match the stored measurements.")
    body = _measurements(after)
    verdict = "Single workload measurement"
    if before is not None:
        verdict = (
            "Output check passed" if report["output_check"] == "passed" else "Output check failed — gains withheld"
        )
        if changes:
            verdict += ". Recorded context changed: " + ", ".join(changes)
        rows = []
        for name, difference in report["totals"].items():
            delta = difference["delta"]
            scale, unit = _display_unit((difference["after"] or difference["before"])["unit"])
            text = "Unavailable" if delta is None else f"{_number(delta, 'delta') * scale:+.5g}"
            rows.append(
                f"<tr><th scope='row'>{escape(_LABELS.get(name, name))}</th><td>{text}</td><td>{escape(unit)}</td></tr>"
            )
        body = (
            "<h3>Baseline</h3>"
            + _measurements(before)
            + "<h3>Candidate</h3>"
            + body
            + "<h3>Measured changes</h3><div class='table-scroll'><table><caption>Candidate minus baseline</caption>"
            + "<thead><tr><th>Metric</th><th>Delta</th><th>Unit</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
            + f"<p>Latency change: {escape(report['latency_change'].replace('_', ' '))}. "
            "IQR is descriptive variability, not a significance test. Task accuracy is unmeasured.</p>"
            + "<details><summary>Changed execution context</summary>"
            + _metadata(changes)
            + "</details>"
        )
    timings = {"baseline": before["measurement"], "candidate": after["measurement"]} if before else after["measurement"]
    profile_heading = "Candidate bottleneck evidence" if comparison else "Bottleneck evidence"
    profiles = f"<section><h2>{profile_heading}</h2>{_profile(after)}</section>"
    if before is not None and before.get("profile") is not None:
        profiles = f"<section><h2>Baseline bottleneck evidence</h2>{_profile(before)}</section>" + profiles
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>{escape(title)}</title><style>{STYLE}"
        "h1{font-size:2rem}header p{max-width:72ch}"
        "</style></head><body><a class='skip' href='#measurements'>Skip to measurements</a><main>"
        f"<header><h1>{escape(title)}</h1><p>{escape(verdict)}</p>"
        "<p>Recorded workload timing and any supplied memory or profiler evidence. "
        "Values retain their methods; unavailable data stays visible.</p></header>"
        f"<section id='measurements'><h2>Measured values</h2>{body}</section>"
        f"{profiles}"
        "<section><h2>Execution context</h2><details><summary>Inputs and hardware</summary>"
        f"{_metadata({'inputs': after['inputs'], 'context': after['context']})}</details>"
        "<details><summary>Raw timing blocks</summary>" + _metadata(timings) + "</details></section>"
        "</main></body></html>\n"
    )
