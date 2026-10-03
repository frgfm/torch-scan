# Offline module cost explorer

`render_report` turns an existing schema-v1 `AnalysisReport` into self-contained HTML or standalone SVG.
It does not execute the model or change the report. No server, external scripts, fonts, CDN, or extra dependencies are needed.

```python
from pathlib import Path
import webbrowser
from torch import nn
from torchscan import crawl_module, render_report

report = crawl_module(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)), (4,))
path = Path("model-report.html").resolve()
path.write_text(render_report(report), encoding="utf-8")
webbrowser.open(path.as_uri())  # Or open the saved file manually.
Path("model-report.svg").write_text(render_report(report, format="svg", metric="parameters"), encoding="utf-8")
```

```python
render_report(report, *, format="html", before=None,
              title="TorchScan analysis", metric="module_flops") -> str
```

`format` accepts `"html"` or `"svg"`. `metric` accepts `"module_flops"`, `"macs"`, `"dmas"`, `"parameters"`, or
`"parameter_bytes"` and selects the initial HTML view or SVG map. The renderer validates schema versions, formats/views,
JSON/text, finite numbers, measurement states, call identities, operator totals, and device/dtype context lists.
Invalid inputs raise `ValueError`.

## Read the map

Nested rectangles show the **observed module hierarchy**, not a computational graph. Rows show module depth;
each child sits below its parent. Width shows recorded additive contributions for the selected metric. Parent
widths are **derived recorded subtotals**, not new measurements. A missing container formula stays unavailable in
the evidence without implying missing child work. Each compatible method, unit, and scope has a separate map.
Model totals remain authoritative. Operator counts are separate: upstream module counts include child work and
must not be summed or joined to additive layer contributions.

| Status | Meaning |
| --- | --- |
| Complete | Full value for the stated method. A complete zero is valid. |
| Partial | Only a lower bound is known. Even a zero lower bound leaves the full cost unknown. |
| Unavailable | No measurement was produced. |

Positive partial bounds size only the recorded segment, not a fraction of full model cost. When no positive cost is
known, a neutral map shows structure without a numeric scale. The **unscaled status rail** keeps unknown, tiny, and
complete-zero contributions selectable. Words, patterns, and outlines supplement color. Supporting rankings include
complete calls only; bars compare with the largest complete call in their method group, not a coverage percentage.

Select a module for shapes, individual calls, methods, diagnostics, and linked evidence. Repeated calls share one
block with a call-count marker; their compute accumulates. The schema cannot connect a child invocation to a specific
parent invocation. Parameter views show **first-attributed unique storage**: shared tensors count once in execution
order. Later calls can have zero new attribution without being parameter-free. Uncalled parameters can appear only
in model totals; the schema does not identify all shared owners or aliases.

Suggestions link **recorded facts** to **experiments to try**. Check output quality and benchmark the intended workload
on target hardware. FLOPs do not establish latency. Static parameter/buffer bytes do not establish measured peak memory.

## Controls

Tab reaches native view controls, map/evidence links, and hierarchy summaries. Radio arrows change the metric;
Enter activates links, and Enter/Space expands a summary. With a map rectangle focused, Up/Down moves through visible
modules, Left selects the parent, Right expands and selects a visible child, Home/End selects the first/last module,
and Enter/Space selects its evidence. Buttons collapse, zoom, select the parent, or reset all groups in the current view.
Status cards stay selectable when branches collapse. On narrow screens, the inspector stacks below the map.

All contents work offline. With JavaScript disabled, native views, maps, hierarchy, and call evidence remain available;
expand linked details manually. SVG is a static, searchable snapshot with internal evidence links, without live
selection, view switching, or collapse. Large HTML reports are not virtualized and can require scrolling.

## Before / after

```python
before = crawl_module(nn.Linear(4, 8, bias=False), (4,))
after = crawl_module(nn.Linear(4, 4, bias=False), (4,))
Path("comparison.html").write_text(render_report(after, before=before), encoding="utf-8")
```

The renderer uses `compare_reports(before, report)` to compare totals and match calls by path and call index.
Incompatible schema versions, methods, units, or scopes raise `ValueError`. Deltas are after minus before and require
**two complete comparable measurements**. Incomplete or missing results have unknown deltas; added/removed calls
have snapshots without numeric deltas. Both reports share fixed hierarchy geometry and a common bar scale.

Execution deltas are also withheld when inputs, execution/analysis modes, software versions (TorchScan/PyTorch/Python),
devices, or dtypes differ or are missing. Absent `analysis_mode` means `full`. Complete compatible storage totals can
be compared independently of execution inputs. Per-module parameter deltas are withheld because first attribution
can move without changing shared storage. Both input/measurement contexts are shown. This presentation rule does not
change `compare_reports`. Matching context still does not establish equal latency or peak memory: the schema lacks
those benchmarks, hardware identity, model revision, and full custom-formula provenance.

## Examples and validation

Run `python scripts/visual_report.py` from the repository root. Open `examples/visual-report/explorer.html` or `.svg`
for a locally initialized CPU model with nested convolution blocks, repeated calls, and a custom `Sine` probe with
unknown full cost. `explorer-comparison` compares channel widths 8 and 4 on the same input `[1, 3, 16, 16]`.
`explorer-complete`, `explorer-slim`, `shared`, and `structure` cover complete module estimates, shared weights, and
unrequested compute. Each model has source JSON. Generated reports are ignored by Git; no weights are downloaded.
PyTorch may lack an `AvgPool2d` operator formula, leaving operator counts partial even when module estimates are complete.

```shell
pytest tests/test_render.py tests/test_render_map.py
PYTHON=.venv/bin/python CHROMIUM_PATH=/usr/bin/chromium node tests/render_browser.cjs
```

Playwright and Chromium are optional development tools. Tests cover geometry, sharing, status/comparison rules,
keyboard selection, collapse/reset, parent zoom, mobile layout, evidence focus, zero network requests, hostile
HTML/SVG content, and JavaScript-disabled controls.

## Scope and security

Untrusted text/attributes are escaped; anchors use generated IDs. Embedded JSON escapes script delimiters,
ampersands, and Unicode line separators. HTML has a restrictive Content Security Policy for its fixed, hash-authorized
script and inline styles. SVG has no scripts or external references. Inspect supplied metadata before sharing.
Only schema-v1 analysis reports are supported, not standalone `FlopReport`, `ReportDiff`, peak-memory reports,
or legacy summaries. The renderer does not reconstruct topology, infer coverage, benchmark runtime, or claim
that an experiment preserves accuracy or improves performance.
