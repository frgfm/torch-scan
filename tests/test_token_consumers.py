import copy
import json
import re

# Only parse XML emitted by the renderer, never caller-supplied XML.
from xml.etree import ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]

import pytest
import torch
from torch import nn

from torchscan import compare_reports, crawl_module, render_report, utils


def _dependencies(kind="all", **options):
    return {
        "status": "complete",
        "scope": "module_call",
        "method": "torchscan_token_dependency_v1",
        "output": {"sequence_axis": 1, "length": 2},
        "sources": [
            {"arguments": ["query"], "sequence_axis": 1, "length": 2, "relation": {"kind": "same_position"}},
            {"arguments": ["key", "value"], "sequence_axis": 0, "length": 3, "relation": {"kind": kind, **options}},
        ],
        "assumptions": ["Main activation output only; returned attention weights are excluded."],
    }


@pytest.fixture
def report():
    result = crawl_module(nn.Identity(), args=(torch.ones(1, 2, 4),))
    result["layers"][0]["token_dependencies"] = _dependencies()
    return result


def test_token_only_comparison_and_legacy_absence(report):
    view = utils.aggregate_info(report, 0)
    view["layers"][0]["token_dependencies"]["sources"][0]["arguments"].append("other")
    assert report["layers"][0]["token_dependencies"]["sources"][0]["arguments"] == ["query"]
    after = copy.deepcopy(report)
    after["layers"][0]["token_dependencies"] = _dependencies("prefix", first_position=1, limit=2)
    change = json.loads(json.dumps(compare_reports(report, after)))["layers"]["changed"][0]
    assert change["metrics"] == {}
    assert change["token_dependencies"] == {
        "before": report["layers"][0]["token_dependencies"],
        "after": after["layers"][0]["token_dependencies"],
    }
    del after["layers"][0]["token_dependencies"]
    assert compare_reports(after, after)["layers"]["changed"] == []
    assert compare_reports(report, after)["layers"]["changed"][0]["token_dependencies"]["after"] is None
    assert compare_reports(after, report)["layers"]["changed"][0]["token_dependencies"]["before"] is None
    assert "Token dependencies</h3>" not in render_report(after)
    after["layers"][0]["call_index"] = 1
    added = compare_reports(after, report)["layers"]["added"][0]["token_dependencies"]
    assert added == report["layers"][0]["token_dependencies"]
    assert added is not report["layers"][0]["token_dependencies"]


@pytest.mark.parametrize(
    ("kind", "options", "expected"),
    [
        ("all", {}, "local + source"),
        ("prefix", {}, "prefix"),
        ("all", {"limit": 2}, "token subset"),
        ("none", {}, "token-local"),
    ],
)
def test_summary_token_relations_and_unavailable_spatial_fields(report, kind, options, expected):
    report["layers"][0]["token_dependencies"] = _dependencies(kind, **options)
    columns = utils.format_line_str(report["layers"][0], receptive_field=True, effective_rf_stats=True)
    assert expected in columns[5]
    assert columns[6:] == ["?", "?"]
    assert "not graph-wide effective receptive fields" in utils.format_info(report, receptive_field=True)
    assert "Token dependencies are module-local" not in utils.format_info(report)


def test_html_token_details_comparison_and_json(report):
    before = copy.deepcopy(report)
    report["layers"][0]["token_dependencies"] = _dependencies("prefix", first_position=1, limit=2)
    report["layers"][0]["token_dependencies"]["assumptions"].append("<img src=x onerror=alert(1)>")
    html = render_report(report, before=before)
    embedded = json.loads(
        re.search(r'<script id="torchscan-data" type="application/json">(.*?)</script>', html).group(1)
    )
    assert embedded["report"] == json.loads(json.dumps(report))
    assert "prefix through the output position (inclusive), at most 2 tokens, starting at output position 1" in html
    assert "query: same position (token axis 1, length 2)" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "Module-local token dependencies changed." in html
    assert "Before token dependencies" in html
    assert "After token dependencies" in html
    calls = embedded["maps"]["module_flops"]["groups"][0]["nodes"][0]["calls"]
    assert calls[0]["token_dependency_text"].startswith("Module-local token dependencies")
    report["layers"][0]["token_dependencies"] = _dependencies("all", limit=2)
    assert "first 2 source tokens" in render_report(report)


def test_svg_token_details_are_visible_and_escaped():
    module, query = nn.MultiheadAttention(4, 2, batch_first=True), torch.ones(1, 3, 4)
    report = crawl_module(
        module,
        args=(query,) * 3,
        kwargs={"attn_mask": torch.ones(3, 3, dtype=torch.bool).triu(1), "need_weights": False},
    )
    report["layers"][0]["token_dependencies"]["assumptions"].append("<img src=x onerror=alert(1)>")
    svg = render_report(report, format="svg")
    root = ET.fromstring(svg)  # ruff: ignore[suspicious-xml-element-tree-usage]
    visible = " ".join("".join(node.itertext()) for node in root.findall(".//{*}text"))
    assert "Module-local token dependencies" in visible
    assert "query/key/value: prefix through the output position (inclusive)" in visible
    assert "<img src=x onerror=alert(1)>" not in svg
    assert not root.findall(".//{*}img")


def test_native_transformer_html_combines_token_dependencies_and_ownership():
    module = nn.Transformer(
        d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=8, dropout=0, batch_first=True
    )
    report = crawl_module(
        module,
        args=(torch.ones(1, 3, 4), torch.ones(1, 2, 4)),
        kwargs={"tgt_mask": torch.ones(2, 2, dtype=torch.bool).triu(1), "tgt_is_causal": True},
    )
    html = render_report(report, metric="macs")
    data = json.loads(re.search(r'<script id="torchscan-data" type="application/json">(.*?)</script>', html).group(1))
    owner = report["layers"][0]
    assert owner["token_dependencies"]["scope"] == "module_call"
    assert [source["relation"]["kind"] for source in owner["token_dependencies"]["sources"]] == ["prefix", "all"]
    assert "tgt: prefix through the output position (inclusive)" in html
    assert "src: all 3 source tokens" in html
    for metric in ("module_flops", "macs", "dmas", "parameters"):
        groups = data["maps"][metric]["groups"]
        assert sum(group["known_total"] for group in groups) == report["totals"][metric]["value"]
        if metric == "parameters":
            assert all(node["coverage"] is None for group in groups for node in group["nodes"])
        else:
            assert owner["metric_ownership"][metric] == "subtree"
            encoder = next(node for group in groups for node in group["nodes"] if node["path"] == "encoder")
            assert encoder["coverage"] == "included in (root) · call #0 inclusive estimate"
    assert "Covered metrics" in html


def test_unavailable_dependencies_without_output(report):
    report["layers"][0]["token_dependencies"] = {
        "status": "unavailable",
        "scope": "module_call",
        "method": "torchscan_token_dependency_v1",
        "assumptions": ["This mask's token relations are unsupported."],
    }
    assert "Module-local token dependencies: unavailable." in render_report(report)
    columns = utils.format_line_str(report["layers"][0], receptive_field=True, effective_rf_stats=True)
    assert columns[5:] == ["?", "?", "?"]


@pytest.mark.parametrize(
    "change",
    [
        lambda dep: dep.update(status="partial"),
        lambda dep: dep.update(scope="graph"),
        lambda dep: dep.update(assumptions="text"),
        lambda dep: dep.pop("output"),
        lambda dep: dep.update(sources={}),
        lambda dep: dep["output"].update(sequence_axis=True),
        lambda dep: dep["sources"][0].update(arguments=[]),
        lambda dep: dep["sources"][0].update(arguments=[7]),
        lambda dep: dep["sources"][0]["relation"].update(kind="convolution"),
        lambda dep: dep["sources"][0]["relation"].update(first_position=-1),
        lambda dep: dep["sources"][0]["relation"].update(limit=1.5),
    ],
)
def test_renderer_rejects_malformed_optional_dependencies(report, change):
    change(report["layers"][0]["token_dependencies"])
    with pytest.raises(ValueError, match="token_dependencies"):
        render_report(report)
