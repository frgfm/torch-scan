import copy
import json
import re

import pytest
import torch
from torch import nn

from torchscan import compare_reports, crawl_module, render_report, utils


def _dependencies(kind="all", **relation_options):
    return {
        "status": "complete",
        "scope": "module_call",
        "method": "torchscan_token_dependency_v1",
        "output": {"sequence_axis": 1, "length": 2},
        "sources": [
            {
                "arguments": ["query"],
                "sequence_axis": 1,
                "length": 2,
                "relation": {"kind": "same_position"},
            },
            {
                "arguments": ["key", "value"],
                "sequence_axis": 0,
                "length": 3,
                "relation": {"kind": kind, **relation_options},
            },
        ],
        "assumptions": ["Main activation output only; returned attention weights are excluded."],
    }


@pytest.fixture
def report():
    result = crawl_module(nn.Identity(), args=(torch.ones(1, 2, 4),))
    # Model a saved schema-v1 report with optional module-local token evidence.
    # Production formulas and semantic tests live in the Transformer test files.
    result["layers"][0]["token_dependencies"] = _dependencies()
    return result


def test_dependency_only_change_is_detected_and_serializable(report):
    before = copy.deepcopy(report)
    after = copy.deepcopy(report)
    after["layers"][0]["token_dependencies"] = _dependencies("prefix", first_position=1, limit=2)
    expected_before = copy.deepcopy(before)
    expected_after = copy.deepcopy(after)

    difference = compare_reports(before, after)
    changed = json.loads(json.dumps(difference))["layers"]["changed"]

    assert len(changed) == 1
    assert changed[0]["metrics"] == {}
    assert changed[0]["token_dependencies"] == {
        "before": before["layers"][0]["token_dependencies"],
        "after": after["layers"][0]["token_dependencies"],
    }
    assert "delta" not in changed[0]["token_dependencies"]
    difference["layers"]["changed"][0]["token_dependencies"]["after"]["sources"][0]["arguments"].append("other")
    assert before == expected_before
    assert after == expected_after


def test_older_reports_and_optional_dependency_removal_remain_compatible(report):
    older = copy.deepcopy(report)
    del older["layers"][0]["token_dependencies"]

    assert compare_reports(older, older)["layers"]["changed"] == []
    assert compare_reports(report, report)["layers"]["changed"] == []
    added = compare_reports(older, report)["layers"]["changed"][0]["token_dependencies"]
    removed = compare_reports(report, older)["layers"]["changed"][0]["token_dependencies"]
    assert added == {"before": None, "after": report["layers"][0]["token_dependencies"]}
    assert removed == {"before": report["layers"][0]["token_dependencies"], "after": None}
    assert "Token dependencies</h3>" not in render_report(older)


def test_added_and_removed_snapshots_preserve_optional_dependency_evidence(report):
    after = copy.deepcopy(report)
    after["layers"][0]["path"] = "new"
    layers = compare_reports(report, after)["layers"]

    assert layers["removed"][0]["token_dependencies"] == report["layers"][0]["token_dependencies"]
    assert layers["added"][0]["token_dependencies"] == after["layers"][0]["token_dependencies"]
    layers["removed"][0]["token_dependencies"]["sources"][0]["arguments"].append("other")
    assert report["layers"][0]["token_dependencies"]["sources"][0]["arguments"] == ["query"]


@pytest.mark.parametrize(
    ("kind", "options", "expected"),
    [
        ("all", {}, "local + source"),
        ("prefix", {"first_position": 1}, "prefix"),
        ("all", {"limit": 2}, "token subset"),
        ("none", {}, "token-local"),
    ],
)
def test_summary_shows_token_relations_and_leaves_spatial_fields_unavailable(report, kind, options, expected):
    report["layers"][0]["token_dependencies"] = _dependencies(kind, **options)
    columns = utils.format_line_str(report["layers"][0], receptive_field=True, effective_rf_stats=True)
    formatted = utils.format_info(report, receptive_field=True, effective_rf_stats=True)

    assert expected in columns[5]
    assert columns[6:] == ["?", "?"]
    assert "Token dependencies are module-local" in formatted
    assert "not graph-wide effective receptive fields" in formatted
    assert "Token dependencies are module-local" not in utils.format_info(report)


def test_depth_limited_views_copy_optional_dependencies(report):
    view = utils.aggregate_info(report, 0)
    view["layers"][0]["token_dependencies"]["sources"][0]["arguments"].append("other")

    assert report["layers"][0]["token_dependencies"]["sources"][0]["arguments"] == ["query"]


def test_html_exposes_readable_dependency_metadata_and_preserves_json(report):
    report["layers"][0]["token_dependencies"] = _dependencies("prefix", first_position=1, limit=2)
    report["layers"][0]["token_dependencies"]["assumptions"].append("<img src=x onerror=alert(1)>")
    saved = json.loads(json.dumps(report))

    html = render_report(saved)
    embedded = json.loads(
        re.search(r'<script id="torchscan-data" type="application/json">(.*?)</script>', html).group(1)
    )

    assert embedded["report"] == saved
    assert "prefix through the output position (inclusive), at most 2 tokens, starting at output position 1" in html
    assert "query: same position (token axis 1, length 2)" in html
    assert "Module-local token dependencies" in html
    assert "torchscan_token_dependency_v1" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    calls = embedded["maps"]["module_flops"]["groups"][0]["nodes"][0]["calls"]
    assert calls[0]["token_dependency_text"].startswith("Module-local token dependencies")
    assert "if (call.token_dependency_text) row.append(el('p', call.token_dependency_text, 'muted'));" in html
    assert render_report(saved, format="svg").startswith("<svg")


def test_html_describes_limited_all_relation_and_metadata_only_comparison(report):
    before = copy.deepcopy(report)
    report["layers"][0]["token_dependencies"] = _dependencies("all", limit=2)

    html = render_report(report, before=before)

    assert "first 2 source tokens" in html
    assert "Changed calls (1)" in html
    assert "Module-local token dependencies changed." in html
    assert "Before token dependencies" in html
    assert "After token dependencies" in html


def test_native_transformer_html_combines_token_dependencies_and_metric_ownership():
    module = nn.Transformer(
        d_model=4,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
        batch_first=True,
    )
    report = crawl_module(
        module,
        args=(torch.ones(1, 3, 4), torch.ones(1, 2, 4)),
        kwargs={"tgt_mask": torch.ones(2, 2, dtype=torch.bool).triu(1), "tgt_is_causal": True},
    )
    saved = json.loads(json.dumps(report))
    html = render_report(saved, metric="macs")
    data = json.loads(re.search(r'<script id="torchscan-data" type="application/json">(.*?)</script>', html).group(1))

    assert data["report"] == saved
    owner = saved["layers"][0]
    assert owner["token_dependencies"]["scope"] == "module_call"
    assert [source["relation"]["kind"] for source in owner["token_dependencies"]["sources"]] == ["prefix", "all"]
    assert "tgt: prefix through the output position (inclusive)" in html
    assert "src: all 3 source tokens" in html
    for metric in ("module_flops", "macs", "dmas"):
        assert owner["metric_ownership"][metric] == "subtree"
        groups = data["maps"][metric]["groups"]
        assert sum(group["known_total"] for group in groups) == saved["totals"][metric]["value"]
        encoder = next(node for group in groups for node in group["nodes"] if node["path"] == "encoder")
        assert encoder["coverage"] == "included in (root) · call #0 inclusive estimate"
    assert "Covered metrics" in html
    parameter_groups = data["maps"]["parameters"]["groups"]
    assert sum(group["known_total"] for group in parameter_groups) == saved["totals"]["parameters"]["value"]
    assert all(node["coverage"] is None for group in parameter_groups for node in group["nodes"])


def test_unavailable_dependencies_without_output_are_valid(report):
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
