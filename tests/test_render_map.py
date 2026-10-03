import copy

import pytest
from torch import nn

from torchscan import compare_reports, crawl_module, metric_result
from torchscan._render_map import build_maps


@pytest.fixture
def report():
    return crawl_module(nn.Sequential(nn.Linear(4, 8, bias=False), nn.Identity(), nn.Linear(8, 2, bias=False)), (4,))


def _node(group, path):
    return next(node for node in group["nodes"] if node["path"] == path)


def _metric(value, *, status="complete", method="torchscan_module_formula", unit="FLOPs", scope="module_call"):
    return metric_result(
        status=status,
        value=value,
        known_value=value,
        method=method,
        unit=unit,
        scope=scope,
    )


def test_inclusive_operator_counts_do_not_affect_any_map(report):
    before = build_maps(report)
    report["operator_flops"]["by_module"] = {"Container": 1000000, "Container.Child": 500000}
    report["operator_flops"]["by_operator"] = {"aten.mm": 1000000}
    assert build_maps(report) == before


def test_partial_zero_unavailable_and_complete_zero_keep_distinct_rails(report):
    for layer, result in zip(
        report["layers"][1:],
        [_metric(0, status="partial"), _metric(None, status="unavailable"), _metric(0)],
        strict=True,
    ):
        layer["metrics"]["module_flops"] = result
    group = build_maps(report)["module_flops"]["groups"][0]
    assert group["scale"] == "structure"
    partial, unavailable, zero = [_node(group, path) for path in ("0", "1", "2")]
    assert partial["known"] == 0
    assert partial["status"] == "partial"
    assert partial["rail"] == ["unknown"]
    assert unavailable["known"] is None
    assert unavailable["status"] == "unavailable"
    assert unavailable["rail"] == ["unknown"]
    assert zero["status"] == "complete"
    assert zero["rail"] == ["zero"]
    assert all(node["width"] > 0 for node in (partial, unavailable, zero))
    assert group["known_total"] == 0
    assert _node(group, "")["direct"] is None
    assert _node(group, "")["status"] == "partial"
    assert _node(group, "")["subtotal"]["method"] == "derived_recorded_subtotal"
    assert _node(group, "")["subtotal"]["scope"] == "observed_subtree"


def test_small_costs_appear_in_rail_and_direct_plus_descendant_segments_do_not_overlap(report):
    report["layers"][0]["metrics"]["module_flops"] = _metric(4)
    report["layers"][1]["metrics"]["module_flops"] = _metric(1000)
    report["layers"][3]["metrics"]["module_flops"] = _metric(1)
    group = build_maps(report)["module_flops"]["groups"][0]
    root, first, last = [_node(group, path) for path in ("", "0", "2")]
    assert group["layout_total"] == 1005
    assert root["direct_width"] == pytest.approx(4 / 1005)
    assert first["x"] == pytest.approx(root["direct_x"] + root["direct_width"])
    assert last["rail"] == ["small"]
    assert root["width"] == pytest.approx(
        root["direct_width"] + sum(_node(group, path)["width"] for path in ("0", "1", "2"))
    )
    assert last["x"] + last["width"] == pytest.approx(1)


@pytest.mark.parametrize("field", ["method", "unit", "scope"])
def test_incompatible_measurements_have_separate_scales(report, field):
    report["layers"][1]["metrics"]["module_flops"][field] = "other"
    groups = build_maps(report)["module_flops"]["groups"]
    assert len(groups) == 2
    assert sorted(group["known_total"] for group in groups) == [30, 56]
    assert all(group["nodes"][0]["width"] == 1 for group in groups)
    assert sum(group["known_total"] for group in groups) == 86  # No individual map mixes scales.


@pytest.mark.parametrize("missing_leaf", [False, True])
def test_user_method_cannot_collide_with_synthetic_group_identity(report, missing_leaf):
    for layer in report["layers"][1:]:
        layer["metrics"]["module_flops"]["method"] = "no_recorded_measurement"
    if missing_leaf:
        del report["layers"][2]["metrics"]["module_flops"]
    after = copy.deepcopy(report)
    after["layers"][1]["metrics"]["module_flops"] = _metric(60, method="no_recorded_measurement")
    groups = build_maps(after, before=report, comparison=(compare_reports(report, after), []))["module_flops"]["groups"]
    recorded = next(group for group in groups if group["kind"] == "recorded")
    assert recorded["method"] == "no_recorded_measurement"
    assert recorded["known_total"] == 90
    assert recorded["scale"] == "known"
    assert _node(recorded, "0")["delta"] == 4
    assert _node(recorded, "")["delta"] == 4
    if missing_leaf:
        assert len(groups) == 2
        synthetic = next(group for group in groups if group["kind"] == "unrecorded")
        assert synthetic["method"] == recorded["method"]
        assert synthetic["id"] != recorded["id"]
        assert synthetic["known_total"] is None
        assert _node(synthetic, "1")["rail"] == ["unknown"]
        assert all(node["delta"] is None for node in synthetic["nodes"])
    else:
        assert len(groups) == 1


def test_comparison_uses_union_max_geometry_and_only_complete_comparable_deltas(report):
    before, after = copy.deepcopy(report), copy.deepcopy(report)
    for snapshot, values in ((before, (100, 50)), (after, (10, 90))):
        for layer, value in zip((snapshot["layers"][1], snapshot["layers"][3]), values, strict=True):
            layer["metrics"]["module_flops"] = _metric(value)
    comparison = (compare_reports(before, after), [])
    maps = build_maps(after, before=before, comparison=comparison)
    group = maps["module_flops"]["groups"][0]
    a, b, root = [_node(group, path) for path in ("0", "2", "")]
    assert group["layout_total"] == 190
    assert a["width"] == pytest.approx(100 / 190)
    assert b["x"] == pytest.approx(100 / 190)
    assert a["before_known"] == 100
    assert a["known"] == 10
    assert a["delta"] == -90
    assert b["delta"] == 40
    assert root["delta"] == -50
    assert all(node["delta"] is None for node in maps["parameters"]["groups"][0]["nodes"])
    assert maps["parameters"]["comparison_delta"]["delta"] == 0
    assert all(node["delta"] is None for node in build_maps(after, before=before)["module_flops"]["groups"][0]["nodes"])


def test_comparison_partial_and_context_gates_withhold_deltas(report):
    before = copy.deepcopy(report)
    before["layers"][1]["metrics"]["module_flops"] = _metric(0, status="partial")
    group = build_maps(report, before=before, comparison=(compare_reports(before, report), []))["module_flops"][
        "groups"
    ][0]
    assert _node(group, "0")["delta"] is None
    assert _node(group, "")["delta"] is None
    assert _node(group, "2")["delta"] == 0
    gated = build_maps(report, before=report, comparison=(compare_reports(report, report), ["input metadata differs"]))
    assert all(node["delta"] is None for node in gated["module_flops"]["groups"][0]["nodes"])
    assert "input metadata differs" in _node(gated["module_flops"]["groups"][0], "0")["delta_reason"]


def test_added_and_removed_calls_are_visible_without_invented_deltas(report):
    before = copy.deepcopy(report)
    before["layers"][1]["path"] = "old.linear"
    group = build_maps(report, before=before, comparison=(compare_reports(before, report), []))["module_flops"][
        "groups"
    ][0]
    assert _node(group, "old")["change"] == "removed"
    assert _node(group, "old.linear")["change"] == "removed"
    assert _node(group, "0")["change"] == "added"
    assert _node(group, "old.linear")["delta"] is None
    assert _node(group, "0")["delta"] is None
    assert _node(group, "")["delta"] is None
    assert _node(group, "2")["delta"] == 0
    assert _node(group, "old.linear")["width"] > 0


@pytest.mark.parametrize("before", [False, True])
def test_negative_additive_costs_rejected(report, before):
    negative = copy.deepcopy(report)
    negative["layers"][1]["metrics"]["module_flops"] = _metric(-1)
    with pytest.raises(ValueError, match="negative additive cost"):
        build_maps(report, before=negative) if before else build_maps(negative)


def test_preorder_geometry_handles_punctuation_in_valid_module_names(report):
    report["layers"][1]["path"] = "a.child"
    report["layers"][2]["path"] = "a-child"
    report["layers"][3]["path"] = "a!child"
    group = build_maps(report)["module_flops"]["groups"][0]
    positions = {node["id"]: index for index, node in enumerate(group["nodes"])}
    assert all(node["parent"] is None or positions[node["parent"]] < positions[node["id"]] for node in group["nodes"])
    for node in group["nodes"]:
        assert 0 <= node["x"] <= 1
        assert 0 <= node["width"] <= 1


def test_nested_failures_propagate_first_reason_without_blocking_complete_siblings():
    class Punctuation(nn.Module):
        def __init__(self):
            super().__init__()
            self.add_module("a!lane", nn.Sequential(nn.Linear(4, 4, bias=False), nn.Identity()))
            self.add_module("a-lane", nn.Sequential(nn.Linear(4, 4, bias=False), nn.Identity()))

        def forward(self, inputs):
            return getattr(self, "a-lane")(getattr(self, "a!lane")(inputs))

    before = crawl_module(Punctuation(), (4,))
    after = copy.deepcopy(before)
    first = next(layer for layer in before["layers"] if layer["path"] == "a!lane.0")
    first["metrics"]["module_flops"] = _metric(0, status="partial")
    removed = next(layer for layer in after["layers"] if layer["path"] == "a-lane.0")
    removed["path"] = "a-lane.added"
    group = build_maps(after, before=before, comparison=(compare_reports(before, after), []))["module_flops"]["groups"][
        0
    ]
    assert _node(group, "a!lane")["delta"] is None
    assert _node(group, "a-lane")["delta"] is None
    assert _node(group, "")["delta"] is None
    assert _node(group, "")["delta_reason"] == "Two comparable complete measurements are required"
    assert _node(group, "a!lane")["delta_reason"] == _node(group, "")["delta_reason"]
    assert "Added, removed" in _node(group, "a-lane")["delta_reason"]
    assert _node(group, "a!lane.1")["delta"] == 0
    assert _node(group, "a-lane.1")["delta"] == 0
    assert _node(group, "a-lane.added")["delta"] is None
