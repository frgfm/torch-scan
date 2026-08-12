import copy
import json

import pytest

from torchscan.compare import compare_reports


def _metric(status="complete", value=0, known_value=None, **metadata):
    if status == "complete" and known_value is None:
        known_value = value
    return {"status": status, "value": value, "known_value": known_value, **metadata}


def _report(*, version=1, totals=None, layers=None):
    return {"schema_version": version, "totals": totals or {}, "layers": layers or []}


def _layer(path, call_index, **metrics):
    return {"path": path, "call_index": call_index, "metrics": metrics}


def test_compare_reports_numeric_totals_and_deterministic_order():
    before = _report(totals={"parameters": _metric(value=8), "flops": _metric(value=10, unit="count")})
    after = _report(totals={"flops": _metric(value=15, unit="count"), "parameters": _metric(value=8)})

    result = compare_reports(before, after)

    assert list(result["totals"]) == ["flops", "parameters"]
    assert result["totals"]["flops"] == {
        "status": "complete",
        "delta": 5,
        "before": _metric(value=10, unit="count"),
        "after": _metric(value=15, unit="count"),
    }
    assert result["totals"]["parameters"]["delta"] == 0


def test_compare_reports_added_removed_and_reused_layer_calls():
    before = _report(
        layers=[
            _layer("old", 0, flops=_metric(value=3)),
            _layer("shared", 1, flops=_metric(value=10)),
            _layer("shared", 0, flops=_metric(value=5)),
        ]
    )
    after = _report(
        layers=[
            _layer("new", 0, flops=_metric(value=7)),
            _layer("shared", 0, flops=_metric(value=5)),
            _layer("shared", 1, flops=_metric(value=14)),
        ]
    )

    layers = compare_reports(before, after)["layers"]

    assert [(item["path"], item["call_index"]) for item in layers["added"]] == [("new", 0)]
    assert [(item["path"], item["call_index"]) for item in layers["removed"]] == [("old", 0)]
    assert layers["changed"] == [
        {
            "path": "shared",
            "call_index": 1,
            "metrics": {
                "flops": {
                    "status": "complete",
                    "delta": 4,
                    "before": _metric(value=10),
                    "after": _metric(value=14),
                }
            },
        }
    ]


def test_compare_reports_propagates_incomplete_metrics():
    before = _report(
        totals={
            "flops": _metric(value=100),
            "macs": _metric(status="partial", value=None, known_value=80),
        }
    )
    after = _report(
        totals={
            "flops": _metric(status="partial", value=None, known_value=120),
            "macs": _metric(status="unavailable", value=None),
        }
    )

    totals = compare_reports(before, after)["totals"]

    assert totals["flops"]["status"] == "partial"
    assert totals["flops"]["delta"] is None
    assert totals["flops"]["after"]["known_value"] == 120
    assert totals["macs"]["status"] == "unavailable"
    assert totals["macs"]["delta"] is None


def test_compare_reports_missing_metric_is_unavailable():
    result = compare_reports(
        _report(totals={"flops": _metric(value=10)}),
        _report(totals={"parameters": _metric(value=4)}),
    )

    assert result["totals"]["flops"] == {
        "status": "unavailable",
        "delta": None,
        "before": _metric(value=10),
        "after": None,
    }
    assert result["totals"]["parameters"]["status"] == "unavailable"


def test_compare_reports_rejects_schema_mismatch_and_malformed_essentials():
    with pytest.raises(ValueError, match="schema versions differ"):
        compare_reports(_report(version=1), _report(version=2))

    with pytest.raises(TypeError, match=r"before\['totals'\] must be a mapping"):
        compare_reports({"schema_version": 1, "totals": None, "layers": []}, _report())

    duplicate = _report(layers=[_layer("block", 0), _layer("block", 0)])
    with pytest.raises(ValueError, match="duplicate layer call"):
        compare_reports(duplicate, _report())

    with pytest.raises(ValueError, match="unsupported schema version"):
        compare_reports(_report(version=99), _report(version=99))


@pytest.mark.parametrize(
    "metric",
    [
        {"status": "complete", "value": 1, "known_value": None},
        {"status": "partial", "value": 1, "known_value": 1},
        {"status": "partial", "value": None, "known_value": None},
        {"status": "unavailable", "value": 1, "known_value": None},
    ],
)
def test_compare_reports_rejects_invalid_metric_invariants(metric):
    with pytest.raises(ValueError):
        compare_reports(_report(totals={"flops": metric}), _report())


@pytest.mark.parametrize("field", ["unit", "scope", "method"])
def test_compare_reports_rejects_incompatible_metric_metadata(field):
    before = _report(totals={"flops": _metric(**{field: "before"})})
    after = _report(totals={"flops": _metric(**{field: "after"})})

    with pytest.raises(ValueError, match=f"incompatible {field}"):
        compare_reports(before, after)


def test_compare_reports_is_json_serializable_and_does_not_mutate_inputs():
    before = _report(
        totals={"flops": _metric(value=10, method="module")},
        layers=[_layer("block", 0, flops=_metric(value=10))],
    )
    after = _report(
        totals={"flops": _metric(value=15, method="module")},
        layers=[_layer("block", 0, flops=_metric(value=15))],
    )
    expected_before = copy.deepcopy(before)
    expected_after = copy.deepcopy(after)

    result = compare_reports(before, after)
    json.dumps(result)
    result["totals"]["flops"]["before"]["value"] = -1

    assert before == expected_before
    assert after == expected_after
