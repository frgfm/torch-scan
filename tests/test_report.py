import json
from typing import Any, cast

import pytest
import torch
from torch import nn

from torchscan import IncompleteAnalysisError, crawl_module, summary
from torchscan.report import metric_result

STATUSES = {"complete", "partial", "unavailable"}


def _metric_results(report):
    for layer in report["layers"]:
        yield from layer["metrics"].values()
    yield from report["totals"].values()


def _assert_metric_result(result):
    assert result["status"] in STATUSES
    assert "value" in result
    assert "known_value" in result
    if result["status"] == "complete":
        assert result["value"] is not None
    else:
        assert result["value"] is None


def test_report_contract_is_json_serializable_and_deterministic():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Identity()).eval()
    input_t = torch.full((2, 4), 12_345.25)

    first = crawl_module(model, args=(input_t,))
    second = crawl_module(model, args=(input_t,))

    assert first["schema_version"] == 1
    assert {"context", "inputs", "layers", "totals", "diagnostics"} <= first.keys()
    assert first["context"]
    assert first["inputs"]
    assert first["layers"]
    assert all(isinstance(layer["path"], str) for layer in first["layers"])
    assert all(isinstance(layer["call_index"], int) for layer in first["layers"])
    assert all(layer["metrics"] for layer in first["layers"])
    assert all(diagnostic["severity"] for diagnostic in first["diagnostics"])
    assert all(diagnostic["code"] for diagnostic in first["diagnostics"])
    for result in _metric_results(first):
        _assert_metric_result(result)

    first_json = json.dumps(first, sort_keys=True)
    assert first_json == json.dumps(second, sort_keys=True)
    assert "12345.25" not in first_json


def test_summary_returns_the_analysis_report(capsys):
    model = nn.Linear(4, 2).eval()
    input_t = torch.ones(2, 4)

    expected = crawl_module(model, args=(input_t,))
    actual = summary(model, args=(input_t,))

    assert actual == expected
    assert capsys.readouterr().out


def test_reused_module_has_stable_path_and_distinct_call_indexes():
    class ReusedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.ReLU()

        def forward(self, input_t):
            return self.shared(self.shared(input_t))

    model = ReusedModule().eval()

    first = crawl_module(model, args=(torch.randn(2, 4),))
    second = crawl_module(model, args=(torch.randn(2, 4),))
    first_calls = [(layer["path"], layer["call_index"]) for layer in first["layers"] if layer["path"] == "shared"]
    second_calls = [(layer["path"], layer["call_index"]) for layer in second["layers"] if layer["path"] == "shared"]

    assert first_calls == second_calls
    assert len(first_calls) == 2
    assert len({call_index for _, call_index in first_calls}) == 2


def test_true_zero_is_complete_but_unsupported_work_is_partial():
    class UnsupportedIdentity(nn.Module):
        def forward(self, input_t):
            return input_t

    input_t = torch.randn(2, 4)

    identity_report = crawl_module(nn.Identity(), args=(input_t,))
    assert any(result["status"] == "complete" and result["value"] == 0 for result in _metric_results(identity_report))

    unsupported_report = crawl_module(UnsupportedIdentity(), args=(input_t,))
    partial_results = [result for result in _metric_results(unsupported_report) if result["status"] == "partial"]
    assert partial_results
    assert all(result["value"] is None for result in partial_results)
    assert all(result["known_value"] is not None for result in partial_results)
    assert unsupported_report["diagnostics"]


def test_strict_mode_rejects_incomplete_analysis():
    class UnsupportedIdentity(nn.Module):
        def forward(self, input_t):
            return input_t

    with pytest.raises(IncompleteAnalysisError):
        crawl_module(UnsupportedIdentity(), args=(torch.randn(2, 4),), strict=True)


def test_metric_result_rejects_invalid_status():
    with pytest.raises(ValueError, match="status must be"):
        metric_result(status=cast("Any", "bogus"), unit="FLOPs", scope="forward", method="test")
