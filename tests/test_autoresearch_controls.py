import runpy
from copy import deepcopy
from pathlib import Path

import pytest
import torch

PACK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/autoresearch/control_pack.py"))


def test_toy_controls_have_exact_and_incorrect_outputs():
    inputs = torch.randn(1, 4096)
    expected = PACK["baseline"]()(inputs)
    assert torch.equal(expected, PACK["faster"]()(inputs))
    assert torch.all((PACK["Wrong"]()(inputs) - expected).abs() > 0.9)


def control_reports():
    reports = {
        f"identical-{index:02}": {
            "decision": "inconclusive",
            "output_error_policy": "enforced",
            "trials": [{"output_check": "passed"}] * 3,
        }
        for index in range(10)
    }
    reports["incorrect"] = {"decision": "discard", "initial_checks": {"candidate": {"output_check": "failed"}}}
    return reports


def test_control_release_gates_preserve_incomplete_evidence():
    reports = control_reports()
    original = deepcopy(reports)
    assert set(PACK["release_gates"](reports).values()) == {"passed"}
    assert reports == original
    reports["identical-09"]["decision"] = "keep"
    assert PACK["release_gates"](reports)["zero_false_keeps"] == "failed"
    reports["identical-09"].update(decision="inconclusive", trials=[])
    assert PACK["release_gates"](reports)["zero_false_keeps"] == "inconclusive"
    reports["incorrect"] = {"decision": "inconclusive"}
    assert PACK["release_gates"](reports)["incorrect_output_rejected"] == "inconclusive"
    reports["incorrect"] = {"decision": "keep"}
    assert PACK["release_gates"](reports)["incorrect_output_rejected"] == "failed"


def test_control_pack_refuses_checkout_evidence():
    with pytest.raises(ValueError, match="outside the checkout"):
        PACK["run_pack"](Path(__file__).parent / "control-evidence", "cpu")


def test_missing_cuda_retains_unvalidated_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = PACK["run_pack"](tmp_path / "cuda", "cuda")
    assert report["cuda_gate"] == "unvalidated"
    assert (tmp_path / "cuda/summary.json").is_file()


def test_native_control_retains_raw_warmed_and_first_call_timing():
    reference = str(Path(PACK["__file__"]).resolve())
    contract = {
        "model": f"{reference}:baseline",
        "device": "cpu",
        "threads": 1,
        "seed": 0,
        "args": [{"shape": [1, 4096]}],
        "check": {"absolute_error": {"max": 0}},
    }
    result = PACK["native_timing"](contract, f"{reference}:faster")
    assert result["max_absolute_error"] == 0
    assert result["first_call_seconds"] > 0
    assert len(result["raw_times_seconds"]) >= 5
    assert result["median_seconds"] > 0
