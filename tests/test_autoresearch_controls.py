import runpy
from pathlib import Path

import pytest
import torch

PACK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/autoresearch/control_pack.py"))


def test_toy_controls_have_exact_and_incorrect_outputs():
    inputs = torch.randn(1, 4096)
    expected = PACK["Redundant"]()(inputs)
    assert torch.equal(expected, torch.nn.ReLU()(inputs))
    assert torch.all((PACK["Wrong"]()(inputs) - expected).abs() > 0.9)


def test_control_release_gates_preserve_incomplete_evidence():
    trial = {"output_check": "passed", "totals": {"latency": {"delta": 0}}}
    reports = {
        f"identical-{index:02}": {
            "decision": "inconclusive",
            "output_error_policy": "enforced",
            "trials": [trial] * 3,
        }
        for index in range(10)
    }
    checks = {"baseline": {"output_check": "passed"}, "candidate": {"output_check": "failed"}}
    reports["incorrect"] = {"decision": "discard", "output_error_policy": "enforced", "initial_checks": checks}
    gates = PACK["release_gates"]
    assert set(gates(reports).values()) == {"passed"}
    reports["identical-09"]["decision"] = "keep"
    assert gates(reports)["zero_false_keeps"] == "failed"
    reports["identical-09"].update(decision="inconclusive", trials=[])
    assert gates(reports)["zero_false_keeps"] == "inconclusive"
    checks["baseline"]["output_check"] = "failed"
    assert gates(reports)["incorrect_output_rejected"] == "inconclusive"
    checks["candidate"]["output_check"] = "passed"
    reports["incorrect"]["decision"] = "inconclusive"
    assert gates(reports)["incorrect_output_rejected"] == "failed"


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
        "model": f"{reference}:Redundant",
        "device": "cpu",
        "threads": 1,
        "seed": 0,
        "args": [{"shape": [1, 4096]}],
        "check": {"absolute_error": {"max": 0}},
    }
    result = PACK["native_timing"](contract, "torch.nn:ReLU")
    assert result["max_absolute_error"] == 0
    assert result["first_call_seconds"] > 0
    assert len(result["raw_times_seconds"]) >= 5
    assert result["median_seconds"] > 0
