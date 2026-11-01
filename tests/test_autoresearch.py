import json
import os
import py_compile
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from torchscan import autoresearch, measure_workload
from torchscan._model_workload import _tensor


def workload(tmp_path, **updates):
    (tmp_path / "models.py").write_text("""import torch
def baseline(): return torch.nn.Linear(4, 2)
def candidate():
    model = baseline()
    torch.randn(100)
    return model
def wrong(): return torch.nn.Linear(4, 3)
def shifted():
    model = baseline()
    model.register_forward_hook(lambda m, a, out: out + 0.09)
    return model
def drifted():
    model = baseline()
    model.register_forward_hook(lambda m, a, out: out + 0.18)
    return model
class Stateful(torch.nn.Module):
    def __init__(self, fails_at):
        super().__init__()
        self.model, self.calls, self.fails_at = baseline(), 0, fails_at
    def forward(self, inputs):
        self.calls += 1
        output = self.model(inputs)
        return output if self.calls < self.fails_at else output + 1
def warmed(): return Stateful(2)
def measured(): return Stateful(3)
""")
    path = tmp_path / "workload.json"
    path.write_text(
        json.dumps({
            "model": "models.py:baseline",
            "args": [{"shape": [1, 4]}],
            "settings": {"trials": 2, "warmup": 0, "min_run_time": 0.001, "min_repeats": 2, "min_improvement": 0.99},
            **updates,
        })
    )
    return path


def test_probe_and_identical_candidate_retain_measurements_and_checks(tmp_path):
    path = workload(tmp_path)
    probe = autoresearch.evaluate_model(path, output=tmp_path / "probe")
    assert probe["decision"] == "measured"
    assert probe["feedback"]["latency_ms"] > 0
    if hasattr(os, "wait4"):
        assert probe["feedback"]["peak_rss_mib"] > 0
    run = tmp_path / "run"
    report = autoresearch.evaluate_model(path, candidate_model="models.py:candidate", output=run)
    assert report["decision"] == "inconclusive", report["reasons"]
    assert len(report["trials"]) == 2
    assert report["feedback"]["after"]["output_errors"]["/"]["max"] == 0
    assert report["feedback"]["after"]["output_check_phase"] == "after_measurement"
    assert (run / "trial-2.html").is_file()
    assert json.loads((run / "trial-2-candidate.result.json").read_text())["totals"]
    assert json.loads((run / "report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        autoresearch.evaluate_model(path, output=run)


@pytest.mark.parametrize(
    ("candidate", "phase"), [("wrong", "first_call"), ("warmed", "warmed"), ("measured", "after_measurement")]
)
def test_bad_candidates_cannot_pass_fresh_or_measured_state(tmp_path, candidate, phase):
    path = workload(tmp_path, check={"absolute_error": {"max": 0}})
    report = autoresearch.evaluate_model(path, candidate_model=f"models.py:{candidate}", output=tmp_path / "run")
    assert report["decision"] == "discard", report["reasons"]
    assert report["feedback"]["after"]["output_check_phase"] == phase
    assert report["feedback"]["after"]["output_errors"]["/"]
    if phase == "after_measurement":
        assert report["trials"][0]["totals"]["latency"]["delta"] is None
    assert report["confirmation"] is None


@pytest.mark.parametrize(
    ("baseline", "candidate", "decision"),
    [
        ("baseline", "shifted", None),
        ("shifted", "drifted", "discard"),
        ("measured", "baseline", "inconclusive"),
    ],
)
def test_fixed_reference_and_incumbent_failure(tmp_path, baseline, candidate, decision):
    path = workload(tmp_path, check={"absolute_error": {"max": 0.1}})
    report = autoresearch.evaluate_model(
        path, baseline_model=f"models.py:{baseline}", candidate_model=f"models.py:{candidate}", output=tmp_path / "run"
    )
    if decision is None:
        assert all(pair["output_check"] == "passed" for pair in report["trials"])
    else:
        assert report["decision"] == decision, report["reasons"]
    side = "before" if baseline == "measured" else "after"
    assert report["feedback"][side]["output_errors"]["/"]["max"] > 0


def test_no_implicit_error_ceiling(tmp_path):
    path = workload(tmp_path)
    report = autoresearch.evaluate_model(path, candidate_model="models.py:shifted", output=tmp_path / "run")
    assert report["output_error_policy"] == "report_only"
    assert all(pair["output_check"] == "passed" for pair in report["trials"])
    assert report["feedback"]["after"]["output_errors"]["/"]["mean"] == pytest.approx(0.09)


def test_saved_inputs_and_integer_descriptors(tmp_path):
    assert _tensor({"shape": [1, 8], "dtype": "bool"}, "cpu").all()
    assert _tensor({"shape": [1, 8], "dtype": "int64", "low": 0, "high": 4}, "cpu").max() < 4
    with pytest.raises(ValueError, match="bounds"):
        _tensor({"shape": [1, 8], "dtype": "int64"}, "cpu")
    torch.save({"args": (torch.ones(1, 4),)}, tmp_path / "inputs.pt")
    path = workload(tmp_path, inputs="inputs.pt")
    contract = json.loads(path.read_text())
    contract.pop("args")
    path.write_text(json.dumps(contract))
    report = autoresearch.evaluate_model(path, output=tmp_path / "run")
    assert report["decision"] == "measured", report["reasons"]
    assert str(tmp_path / "inputs.pt") in report["identities"]
    assert _tensor({"shape": [1, 4]}, "cpu", descriptors=False) == {"shape": [1, 4]}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_fp32_input_contract(dtype):
    with pytest.raises(ValueError, match="FP32"):
        _tensor(torch.ones(1, dtype=dtype), "cpu", descriptors=False)


def test_stale_bytecode_and_mid_run_source_changes_are_not_accepted(tmp_path):
    path = workload(tmp_path)
    source = tmp_path / "models.py"
    (tmp_path / "reference.py").write_bytes(source.read_bytes())
    py_compile.compile(str(source), doraise=True)
    stamp = source.stat()
    hashes = autoresearch._identities([source])
    source.write_text(source.read_text().replace("Linear(4, 2)", "Linear(4, 3)"))
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="changed"):
        autoresearch._verify_files(hashes)
    path.write_text(json.dumps(json.loads(path.read_text()) | {"model": "reference.py:baseline"}))
    report = autoresearch.evaluate_model(path, candidate_model="models:baseline", output=tmp_path / "run")
    assert report["decision"] == "discard", report["reasons"]


def test_worker_failure_after_result_publication_is_not_accepted(tmp_path):
    path = workload(tmp_path)
    with (tmp_path / "models.py").open("a") as source:
        source.write("import atexit, os\natexit.register(os._exit, 7)\n")
    report = autoresearch.evaluate_model(path, output=tmp_path / "run")
    assert report["decision"] == "inconclusive"
    assert "status 7" in report["reasons"][0]


def test_timeout_kills_worker_process_group(monkeypatch, tmp_path):
    child = MagicMock(pid=123, returncode=-9)
    child.__enter__.return_value = child
    child.wait.side_effect = [autoresearch.subprocess.TimeoutExpired("worker", 0.01), -9]
    monkeypatch.setattr(autoresearch.subprocess, "Popen", lambda *_args, **_kwargs: child)
    kill = MagicMock()
    monkeypatch.setattr(autoresearch.os, "name", "posix")
    monkeypatch.setattr(autoresearch.os, "killpg", kill)
    with pytest.raises(autoresearch.subprocess.TimeoutExpired):
        autoresearch._run_worker({"experiment": str(tmp_path / "workload.json")}, tmp_path / "worker.json", 0.01)
    kill.assert_called_once_with(123, autoresearch.signal.SIGKILL)
    assert child.wait.call_count == 2


def test_confirmation_failure_overrides_tentative_keep(monkeypatch, tmp_path):
    path = workload(tmp_path)
    measured = dict(
        measure_workload(
            lambda: torch.ones(1) + 1,
            device="cpu",
            inputs=torch.ones(1),
            metrics=("latency", "memory"),
            min_run_time=0.001,
            min_repeats=2,
            print_summary=False,
        )
    )

    def worker(request, artifact, _timeout):
        if request["mode"] == "inspect":
            return {"settings": autoresearch._settings(json.loads(path.read_text())), "files": []}
        if request["mode"] == "reference":
            Path(request["reference_outputs"]).write_bytes(b"reference")
            return {}
        result = deepcopy(measured)
        result.update(
            output_check="failed" if artifact.name == "confirmation-candidate.json" else "passed",
            _process_peak_rss=result["totals"]["process_peak_rss"],
        )
        return result

    monkeypatch.setattr(autoresearch, "_run_worker", worker)
    monkeypatch.setattr(autoresearch, "_decide", lambda _trials, _settings: ("keep", []))
    report = autoresearch.evaluate_model(path, candidate_model="models.py:candidate", output=tmp_path / "run")
    assert report["decision"] == "discard"
    assert report["confirmation"]["output_check"] == "failed"


@pytest.mark.parametrize(
    "settings", [{"objective": "memory"}, {"trials": 1}, {"min_improvement": float("nan")}, {"max_memory_bytes": True}]
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        autoresearch._settings({"settings": settings})


@pytest.mark.parametrize(
    ("after", "label", "memory", "changes", "decision"),
    [
        (0.8, "faster", 90, {}, "keep"),
        (0.97, "faster", 90, {}, "inconclusive"),
        (0.8, "within_variability", 90, {}, "inconclusive"),
        (1.2, "slower", 90, {}, "discard"),
        (0.8, "faster", 101, {}, "discard"),
        (0.8, "faster", None, {}, "inconclusive"),
        (0.8, "faster", 90, {"num_threads": {}}, "inconclusive"),
    ],
)
def test_latency_and_memory_gates(after, label, memory, changes, decision):
    trial = {
        "before": {"totals": {"latency": {"value": 1}}},
        "after": {
            "totals": {"peak_memory": {"status": "complete" if memory is not None else "unavailable", "value": memory}}
        },
        "totals": {"latency": {"delta": after - 1}},
        "latency_change": label,
        "context_changes": changes,
    }
    assert autoresearch._decide([trial, trial], {"min_improvement": 0.03, "max_memory_bytes": 100})[0] == decision
