import json
from copy import deepcopy
from types import ModuleType
from unittest.mock import Mock

import pytest

from torchscan import _experiment_decision, autoresearch


def experiment_files(tmp_path, *, scale=1, check="valid"):
    experiment = tmp_path / "experiment.py"
    baseline = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    reference = tmp_path / "reference.py"
    baseline.write_text("SCALE = 1\n", encoding="utf-8")
    candidate.write_text(f"SCALE = {scale}\n", encoding="utf-8")
    reference.write_text("SCALE = 1\n", encoding="utf-8")
    experiment.write_text(
        f"""
import os
from pathlib import Path
import torch
from reference import SCALE as REFERENCE_SCALE

SETTINGS = {{
    "device": "cpu",
    "objective": "latency",
    "min_improvement": 0.99,
    "max_memory_bytes": 100_000_000,
    "trials": 2,
    "timeout": 60,
    "warmup": 0,
    "min_run_time": 0.001,
    "min_repeats": 2,
    "workload_id": "fixed-test-input-and-weights-v1",
}}
FILES = ["reference.py"]
EVENTS = Path(__file__).with_name("events.log")

def build(implementation):
    value = torch.ones(1, 4)
    with EVENTS.open("a", encoding="utf-8") as events:
        events.write(f"measure:{{os.getpid()}}\\n")

    def workload():
        return value * implementation.SCALE

    return workload, value

def check(implementation):
    with EVENTS.open("a", encoding="utf-8") as events:
        events.write(f"check:{{os.getpid()}}\\n")
    torch.ones(1024, 1024)  # Check allocations must not enter the measurement worker.
    {"return False" if check == "false" else "return 'pass'" if check == "invalid" else "return implementation.SCALE == REFERENCE_SCALE"}
""",
        encoding="utf-8",
    )
    return experiment, baseline, candidate, reference


def test_candidate_check_failure_returns_discard_without_running_measurements(tmp_path):
    experiment, baseline, candidate, _ = experiment_files(tmp_path, scale=2)

    report = autoresearch.evaluate_experiment(experiment, baseline, candidate, output=tmp_path / "evidence")

    assert report["decision"] == "discard"
    assert report["checks"]["initial"] == {"baseline": "passed", "candidate": "failed"}
    assert report["trials"] == []
    assert report["confirmation"] is None
    assert not list((tmp_path / "evidence").glob("trial-*.html"))
    saved = json.loads((tmp_path / "evidence" / "report.json").read_text(encoding="utf-8"))
    assert saved == report


def test_identical_candidate_stays_unpromoted_and_check_workers_are_isolated(tmp_path):
    experiment, baseline, candidate, _ = experiment_files(tmp_path)
    candidate.write_bytes(baseline.read_bytes())

    report = autoresearch.evaluate_experiment(experiment, baseline, candidate, output=tmp_path / "evidence")

    assert report["decision"] != "keep"
    assert len(report["trials"]) == 2
    assert all(trial["output_check"] == "passed" for trial in report["trials"])
    assert all((tmp_path / "evidence" / f"trial-{index}.html").is_file() for index in (1, 2))
    events = [line.split(":") for line in (tmp_path / "events.log").read_text(encoding="utf-8").splitlines()]
    checks = {pid for kind, pid in events if kind == "check"}
    measurements = {pid for kind, pid in events if kind == "measure"}
    assert checks
    assert measurements
    assert checks.isdisjoint(measurements)
    assert json.loads((tmp_path / "evidence" / "report.json").read_text(encoding="utf-8")) == report


def test_experiment_settings_reject_unknown_nonfinite_and_unused_limits():
    experiment = ModuleType("experiment")
    experiment.build = lambda _implementation: None
    experiment.check = lambda _implementation: None
    experiment.SETTINGS = {
        "device": "cpu",
        "objective": "latency",
        "min_improvement": 0.1,
        "max_memory_bytes": 100,
        "workload_id": "test",
    }

    for invalid in (
        {"unknown": True},
        {"min_improvement": float("nan")},
        {"max_memory_bytes": 100, "max_latency_seconds": 1},
        {"max_memory_bytes": True},
    ):
        experiment.SETTINGS.update(invalid)
        with pytest.raises(ValueError):
            autoresearch._settings(experiment)
        for key in invalid:
            experiment.SETTINGS.pop(key)


def test_unexpected_quality_check_return_is_inconclusive(tmp_path):
    experiment, baseline, candidate, _ = experiment_files(tmp_path, check="invalid")

    report = autoresearch.evaluate_experiment(experiment, baseline, candidate, output=tmp_path / "evidence")

    assert report["decision"] == "inconclusive"
    assert "check must return" in report["reasons"][0]
    assert report["trials"] == []


def test_confirmation_failure_changes_tentative_keep_to_discard(monkeypatch, tmp_path):
    experiment, baseline, candidate, _ = experiment_files(tmp_path)
    experiment_module = ModuleType("experiment")
    experiment_module.SETTINGS = {
        "device": "cpu",
        "objective": "latency",
        "min_improvement": 0.1,
        "max_memory_bytes": 10_000,
        "workload_id": "test",
        "trials": 2,
        "timeout": 1,
    }
    experiment.write_text(
        "SETTINGS = " + repr(experiment_module.SETTINGS) + "\n"
        "def build(implementation): return None, {'kind': 'tensor', 'shape': [1], "
        "'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False}\n"
        "def check(implementation): return True\n",
        encoding="utf-8",
    )

    import torch

    from torchscan import measure_latency

    tensor = torch.ones(1)
    before = measure_latency(lambda: tensor + 1, device="cpu", inputs=tensor, min_run_time=0.001, min_repeats=2)
    after = deepcopy(before)
    monkeypatch.setattr(_experiment_decision, "decide", lambda _trials, _settings: ("keep", ["tentative"]))
    requests = []

    def worker(request, path, _timeout):
        requests.append(request)
        if request["mode"] == "inspect":
            return {"settings": experiment_module.SETTINGS, "files": []}
        if request["mode"] == "check":
            return {
                "output_check": "failed"
                if path.name.startswith("confirmation") and request["implementation"] == str(candidate)
                else "passed"
            }
        return before if request["implementation"] == str(baseline) else after

    monkeypatch.setattr(autoresearch, "_run_worker", worker)
    report = autoresearch.evaluate_experiment(experiment, baseline, candidate, output=tmp_path / "evidence")

    assert report["decision"] == "discard"
    assert report["confirmation"]["output_check"] == "failed"
    assert report["confirmation"]["totals"]["latency"]["delta"] is None
    assert len([request for request in requests if request["mode"] == "measure"]) == 6


@pytest.mark.parametrize(("failed_role", "decision"), [("candidate", "discard"), ("baseline", "inconclusive")])
def test_trial_check_failure_is_saved_with_deltas_withheld(monkeypatch, failed_role, decision, tmp_path):
    experiment, baseline, candidate, _ = experiment_files(tmp_path)
    experiment_module = ModuleType("experiment")
    experiment_module.SETTINGS = {
        "device": "cpu",
        "objective": "latency",
        "min_improvement": 0.1,
        "max_memory_bytes": 10_000,
        "workload_id": "test",
        "trials": 3,
        "timeout": 1,
    }
    experiment.write_text(
        "SETTINGS = " + repr(experiment_module.SETTINGS) + "\n"
        "def build(implementation): return None, {'kind': 'tensor', 'shape': [1], "
        "'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False}\n"
        "def check(implementation): return True\n",
        encoding="utf-8",
    )
    import torch

    from torchscan import measure_latency

    tensor = torch.ones(1)
    measurement = measure_latency(lambda: tensor + 1, device="cpu", inputs=tensor, min_run_time=0.001, min_repeats=2)
    monkeypatch.setattr(_experiment_decision, "decide", lambda _trials, _settings: ("keep", ["tentative"]))

    def worker(request, path, _timeout):
        role = "baseline" if request["implementation"] == str(baseline) else "candidate"
        if request["mode"] == "inspect":
            return {"settings": experiment_module.SETTINGS, "files": []}
        if request["mode"] == "check":
            failed = path.name.startswith("trial-1") and role == failed_role
            return {"output_check": "failed" if failed else "passed"}
        return measurement

    monkeypatch.setattr(autoresearch, "_run_worker", worker)
    report = autoresearch.evaluate_experiment(experiment, baseline, candidate, output=tmp_path / "evidence")

    assert report["decision"] == decision
    assert len(report["trials"]) == 1
    trial = report["trials"][0]
    assert trial["output_check"] == "failed"
    assert trial["totals"]["latency"]["delta"] is None
    assert all(metric["delta"] is None for metric in trial["totals"].values())
    assert report["confirmation"] is None


def test_worker_timeout_terminates_process_group(monkeypatch, tmp_path):
    request = {"experiment": str(tmp_path / "experiment.py")}
    path = tmp_path / "timeout.json"
    child = Mock()
    child.pid = 123
    child.returncode = -9
    child.wait.side_effect = [
        autoresearch.subprocess.TimeoutExpired(["python"], timeout=0.01),
        -9,
    ]
    process = Mock()
    process.__enter__ = Mock(return_value=child)
    process.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(autoresearch.subprocess, "Popen", Mock(return_value=process))
    kill_group = Mock()
    monkeypatch.setattr(autoresearch.os, "name", "posix")
    monkeypatch.setattr(autoresearch.os, "killpg", kill_group)

    with pytest.raises(autoresearch.subprocess.TimeoutExpired):
        autoresearch._run_worker(request, path, 0.01)

    kill_group.assert_called_once_with(123, autoresearch.signal.SIGKILL)
    assert child.wait.call_count == 2


def test_candidate_change_during_evaluation_is_inconclusive(monkeypatch, tmp_path):
    experiment, baseline, candidate, _ = experiment_files(tmp_path)
    experiment_module = ModuleType("experiment")
    experiment_module.SETTINGS = {
        "device": "cpu",
        "objective": "latency",
        "min_improvement": 0.1,
        "max_memory_bytes": 10_000,
        "workload_id": "test",
        "trials": 2,
        "timeout": 1,
    }
    experiment.write_text(
        "SETTINGS = " + repr(experiment_module.SETTINGS) + "\n"
        "def build(implementation): return None, {'kind': 'tensor', 'shape': [1], "
        "'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False}\n"
        "def check(implementation): return True\n",
        encoding="utf-8",
    )
    import torch

    from torchscan import measure_latency

    tensor = torch.ones(1)
    before = measure_latency(lambda: tensor + 1, device="cpu", inputs=tensor, min_run_time=0.001, min_repeats=2)
    after = deepcopy(before)
    monkeypatch.setattr(_experiment_decision, "decide", lambda _trials, _settings: ("keep", ["tentative"]))
    calls = 0

    def worker(request, _path, _timeout):
        nonlocal calls
        if request["mode"] == "inspect":
            return {"settings": experiment_module.SETTINGS, "files": []}
        if request["mode"] == "check":
            return {"output_check": "passed"}
        calls += 1
        if calls == 2:
            candidate.write_text("SCALE = 2\n", encoding="utf-8")
        return before if request["implementation"] == str(baseline) else after

    monkeypatch.setattr(autoresearch, "_run_worker", worker)
    report = autoresearch.evaluate_experiment(experiment, baseline, candidate, output=tmp_path / "evidence")

    assert report["decision"] == "inconclusive"
    assert "changed during evaluation" in report["reasons"][0]


def test_output_directory_is_never_overwritten(tmp_path):
    experiment, baseline, candidate, _ = experiment_files(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        autoresearch.evaluate_experiment(experiment, baseline, candidate, output=output)

    assert marker.read_text(encoding="utf-8") == "existing"
