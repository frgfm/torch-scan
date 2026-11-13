import json
import os
import py_compile
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from torchscan import autoresearch, measure_workload
from torchscan._model_workload import _tensor, model_worker


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
def selective():
    model = baseline()
    model.register_forward_hook(lambda m, a, out: out + a[0][:, :2] * 0.2)
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


def case_manifest(tmp_path, **updates):
    path = workload(tmp_path, **updates)
    contract = json.loads(path.read_text())
    contract.pop("args")
    contract["cases"] = {
        "zeros": {"args": [{"shape": [1, 4], "fill": 0}]},
        "ones": {"args": [{"shape": [1, 4], "fill": 1}]},
    }
    path.write_text(json.dumps(contract))
    return path


@pytest.mark.parametrize("enforced", [True, False])
def test_candidate_passing_one_input_but_failing_another(tmp_path, enforced):
    path = case_manifest(tmp_path, **({"check": {"absolute_error": {"max": 0}}} if enforced else {}))
    torch.save({"args": (torch.ones(1, 4),)}, tmp_path / "inputs.pt")
    contract = json.loads(path.read_text())
    contract["cases"]["ones"] = {"inputs": "inputs.pt"}
    path.write_text(json.dumps(contract))
    report = autoresearch.evaluate_model(path, candidate_model="models.py:selective", output=tmp_path / "run")
    if enforced:
        assert report["decision"] == "discard", report["reasons"]
    assert report["output_error_policy"] == ("enforced" if enforced else "report_only")
    zeros, ones = report["cases"].values()
    assert all(pair["output_check"] == "passed" for pair in zeros["trials"])
    assert zeros["feedback"]["after"]["latency_ms"] > 0
    assert zeros["feedback"]["after"]["pytorch_memory_scope"] == "pytorch_tensor_bytes"
    assert zeros["feedback"]["after"]["output_errors"]["/"]["max"] == 0
    if enforced:
        assert ones["decision"] == "discard"
    else:
        assert len(ones["trials"]) == 2
        assert all(pair["output_check"] == "passed" for pair in ones["trials"])
    assert ones["feedback"]["after"]["output_errors"]["/"]["max"] == pytest.approx(0.2)
    assert str(tmp_path / "inputs.pt") in zeros["identities"]
    for index, case in enumerate(report["cases"].values(), 1):
        saved = json.loads(Path(case["report"]).read_text())
        assert saved == {key: value for key, value in case.items() if key != "report"}
        assert (tmp_path / "run" / f"case-{index}" / "reference.pt").is_file()
    assert json.loads((tmp_path / "run" / "report.json").read_text()) == report


def test_case_probe_and_fixed_original_reference(tmp_path):
    path = case_manifest(tmp_path, check={"absolute_error": {"max": 0.1}})
    probe = autoresearch.evaluate_model(path, output=tmp_path / "probe")
    assert probe["decision"] == "measured", probe["reasons"]
    assert all(case["feedback"]["latency_ms"] > 0 for case in probe["cases"].values())
    report = autoresearch.evaluate_model(
        path, baseline_model="models.py:shifted", candidate_model="models.py:drifted", output=tmp_path / "run"
    )
    assert report["decision"] == "discard", report["reasons"]
    for case in report["cases"].values():
        assert case["feedback"]["before"]["output_errors"]["/"]["max"] == pytest.approx(0.09)
        assert case["feedback"]["after"]["output_errors"]["/"]["max"] == pytest.approx(0.18)


@pytest.fixture
def case_worker(monkeypatch):
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

    def install(outcomes):
        def worker(request, _artifact, _timeout):
            autoresearch._verify_files(request["identities"])
            if request["mode"] == "inspect":
                return model_worker(request)
            if request["mode"] == "reference":
                Path(request["reference_outputs"]).write_bytes(b"reference")
                return {}
            latency, memory = outcomes[request["case"]]
            result = deepcopy(measured)
            result.update(output_check="passed", _process_peak_rss=result["totals"]["process_peak_rss"])
            value = 1 if request["implementation"] == request["baseline"] else latency
            result["totals"]["latency"].update(value=value, known_value=value)
            result["totals"]["latency_iqr"].update(value=0.001, known_value=0.001)
            result["totals"]["peak_memory"].update(
                status="complete" if memory is not None else "unavailable", value=memory, known_value=memory
            )
            return result

        monkeypatch.setattr(autoresearch, "_run_worker", worker)
        return worker

    return install


@pytest.mark.parametrize(
    ("second", "decision"),
    [((0.8, 90), "keep"), ((1.2, 90), "discard"), ((1, 90), "inconclusive"), ((0.8, None), "inconclusive")],
)
def test_all_case_performance_and_memory_gates(tmp_path, case_worker, second, decision):
    path = case_manifest(tmp_path, settings={"trials": 2, "min_improvement": 0.03, "max_memory_bytes": 100})
    case_worker({"zeros": (0.8, 90), "ones": second})
    report = autoresearch.evaluate_model(path, candidate_model="models.py:candidate", output=tmp_path / "run")
    assert report["decision"] == decision, report
    assert report["cases"]["zeros"]["decision"] == "keep", report["cases"]["zeros"]["reasons"]
    assert report["cases"]["ones"]["decision"] == decision
    assert len(report["cases"]["ones"]["trials"]) == 2


def test_discard_dominates_inconclusive_and_later_cases_still_run(tmp_path, case_worker):
    path = case_manifest(tmp_path, settings={"trials": 2, "max_memory_bytes": 100})
    case_worker({"zeros": (1.2, 90), "ones": (0.8, None)})
    report = autoresearch.evaluate_model(path, candidate_model="models.py:candidate", output=tmp_path / "run")
    assert [case["decision"] for case in report["cases"].values()] == ["discard", "inconclusive"]
    assert report["decision"] == "discard"


@pytest.mark.parametrize("changed", ["workload.json", "models.py", "inputs.pt", "reference.pt"])
def test_changes_between_cases_cannot_become_a_new_fixed_experiment(tmp_path, monkeypatch, case_worker, changed):
    path = case_manifest(tmp_path, settings={"trials": 2})
    contract = json.loads(path.read_text())
    contract["cases"]["ones"] = {"inputs": "inputs.pt"}
    path.write_text(json.dumps(contract))
    (tmp_path / "inputs.pt").write_bytes(b"saved inputs")
    worker = case_worker({"zeros": (0.8, 90), "ones": (0.8, 90)})

    def mutate(request, artifact, timeout):
        result = worker(request, artifact, timeout)
        if request["case"] == "ones" and request["mode"] == "inspect":
            target = tmp_path / "run" / "case-1" / changed if changed == "reference.pt" else tmp_path / changed
            target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(autoresearch, "_run_worker", mutate)
    report = autoresearch.evaluate_model(path, candidate_model="models.py:candidate", output=tmp_path / "run")
    assert report["cases"]["zeros"]["decision"] == "keep"
    assert report["cases"]["ones"]["decision"] == "inconclusive"
    assert "changed" in report["cases"]["ones"]["reasons"][0]
    assert report["decision"] == "inconclusive"


@pytest.mark.parametrize(
    "cases",
    [{}, [], {"": {"args": []}}, {"sample": {}}, {"sample": {"seed": 1}}, {"sample": {"inputs": "a.pt", "args": []}}],
)
def test_invalid_case_manifests(tmp_path, cases):
    path = workload(tmp_path)
    contract = json.loads(path.read_text())
    contract.pop("args")
    path.write_text(json.dumps(contract | {"cases": cases}))
    with pytest.raises(ValueError):
        autoresearch.evaluate_model(path, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_case_manifest_cannot_mix_top_level_inputs(tmp_path):
    path = workload(tmp_path, cases={"sample": {"args": []}})
    with pytest.raises(ValueError, match="top level"):
        autoresearch.evaluate_model(path, output=tmp_path / "run")


@pytest.mark.skipif(os.name != "posix", reason="Symlink creation may require extra privileges on Windows.")
@pytest.mark.parametrize("name", ["models.py", "inputs.pt", "extra.txt"])
def test_retargeted_symlinks_cannot_change_the_fixed_cases(tmp_path, monkeypatch, case_worker, name):
    path = case_manifest(tmp_path, settings={"trials": 2}, files=["extra.txt"])
    contract = json.loads(path.read_text())
    contract["cases"]["ones"] = {"inputs": "inputs.pt"}
    path.write_text(json.dumps(contract))
    (tmp_path / "inputs.pt").write_bytes(b"saved inputs")
    (tmp_path / "extra.txt").write_bytes(b"declared dependency")
    alias = tmp_path / name
    original = alias.with_name("original-" + name)
    alias.rename(original)
    replacement = alias.with_name("changed-" + name)
    replacement.write_bytes(original.read_bytes() + b" ")
    alias.symlink_to(original)
    worker = case_worker({"zeros": (0.8, 90), "ones": (0.8, 90)})

    def mutate(request, artifact, timeout):
        result = worker(request, artifact, timeout)
        if request["case"] == "ones" and request["mode"] == "inspect":
            alias.unlink()
            alias.symlink_to(replacement)
        return result

    monkeypatch.setattr(autoresearch, "_run_worker", mutate)
    report = autoresearch.evaluate_model(path, candidate_model="models.py:candidate", output=tmp_path / "run")
    assert report["cases"]["zeros"]["decision"] == "keep"
    assert report["decision"] == "inconclusive"
    assert "changed" in report["cases"]["ones"]["reasons"][0]


@pytest.mark.parametrize("multiple", [False, True])
def test_cli_preserves_feedback_and_case_report_paths(tmp_path, monkeypatch, capsys, multiple):
    feedback = {"after": {"latency_ms": 2, "peak_rss_mib": 3, "output_errors": {"/": {"max": 0.2}}}}
    result = {"decision": "discard", "reasons": ["Failed check"], "output_error_policy": "enforced"}
    if multiple:
        result["cases"] = {"sample": {**result, "feedback": feedback, "report": "case-1/report.json", "trials": []}}
    else:
        result["feedback"] = feedback
    evaluate = MagicMock(return_value=result)
    monkeypatch.setattr(autoresearch, "evaluate_model", evaluate)
    monkeypatch.setattr(autoresearch.sys, "argv", ["autoresearch", "cases.json", "--output", str(tmp_path)])
    autoresearch.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["decision"] == "discard"
    assert summary["report"] == str(tmp_path / "report.json")
    if multiple:
        assert summary["cases"]["sample"] == {
            "decision": "discard",
            "reasons": ["Failed check"],
            "feedback": feedback,
            "report": "case-1/report.json",
        }
    else:
        assert summary["feedback"] == feedback
        assert "cases" not in summary
