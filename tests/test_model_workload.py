import json
import os
import py_compile

import pytest
import torch

from torchscan._model_workload import _tensor, model_worker
from torchscan.autoresearch import _identities, evaluate_model


def contract(tmp_path, **updates):
    source = tmp_path / "models.py"
    source.write_text(
        "import torch\n"
        "def baseline(): return torch.nn.Linear(4, 2)\n"
        "def candidate():\n"
        "    model = baseline()\n"
        "    torch.randn(100)\n"
        "    return model\n"
        "def wrong(): return torch.nn.Linear(4, 3)\n",
        encoding="utf-8",
    )
    specification = {
        "model": "models.py:baseline",
        "args": [{"shape": [2, 4]}],
        "workload": "inference",
        "settings": {"trials": 2, "warmup": 0, "min_run_time": 0.001, "min_repeats": 2},
        **updates,
    }
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(specification), encoding="utf-8")
    return path


def request(path, mode="inspect", implementation="models.py:baseline"):
    return {
        "experiment": str(path),
        "identities": _identities([path]),
        "mode": mode,
        "baseline": "models.py:baseline",
        "candidate": implementation,
        "implementation": implementation,
        "reference_outputs": str(path.parent / "reference.pt"),
    }


def test_zero_script_probe_collects_clean_rss_and_latency(tmp_path):
    path = contract(tmp_path)
    report = evaluate_model(path, output=tmp_path / "run")
    assert report["decision"] == "measured", report["reasons"]
    assert report["feedback"]["latency_ms"] > 0
    rss = report["measurement"]["totals"]["process_peak_rss"]
    if hasattr(os, "wait4"):
        assert rss["status"] == "complete"
        assert rss["value"] > 0
    assert "no instrumentation" in report["measurement"]["context"]["rss_boundary"]
    assert (tmp_path / "run" / "measurement-rss.log").is_file()
    assert (tmp_path / "run" / "measurement.html").is_file()
    assert json.loads((tmp_path / "run" / "report.json").read_text()) == report


def test_automatic_check_rejects_wrong_model_without_user_callbacks(tmp_path):
    path = contract(tmp_path)
    report = evaluate_model(path, candidate_model="models.py:wrong", output=tmp_path / "run")
    assert report["decision"] == "discard", report["reasons"]
    assert report["checks"]["initial"]["candidate"] == "failed"
    assert report["trials"] == []


def test_imported_candidate_does_not_execute_stale_bytecode(tmp_path):
    path = contract(tmp_path)
    source = tmp_path / "models.py"
    (tmp_path / "reference.py").write_bytes(source.read_bytes())
    py_compile.compile(str(source), doraise=True)
    timestamp = source.stat()
    source.write_text(source.read_text().replace("Linear(4, 2)", "Linear(4, 3)"))
    os.utime(source, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    config = json.loads(path.read_text())
    config["model"] = "reference.py:baseline"
    path.write_text(json.dumps(config))
    report = evaluate_model(path, candidate_model="models:baseline", output=tmp_path / "run")
    assert report["decision"] == "discard", report["reasons"]


def test_inputs_do_not_depend_on_candidate_constructor_rng(tmp_path):
    path = contract(tmp_path)
    base = request(path, "reference")
    settings = model_worker(request(path))["settings"]
    base["settings"] = settings
    model_worker(base)
    other = request(path, "check", "models.py:candidate")
    other["settings"] = settings
    assert model_worker(other)["output_check"] == "passed"


def test_incumbent_does_not_replace_the_original_numerical_reference(tmp_path):
    path = contract(tmp_path, check={"rtol": 0, "atol": 0.1})
    with (tmp_path / "models.py").open("a") as source:
        source.write(
            "def incumbent():\n"
            "    model = baseline()\n"
            "    model.register_forward_hook(lambda m, a, out: out + 0.09)\n"
            "    return model\n"
            "def drifted():\n"
            "    model = baseline()\n"
            "    model.register_forward_hook(lambda m, a, out: out + 0.18)\n"
            "    return model\n"
        )
    report = evaluate_model(
        path, baseline_model="models.py:incumbent", candidate_model="models.py:drifted", output=tmp_path / "run"
    )
    assert report["checks"]["initial"]["baseline"] == "passed"
    assert report["checks"]["initial"]["candidate"] == "failed"
    assert report["decision"] == "discard"


def test_integer_mask_and_saved_nested_inputs(tmp_path):
    tokens = _tensor({"shape": [2, 8], "dtype": "int64", "low": 0, "high": 32}, "cpu")
    mask = _tensor({"shape": [2, 8], "dtype": "bool"}, "cpu")
    assert tokens.dtype == torch.int64
    assert tokens.min() >= 0
    assert tokens.max() < 32
    assert mask.dtype == torch.bool
    assert mask.all()
    with pytest.raises(ValueError, match="bounds"):
        _tensor({"shape": [2, 8], "dtype": "int64"}, "cpu")
    with pytest.raises(ValueError, match="bounds"):
        _tensor({"shape": [2, 8], "dtype": "float32", "low": 0, "high": 1}, "cpu")
    batch = {"args": (torch.ones(2, 4),), "kwargs": {}}
    torch.save(batch, tmp_path / "batch.pt")
    path = contract(tmp_path, inputs="batch.pt")
    config = json.loads(path.read_text())
    config.pop("args")
    path.write_text(json.dumps(config))
    inspected = model_worker(request(path))
    assert str(tmp_path / "batch.pt") in inspected["files"]
    phase = request(path, "reference")
    phase["settings"] = inspected["settings"]
    model_worker(phase)
    assert (tmp_path / "reference.pt").is_file()
    assert _tensor({"shape": [2, 4]}, "cpu", descriptors=False) == {"shape": [2, 4]}


def test_existing_model_instance_tracks_its_owner_module(tmp_path):
    path = contract(tmp_path, model="models.py:instance")
    with (tmp_path / "models.py").open("a") as source:
        source.write("instance = torch.nn.Linear(4, 2)\n")
    inspected = model_worker(request(path, implementation="models.py:instance"))
    assert str(tmp_path / "models.py") in inspected["files"]


def test_worker_failure_after_publishing_a_result_cannot_pass(tmp_path):
    path = contract(tmp_path)
    with (tmp_path / "models.py").open("a") as source:
        source.write("import atexit, os\natexit.register(os._exit, 7)\n")
    report = evaluate_model(path, output=tmp_path / "run")
    assert report["decision"] == "inconclusive"
    assert "status 7" in report["reasons"][0]


@pytest.mark.parametrize(
    "updates", [{"workload": "training"}, {"unknown": True}, {"seed": True}, {"files": "model.py"}]
)
def test_invalid_contract_cannot_silently_change_work(tmp_path, updates):
    path = contract(tmp_path, **updates)
    with pytest.raises(ValueError):
        model_worker(request(path))
