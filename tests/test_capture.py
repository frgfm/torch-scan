import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest
import torch

from torchscan import analyze_capture, capture_workload, measure_flops, measure_workload


def test_replay_without_pytorch(tmp_path):
    def tensor(shape):
        return {"shape": shape, "floating": True, "complex": False, "dtype": "Float", "device": "cpu"}

    capture = {
        "schema_version": 1,
        "report_type": "operator_capture",
        "inputs": {},
        "dropped_events": 0,
        "errors": 0,
        "context": {
            "method": "ATen.RecordFunction/torchscan_flops_v1",
            "scope": "calling_thread_eager_ops",
            "boundary": "outermost_native_operator",
            "torch_version": "2.1.0",
        },
        "operators": [
            {
                "operator": "aten::mm",
                "overload": "",
                "module": "Linear",
                "kwargs": {},
                "inputs": [tensor([2, 4]), tensor([4, 3])],
                "outputs": [tensor([2, 3])],
            }
        ],
    }
    assert analyze_capture(capture)["total"]["value"] == 48
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture), encoding="utf-8")
    script = (
        "import sys,json; sys.modules['torch']=None; from torchscan import analyze_capture; "
        "r=analyze_capture(json.load(open(sys.argv[1]))); assert r['total']['value']==48"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]
    capture["dropped_events"] = 1
    assert analyze_capture(capture)["total"]["status"] == "partial"
    capture["schema_version"] = 2
    with pytest.raises(ValueError):
        analyze_capture(capture)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_native_counts_and_roundtrip(device):
    pytest.importorskip("_torchscan_native")
    if (device == "cuda" and not torch.cuda.is_available()) or (
        device == "mps" and not torch.backends.mps.is_available()
    ):
        pytest.skip(f"{device} hardware unavailable")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU(approximate="tanh")).to(device)
    inputs = torch.ones(2, 4, device=device)
    workload = lambda: model(inputs)
    capture = capture_workload(workload, inputs=inputs, modules=model)
    result = analyze_capture(json.loads(json.dumps(capture)))
    assert result["total"]["value"] == measure_flops(workload)["total"]["value"] == 132
    assert len(capture["operators"]) == 2  # No linear/addmm or GELU implementation double-counting.
    assert result["by_module"]["Sequential"] == 132
    report = measure_workload(workload, device=device, metrics=("flops",), flops_backend="native", print_summary=False)
    assert analyze_capture(report["operator_flops"]["capture"])["total"] == report["totals"]["operator_flops"]


@pytest.mark.parametrize("limits", [{"max_events": 1}, {"max_bytes": 1}])
def test_native_loss_is_partial(limits):
    pytest.importorskip("_torchscan_native")
    value = torch.ones(4)
    capture = capture_workload(lambda: (value + 1).relu(), **limits)
    assert capture["dropped_events"] > 0
    assert analyze_capture(capture)["total"]["status"] == "partial"


def test_native_cleanup_and_tensor_lifetime():
    pytest.importorskip("_torchscan_native")
    references = []
    calls = []

    def workload():
        calls.append(1)
        value = torch.ones(4) + 1
        references.append(weakref.ref(value))
        return value

    capture_workload(workload)
    assert len(calls) == 1
    assert references[0]() is None

    def broken():
        workload()
        raise RuntimeError("workload failed")

    with pytest.raises(RuntimeError, match="workload failed"):
        capture_workload(broken)
    assert analyze_capture(capture_workload(workload))["total"]["status"] == "complete"
    with pytest.raises(RuntimeError, match="already active"):
        capture_workload(lambda: capture_workload(workload))


def test_native_unknown_and_failed_ops():
    pytest.importorskip("_torchscan_native")
    value = torch.eye(4)
    report = analyze_capture(capture_workload(lambda: torch.linalg.svd(value)))
    assert report["total"]["status"] == "partial"
    assert report["diagnostics"]

    def catches_error():
        with suppress(RuntimeError):
            value @ torch.ones(3, 2)

    assert analyze_capture(capture_workload(catches_error))["total"]["status"] == "partial"


def test_native_module_calls_on_other_threads():
    pytest.importorskip("_torchscan_native")
    model, inputs = torch.nn.Linear(4, 3), torch.ones(2, 4)
    with ThreadPoolExecutor(max_workers=1) as pool:
        capture = capture_workload(lambda: pool.submit(model, inputs).result(), modules=model)
    assert capture["context"]["unobserved_module_calls"] == 1
    assert analyze_capture(capture)["total"]["status"] == "partial"


@pytest.mark.parametrize("name", ["resnet18", "bert"])
def test_native_model_parity(name):
    pytest.importorskip("_torchscan_native")
    pytest.importorskip("torchvision" if name == "resnet18" else "transformers")
    from scripts.benchmark_comparison import build_model

    model, inputs, *_ = build_model(name, torch.device("cpu"))
    workload = lambda: model(**inputs) if name == "bert" else model(inputs["input"])
    with torch.inference_mode():
        before = measure_flops(workload)
        after = measure_flops(workload, modules=model, backend="native")
    assert after["total"]["status"] == before["total"]["status"] == "complete"
    assert after["total"]["value"] == before["total"]["value"]
