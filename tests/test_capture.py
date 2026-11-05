import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest
import torch

from torchscan import analyze_capture, capture_workload, measure_flops, measure_workload


def test_native_counter_after_python_exits(tmp_path):
    path = tmp_path / "capture.json"
    script = (
        "import json,sys,torch; from torchscan import capture_workload; "
        "x=torch.ones(2,4); y=torch.ones(4,3); "
        "capture=capture_workload(lambda:x@y); "
        "assert 'torchscan._analysis' not in sys.modules; "
        "open(sys.argv[1],'w').write(json.dumps(capture))"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]
    counter = shutil.which("torchscan-count")
    assert counter is not None
    # A native executable must work without finding Python or its libraries.
    environment = {**os.environ, "PATH": "", "PYTHONHOME": str(tmp_path / "no-python")}
    output = tmp_path / "report.json"
    capture = json.loads(path.read_text())
    for change, code in [({}, 0), ({"dropped_events": 1}, 2), ({"schema_version": 2}, 2)]:
        path.write_text(json.dumps({**capture, **change}))
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [counter, str(path), "-o", str(output)], env=environment, capture_output=True, check=False
        )
        assert result.returncode == code
        if "schema_version" not in change:
            assert json.loads(output.read_text()) == analyze_capture({**capture, **change})
    path.write_text("{broken")
    assert subprocess.run([counter, str(path)], env=environment, capture_output=True, check=False).returncode == 2  # ruff: ignore[subprocess-without-shell-equals-true]


def test_exact_large_native_counts(tmp_path):
    x, y = torch.ones(2, 4), torch.ones(4, 3)
    capture = capture_workload(lambda: x @ y)
    event = next(event for event in capture["operators"] if event["operator"] == "aten::mm")
    event["inputs"][0]["shape"][0] = event["outputs"][0]["shape"][0] = 2**62
    assert analyze_capture(capture)["total"]["value"] == 2**62 * 4 * 3 * 2
    event["custom_flops"] = 10**100
    assert analyze_capture(capture)["total"]["value"] == 10**100
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    result = subprocess.check_output([shutil.which("torchscan-count"), str(path)])  # ruff: ignore[subprocess-without-shell-equals-true]
    assert json.loads(result)["total"]["value"] == 10**100
    event["custom_flops"] = True
    assert analyze_capture(capture)["total"]["status"] == "partial"


@pytest.mark.parametrize(("operator", "weight"), [("mm", [4, 3]), ("linear", [3, 4])])
def test_replay_without_pytorch(tmp_path, operator, weight):
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
            "scope": "calling_thread_and_autograd_eager_ops",
            "boundary": "dispatcher_operator",
            "torch_version": "2.1.0",
        },
        "operators": [
            {
                "operator": f"aten::{operator}",
                "overload": "",
                "module": "Linear",
                "kwargs": {},
                "inputs": [tensor([2, 4]), tensor(weight)],
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
    for malformed in (
        None,
        {**capture, "schema_version": 2},
        {**capture, "errors": True},
        {**capture, "operators": [{}]},
        {**capture, "inputs": {1: "invalid key"}},
        {**capture, "inputs": {"value": float("nan")}},
    ):
        with pytest.raises(ValueError):
            analyze_capture(malformed)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_native_counts_and_roundtrip(device):
    if (device == "cuda" and not torch.cuda.is_available()) or (
        device == "mps" and not torch.backends.mps.is_available()
    ):
        pytest.skip(f"{device} hardware unavailable")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU(approximate="tanh")).to(device)
    inputs = torch.ones(2, 4, device=device)
    workload = lambda: model(inputs)
    capture = capture_workload(workload, inputs=inputs, modules=model)
    result = analyze_capture(json.loads(json.dumps(capture)))
    assert result["total"]["value"] == measure_flops(workload)["total"]["value"] == 132, result
    product = "aten.linear" if device == "mps" else "aten.addmm"
    assert result["by_operator"] == {product: 48, "aten.gelu": 84}
    assert result["by_module"]["Sequential"] == 132
    report = measure_workload(workload, device=device, metrics=("flops",), print_summary=False)
    assert analyze_capture(report["operator_flops"]["capture"])["total"] == report["totals"]["operator_flops"]


@pytest.mark.parametrize("limits", [{"max_events": 1}, {"max_bytes": 1}])
def test_native_loss_is_partial(limits):
    value = torch.ones(4)
    capture = capture_workload(lambda: (value + 1).relu(), **limits)
    assert capture["dropped_events"] > 0
    assert analyze_capture(capture)["total"]["status"] == "partial"


def test_native_cleanup_and_tensor_lifetime():
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


@pytest.mark.parametrize("right_shape", [(3, 2), ()])
def test_native_unknown_and_failed_ops(right_shape):
    value = torch.eye(4)
    report = analyze_capture(capture_workload(lambda: torch.linalg.svd(value)))
    assert report["total"]["status"] == "partial"
    assert report["diagnostics"]
    negative = torch._neg_view(value)
    assert analyze_capture(capture_workload(lambda: torch.resolve_neg(negative)))["total"]["status"] == "partial"

    def catches_error():
        with suppress(RuntimeError):
            value @ torch.ones(right_shape)

    assert analyze_capture(capture_workload(catches_error))["total"]["status"] == "partial"
    captured = capture_workload(catches_error, custom_mapping={torch.ops.aten.mm: lambda *_args, **_kwargs: 99})
    assert analyze_capture(captured)["total"]["status"] == "partial"


def test_native_module_calls_on_other_threads():
    model, inputs = torch.nn.Linear(4, 3), torch.ones(2, 4)
    with ThreadPoolExecutor(max_workers=1) as pool:
        capture = capture_workload(lambda: pool.submit(model, inputs).result(), modules=model)
    assert capture["context"]["unobserved_module_calls"] == 1
    assert analyze_capture(capture)["total"]["status"] == "partial"
