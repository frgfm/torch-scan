import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchscan import process
from torchscan.process import memory as process_memory


def _assert_memory_stats(stats, *, device, metric, accelerator=False):
    expected_keys = {"device", "metric", "baseline_bytes", "peak_bytes", "delta_bytes"}
    if accelerator:
        expected_keys.add("allocated_peak_bytes")

    assert stats.keys() == expected_keys
    assert stats["device"] == device
    assert stats["metric"] == metric
    assert all(type(stats[key]) is int for key in expected_keys - {"device", "metric"})
    assert stats["peak_bytes"] >= stats["baseline_bytes"]
    assert stats["delta_bytes"] == stats["peak_bytes"] - stats["baseline_bytes"]


def test_measure_peak_memory_cpu_inference():
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 32))
    inputs = torch.randn(16, 64)
    calls = 0

    def workload():
        nonlocal calls
        calls += 1
        with torch.inference_mode():
            model(inputs)

    stats = process.measure_peak_memory(workload, device=torch.device("cpu:0"))

    _assert_memory_stats(stats, device="cpu", metric="pytorch_tensor_bytes")
    assert stats["delta_bytes"] > 0
    assert calls == 1
    assert json.loads(json.dumps(stats)) == stats


def test_measure_peak_memory_cpu_training():
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 32))
    inputs = torch.randn(16, 64)
    targets = torch.randn(16, 32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def workload():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

    stats = process.measure_peak_memory(workload, device="cpu")

    _assert_memory_stats(stats, device="cpu", metric="pytorch_tensor_bytes")
    assert stats["delta_bytes"] > 0


def test_measure_peak_memory_cpu_without_memory_events():
    with pytest.raises(RuntimeError, match="no memory timeline points for 'cpu'") as exc_info:
        process.measure_peak_memory(lambda: None, device="cpu")

    assert torch.__version__ in str(exc_info.value)


def test_measure_peak_memory_preserves_workload_exception_and_cleans_temporary_file(monkeypatch, tmp_path):
    created_paths = []
    named_temporary_file = process_memory.tempfile.NamedTemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        temporary_file = named_temporary_file(*args, dir=tmp_path, **kwargs)
        created_paths.append(Path(temporary_file.name))
        return temporary_file

    monkeypatch.setattr(process_memory.tempfile, "NamedTemporaryFile", tracked_temporary_file)
    error = RuntimeError("workload failed")

    def workload():
        raise error

    with pytest.raises(RuntimeError) as exc_info:
        process.measure_peak_memory(workload, device="cpu")

    assert exc_info.value is error
    assert created_paths
    assert all(not path.exists() for path in created_paths)


def test_measure_peak_memory_cuda_order_and_device(monkeypatch):
    events = []
    current_device = Mock(return_value=1)
    empty_cache = Mock()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("synchronize", device)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: events.append(("memory_reserved", device)) or 100)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: events.append(("reset_peak_memory_stats", device)),
    )
    monkeypatch.setattr(
        torch.cuda, "max_memory_reserved", lambda device: events.append(("max_memory_reserved", device)) or 180
    )
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda device: events.append(("max_memory_allocated", device)) or 140
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    def workload():
        events.append(("workload", None))

    stats = process.measure_peak_memory(workload, device="cuda")
    device = torch.device("cuda:1")

    assert stats == {
        "device": "cuda:1",
        "metric": "pytorch_reserved_bytes",
        "baseline_bytes": 100,
        "peak_bytes": 180,
        "delta_bytes": 80,
        "allocated_peak_bytes": 140,
    }
    assert events == [
        ("synchronize", device),
        ("memory_reserved", device),
        ("reset_peak_memory_stats", device),
        ("workload", None),
        ("synchronize", device),
        ("max_memory_reserved", device),
        ("max_memory_allocated", device),
    ]
    current_device.assert_called_once_with()
    empty_cache.assert_not_called()


def test_measure_peak_memory_mps_order_and_device(monkeypatch):
    events = []
    memory = SimpleNamespace(
        memory_reserved=lambda device: events.append(("memory_reserved", device)) or 200,
        reset_peak_memory_stats=lambda device: events.append(("reset_peak_memory_stats", device)),
        max_memory_reserved=lambda device: events.append(("max_memory_reserved", device)) or 260,
        max_memory_allocated=lambda device: events.append(("max_memory_allocated", device)) or 220,
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append(("synchronize", None)))
    monkeypatch.setattr(torch, "accelerator", SimpleNamespace(memory=memory), raising=False)

    def workload():
        events.append(("workload", None))

    stats = process.measure_peak_memory(workload, device=torch.device("mps:0"))
    device = torch.device("mps")

    assert stats == {
        "device": "mps",
        "metric": "pytorch_reserved_bytes",
        "baseline_bytes": 200,
        "peak_bytes": 260,
        "delta_bytes": 60,
        "allocated_peak_bytes": 220,
    }
    assert events == [
        ("synchronize", None),
        ("memory_reserved", device),
        ("reset_peak_memory_stats", device),
        ("workload", None),
        ("synchronize", None),
        ("max_memory_reserved", device),
        ("max_memory_allocated", device),
    ]


def test_measure_peak_memory_missing_cuda_api(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", None)

    with pytest.raises(NotImplementedError) as exc_info:
        process.measure_peak_memory(Mock(), device="cuda:0")

    assert "cuda:0" in str(exc_info.value)
    assert torch.__version__ in str(exc_info.value)
    assert "torch.cuda.max_memory_reserved" in str(exc_info.value)


def test_measure_peak_memory_missing_mps_api(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "accelerator", SimpleNamespace(memory=SimpleNamespace()), raising=False)

    with pytest.raises(NotImplementedError) as exc_info:
        process.measure_peak_memory(Mock(), device="mps")

    assert "mps" in str(exc_info.value)
    assert torch.__version__ in str(exc_info.value)
    assert "torch.accelerator.memory.memory_reserved" in str(exc_info.value)


def test_measure_peak_memory_unimplemented_mps_api(monkeypatch):
    workload = Mock()
    memory = SimpleNamespace(
        memory_reserved=Mock(side_effect=RuntimeError("Allocator for mps is not a DeviceAllocator")),
        reset_peak_memory_stats=Mock(),
        max_memory_reserved=Mock(),
        max_memory_allocated=Mock(),
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", Mock())
    monkeypatch.setattr(torch, "accelerator", SimpleNamespace(memory=memory), raising=False)

    with pytest.raises(NotImplementedError) as exc_info:
        process.measure_peak_memory(workload, device="mps")

    message = str(exc_info.value)
    assert "mps" in message
    assert torch.__version__ in message
    assert "torch.accelerator.memory.memory_reserved" in message
    workload.assert_not_called()


def test_measure_peak_memory_unavailable_cuda(monkeypatch):
    workload = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA device 'cuda:0' is unavailable"):
        process.measure_peak_memory(workload, device="cuda:0")

    workload.assert_not_called()


def test_measure_peak_memory_unavailable_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="CUDA device 'cuda:2' is unavailable"):
        process.measure_peak_memory(Mock(), device="cuda:2")


def test_measure_peak_memory_unavailable_mps(monkeypatch):
    workload = Mock()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS device 'mps' is unavailable"):
        process.measure_peak_memory(workload, device="mps")

    workload.assert_not_called()


def test_measure_peak_memory_unavailable_mps_index():
    with pytest.raises(RuntimeError, match="MPS device 'mps:1' is unavailable"):
        process.measure_peak_memory(Mock(), device="mps:1")


def test_measure_peak_memory_unsupported_backend():
    with pytest.raises(NotImplementedError, match="not implemented for 'meta'"):
        process.measure_peak_memory(Mock(), device="meta")


@pytest.mark.parametrize("device_type", ["mps", "cuda"])
@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_measure_peak_memory_accelerator_smoke(device_type, training):
    if device_type == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS hardware is unavailable")
        memory = getattr(getattr(torch, "accelerator", None), "memory", None)
        required_apis = (
            "memory_reserved",
            "reset_peak_memory_stats",
            "max_memory_reserved",
            "max_memory_allocated",
        )
        if any(not callable(getattr(memory, name, None)) for name in required_apis):
            pytest.skip(f"PyTorch {torch.__version__} lacks resettable MPS peak allocator statistics")
        device = torch.device("mps")
    else:
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware is unavailable")
        device = torch.device("cuda:0")

    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 32)).to(device)
    inputs = torch.randn(16, 64, device=device)
    targets = torch.randn(16, 32, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if training:

        def workload():
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()

    else:

        def workload():
            with torch.inference_mode():
                model(inputs)

    stats = process.measure_peak_memory(workload, device=device)

    _assert_memory_stats(stats, device=str(device), metric="pytorch_reserved_bytes", accelerator=True)
