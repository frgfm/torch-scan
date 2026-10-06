import builtins
import json
from unittest.mock import Mock

import pytest
import torch
from torch.utils.benchmark import Measurement, Timer

from torchscan import benchmark as timing
from torchscan import measure_latency


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_workload_report(device):
    if (device == "cuda" and not torch.cuda.is_available()) or (
        device == "mps" and not torch.backends.mps.is_available()
    ):
        pytest.skip(f"{device} hardware unavailable")
    inputs = torch.ones(8, 64, device=device)
    model = torch.nn.Linear(64, 16).to(device).eval()
    threads = torch.get_num_threads()

    def workload():
        with torch.inference_mode():
            return model(inputs)

    report = measure_latency(
        workload, device=device, inputs=inputs, work_units=8, work_unit="samples", min_run_time=0.02
    )
    assert all(metric["status"] == "complete" for metric in report["totals"].values())
    assert report["totals"]["latency"]["value"] > 0
    assert len(report["measurement"]["raw_times_seconds"]) >= 5
    assert report["inputs"]["shape"] == [8, 64]
    assert report["context"]["num_threads"] == threads == torch.get_num_threads()
    assert not model.training
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("invalid", [None, "duration", "throughput"])
def test_completed_timing_and_statistics(monkeypatch, device, invalid):
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test GPU")
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", sync)
    spec = Timer(stmt="pass")._task_spec
    timer = Mock()
    timer.blocked_autorange.return_value = Measurement(number_per_run=4, raw_times=[0.02, 0.04, 0.18], task_spec=spec)
    timer.timeit.return_value = Measurement(number_per_run=4, raw_times=[0.08], task_spec=spec)
    factory = Mock(return_value=timer)
    monkeypatch.setattr(timing, "_benchmark_tools", lambda: (factory, Measurement))
    ticks = iter([1.0, 1.0 if invalid == "duration" else 1.5, 2.0])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    options = {"device": device, "warmup": 0, "work_units": 1e308 if invalid == "throughput" else 8}
    if invalid:
        with pytest.raises(RuntimeError, match=r"durations|Throughput"):
            measure_latency(lambda: None, **options)
        return
    report = measure_latency(lambda: None, **options)
    factory.call_args.kwargs["timer"]()  # The native timer receives the selected-device completion clock.
    assert sync.call_count == 3
    assert sync.call_args.args == ((torch.device("cuda:1"),) if device == "cuda" else ())
    assert timer.timeit.call_count == 2
    assert report["totals"]["first_call_latency"]["value"] == pytest.approx(0.5)
    assert report["totals"]["latency"]["value"] == pytest.approx(0.02)
    assert report["totals"]["latency_iqr"]["value"] == pytest.approx(0.01)
    assert report["totals"]["throughput"]["value"] == pytest.approx(400)


def test_failures_preserve_threads():
    with pytest.raises(TypeError):
        measure_latency(None, device="cpu")
    threads = torch.get_num_threads()
    error = RuntimeError("workload failed")
    with pytest.raises(RuntimeError) as caught:
        measure_latency(Mock(side_effect=error), device="cpu")
    assert caught.value is error
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize(
    ("settings", "error_type"),
    [
        ({"work_units": 0}, ValueError),
        ({"work_units": True}, ValueError),
        ({"work_units": 10**1000}, ValueError),
        ({"work_unit": " "}, ValueError),
        ({"warmup": -1}, ValueError),
        ({"min_run_time": float("inf")}, ValueError),
        ({"min_repeats": 1}, ValueError),
        ({"device": "meta"}, NotImplementedError),
        ({"device": "cuda:2"}, RuntimeError),
        ({"device": "mps:1"}, RuntimeError),
    ],
)
def test_invalid_configuration_does_not_run(monkeypatch, settings, error_type):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    workload = Mock()
    with pytest.raises(error_type):
        measure_latency(workload, **({"device": "cpu"} | settings))
    workload.assert_not_called()


@pytest.mark.parametrize("system", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("available", [True, False])
def test_processor_metadata(monkeypatch, system, available):
    monkeypatch.setattr(timing.sys, "platform", system)
    monkeypatch.setattr(timing.platform, "processor", lambda: "unknown CPU")
    query = Mock(return_value="test CPU") if available else Mock(side_effect=OSError())
    monkeypatch.setattr(timing.subprocess, "check_output", query)
    monkeypatch.setattr(timing.Path, "read_text", Mock(return_value="model name: test CPU") if available else query)
    assert timing._processor() == ("test CPU" if available and system != "win32" else "unknown CPU")


def test_missing_benchmark_dependencies(monkeypatch):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "torch.utils.benchmark":
            raise ImportError("benchmark unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    workload = Mock()
    with pytest.raises(ImportError, match="upgrade PyTorch"):
        measure_latency(workload, device="cpu")
    workload.assert_not_called()
