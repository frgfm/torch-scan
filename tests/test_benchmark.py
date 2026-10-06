import builtins
import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from unittest.mock import Mock

import pytest
import torch
from torch.utils.benchmark import Measurement, Timer

from torchscan import benchmark as benchmark_module
from torchscan import measure_latency


def test_cpu_first_call_warm_timing_and_metadata(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=AssertionError("CPU must not synchronize CUDA")))
    monkeypatch.setattr(torch.mps, "synchronize", Mock(side_effect=AssertionError("CPU must not synchronize MPS")))
    previous_threads = torch.get_num_threads()
    threads = previous_threads if "native thread pool" in torch.__config__.parallel_info() else 1
    inputs = torch.ones(8, 4)
    calls = 0

    def workload():
        nonlocal calls
        assert torch.get_num_threads() == threads
        calls += 1
        time.sleep(0.04 if calls == 1 else 0.002)
        return inputs + 1

    report = measure_latency(
        workload, device="cpu", inputs=inputs, work_units=8, work_unit="images", num_threads=threads, min_run_time=0.03
    )
    assert torch.get_num_threads() == previous_threads
    totals = report["totals"]
    assert totals["first_call_latency"]["value"] >= 0.04
    assert totals["latency"]["value"] >= 0.002
    assert totals["latency"]["scope"] == "warmed_block_average"
    assert all(metric["status"] == "complete" for metric in totals.values())
    measurement = report["measurement"]
    assert len(measurement["raw_times_seconds"]) >= 5
    assert calls > 1 + 5 + measurement["measured_calls"]  # Native timer calibration also invokes the workload.
    assert totals["throughput"]["unit"] == "images/s"
    assert totals["throughput"]["value"] == pytest.approx(
        8 * measurement["measured_calls"] / sum(measurement["raw_times_seconds"])
    )
    assert report["inputs"]["shape"] == [8, 4]
    assert report["inputs"]["dtype"] == "torch.float32"
    assert report["context"]["inputs_source"] == "caller_supplied"
    assert report["context"]["device"] == "cpu"
    assert report["context"]["num_threads"] == threads
    assert report["context"]["torch_version"] == torch.__version__
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "mps:0"])
def test_selected_device_synchronization_and_block_statistics(monkeypatch, device):
    events = []
    threads = torch.get_num_threads()
    if device.startswith("cuda"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test GPU")
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("sync", str(device))))
        expected_device = "cuda:1"
    else:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append(("sync", "mps")))
        expected_device = "mps"
    ticks = iter([1.0, 1.5, 2.0, 2.1])
    monkeypatch.setattr(benchmark_module.time, "perf_counter", lambda: next(ticks))

    class ControlledTimer:
        def __init__(self, **kwargs):
            assert kwargs["num_threads"] == threads
            self.kwargs = kwargs
            self.task_spec = Timer(**kwargs)._task_spec

        def blocked_autorange(self, *, min_run_time):
            assert min_run_time == pytest.approx(0.1)
            self.kwargs["timer"]()
            self.kwargs["globals"]["workload"]()
            self.kwargs["timer"]()
            return Measurement(number_per_run=4, raw_times=[0.02, 0.04, 0.18], task_spec=self.task_spec)

    monkeypatch.setattr(benchmark_module, "_benchmark_tools", lambda: (ControlledTimer, Measurement))
    report = measure_latency(
        lambda: events.append(("work", torch.get_num_threads())),
        device=device,
        work_units=8,
        work_unit="images",
        num_threads=threads,
        warmup=1,
        min_run_time=0.1,
        min_repeats=3,
    )
    assert events == [
        ("sync", expected_device),
        ("work", threads),
        ("sync", expected_device),
        ("work", threads),
        ("sync", expected_device),
        ("work", threads),
        ("sync", expected_device),
    ]
    assert report["context"]["device"] == expected_device
    assert report["totals"]["first_call_latency"]["value"] == pytest.approx(0.5)
    assert report["totals"]["latency"]["value"] == pytest.approx(0.01)
    assert report["totals"]["latency_iqr"]["value"] == pytest.approx(0.02)
    assert report["totals"]["throughput"]["value"] == pytest.approx(400)


def test_minimum_repeats_and_default_threads(monkeypatch):
    threads = torch.get_num_threads()
    real_timer = Timer(stmt="pass", num_threads=threads)
    repeats = []

    class ControlledTimer:
        def __init__(self, **kwargs):
            assert kwargs["num_threads"] == threads

        def blocked_autorange(self, **_kwargs):
            return Measurement(number_per_run=10, raw_times=[0.01], task_spec=real_timer._task_spec)

        def timeit(self, number):
            repeats.append(number)
            return Measurement(number_per_run=number, raw_times=[0.02], task_spec=real_timer._task_spec)

    monkeypatch.setattr(benchmark_module, "_benchmark_tools", lambda: (ControlledTimer, Measurement))
    report = measure_latency(lambda: None, device="cpu", warmup=0, min_repeats=3)
    assert repeats == [10, 10]
    assert report["measurement"]["raw_times_seconds"] == [0.01, 0.02, 0.02]
    assert report["measurement"]["measured_calls"] == 30
    assert report["context"]["num_threads"] == threads
    assert report["context"]["inputs_source"] == "not_provided"


def test_workload_failure_restores_threads_and_preserves_error():
    previous_threads = torch.get_num_threads()
    error = RuntimeError("workload failed")
    with pytest.raises(RuntimeError) as caught:
        measure_latency(Mock(side_effect=error), device="cpu", num_threads=previous_threads)
    assert caught.value is error
    assert torch.get_num_threads() == previous_threads


def test_non_restorable_thread_change_rejected_before_mutation(monkeypatch):
    monkeypatch.setattr(torch.__config__, "parallel_info", lambda: "ATen parallel backend: native thread pool")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    setter = Mock()
    monkeypatch.setattr(torch, "set_num_threads", setter)
    workload = Mock()
    with pytest.raises(NotImplementedError, match="cannot restore"):
        measure_latency(workload, device="cpu", num_threads=1)
    setter.assert_not_called()
    workload.assert_not_called()


def test_missing_native_benchmark_dependencies_leave_analysis_usable(monkeypatch):
    script = """
import builtins
original_import = builtins.__import__
def import_without_benchmark(name, *args, **kwargs):
    if name == "torch.utils.benchmark":
        raise ImportError("missing native benchmark dependency")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_benchmark
import torch
import torchscan
report = torchscan.crawl_module(torch.nn.Linear(4, 2), (4,))
assert report["totals"]["parameters"]["value"] == 10
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    original_import = builtins.__import__

    def import_without_benchmark(name, *args, **kwargs):
        if name == "torch.utils.benchmark":
            raise ImportError("missing native benchmark dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_benchmark)
    workload = Mock()
    with pytest.raises(ImportError, match="upgrade PyTorch"):
        measure_latency(workload, device="cpu")
    workload.assert_not_called()


@pytest.mark.parametrize(
    ("system", "available"),
    [
        ("darwin", True),
        ("darwin", False),
        ("linux", True),
        ("linux", False),
        ("win32", False),
    ],
)
def test_hardware_identity_and_unavailable_os_queries(monkeypatch, system, available):
    monkeypatch.setattr(benchmark_module.sys, "platform", system)
    monkeypatch.setattr(benchmark_module.platform, "processor", lambda: "fallback CPU")
    cpu_query = Mock(return_value="test CPU\n") if available else Mock(side_effect=OSError("not available"))
    monkeypatch.setattr(benchmark_module.subprocess, "check_output", cpu_query)
    cpu_file = Mock(return_value="processor: 0\nmodel name: test CPU\n") if available else Mock(side_effect=OSError())
    monkeypatch.setattr(benchmark_module.Path, "read_text", cpu_file)
    report = measure_latency(lambda: None, device="cpu", warmup=0, min_run_time=0.001, min_repeats=2)
    expected = "test CPU" if available else "fallback CPU"
    assert report["context"]["processor"] == expected
    assert report["context"]["device_name"] == expected


def test_non_callable_workload_rejected():
    with pytest.raises(TypeError, match="callable"):
        measure_latency(None, device="cpu")


@pytest.mark.parametrize("invalid_first_call", [True, False])
def test_invalid_timing_cannot_produce_a_complete_report(monkeypatch, invalid_first_call):
    real_timer = Timer(stmt="pass", num_threads=torch.get_num_threads())
    measurement = Measurement(number_per_run=1, raw_times=[0.01, 0.02], task_spec=real_timer._task_spec)

    class ControlledTimer:
        def __init__(self, **_kwargs):
            pass

        def blocked_autorange(self, **_kwargs):
            return measurement

    monkeypatch.setattr(benchmark_module, "_benchmark_tools", lambda: (ControlledTimer, Measurement))
    ticks = iter([1.0, 1.0 if invalid_first_call else 1.1])
    monkeypatch.setattr(benchmark_module.time, "perf_counter", lambda: next(ticks))
    message = "durations" if invalid_first_call else "Throughput"
    with pytest.raises(RuntimeError, match=message):
        measure_latency(
            lambda: None, device="cpu", warmup=0, min_repeats=2, work_units=1 if invalid_first_call else 1e308
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"work_units": 0},
        {"work_units": float("nan")},
        {"work_units": 10**1000},
        {"work_units": True},
        {"work_unit": " "},
        {"num_threads": 0},
        {"num_threads": True},
        {"warmup": -1},
        {"warmup": 1.5},
        {"min_run_time": float("inf")},
        {"min_run_time": 0},
        {"min_repeats": 1},
    ],
)
def test_invalid_settings_do_not_execute_workload(settings):
    workload = Mock()
    with pytest.raises(ValueError):
        measure_latency(workload, device="cpu", **settings)
    workload.assert_not_called()


@pytest.mark.parametrize("device", ["meta", "xpu"])
def test_unsupported_device_does_not_execute_workload(device):
    workload = Mock()
    with pytest.raises(NotImplementedError):
        measure_latency(workload, device=device)
    workload.assert_not_called()


@pytest.mark.parametrize("device", ["cuda:0", "mps:0", "mps:1"])
def test_unavailable_device_does_not_execute_workload(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    workload = Mock()
    with pytest.raises(RuntimeError):
        measure_latency(workload, device=device)
    workload.assert_not_called()


def test_unavailable_cuda_index_does_not_execute_workload(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    workload = Mock()
    with pytest.raises(RuntimeError, match="cuda:1"):
        measure_latency(workload, device="cuda:1")
    workload.assert_not_called()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_real_accelerator_completed_workload(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} hardware is unavailable")
    model = torch.nn.Linear(128, 128).to(device).eval()
    inputs = torch.ones(8, 128, device=device)

    def workload():
        with torch.inference_mode():
            return model(inputs)

    report = measure_latency(workload, device=device, inputs=inputs, work_units=8, min_run_time=0.02, min_repeats=3)
    assert report["totals"]["latency"]["value"] > 0
    assert report["totals"]["throughput"]["value"] > 0
    assert report["context"]["device"].startswith(device)
    assert model.training is False
