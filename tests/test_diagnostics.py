import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from torchscan import profile_workload
from torchscan.process import measure_peak_rss


def test_child_rss_is_independent_and_failures_propagate():
    if sys.platform not in {"darwin", "linux"}:
        pytest.skip("OS per-child accounting unavailable")
    small = [sys.executable, "-c", "pass"]
    large = [sys.executable, "-c", "data = bytearray(64 * 1024 * 1024)"]
    before = measure_peak_rss(small)
    peak = measure_peak_rss(large)
    after = measure_peak_rss(small)
    assert peak["value"] > before["value"] + 32 * 1024 * 1024
    assert after["value"] < peak["value"] - 32 * 1024 * 1024
    assert peak["scope"] == "child_process_lifetime"
    assert peak["unit"] == "bytes"
    with pytest.raises(subprocess.CalledProcessError):
        measure_peak_rss([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(ValueError):
        measure_peak_rss("not a command list")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_operator_profile_and_trace(tmp_path, device):
    if (device == "mps" and not torch.backends.mps.is_available()) or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        pytest.skip(f"{device} unavailable")
    inputs = torch.ones(32, 64, device=device)
    weights = torch.ones(64, 64, device=device)
    calls = []

    def workload():
        calls.append(1)
        return inputs @ weights

    trace = tmp_path / "trace.json"
    report = profile_workload(workload, device=device, trace_path=trace)
    assert calls == [1]
    assert any(row["operator"] == "aten::mm" for row in report["operators"])
    assert json.loads(trace.read_text())
    assert json.loads(json.dumps(report)) == report
    with pytest.raises(FileExistsError):
        profile_workload(workload, device=device, trace_path=trace)
    assert calls == [1]
    if device == "mps":
        assert report["diagnostics"][0]["code"] == "mps_gpu_trace_unavailable"
        assert all(row["device_self_seconds"] is None for row in report["operators"])
    with pytest.raises(ValueError):
        profile_workload(workload, device=device, limit=0)


def test_rss_unavailable_platform_does_not_launch(monkeypatch):
    from torchscan.process import rss

    monkeypatch.setattr(rss.sys, "platform", "win32")
    with pytest.raises(NotImplementedError):
        measure_peak_rss([sys.executable, "-c", "raise AssertionError('must not run')"])


@pytest.mark.parametrize("cuda_trace", [True, False])
def test_cuda_profile_units_and_unavailable_time(monkeypatch, cuda_trace):
    from torchscan import profiler as profiling

    monkeypatch.setattr(profiling, "_synchronizer", lambda _device: (torch.device("cuda:0"), Mock()))
    supported = {torch.profiler.ProfilerActivity.CPU}
    if cuda_trace:
        supported.add(torch.profiler.ProfilerActivity.CUDA)
    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: supported)
    native = MagicMock()
    native.__enter__.return_value = native
    native.key_averages.return_value = [
        SimpleNamespace(
            key="aten::mm",
            count=1,
            input_shapes=[[2, 2]],
            self_cpu_time_total=50,
            self_cuda_time_total=100,
            self_cpu_memory_usage=16,
        )
    ]
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: native)
    report = profile_workload(lambda: None, device="cuda")
    row = report["operators"][0]
    assert row["cpu_self_seconds"] == pytest.approx(50e-6)
    assert row["device_self_seconds"] == (pytest.approx(100e-6) if cuda_trace else None)
    assert bool(report["diagnostics"]) is not cuda_trace
    with pytest.raises(TypeError):
        profile_workload(None, device="cuda")
