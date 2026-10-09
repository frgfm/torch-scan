import json
import os
import sys

import torch

from torchscan import compare_benchmarks, measure_workload, render_report


def test_workload_evidence_round_trip(capsys):
    model = torch.nn.Linear(4, 2).double().train()
    inputs = torch.ones(3, 4, dtype=torch.float64)
    threads = torch.get_num_threads()
    calls = 0

    def workload():
        nonlocal calls
        calls += 1
        assert model.training
        assert not torch.is_grad_enabled()
        assert torch.get_num_threads() == threads
        return model(inputs).sinc()

    with torch.no_grad():
        report = measure_workload(
            workload,
            device=inputs.device,
            inputs=inputs,
            work_units=3,
            min_run_time=0.001,
            min_repeats=2,
            profile=True,
            rss_command=[sys.executable, "-c", "values = bytearray(1024 * 1024)"],
        )
    assert calls > 3
    assert model.training
    assert model.weight.dtype == torch.float64
    assert torch.get_num_threads() == threads
    assert report["inputs"]["shape"] == [3, 4]
    assert report["inputs"]["dtype"] == "torch.float64"
    assert report["totals"]["latency"]["status"] == "complete"
    assert report["totals"]["throughput"]["unit"] == "samples/s"
    assert report["totals"]["operator_flops"] == report["operator_flops"]["total"]
    assert report["totals"]["operator_flops"]["status"] == "partial"
    assert report["totals"]["operator_flops"]["value"] is None
    assert report["totals"]["operator_flops"]["known_value"] > 0
    assert report["totals"]["peak_memory"]["value"] == report["memory"]["peak_bytes"] > 0
    assert report["totals"]["peak_memory"]["scope"] == "pytorch_tensor_bytes"
    assert report["profile"]["operators"]
    assert report["context"]["profile_status"] == "complete"
    rss = report["totals"]["process_peak_rss"]
    assert rss["scope"] == "child_process_lifetime"
    if sys.platform in {"linux", "darwin"} and hasattr(os, "wait4"):
        assert rss["status"] == "complete"
        assert rss["value"] > 0
    else:
        assert rss["status"] == "unavailable"
        assert rss["value"] is None
    summary = capsys.readouterr().out
    assert all(text in summary for text in ("ms", "samples/s", "MiB", "partial (known lower bound:", "aten.sinc"))
    restored = json.loads(json.dumps(report, allow_nan=False))
    assert restored == report
    html = render_report(restored)
    assert "aten.sinc" in html
    assert "lower bound" in html
    assert "pytorch_tensor_bytes" in html
    calls_before = calls
    skipped = measure_workload(workload, device="cpu", metrics=(), print_summary=False)
    assert calls == calls_before
    assert all(
        metric["status"] == "unavailable" and metric["method"] == "not_requested"
        for metric in skipped["totals"].values()
    )
    assert "unavailable" in render_report(skipped)

    def copying_workload():
        inputs.sinc()
        return inputs.clone()

    options = {"device": "cpu", "inputs": inputs, "min_run_time": 0.001, "min_repeats": 2, "print_summary": False}
    before = measure_workload(copying_workload, **options)
    after = measure_workload(lambda: inputs, **options)
    comparison = compare_benchmarks(before, after, check=lambda: torch.testing.assert_close(copying_workload(), inputs))
    assert comparison["output_check"] == "passed"
    assert comparison["totals"]["latency"]["status"] == "complete"
    assert comparison["totals"]["peak_memory"]["status"] == "unavailable"
    assert comparison["totals"]["peak_memory"]["delta"] is None
    assert "aten.sinc" in render_report(comparison)
