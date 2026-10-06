# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Compare per-sample and batched inference on your CPU, CUDA, or MPS device without downloads."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from torchscan import compare_benchmarks, measure_latency, profile_workload, render_report
from torchscan.process import measure_peak_memory, measure_peak_rss
from torchscan.report import metric_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rss", action="store_true", help="Also run each variant in a fresh process (Linux/macOS).")
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "torchscan-comparison")
    parser.add_argument("--rss-child", choices=["per_sample", "batched"], help=argparse.SUPPRESS)
    options = parser.parse_args()
    device = torch.device(options.device)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = torch.nn.Linear(128, 128).eval().to(options.device)
    inputs = torch.randn(32, 128, device=options.device)

    @torch.inference_mode()
    def per_sample():
        return torch.stack([model(row) for row in inputs])

    @torch.inference_mode()
    def batched():
        return model(inputs)

    if options.rss_child:
        # Includes imports, model/input loading, and 100 completed calls; no profiler in this RSS trial.
        for _ in range(100):
            {"per_sample": per_sample, "batched": batched}[options.rss_child]()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        return

    reports = []
    for name, workload in (("per_sample", per_sample), ("batched", batched)):
        report = measure_latency(workload, device=options.device, inputs=inputs, work_units=32, work_unit="samples")
        report["context"].update(experiment=name, model="Linear(128,128), seed=0", execution_mode="eval/inference_mode")
        method = {
            "cpu": "torch.profiler.export_memory_timeline",
            "cuda": "torch.cuda.max_memory_reserved",
            "mps": "torch.accelerator.memory.max_memory_reserved",
        }[device.type]
        scope = "workload_tensor_peak" if device.type == "cpu" else "allocator_reserved_peak"
        try:
            memory = measure_peak_memory(workload, device=options.device)
        except NotImplementedError as error:
            report["totals"]["pytorch_peak_memory"] = metric_result(
                status="unavailable", unit="bytes", scope=scope, method=method
            )
            report["context"]["memory_unavailable_reason"] = str(error)
        else:
            report["totals"]["pytorch_peak_memory"] = metric_result(
                status="complete", value=memory["peak_bytes"], unit="bytes", scope=scope, method=method
            )
            report["context"]["memory_pass"] = memory
            if "allocated_peak_bytes" in memory:
                report["totals"]["pytorch_allocated_peak"] = metric_result(
                    status="complete",
                    value=memory["allocated_peak_bytes"],
                    unit="bytes",
                    scope="allocator_allocated_peak",
                    method=method.replace("reserved", "allocated"),
                )
        if options.rss:
            report["totals"]["process_peak_rss"] = measure_peak_rss([
                sys.executable,
                str(Path(__file__).resolve()),
                "--device",
                options.device,
                "--rss-child",
                name,
            ])
        reports.append(report)

    before, after = reports
    after["profile"] = profile_workload(batched, device=options.device, limit=10)
    comparison = compare_benchmarks(
        before, after, check=lambda: torch.testing.assert_close(per_sample(), batched(), rtol=1e-4, atol=1e-5)
    )
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (options.output / "comparison.html").write_text(
        render_report(comparison, title="Per-sample versus batched inference"), encoding="utf-8"
    )
    print(f"PyTorch {torch.__version__}; {after['context']['device_name']}; device={options.device}; threads=1")
    print("Variant | first call ms | warmed median ms | IQR ms | samples/s | peak process RSS MiB")
    for name, report in zip(("per_sample", "batched"), reports, strict=True):
        totals = report["totals"]
        rss = totals.get("process_peak_rss")
        rss_text = f"{rss['value'] / 1024**2:.2f}" if rss else "not requested"
        print(
            f"{name} | {totals['first_call_latency']['value'] * 1000:.4f} | "
            f"{totals['latency']['value'] * 1000:.4f} | {totals['latency_iqr']['value'] * 1000:.4f} | "
            f"{totals['throughput']['value']:.0f} | {rss_text}"
        )
    print(f"Output check: {comparison['correctness']}; latency change: {comparison['latency_change']}")
    print(f"Saved {options.output.resolve()}/comparison.json and comparison.html")


if __name__ == "__main__":
    main()
