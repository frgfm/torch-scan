# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Measure clean execution and counting overhead for a checkout; no model downloads."""

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

import torch

import torchscan
from scripts.benchmark_comparison import build_model
from torchscan.benchmark import _processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pointwise", "resnet18", "bert"], default="pointwise")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = None
    if args.model == "pointwise":
        value = torch.randn(1024)

        def forward():
            output = value
            for _ in range(200):
                output = output.sin().relu() + 0.1
            return output

    else:
        model, inputs, *_ = build_model(args.model, torch.device("cpu"))

        def forward():
            return model(**inputs) if args.model == "bert" else model(inputs["input"])

    workload = torch.inference_mode()(forward)
    report = torchscan.measure_flops(workload, modules=model)
    jobs = {"baseline": workload, "counter": lambda: torchscan.measure_flops(workload, modules=model)}
    capture = None
    if hasattr(torchscan, "capture_workload"):
        capture = torchscan.capture_workload(workload, modules=model)
        jobs.update({
            "capture": lambda: torchscan.capture_workload(workload, modules=model),
            "replay": lambda: torchscan.analyze_capture(capture),
        })
    for job in jobs.values():
        for _ in range(3):
            job()
    samples = {name: [] for name in jobs}
    order = list(jobs)
    rng = random.Random(0)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    for _ in range(args.repeats):
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter()
            result = jobs[name]()
            samples[name].append(time.perf_counter() - start)
            del result
    result = {
        "model": args.model,
        "device": "cpu",
        "threads": args.threads,
        "seed": 0,
        "processor": _processor(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "events": len(capture["operators"]) if capture else None,
        "capture_bytes": len(json.dumps(capture).encode()) if capture else None,
        "flops": report["total"]["known_value"],
        "status": report["total"]["status"],
        "by_operator": report["by_operator"],
        "samples_seconds": samples,
        "median_ms": {name: statistics.median(values) * 1000 for name, values in samples.items()},
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples_seconds"}, indent=2))


if __name__ == "__main__":
    main()
