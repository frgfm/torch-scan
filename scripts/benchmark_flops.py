# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Measure complete FLOP-analysis calls on locally initialized CPU models."""

import argparse
import json
import platform
import random
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["pointwise", "mlp", "resnet18_b1", "resnet18_b8", "bert_base"], default="mlp"
    )
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--once", action="store_true", help="One analysis, without warmup, for fresh-process timing/RSS."
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")

    import torch

    from torchscan import measure_flops

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = None
    if args.model == "pointwise":
        inputs = torch.randn(1024)

        def forward():
            output = inputs
            for _ in range(200):
                output = output.sin().relu() + 0.1
            return output

    elif args.model == "bert_base":
        from transformers import BertConfig, BertModel

        model = BertModel(BertConfig()).eval()
        inputs = torch.arange(128).reshape(1, 128)
        mask = torch.ones_like(inputs)
        forward = lambda: model(input_ids=inputs, attention_mask=mask, return_dict=False)
    else:
        if args.model == "mlp":
            model = torch.nn.Sequential(
                torch.nn.Linear(256, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 10),
            ).eval()
            inputs = torch.randn(32, 256)
        else:
            from torchvision.models import resnet18

            model = resnet18(weights=None).eval()
            inputs = torch.randn(8 if args.model.endswith("b8") else 1, 3, 224, 224)
        forward = lambda: model(inputs)
    workload = torch.inference_mode()(forward)
    start = time.perf_counter_ns()
    report = measure_flops(workload, modules=model)
    first_ms = (time.perf_counter_ns() - start) / 1e6
    assert report["total"]["status"] == "complete", report["diagnostics"]
    result = {
        "model": args.model,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "threads": args.threads,
        "seed": 0,
        "first_call_ms": first_ms,
        "total": report["total"],
        "by_operator": report["by_operator"],
        "samples_ms": {},
    }
    del report
    if not args.once:
        jobs = {"plain_model": workload, "full_call": lambda: measure_flops(workload, modules=model)}
        for _ in range(5):
            for job in jobs.values():
                job()
        samples = {name: [] for name in jobs}
        order = list(jobs)
        rng = random.Random(196)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        for _ in range(args.repeats):
            rng.shuffle(order)
            for name in order:
                start = time.perf_counter_ns()
                measured = jobs[name]()
                samples[name].append((time.perf_counter_ns() - start) / 1e6)
                del measured
        result.update(
            samples_ms=samples, median_ms={name: statistics.median(values) for name, values in samples.items()}
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
