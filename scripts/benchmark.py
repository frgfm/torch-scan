# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""No-download CNN benchmark: python scripts/benchmark.py --json /tmp/torchscan-matrix.json."""

import argparse
import json
from pathlib import Path

import torch

from torchscan import crawl_module

TORCHVISION_MODELS = ["resnet18", "mobilenet_v2", "resnext50_32x4d"]


def metric_cell(metric, scale=1):
    if metric["status"] == "complete":
        return f"{metric['value'] / scale:.6g} [complete]"
    if metric["status"] == "partial":
        return f">={metric['known_value'] / scale:.6g} [partial]"
    return "n/a [unavailable]"


def main():
    from torchvision import models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, help="Save full reports and diagnostics.")
    options = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    columns = [
        ("parameters", "Params (M)", 1e6),
        ("module_flops", "Module FLOPs (G)", 1e9),
        ("operator_flops", "Operator FLOPs (G)", 1e9),
        ("macs", "MACs (G)", 1e9),
        ("dmas", "DMAs (G)", 1e9),
    ]
    print(f"PyTorch {torch.__version__}; device={options.device}; seed=0; eval/no_grad; float32; input=(1,3,32,32)")
    print("Partial values are lower bounds. FLOPs do not measure latency.")
    print(" | ".join([f"{'Model':20}"] + [f"{label:28}" for _, label, _ in columns]))
    reports = {}
    for name in TORCHVISION_MODELS:
        model = getattr(models, name)(weights=None).to(options.device)
        report = crawl_module(model, args=(torch.ones(1, 3, 32, 32, device=options.device),))
        reports[name] = report
        cells = [f"{name:20}"] + [f"{metric_cell(report['totals'][metric], scale):28}" for metric, _, scale in columns]
        print(" | ".join(cells))
        for item in report["operator_flops"]["diagnostics"]:
            print(f"  {item['code']}: {item['message']}")
    if options.json:
        options.json.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
