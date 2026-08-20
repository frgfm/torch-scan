# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Torchvision benchmark."""

import torch
from torchvision import models

from torchscan import crawl_module

TORCHVISION_MODELS = [
    "alexnet",
    "googlenet",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inception_v3",
    "squeezenet1_0",
    "squeezenet1_1",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "mobilenet_v2",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
]


def _known(report, name):
    metric = report["totals"][name]
    return metric["value"] if metric["status"] == "complete" else metric["known_value"] or 0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    headers = ["Model", "Params (M)", "FLOPs (G)", "MACs (G)", "DMAs (G)"]
    widths = [20, 10, 10, 10, 10]
    print(" | ".join(f"{header:<{width}}" for header, width in zip(headers, widths, strict=True)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for name in TORCHVISION_MODELS:
        model = models.__dict__[name](weights=None).eval().to(device)
        input_shape = (3, 299, 299) if "inception" in name else (3, 224, 224)
        report = crawl_module(model, input_shape)
        values = [
            name,
            f"{_known(report, 'parameters') / 1e6:.2f}",
            f"{_known(report, 'module_flops') / 1e9:.2f}",
            f"{_known(report, 'macs') / 1e9:.2f}",
            f"{_known(report, 'dmas') / 1e9:.2f}",
        ]
        print(" | ".join(f"{value:<{width}}" for value, width in zip(values, widths, strict=True)))


if __name__ == "__main__":
    main()
