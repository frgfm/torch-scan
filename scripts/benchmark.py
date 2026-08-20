# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Torchvision metrics and crawler timing benchmark."""

import os
import platform
import statistics
import time
import warnings

import torch
from torch import nn
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


def _benchmark_device():
    default = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(os.environ.get("TORCHSCAN_BENCH_DEVICE", default))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TORCHSCAN_BENCH_DEVICE requests unavailable CUDA.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("TORCHSCAN_BENCH_DEVICE requests unavailable MPS.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("TORCHSCAN_BENCH_DEVICE must select cpu, cuda, or mps.")
    return device


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _format_metric(result, scale=1, precision=2):
    value = result["known_value"]
    if value is None:
        return "-"
    prefix = ">=" if result["status"] == "partial" else ""
    return f"{prefix}{value / scale:.{precision}f}"


def _print_model_metrics(device):
    headers = ["Model", "Params (M)", "FLOPs (G)", "MACs (G)", "DMAs (G)", "RF"]
    widths = [20, 12, 12, 12, 12, 10]
    print("\nTorchvision model metrics")
    print(" | ".join(f"{header:<{width}}" for header, width in zip(headers, widths, strict=True)))
    print("-+-".join("-" * width for width in widths))
    for name in TORCHVISION_MODELS:
        model = models.__dict__[name](weights=None).eval().to(device)
        input_shape = (3, 299, 299) if "inception" in name else (3, 224, 224)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = crawl_module(model, input_shape)
        receptive_field = next(
            (
                layer["metrics"]["receptive_field"]
                for layer in report["layers"]
                if "receptive_field" in layer["metrics"]
            ),
            None,
        )
        values = [
            name,
            _format_metric(report["totals"]["parameters"], 1e6),
            _format_metric(report["totals"]["module_flops"], 1e9),
            _format_metric(report["totals"]["macs"], 1e9),
            _format_metric(report["totals"]["dmas"], 1e9),
            "-" if receptive_field is None else _format_metric(receptive_field, precision=0),
        ]
        print(" | ".join(f"{value:<{width}}" for value, width in zip(values, widths, strict=True)))


def _timing_cases(device):
    shared = nn.ReLU()
    yield (
        "Shared ReLU, 100 calls",
        nn.Sequential(*([shared] * 100)).to(device),
        (torch.rand(1, 16, device=device),),
    )
    for module_count in (20, 200, 1_000):
        layers = []
        for _ in range(module_count // 2):
            layers.extend((nn.Linear(16, 16), nn.ReLU()))
        yield (
            f"Sequential, {module_count:,} modules",
            nn.Sequential(*layers).eval().to(device),
            (torch.rand(1, 16, device=device),),
        )
    yield (
        "ResNet-18",
        models.resnet18(weights=None).eval().to(device),
        (torch.rand(1, 3, 224, 224, device=device),),
    )
    yield (
        "MobileNetV2",
        models.mobilenet_v2(weights=None).eval().to(device),
        (torch.rand(1, 3, 224, 224, device=device),),
    )
    yield (
        "Native Transformer",
        nn
        .Transformer(
            d_model=64,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        .eval()
        .to(device),
        (torch.rand(1, 32, 64, device=device), torch.rand(1, 16, 64, device=device)),
    )


def _timed(function, device):
    for _ in range(5):
        _synchronize(device)
        function()
        _synchronize(device)

    samples = []
    for _ in range(15):
        _synchronize(device)
        started = time.perf_counter_ns()
        function()
        _synchronize(device)
        samples.append(time.perf_counter_ns() - started)
    p95 = statistics.quantiles(samples, n=20, method="inclusive")[18]
    return statistics.median(samples) / 1e6, p95 / 1e6


def _print_timings(device):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    headers = ["Workload", "Direct med/p95 (ms)", "Crawl med/p95 (ms)", "Overhead"]
    widths = [30, 24, 24, 10]
    print("\nCrawler timing")
    print(" | ".join(f"{header:<{width}}" for header, width in zip(headers, widths, strict=True)))
    print("-+-".join("-" * width for width in widths))
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model, args in _timing_cases(device):
            direct = lambda model=model, args=args: model(*args)
            crawl = lambda model=model, args=args: crawl_module(model, args=args)
            direct_median, direct_p95 = _timed(direct, device)
            crawl_median, crawl_p95 = _timed(crawl, device)
            values = [
                name,
                f"{direct_median:.3f} / {direct_p95:.3f}",
                f"{crawl_median:.3f} / {crawl_p95:.3f}",
                f"{crawl_median / direct_median:.2f}x",
            ]
            print(" | ".join(f"{value:<{width}}" for value, width in zip(values, widths, strict=True)))


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = _benchmark_device()
    source = os.path.realpath(crawl_module.__code__.co_filename)
    print(
        f"Python={platform.python_version()} | Torch={torch.__version__} | device={device} | "
        f"threads={torch.get_num_threads()} | source={source}:{crawl_module.__code__.co_firstlineno}"
    )
    _print_model_metrics(device)
    _print_timings(device)


if __name__ == "__main__":
    main()
