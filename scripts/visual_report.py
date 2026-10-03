# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Generate reproducible local-model examples; no network or pretrained weights."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from torchscan import crawl_module, render_report


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4, bias=False))
        self.tied = nn.Linear(4, 4, bias=False)
        self.tied.weight = self.block[0].weight

    def forward(self, inputs):
        return self.tied(self.block(self.block(inputs)))


class Sine(nn.Module):
    def forward(self, inputs):
        return torch.sin(inputs)


class RepeatedConv(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1, bias=False)

    def forward(self, inputs):
        return self.conv(self.conv(inputs))


def explorer_model(width=8, *, incomplete=False):
    """A nested model with depthwise work, repeated calls, and optional unknown work."""
    expanded = width * 2
    return nn.Sequential(
        OrderedDict(
            stem=nn.Sequential(OrderedDict(conv=nn.Conv2d(3, width, 3, padding=1, bias=False), identity=nn.Identity())),
            features=nn.Sequential(
                OrderedDict(
                    block1=nn.Sequential(
                        OrderedDict(
                            expand=nn.Conv2d(width, expanded, 1, bias=False),
                            depthwise=nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
                            project=nn.Conv2d(expanded, width, 1, bias=False),
                        )
                    ),
                    block2=RepeatedConv(width),
                )
            ),
            head=nn.Sequential(
                OrderedDict(pool=nn.AvgPool2d(16), flatten=nn.Flatten(), linear=nn.Linear(width, 4, bias=False))
            ),
            probe=Sine() if incomplete else nn.Identity(),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/visual-report"))
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    inputs = torch.ones(2, 4)
    image_inputs = torch.ones(1, 3, 16, 16)
    explorer_before = crawl_module(explorer_model(8), args=(image_inputs,))
    explorer_after = crawl_module(explorer_model(4), args=(image_inputs,))
    reports = {
        "shared": crawl_module(SharedModel(), args=(inputs,)),
        "structure": crawl_module(explorer_model(8), args=(image_inputs,), mode="structure"),
        "explorer": crawl_module(explorer_model(8, incomplete=True), args=(image_inputs,)),
        "explorer-complete": explorer_before,
        "explorer-slim": explorer_after,
    }
    for name, report in reports.items():
        (options.output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for output_format in ("html", "svg"):
            artifact = options.output / f"{name}.{output_format}"
            artifact.write_text(
                render_report(report, format=output_format, title=f"TorchScan · {name}"), encoding="utf-8"
            )
    for output_format in ("html", "svg"):
        (options.output / f"explorer-comparison.{output_format}").write_text(
            render_report(
                explorer_after,
                before=explorer_before,
                format=output_format,
                title="TorchScan · convolution width 8 → 4",
            ),
            encoding="utf-8",
        )
    print(f"Examples saved to {options.output.resolve()}")


if __name__ == "__main__":
    main()
