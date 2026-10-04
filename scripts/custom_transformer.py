# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Analyze a native encoder and a custom head. Requires unreleased main; no downloads."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from torchscan import ModuleCall, ModuleEstimates, ModuleHandler, crawl_module


class SineHead(nn.Module):
    """Project each token, apply sine, and scale the result."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 2, bias=False)

    def forward(self, tokens, *, scale: float, tag: str):
        logits = torch.sin(self.projection(tokens)) * scale
        return {"logits": (logits,), "tag": tag}


class CustomTransformer(nn.Module):
    """Use built-in encoder estimates and a caller-supplied head estimate."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(4, 2, 8, dropout=0, batch_first=True)
        self.head = SineHead()

    def forward(self, source, *, scale: float = 0.5, tag: str = "demo", src_mask=None):
        tokens = self.encoder(source, src_mask=src_mask)
        return self.head(tokens, scale=scale, tag=tag)


def estimate_head(call: ModuleCall) -> ModuleEstimates:
    """Count the whole head under the stated dense real arithmetic convention."""
    head = call.module
    tokens = call.args[0]
    logits = call.output["logits"][0]
    assert type(head) is SineHead
    assert tokens.is_floating_point()
    assert not tokens.is_complex()
    assert tokens.shape[-1] == head.projection.in_features
    assert logits.shape == (*tokens.shape[:-1], head.projection.out_features)
    assert call.output["tag"] == call.kwargs["tag"]
    assert isinstance(call.kwargs["scale"], float)
    elements = logits.numel()
    terms = head.projection.in_features
    return {
        # K multiplies and K-1 adds per projected element; sine and scale add two operations.
        "module_flops": elements * (2 * terms - 1) + 2 * elements,
        "macs": elements * terms,
        # Linear reads inputs/weights and writes outputs. Sine and scale each read/write outputs.
        "dmas": tokens.numel() + head.projection.weight.numel() + 5 * elements,
        # The head does not mix token positions. These fields describe the head call only.
        "receptive_field": 1,
        "effective_stride": 1,
        "effective_padding": 0,
    }


HEAD_HANDLER = ModuleHandler(
    estimate_head,
    subtree_metrics=frozenset({
        "module_flops",
        "macs",
        "dmas",
        "receptive_field",
        "effective_stride",
        "effective_padding",
    }),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Save the full report, including diagnostics.")
    options = parser.parse_args()
    model = CustomTransformer()
    report = crawl_module(
        model,
        args=(torch.ones(1, 3, 4),),
        kwargs={"scale": 0.5, "tag": "demo"},
        custom_modules={SineHead: HEAD_HANDLER},
    )
    print("CPU float32; batch=1; tokens=3; encoder width=4; heads=2; hidden width=8; eval/no_grad")
    print("DMAs count logical element reads/writes. Sine costs one FLOP per output element.")
    print("Module | MACs | DMAs | Count includes")
    for path, includes in (("encoder", "encoder children"), ("head", "projection, sine, scale")):
        row = next(layer for layer in report["layers"] if layer["path"] == path)
        print(f"{path} | {row['metrics']['macs']['value']} | {row['metrics']['dmas']['value']} | {includes}")
    print(f"Total | {report['totals']['macs']['value']} | {report['totals']['dmas']['value']} | each block once")
    operators = report["totals"]["operator_flops"]
    print(f"Operator FLOPs: {operators['status']}. Check diagnostics before you use the count.")
    for diagnostic in report["diagnostics"]:
        if "operator" in diagnostic:
            print(f"  {diagnostic['operator']}: {diagnostic['message']}")
    if options.json:
        options.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
