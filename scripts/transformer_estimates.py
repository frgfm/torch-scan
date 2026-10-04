# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Tiny native Transformer estimates: python scripts/transformer_estimates.py --json /tmp/transformers.json."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from torchscan import crawl_module


class WrappedAttention(nn.Module):
    """Preserve the complete attention call through a custom container."""

    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(4, 2, batch_first=True)

    def forward(self, query, key, value, **kwargs):
        return self.attention(query, key, value, **kwargs)


def encoder_layer():
    return nn.TransformerEncoderLayer(4, 2, 8, dropout=0, batch_first=True)


def decoder_layer():
    return nn.TransformerDecoderLayer(4, 2, 8, dropout=0, batch_first=True)


def native_transformer():
    # Explicitly disable padding-based nested packing in the native encoder.
    encoder = nn.TransformerEncoder(encoder_layer(), 1, norm=nn.LayerNorm(4), enable_nested_tensor=False)
    decoder = nn.TransformerDecoder(decoder_layer(), 1, norm=nn.LayerNorm(4))
    return nn.Transformer(
        d_model=4,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
        batch_first=True,
        custom_encoder=encoder,
        custom_decoder=decoder,
    )


def metric_cell(metric):
    if metric["status"] == "complete":
        return str(metric["value"])
    if metric["status"] == "partial":
        return f">={metric['known_value']} [partial]"
    return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Save complete JSON reports and diagnostics.")
    options = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    source = torch.ones(1, 3, 4)
    target = torch.ones(1, 2, 4)
    causal_source = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    causal_target = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)
    source_padding = torch.zeros(1, 3, dtype=torch.bool)
    target_padding = torch.zeros(1, 2, dtype=torch.bool)
    sequence_first = source.transpose(0, 1)
    reports = {
        "self": crawl_module(
            nn.MultiheadAttention(4, 2, batch_first=True),
            args=(source, source, source),
            kwargs={"need_weights": False},
        ),
        "causal-self-sequence-first": crawl_module(
            nn.MultiheadAttention(4, 2, batch_first=False),
            args=(sequence_first, sequence_first, sequence_first),
            kwargs={"attn_mask": causal_source, "is_causal": True, "need_weights": True, "average_attn_weights": False},
        ),
        "cross-unequal-widths": crawl_module(
            nn.MultiheadAttention(4, 2, kdim=6, vdim=5, batch_first=True),
            args=(target, torch.ones(1, 3, 6), torch.ones(1, 3, 5)),
            kwargs={"need_weights": False},
        ),
        "wrapped-cross": crawl_module(
            WrappedAttention(), args=(target, source, source), kwargs={"need_weights": False}
        ),
        "encoder-layer": crawl_module(encoder_layer(), args=(source,)),
        "decoder-layer": crawl_module(decoder_layer(), args=(target, source)),
        "encoder-stack-causal": crawl_module(
            nn.TransformerEncoder(encoder_layer(), 1, norm=nn.LayerNorm(4), enable_nested_tensor=False),
            args=(source,),
            kwargs={"mask": causal_source, "src_key_padding_mask": source_padding, "is_causal": True},
        ),
        "decoder-stack-causal": crawl_module(
            nn.TransformerDecoder(decoder_layer(), 1, norm=nn.LayerNorm(4)),
            args=(target, source),
            kwargs={
                "tgt_mask": causal_target,
                "memory_mask": None,
                "tgt_key_padding_mask": target_padding,
                "memory_key_padding_mask": source_padding,
                "tgt_is_causal": True,
                "memory_is_causal": False,
            },
        ),
        "transformer": crawl_module(native_transformer(), args=(source, target)),
        "transformer-masked": crawl_module(
            native_transformer(),
            args=(source, target),
            kwargs={
                "src_mask": None,
                "tgt_mask": causal_target,
                "memory_mask": None,
                "src_key_padding_mask": source_padding,
                "tgt_key_padding_mask": target_padding,
                "memory_key_padding_mask": source_padding,
                "src_is_causal": False,
                "tgt_is_causal": True,
                "memory_is_causal": False,
            },
        ),
    }
    print(f"PyTorch {torch.__version__}; CPU float32; seed=0; batch=1; E=4; H=2; F=8; eval/no_grad")
    print("MACs are independent contraction estimates. DMAs are logical element accesses, not measured traffic.")
    for name, report in reports.items():
        totals = report["totals"]
        print(
            f"{name}: MACs={metric_cell(totals['macs'])}; DMAs={metric_cell(totals['dmas'])}; "
            f"module FLOPs={metric_cell(totals['module_flops'])}"
        )
        for layer in report["layers"]:
            if "token_dependencies" in layer:
                print(f"  {layer['path'] or '<root>'} token_dependencies={json.dumps(layer['token_dependencies'])}")
        for diagnostic in report["diagnostics"]:
            print(f"  {diagnostic['metric']}: {diagnostic['message']}")
    if options.json:
        options.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
