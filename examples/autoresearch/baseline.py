# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Initial inference implementation: run one image at a time."""

import torch


@torch.inference_mode()
def forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run images separately and combine their logits."""
    return torch.cat([model(image.unsqueeze(0)) for image in inputs], dim=0)
