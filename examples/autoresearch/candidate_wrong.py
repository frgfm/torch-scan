# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Intentionally incorrect candidate: the output check should reject it."""

import torch


@torch.inference_mode()
def forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return zeros with the expected shape instead of model predictions."""
    return torch.zeros((inputs.shape[0], model.fc.out_features), device=inputs.device)
