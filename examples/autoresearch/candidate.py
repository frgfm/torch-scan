# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Candidate inference implementation for the agent to edit."""

import torch


@torch.inference_mode()
def forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run the complete batch in one model call."""
    return model(inputs)
