# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Fixed, locally initialized ResNet18 reference for the autoresearch example."""

import torch


def forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run one inference batch with the reference implementation."""
    return model(inputs)
