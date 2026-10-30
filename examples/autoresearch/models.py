# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Existing model factories used by the declarative autoresearch example."""

import torch
from torchvision.models import resnet18


class PerImage(torch.nn.Module):
    """The initial implementation runs one image at a time."""

    def __init__(self) -> None:
        super().__init__()
        self.model = resnet18(weights=None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.model(image.unsqueeze(0)) for image in inputs])


def baseline() -> torch.nn.Module:
    """Construct the initial, locally initialized model without downloads."""
    return PerImage()


def candidate() -> torch.nn.Module:
    """Construct the same model with batched execution."""
    return resnet18(weights=None)
