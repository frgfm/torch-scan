# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Memory-heavy candidate: the output check passes, but a resource gate should reject it."""

import torch


@torch.inference_mode()
def forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Retain a large temporary tensor while returning the reference output."""
    output = model(inputs)
    temporary = torch.zeros(64 * 1024 * 1024, device=inputs.device)
    return output + temporary[0]
