# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Owner-controlled workload and correctness contract for the autoresearch example."""

from collections.abc import Callable
from types import ModuleType

import reference
import torch
from torchvision.models import resnet18

SETTINGS = {
    "workload_id": "resnet18-fp32-inference-b4-v1",
    "device": "cpu",
    "objective": "latency",
    "min_improvement": 0.03,
    "max_memory_bytes": 256 * 1024**2,
    "trials": 3,
    "timeout": 120,
    "warmup": 2,
    "min_run_time": 0.2,
    "min_repeats": 3,
    "work_units": 4,
    "work_unit": "images",
}

FILES = ["reference.py"]


def _model_and_inputs(device: str) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = resnet18(weights=None).eval().to(device)
    inputs = torch.randn(4, 3, 32, 32, device=device)
    return model, inputs


def build(implementation: ModuleType) -> tuple[Callable[[], torch.Tensor], torch.Tensor]:
    """Return one repeatable workload and the metadata for its fixed inputs."""
    device = SETTINGS["device"]
    model, inputs = _model_and_inputs(device)

    @torch.inference_mode()
    def workload() -> torch.Tensor:
        return implementation.forward(model, inputs)

    return workload, inputs


def check(implementation: ModuleType) -> bool:
    """Check the candidate against the unchanged batched reference."""
    device = SETTINGS["device"]
    model, inputs = _model_and_inputs(device)
    candidate = implementation.forward(model, inputs)
    reference_output = reference.forward(model, inputs)
    if not torch.isfinite(candidate).all().item():
        return False
    torch.testing.assert_close(candidate, reference_output, rtol=1e-4, atol=1e-5)
    return True
