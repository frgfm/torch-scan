# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Existing model factories used by the declarative autoresearch example."""

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models import resnet18


def baseline() -> torch.nn.Module:
    """Construct the initial, locally initialized model without downloads."""
    return resnet18(weights=None)


def candidate() -> torch.nn.Module:
    """Fold BatchNorm into convolutions for batch-one inference at fixed FP32."""
    model = baseline().eval()
    for module in model.modules():
        for conv_name, norm_name in (("conv1", "bn1"), ("conv2", "bn2"), ("0", "1")):
            conv, norm = getattr(module, conv_name, None), getattr(module, norm_name, None)
            if isinstance(conv, torch.nn.Conv2d) and isinstance(norm, torch.nn.BatchNorm2d):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, norm))
                setattr(module, norm_name, torch.nn.Identity())
    return model
