# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Pin a wheel to its build-time PyTorch release and expose host extension flags."""

import json
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any


def dynamic_metadata(_settings: Mapping[str, Any], _project: Mapping[str, Any]) -> dict[str, list[str]]:
    """Record the PyTorch release used to compile the collector."""
    return {"dependencies": [f"torch=={version('torch').split('+')[0]}"]}


def dynamic_wheel(_settings: Mapping[str, Any]) -> dict[str, bool]:
    """Allow a source archive to build against a different supported PyTorch release."""
    return {"dependencies": True}


if __name__ == "__main__":
    import torch

    sys.stdout.write(
        json.dumps({
            "root": str(Path(torch.__file__).parent),
            "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
            "pybind": {
                name: getattr(torch._C, f"_PYBIND11_{name}", None) for name in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI")
            },
        })
    )
