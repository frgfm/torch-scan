# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Private shape-formula adapters to the shared native counting engine."""

from collections.abc import Callable
from functools import partial
from importlib import import_module
from typing import Any

_native = import_module("torchscan._analysis")


def _formula(name: str, *args: Any, out_shape: Any = None, **kwargs: Any) -> int:
    return _native.formula(name, args, kwargs, out_shape)


FORMULAS: dict[str, Callable[..., int]] = {name: partial(_formula, name) for name in _native.formula_names()}
_batch_norm = FORMULAS["native_batch_norm"]
