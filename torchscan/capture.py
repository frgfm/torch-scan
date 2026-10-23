# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
import platform
from collections.abc import Callable
from contextlib import ExitStack
from importlib import import_module
from threading import get_ident
from typing import Any

import torch
from torch import nn

from .crawler import _describe, _package_version
from .replay import _METHOD, _SCOPE, OperatorCapture

__all__ = ["capture_workload"]


def capture_workload(
    workload: Callable[[], object],
    *,
    inputs: Any = None,
    modules: nn.Module | list[nn.Module] | None = None,
    max_events: int = 100_000,
    max_bytes: int = 16 * 1024 * 1024,
) -> OperatorCapture:
    """Collect bounded operator metadata in C++, invoking the Python workload once.

    Args:
        workload: Zero-argument callable; its outputs, exceptions, state and execution mode belong to the caller.
        inputs: Optional caller-supplied input description, never forwarded to workload.
        modules: Optional module roots for Python module-entry/exit labels; no Python callback runs per operator.
        max_events: Positive maximum number of retained outermost native events.
        max_bytes: Positive maximum retained metadata payload bytes, excluding container overhead.

    Returns:
        JSON-serializable metadata for analyze_capture. Captures calling-thread eager operations;
        compiled kernels and other threads are outside this scope. Loss is explicit. No tensors are retained.

    Raises:
        ImportError: If the separately built torchscan-native collector is unavailable or incompatible.
        ValueError: If limits are invalid.
        TypeError: If workload or module roots are invalid.
        RuntimeError: If another capture is active on this thread.
    """
    if not callable(workload):
        raise TypeError("workload must be callable.")
    for name, value in (("max_events", max_events), ("max_bytes", max_bytes)):
        if type(value) is not int or not 0 < value < 2**63:
            raise ValueError(f"{name} must be a positive integer below 2**63.")
    roots = [] if modules is None else [modules] if isinstance(modules, nn.Module) else modules
    if not isinstance(roots, list) or any(not isinstance(root, nn.Module) for root in roots):
        raise TypeError("modules must be a module or list of modules.")
    try:
        native = import_module("_torchscan_native")
    except ImportError as error:
        raise ImportError(
            "Build torchscan-native against the installed PyTorch; see the native capture documentation."
        ) from error
    if native.torch_version != str(torch.__version__).split("+")[0]:
        raise ImportError("torchscan-native was built for another PyTorch version; rebuild it in this environment.")
    described = _describe(inputs)
    execution = {
        "grad_enabled_before": torch.is_grad_enabled(),
        "inference_mode_before": torch.is_inference_mode_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }
    session = native.start(max_events, max_bytes)
    owner, foreign_calls = get_ident(), 0

    def label(name: str | None = None) -> None:
        nonlocal foreign_calls
        if get_ident() != owner:
            foreign_calls += int(name is not None)
        elif name is None:
            native.module(session)
        else:
            native.module(session, name)

    try:
        with ExitStack() as hooks:
            seen: set[int] = set()
            for index, root in enumerate(roots):
                for path, module in root.named_modules():
                    if id(module) in seen:
                        continue
                    seen.add(id(module))
                    path = f"{type(root).__name__}{index if len(roots) > 1 else ''}{'.' + path if path else ''}"
                    pre = module.register_forward_pre_hook(lambda _mod, _args, name=path: label(name))
                    post = module.register_forward_hook(lambda _mod, _args, _out: label(), always_call=True)
                    hooks.callback(pre.remove)
                    hooks.callback(post.remove)
            workload()
    finally:
        recorded = json.loads(native.stop(session))
    recorded["errors"] += foreign_calls
    return {
        "schema_version": 1,
        "report_type": "operator_capture",
        "inputs": described,
        **recorded,
        "context": {
            **execution,
            "torch_version": str(torch.__version__),
            "torchscan_version": _package_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "method": _METHOD,
            "scope": _SCOPE,
            "boundary": "outermost_native_operator",
            "max_events": max_events,
            "max_bytes": max_bytes,
            "unobserved_module_calls": foreign_calls,
        },
    }
