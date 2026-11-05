# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
import platform
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from importlib import import_module
from threading import get_ident
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_map

from .crawler import _describe, _package_version
from .replay import _METHOD, _SCOPE, OperatorCapture

__all__ = ["capture_workload"]

# These wrappers consist of dispatcher operations. Unknown composites stay atomic:
# a CompositeImplicitAutograd registration alone does not establish coverage.
_COMPOSITES = tuple(
    "aten::" + name
    for name in (
        "linear",
        "matmul",
        "batch_norm",
        "_batch_norm_impl_index",
        "layer_norm",
        "group_norm",
        "softmax",
        "log_softmax",
        "scaled_dot_product_attention",
        "_scaled_dot_product_attention_math",
        "dropout",
        "selu",
        "selu_",
        "log_sigmoid",
        "rrelu",
        "rrelu_",
        "avg_pool1d",
        "max_pool1d_with_indices",
        "adaptive_avg_pool1d",
        "adaptive_max_pool1d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
        "max_pool2d",
        "max_pool3d",
        "_weight_norm",
        *(f"{prefix}{rank}d" for prefix in ("conv", "conv_transpose") for rank in (1, 2, 3)),
    )
)


def _custom_formula(formula: Callable[..., int | float]) -> Callable[..., int | float]:
    def count(*args: Any, out_val: Any, **kwargs: Any) -> int | float:
        if getattr(formula, "_get_raw", False):
            return formula(*args, out_val=out_val, **kwargs)
        args, kwargs, out_shape = tree_map(
            lambda value: value.shape if isinstance(value, torch.Tensor) else value, (args, kwargs, out_val)
        )
        return formula(*args, out_shape=out_shape, **kwargs)

    return count


def capture_workload(
    workload: Callable[[], object],
    *,
    inputs: Any = None,
    modules: nn.Module | list[nn.Module] | None = None,
    custom_mapping: Mapping[Any, Callable[..., int | float]] | None = None,
    max_events: int = 100_000,
    max_bytes: int = 16 * 1024 * 1024,
) -> OperatorCapture:
    """Collect bounded operator metadata in C++, invoking the Python workload once.

    Args:
        workload: Zero-argument callable; its outputs, exceptions, state and execution mode belong to the caller.
        inputs: Optional caller-supplied input description, never forwarded to workload.
        modules: Module roots for entry/exit labels, or None for automatic discovery. An empty list disables labels.
        custom_mapping: Per-call operator packet formulas. Only these operators call Python;
            callbacks receive shapes, or live tensors with ``_get_raw = True``. Results are saved for replay.
        max_events: Positive maximum number of retained dispatcher events.
        max_bytes: Positive maximum retained metadata payload bytes, excluding container overhead.

    Returns:
        JSON-serializable metadata for analyze_capture. Captures calling-thread eager operations and their autograd workers;
        compiled kernels and unrelated threads are outside this scope. Loss is explicit. No tensors are retained.

    Raises:
        ImportError: If the TorchScan collector is unavailable or incompatible.
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
    mapping = dict(custom_mapping or {})
    if any(getattr(operator, "_overloadpacket", operator) is not operator for operator in mapping):
        raise TypeError("custom_mapping keys must be operator packets, not overloads.")
    if any(not callable(formula) for formula in mapping.values()):
        raise TypeError("Custom formulas must be callable.")
    try:
        native = import_module("_torchscan_native")
    except ImportError as error:
        raise ImportError(
            "Reinstall TorchScan against the installed PyTorch; see the installation documentation."
        ) from error
    if native.torch_version != str(torch.__version__).split("+")[0]:
        raise ImportError("TorchScan was built for another PyTorch version; reinstall it in this environment.")
    described = _describe(inputs)
    execution = {
        "grad_enabled_before": torch.is_grad_enabled(),
        "inference_mode_before": torch.is_inference_mode_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }
    session = native.start(
        max_events,
        max_bytes,
        {str(op).replace(".", "::", 1): _custom_formula(fn) for op, fn in mapping.items()},
        _COMPOSITES,
    )
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
            if modules is None:
                names: dict[int, str] = {}

                def enter(module: nn.Module, _args: Any) -> None:
                    if id(module) not in names:
                        for path, child in module.named_modules():
                            names.setdefault(id(child), f"{type(module).__name__}{'.' + path if path else ''}")
                    label(names[id(module)])

                pre = nn.modules.module.register_module_forward_pre_hook(enter)
                hooks.callback(pre.remove)
                post = nn.modules.module.register_module_forward_hook(
                    lambda _mod, _args, _out: label(), always_call=True
                )
                hooks.callback(post.remove)
            for index, root in enumerate(roots):
                for path, module in root.named_modules():
                    if id(module) in seen:
                        continue
                    seen.add(id(module))
                    path = f"{type(root).__name__}{index if len(roots) > 1 else ''}{'.' + path if path else ''}"
                    pre = module.register_forward_pre_hook(lambda _mod, _args, name=path: label(name))
                    hooks.callback(pre.remove)
                    post = module.register_forward_hook(lambda _mod, _args, _out: label(), always_call=True)
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
            "boundary": "dispatcher_operator",
            "max_events": max_events,
            "max_bytes": max_bytes,
            "unobserved_module_calls": foreign_calls,
        },
    }
