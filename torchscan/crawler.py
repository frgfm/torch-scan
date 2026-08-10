# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import os
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import Module

from .modules import module_dmas, module_flops, module_macs, module_rf
from .process import get_process_gpu_ram
from .utils import aggregate_info, format_info

__all__ = ["crawl_module", "summary"]


def _normalize_output(out: Any) -> tuple[Any, torch.Tensor]:
    """Return recursive shape metadata and the first tensor in a hooked output."""
    primary: torch.Tensor | None = None

    def _shape(value: Any, path: str) -> Any:
        nonlocal primary
        if isinstance(value, torch.Tensor):
            if primary is None:
                primary = value
            return (-1, *value.shape[1:])
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(_shape(item, f"{path}[{idx}]") for idx, item in enumerate(value))
        if isinstance(value, list):
            return [_shape(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: _shape(item, f"{path}[{key!r}]") for key, item in value.items()}
        raise TypeError(
            f"Unsupported output at {path}: {type(value).__name__} is not a tensor, tuple, list, dict, or None"
        )

    output_shape = _shape(out, "output")
    if primary is None:
        raise TypeError("Unsupported output at output: no tensor found")
    return output_shape, primary


def apply(module: Module, fn: Callable[[Module, str], None], name: str | None = None) -> None:
    """Modified version of `torch.nn.Module.apply` method

    Args:
        module: target module
        fn: function to apply to each module
        name: name of the current module
    """
    if name is None:
        name = module.__class__.__name__.lower()
    fn(module, name)
    for n, m in module.named_children():
        apply(m, fn, f"{name}.{n}")


def crawl_module(
    module: Module,
    input_shape: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    dtype: torch.dtype | Iterable[torch.dtype] | None = None,
    *,
    input_data: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
) -> dict[str, Any]:
    """Collect module information using a forward pass.

    Examples:
        >>> import torch.nn as nn
        >>> from torchscan import crawl_module
        >>> mod = nn.Conv2d(3, 8, 3)
        >>> module_info = crawl_module(mod, (3, 224, 224))

    Args:
        module: Module to inspect. It must contain at least one parameter.
        input_shape: Input shape without a batch dimension, or one shape per positional tensor input.
        dtype: One data type for every input, or one data type per input. Defaults to the first parameter's data type.
        input_data: One tensor or a non-empty list or tuple of positional tensor inputs, including all dimensions.

    Returns:
        A dictionary containing per-layer information in `layers`, parameter and buffer totals in `overall`, and
        process and framework memory estimates in `overheads`.

    Raises:
        StopIteration: If the module has no parameters.
        TypeError: If `input_data` is invalid or a hooked output has an unsupported leaf or contains no tensor.
        ValueError: If exactly one input source is not provided, `input_data` is empty, or `dtype` is used with it.

    Notes:
        `input_shape` creates a random batch of one on the first parameter's device. `input_data` is forwarded unchanged.
        Both paths run without gradients in the module's current training mode. Functional operations are not observed
        by module hooks. See the model-support and metrics guides for the complete limitations.
    """
    # Get device and data types from model
    p = next(module.parameters())

    cuda_overhead, framework_overhead = 0.0, 0.0
    if torch.cuda.is_available():
        # Process RAM - allocator RAM
        cuda_overhead = get_process_gpu_ram(os.getpid()) - (torch.cuda.memory_reserved() / 1024**2)
        # Allocator RAM - Used RAM
        framework_overhead = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**2

    # Input
    if (input_shape is None) == (input_data is None):
        raise ValueError("Exactly one of input_shape and input_data must be provided.")
    if input_data is not None:
        if dtype is not None:
            raise ValueError("dtype cannot be used with input_data.")
        if isinstance(input_data, torch.Tensor):
            input_ts = [input_data]
        elif isinstance(input_data, (list, tuple)):
            if not input_data:
                raise ValueError("input_data must not be empty.")
            for idx, input_t in enumerate(input_data):
                if not isinstance(input_t, torch.Tensor):
                    raise TypeError(f"input_data[{idx}] must be a torch.Tensor, got {type(input_t).__name__}.")
            input_ts = list(input_data)
        else:
            raise TypeError(
                "input_data must be a torch.Tensor or a non-empty list/tuple of torch.Tensor objects, "
                f"got {type(input_data).__name__}."
            )
    elif input_shape is not None:
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        if dtype is None:
            dtype = p.data.dtype
        if isinstance(dtype, torch.dtype):
            dtype = [dtype] * len(input_shape)
        input_ts = [
            torch.rand(1, *in_shape).to(dtype=dtype_, device=p.device)
            for in_shape, dtype_ in zip(input_shape, dtype, strict=False)
        ]

    pre_fw_handles, post_fw_handles = [], []
    pre_hook_tracker: dict[int, Any] = {}
    post_hook_tracker: dict[int, Any] = {}
    root_module = module

    def _is_metric_leaf(current: Module) -> bool:
        return (
            not any(current.children())
            or isinstance(current, nn.MultiheadAttention)
            or (current is root_module and isinstance(root_module, nn.Transformer))
        )

    # Hook definition
    def _hook_info(module: Module, name: str) -> None:
        def _pre_hook(module: Module, inp: tuple[torch.Tensor, ...]) -> None:
            """Pre-forward hook"""
            # Check that another hook has not been triggered at this forward stage
            if not pre_hook_tracker[id(module)]["is_used"] and (
                pre_hook_tracker[id(module)]["target"] == pre_hook_tracker[id(module)]["current"]
            ):
                # Add information
                # Params
                grad_params, nograd_params, param_size = 0, 0, 0
                num_buffers, buffer_size = 0, 0
                is_shared = False
                if _is_metric_leaf(module):
                    # Parameters
                    for p in module.parameters():
                        if id(p) not in param_ids:
                            if p.requires_grad:
                                grad_params += p.data.numel()
                            else:
                                nograd_params += p.data.numel()
                            param_size += p.data.numel() * p.data.element_size()
                            param_ids.append(id(p))
                        else:
                            is_shared = True
                    # Buffers
                    for b in module.buffers():
                        if id(b) not in param_ids:
                            num_buffers += b.numel()
                            buffer_size += b.numel() * b.element_size()
                            param_ids.append(id(b))
                        else:
                            is_shared = True

                if call_idxs.get(id(module)) is None:
                    call_idxs[id(module)] = [len(info)]
                else:
                    call_idxs[id(module)].append(len(info))

                info.append({
                    "name": name.rpartition(".")[-1],
                    "depth": len(name.split(".")) - 1,
                    "type": module.__class__.__name__,
                    "input_shape": (-1, *inp[0][0].shape[1:]),
                    "output_shape": None,
                    "grad_params": grad_params,
                    "nograd_params": nograd_params,
                    "param_size": param_size,
                    "num_buffers": num_buffers,
                    "buffer_size": buffer_size,
                    "flops": 0,
                    "macs": 0,
                    "dmas": 0,
                    "rf": 1,
                    "s": 1,
                    "p": 0,
                    "is_shared": is_shared,
                    "is_leaf": _is_metric_leaf(module),
                })
                # Mark the next hook for execution
                pre_hook_tracker[id(module)]["target"] += 1
                # Current pass already used one of the hooks
                pre_hook_tracker[id(module)]["is_used"] = True
            pre_hook_tracker[id(module)]["current"] += 1
            # All the hooks have been checked, reset the temporary values
            if pre_hook_tracker[id(module)]["current"] == len(module._forward_pre_hooks):
                pre_hook_tracker[id(module)]["current"] = 0
                pre_hook_tracker[id(module)]["is_used"] = False

        def _fwd_hook(module: Module, inputs: tuple[torch.Tensor, ...], out: Any) -> None:
            """Post-forward hook"""
            # Check that another hook has not been triggered at this forward stage
            if not post_hook_tracker[id(module)]["is_used"] and (
                post_hook_tracker[id(module)]["target"] == post_hook_tracker[id(module)]["current"]
            ):
                # Write information
                # Retrieve forward index
                if len(call_idxs[id(module)]) == 1:
                    fw_idx = call_idxs[id(module)][0]
                else:
                    # The first dictionary with output_shape=None is the correct one
                    for idx in call_idxs[id(module)]:
                        if info[idx]["output_shape"] is None:
                            fw_idx = idx
                            break

                output_shape, primary_out = _normalize_output(out)

                if not _is_metric_leaf(module):
                    tot_flops, tot_macs, tot_dmas = 0, 0, 0
                    current_rf, current_stride, current_padding = 1.0, 1.0, 0.0
                else:
                    # Compute stats for standalone layers
                    flops_out = out if isinstance(module, nn.MultiheadAttention) else primary_out
                    tot_flops = module_flops(module, inputs, flops_out)
                    tot_macs = module_macs(module, inputs[0], primary_out)
                    tot_dmas = module_dmas(module, inputs[0], primary_out)
                    current_rf, current_stride, current_padding = module_rf(module, inputs[0], primary_out)

                # Update layer information
                info[fw_idx]["output_shape"] = output_shape
                # Add them, since some modules can be used several times
                info[fw_idx]["flops"] = tot_flops
                info[fw_idx]["macs"] = tot_macs
                info[fw_idx]["dmas"] = tot_dmas
                # Compute receptive field
                info[fw_idx]["rf"] = current_rf
                info[fw_idx]["s"] = current_stride
                info[fw_idx]["p"] = current_padding

                # Mark the next hook for execution
                post_hook_tracker[id(module)]["target"] += 1
                # Current pass already used one of the hooks
                post_hook_tracker[id(module)]["is_used"] = True
            post_hook_tracker[id(module)]["current"] += 1
            # All the hooks have been checked, reset the temporary values
            if post_hook_tracker[id(module)]["current"] == len(module._forward_pre_hooks):
                post_hook_tracker[id(module)]["current"] = 0
                post_hook_tracker[id(module)]["is_used"] = False

        pre_fw_handles.append(module.register_forward_pre_hook(_pre_hook))
        post_fw_handles.append(module.register_forward_hook(_fwd_hook))
        # Handle modules that are used multiple times (with several hooks)
        pre_hook_tracker[id(module)] = {"current": 0, "target": 0, "is_used": False}
        post_hook_tracker[id(module)] = {"current": 0, "target": 0, "is_used": False}

    # Hook model
    info: list[dict[str, Any]] = []
    param_ids: list[int] = []
    call_idxs: dict[int, list[int]] = {}
    if isinstance(module, nn.Transformer):
        _hook_info(module, module.__class__.__name__.lower())
    else:
        apply(module, _hook_info)

    # Forward
    try:
        with torch.no_grad():
            module(*input_ts)
    finally:
        # Removes all hooks using their handles
        for handle in (*pre_fw_handles, *post_fw_handles):
            handle.remove()

    reserved_ram, diff_ram = 0.0, 0.0
    if torch.cuda.is_available():
        reserved_ram = torch.cuda.memory_reserved() / 1024**2
        diff_ram = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**2
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    grad_params, nograd_params, param_size = 0, 0, 0
    num_buffers, buffer_size = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            grad_params += p.data.numel()
        else:
            nograd_params += p.data.numel()
        param_size += p.data.numel() * p.data.element_size()
    for b in module.buffers():
        num_buffers += b.numel()
        buffer_size += b.numel() * b.element_size()

    # Update cumulative receptive field
    rf, s, p_ = 1, 1, 0
    for fw_idx, layer in enumerate(info):
        rf += s * (layer["rf"] - 1)
        p_ += s * layer["p"]
        s *= layer["s"]
        info[fw_idx]["rf"] = rf
        info[fw_idx]["s"] = s
        info[fw_idx]["p"] = p_

    return {
        "overheads": {
            "cuda": {
                "pre": cuda_overhead,
                "fwd": get_process_gpu_ram(os.getpid()) - reserved_ram,
            },
            "framework": {"pre": framework_overhead, "fwd": diff_ram},
        },
        "layers": info,
        "overall": {
            "grad_params": grad_params,
            "nograd_params": nograd_params,
            "param_size": param_size,
            "num_buffers": num_buffers,
            "buffer_size": buffer_size,
        },
    }


def summary(
    module: Module,
    input_shape: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    wrap_mode: str = "mid",
    max_depth: int | None = None,
    receptive_field: bool = False,
    effective_rf_stats: bool = False,
    *,
    input_data: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
) -> None:
    """Print a module summary for expected shapes or caller-provided tensors.

    Examples:
        >>> import torch.nn as nn
        >>> from torchscan import summary
        >>> mod = nn.Conv2d(3, 8, 3)
        >>> summary(mod, (3, 224, 224), receptive_field=True)

    Args:
        module: Module to inspect. It must contain at least one parameter.
        input_shape: Input shape without a batch dimension, or one shape per positional tensor input.
        wrap_mode: Wrap long layer names at the middle (`"mid"`) or end (`"end"`).
        max_depth: Maximum depth of layer information.
        receptive_field: Whether to estimate receptive fields.
        effective_rf_stats: If `receptive_field` is true, also display effective stride and padding.
        input_data: One tensor or a non-empty list or tuple of positional tensor inputs, including all dimensions.

    Raises:
        StopIteration: If the module has no parameters.
        TypeError: If `input_data` is invalid or a hooked output has an unsupported leaf or contains no tensor.
        ValueError: If the inputs, `wrap_mode`, or `max_depth` are invalid.

    Notes:
        This function has the same forward-pass and module-hook limitations as `crawl_module`.
    """
    # Get the summary dict
    module_info = crawl_module(module, input_shape, input_data=input_data)
    # Aggregate until max_depth
    if isinstance(max_depth, int):
        module_info = aggregate_info(module_info, max_depth)
    # Format it and print it
    print(format_info(module_info, wrap_mode, receptive_field, effective_rf_stats))  # ruff: ignore[print] T201
