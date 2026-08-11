# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import inspect
import platform
import warnings
from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
from torch import nn
from torch.nn import Module

from .modules import module_dmas, module_flops, module_macs, module_rf
from .report import AnalysisReport, Diagnostic, IncompleteAnalysisError, LayerReport, MetricResult, metric_result
from .utils import aggregate_info, format_info

__all__ = ["crawl_module", "summary"]


def _package_version() -> str:
    try:
        return version("torchscan")
    except PackageNotFoundError:
        return "unknown"


def _describe(value: Any) -> dict[str, Any]:
    """Describe a Python value without retaining its contents."""
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if value is None:
        return {"kind": "none"}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_describe(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_describe(item) for item in value]}
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "type": type(value).__name__,
            "items": [{"key_type": type(key).__name__, "value": _describe(item)} for key, item in value.items()],
        }
    if isinstance(value, (bool, int, float, complex, str, bytes)):
        return {"kind": "scalar", "type": type(value).__name__}
    return {"kind": "object", "type": type(value).__qualname__}


def _describe_call(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "args": [_describe(arg) for arg in args],
        "kwargs": {name: _describe(value) for name, value in kwargs.items()},
    }


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    elif isinstance(value, (tuple, list)):
        for item in value:
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    return None


def _ordered_inputs(module: Module, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return forward arguments in signature order for the legacy formula functions."""
    try:
        signature = inspect.signature(module.forward)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return (*args, *kwargs.values())

    ordered: list[Any] = []
    for name, parameter in signature.parameters.items():
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            ordered.extend(value)
        else:
            ordered.append(value)
    return tuple(ordered)


def _diagnostic(
    diagnostics: list[Diagnostic],
    *,
    code: str,
    metric: str,
    path: str,
    message: str,
) -> None:
    diagnostics.append({
        "code": code,
        "severity": "warning",
        "metric": metric,
        "path": path,
        "message": message,
    })


def _measure_module_metric(
    metric: str,
    unit: str,
    path: str,
    diagnostics: list[Diagnostic],
    measure: Callable[[], int | float],
) -> MetricResult:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = measure()
    except Exception as error:  # ruff: ignore[blind-except] BLE001  # Formula failures are report diagnostics.
        _diagnostic(
            diagnostics,
            code="module_metric_error",
            metric=metric,
            path=path,
            message=f"{type(error).__name__}: {error}",
        )
        return metric_result(status="unavailable", unit=unit, scope="module_call", method="torchscan_module_formula")

    unsupported = next((warning for warning in caught if "Module type not supported" in str(warning.message)), None)
    if unsupported is not None:
        _diagnostic(
            diagnostics,
            code="unsupported_module_metric",
            metric=metric,
            path=path,
            message=str(unsupported.message),
        )
        return metric_result(
            status="partial",
            known_value=value,
            unit=unit,
            scope="module_call",
            method="torchscan_module_formula",
        )
    if caught:
        _diagnostic(
            diagnostics,
            code="module_metric_warning",
            metric=metric,
            path=path,
            message="; ".join(str(warning.message) for warning in caught),
        )
        return metric_result(
            status="partial",
            known_value=value,
            unit=unit,
            scope="module_call",
            method="torchscan_module_formula",
        )
    return metric_result(
        status="complete",
        value=value,
        unit=unit,
        scope="module_call",
        method="torchscan_module_formula",
    )


def _aggregate_metric(layers: list[LayerReport], name: str, unit: str) -> MetricResult:
    results = [layer["metrics"][name] for layer in layers if name in layer["metrics"]]
    if not results:
        return metric_result(status="unavailable", unit=unit, scope="forward", method="torchscan_module_formula")

    known_values = [result["known_value"] for result in results if result["known_value"] is not None]
    known_total = sum(known_values)
    if all(result["status"] == "complete" for result in results):
        return metric_result(
            status="complete",
            value=known_total,
            unit=unit,
            scope="forward",
            method="torchscan_module_formula",
        )
    if known_values:
        return metric_result(
            status="partial",
            known_value=known_total,
            unit=unit,
            scope="forward",
            method="torchscan_module_formula",
        )
    return metric_result(status="unavailable", unit=unit, scope="forward", method="torchscan_module_formula")


def _model_defaults(module: Module) -> tuple[torch.device, torch.dtype]:
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    if tensor is None:
        return torch.device("cpu"), torch.float32
    return tensor.device, tensor.dtype


def _prepare_inputs(
    module: Module,
    input_shape: list[tuple[int, ...]] | tuple[int, ...] | None,
    dtype: torch.dtype | Iterable[torch.dtype] | None,
    args: tuple[Any, ...] | None,
    kwargs: Mapping[str, Any] | None,
    device: str | torch.device | None,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    provided = args is not None or kwargs is not None
    generated = input_shape is not None
    if provided == generated:
        raise ValueError("Exactly one of input_shape or args/kwargs must be provided.")

    if provided:
        if dtype is not None or device is not None:
            raise ValueError("dtype and device apply only to generated input_shape tensors.")
        if args is not None and not isinstance(args, tuple):
            raise TypeError(f"args must be a tuple, got {type(args).__name__}.")
        if kwargs is not None and not isinstance(kwargs, Mapping):
            raise TypeError(f"kwargs must be a mapping, got {type(kwargs).__name__}.")
        call_args = () if args is None else args
        call_kwargs = {} if kwargs is None else dict(kwargs)
        if any(not isinstance(name, str) for name in call_kwargs):
            raise TypeError("kwargs keys must be strings.")
        return call_args, call_kwargs, {"source": "provided", **_describe_call(call_args, call_kwargs)}

    if input_shape is None:
        raise ValueError("input_shape is required when args and kwargs are omitted.")
    shapes = input_shape if isinstance(input_shape, list) else [input_shape]
    if not shapes or any(not isinstance(shape, tuple) for shape in shapes):
        raise TypeError("input_shape must be a tuple or a non-empty list of tuples.")
    if any(any(not isinstance(dimension, int) for dimension in shape) for shape in shapes):
        raise TypeError("Every input_shape dimension must be an integer.")

    default_device, default_dtype = _model_defaults(module)
    target_device = default_device if device is None else torch.device(device)
    if dtype is None:
        dtypes = [default_dtype] * len(shapes)
    elif isinstance(dtype, torch.dtype):
        dtypes = [dtype] * len(shapes)
    else:
        dtypes = list(dtype)
        if len(dtypes) != len(shapes):
            raise ValueError("dtype length must match the number of input shapes.")
        if any(not isinstance(item, torch.dtype) for item in dtypes):
            raise TypeError("Every dtype value must be a torch.dtype.")

    call_args = tuple(
        torch.rand(1, *shape, device=target_device).to(dtype=current_dtype)
        for shape, current_dtype in zip(shapes, dtypes, strict=True)
    )
    call_kwargs: dict[str, Any] = {}
    return call_args, call_kwargs, {"source": "generated", **_describe_call(call_args, call_kwargs)}


def apply(module: Module, fn: Callable[[Module, str], None], name: str | None = None) -> None:
    """Apply a function to a module tree while providing stable dotted paths."""
    if name is None:
        name = module.__class__.__name__.lower()
    fn(module, name)
    for child_name, child in module.named_children():
        apply(child, fn, f"{name}.{child_name}")


def crawl_module(
    module: Module,
    input_shape: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    dtype: torch.dtype | Iterable[torch.dtype] | None = None,
    *,
    args: tuple[Any, ...] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    strict: bool = False,
) -> AnalysisReport:
    """Collect a truthful, machine-readable report from one inference forward pass."""
    call_args, call_kwargs, input_metadata = _prepare_inputs(module, input_shape, dtype, args, kwargs, device)
    diagnostics: list[Diagnostic] = []
    layers: list[LayerReport] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    pending: dict[int, list[int]] = {}
    call_counts: dict[int, int] = {}
    seen_tensor_ids: set[int] = set()
    training_flags = [(child, child.training) for child in module.modules()]
    root_module = module

    def is_metric_leaf(current: Module) -> bool:
        return (
            not any(current.children())
            or isinstance(current, nn.MultiheadAttention)
            or (current is root_module and isinstance(root_module, nn.Transformer))
        )

    def register(current: Module, path: str) -> None:
        def pre_hook(hooked: Module, hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any]) -> None:
            call_index = call_counts.get(id(hooked), 0)
            call_counts[id(hooked)] = call_index + 1
            recurse = isinstance(hooked, nn.MultiheadAttention) or (
                hooked is root_module and isinstance(root_module, nn.Transformer)
            )
            trainable = frozen = parameter_bytes = buffer_elements = buffer_bytes = 0
            parameter_shared = buffer_shared = False
            for parameter in hooked.parameters(recurse=recurse):
                if id(parameter) in seen_tensor_ids:
                    parameter_shared = True
                    continue
                seen_tensor_ids.add(id(parameter))
                if parameter.requires_grad:
                    trainable += parameter.numel()
                else:
                    frozen += parameter.numel()
                parameter_bytes += parameter.numel() * parameter.element_size()
            for buffer in hooked.buffers(recurse=recurse):
                if id(buffer) in seen_tensor_ids:
                    buffer_shared = True
                    continue
                seen_tensor_ids.add(id(buffer))
                buffer_elements += buffer.numel()
                buffer_bytes += buffer.numel() * buffer.element_size()

            layer_path = path
            layers.append({
                "path": layer_path,
                "call_index": call_index,
                "name": layer_path.rpartition(".")[-1] or hooked.__class__.__name__.lower(),
                "depth": 0 if not layer_path else layer_path.count(".") + 1,
                "type": hooked.__class__.__name__,
                "input": _describe_call(hook_args, hook_kwargs),
                "output": {"kind": "pending"},
                "parameters": {
                    "trainable": trainable,
                    "frozen": frozen,
                    "bytes": parameter_bytes,
                    "shared": parameter_shared,
                },
                "buffers": {
                    "elements": buffer_elements,
                    "bytes": buffer_bytes,
                    "shared": buffer_shared,
                },
                "metrics": {
                    "calls": metric_result(
                        status="complete",
                        value=1,
                        unit="calls",
                        scope="module_call",
                        method="pytorch_hook",
                    )
                },
            })
            pending.setdefault(id(hooked), []).append(len(layers) - 1)

        def post_hook(
            hooked: Module,
            hook_args: tuple[Any, ...],
            hook_kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            layer_index = pending[id(hooked)].pop()
            layer = layers[layer_index]
            layer["output"] = _describe(output)
            if not is_metric_leaf(hooked):
                return

            ordered_inputs = _ordered_inputs(hooked, hook_args, hook_kwargs)
            input_tensor = _first_tensor(ordered_inputs)
            output_tensor = _first_tensor(output)
            if input_tensor is None or output_tensor is None:
                for metric, unit in (("module_flops", "FLOPs"), ("macs", "MACs"), ("dmas", "DMAs")):
                    layer["metrics"][metric] = metric_result(
                        status="unavailable",
                        unit=unit,
                        scope="module_call",
                        method="torchscan_module_formula",
                    )
                    _diagnostic(
                        diagnostics,
                        code="missing_metric_tensor",
                        metric=metric,
                        path=layer["path"],
                        message="The module call did not expose both an input and output tensor.",
                    )
                return

            flops_output = output if isinstance(hooked, nn.MultiheadAttention) else output_tensor
            layer["metrics"]["module_flops"] = _measure_module_metric(
                "module_flops",
                "FLOPs",
                layer["path"],
                diagnostics,
                lambda: module_flops(hooked, ordered_inputs, flops_output),
            )
            layer["metrics"]["macs"] = _measure_module_metric(
                "macs",
                "MACs",
                layer["path"],
                diagnostics,
                lambda: module_macs(hooked, input_tensor, output_tensor),
            )
            layer["metrics"]["dmas"] = _measure_module_metric(
                "dmas",
                "DMAs",
                layer["path"],
                diagnostics,
                lambda: module_dmas(hooked, input_tensor, output_tensor),
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    receptive_field, stride, padding = module_rf(hooked, input_tensor, output_tensor)
            except Exception as error:  # ruff: ignore[blind-except] BLE001  # Optional report metric.
                caught = []
                receptive_values: tuple[float, float, float] | None = None
                _diagnostic(
                    diagnostics,
                    code="module_metric_error",
                    metric="receptive_field",
                    path=layer["path"],
                    message=f"{type(error).__name__}: {error}",
                )
            else:
                receptive_values = (receptive_field, stride, padding)
                if caught:
                    _diagnostic(
                        diagnostics,
                        code="unsupported_module_metric",
                        metric="receptive_field",
                        path=layer["path"],
                        message="; ".join(str(warning.message) for warning in caught),
                    )
            for index, name in enumerate(("receptive_field", "effective_stride", "effective_padding")):
                if receptive_values is None or caught:
                    layer["metrics"][name] = metric_result(
                        status="unavailable",
                        unit="elements",
                        scope="module_call",
                        method="torchscan_module_formula",
                    )
                else:
                    layer["metrics"][name] = metric_result(
                        status="complete",
                        value=receptive_values[index],
                        unit="elements",
                        scope="module_call",
                        method="torchscan_module_formula",
                    )

        handles.append(current.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(current.register_forward_hook(post_hook, with_kwargs=True))

    targets = [("", module)] if isinstance(module, nn.Transformer) else list(module.named_modules())
    for module_path, current in targets:
        register(current, module_path)

    try:
        module.eval()
        with torch.inference_mode():
            module(*call_args, **call_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        for child, training in training_flags:
            child.training = training

    parameters = list(module.parameters())
    buffers = list(module.buffers())
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in parameters if not parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    buffer_elements = sum(buffer.numel() for buffer in buffers)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    model_tensors = [*parameters, *buffers]

    report: AnalysisReport = {
        "schema_version": 1,
        "context": {
            "torchscan_version": _package_version(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "model_type": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            "execution_mode": "inference",
            "training_before": training_flags[0][1],
            "devices": sorted({str(tensor.device) for tensor in model_tensors}),
            "dtypes": sorted({str(tensor.dtype) for tensor in model_tensors}),
        },
        "inputs": input_metadata,
        "layers": layers,
        "totals": {
            "parameters": metric_result(
                status="complete", value=trainable + frozen, unit="elements", scope="model", method="pytorch"
            ),
            "trainable_parameters": metric_result(
                status="complete", value=trainable, unit="elements", scope="model", method="pytorch"
            ),
            "frozen_parameters": metric_result(
                status="complete", value=frozen, unit="elements", scope="model", method="pytorch"
            ),
            "parameter_bytes": metric_result(
                status="complete", value=parameter_bytes, unit="bytes", scope="model", method="pytorch"
            ),
            "buffer_elements": metric_result(
                status="complete", value=buffer_elements, unit="elements", scope="model", method="pytorch"
            ),
            "buffer_bytes": metric_result(
                status="complete", value=buffer_bytes, unit="bytes", scope="model", method="pytorch"
            ),
            "module_flops": _aggregate_metric(layers, "module_flops", "FLOPs"),
            "macs": _aggregate_metric(layers, "macs", "MACs"),
            "dmas": _aggregate_metric(layers, "dmas", "DMAs"),
        },
        "diagnostics": diagnostics,
    }
    if strict and (
        report["diagnostics"] or any(result["status"] != "complete" for result in report["totals"].values())
    ):
        raise IncompleteAnalysisError(report)
    return report


def summary(
    module: Module,
    input_shape: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    wrap_mode: str = "mid",
    max_depth: int | None = None,
    receptive_field: bool = False,
    effective_rf_stats: bool = False,
    *,
    dtype: torch.dtype | Iterable[torch.dtype] | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    strict: bool = False,
) -> AnalysisReport:
    """Print and return a truthful module analysis report."""
    report = crawl_module(
        module,
        input_shape,
        dtype,
        args=args,
        kwargs=kwargs,
        device=device,
        strict=strict,
    )
    display_report = aggregate_info(report, max_depth) if isinstance(max_depth, int) else report
    print(format_info(display_report, wrap_mode, receptive_field, effective_rf_stats))  # ruff: ignore[print] T201
    return report
