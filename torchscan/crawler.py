# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import inspect
import platform
import warnings
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from itertools import starmap
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import Module

from .extensions import (
    _METRIC_UNITS,
    ModuleCall,
    ModuleEstimates,
    ModuleHandler,
    _handler_metrics,
    _resolve_handler,
    _validate_handlers,
)
from .flops import FlopReport, measure_flops
from .modules import module_dmas, module_flops, module_macs, module_rf
from .modules._layout import LAYOUT_TYPES
from .modules._primitives import POINTWISE_TYPES, PRIMITIVE_TYPES
from .modules._token_dependencies import module_token_dependencies
from .modules._transformer import dmas_attention, macs_attention, validate_native_attention, validate_native_call
from .report import AnalysisReport, Diagnostic, IncompleteAnalysisError, LayerReport, MetricResult, metric_result
from .utils import aggregate_info, format_info

__all__ = ["crawl_module", "summary"]

_MODULE_METHOD = "torchscan_module_formula"
_NATIVE_TRANSFORMERS = (
    nn.MultiheadAttention,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
    nn.Transformer,
)
_SPATIAL_METRICS = {"receptive_field", "effective_stride", "effective_padding"}


def _builtin_module_handlers() -> Mapping[type[Module], ModuleHandler]:
    """Supply complete-call built-in handlers without a mutable public registry."""
    handler = ModuleHandler(_native_module_estimates, subtree_metrics=frozenset(_METRIC_UNITS))
    handlers: dict[type[Module], ModuleHandler] = dict.fromkeys(_NATIVE_TRANSFORMERS, handler)
    # Gates and norms have no scalar spatial field; activations are pointwise.
    nonspatial = (kind for kind in PRIMITIVE_TYPES if kind not in POINTWISE_TYPES)
    handlers.update(dict.fromkeys(nonspatial, ModuleHandler(_nonspatial_estimates)))
    handlers.update(dict.fromkeys(LAYOUT_TYPES, ModuleHandler(_nonspatial_estimates)))
    return handlers


def _nonspatial_estimates(_call: ModuleCall) -> ModuleEstimates:
    return cast(
        ModuleEstimates,
        {
            name: metric_result(status="unavailable", unit="elements", scope="module_call", method="not_applicable")
            for name in _SPATIAL_METRICS
        },
    )


def _native_module_estimates(
    call: ModuleCall,
    *,
    requested: set[str] | None = None,
    diagnostics: list[Diagnostic] | None = None,
    path: str = "",
) -> ModuleEstimates:
    """Adapt native formulas to the shared complete-call handler contract."""
    requested = set(_METRIC_UNITS) if requested is None else requested
    diagnostics = [] if diagnostics is None else diagnostics
    signature = None
    with suppress(TypeError, ValueError):
        signature = inspect.signature(call.module.forward)
    inputs = _ordered_inputs(signature, call.args, call.kwargs)

    def flops() -> int:
        if any(
            parameter.is_nested or parameter.layout != torch.strided or not parameter.is_floating_point()
            for parameter in call.module.parameters()
        ):
            raise NotImplementedError("Native Transformer FLOPs require dense real floating-point parameters.")
        # Preserve the legacy MHA input/option FLOP convention, including zero
        # batches, while validating the native algorithm and parameter shapes.
        # Composite dense estimates also exclude packing and modified children.
        if type(call.module) is nn.MultiheadAttention:
            validate_native_attention(call.module)
        else:
            validate_native_call(call.module, inputs, call.output)
        return module_flops(call.module, inputs, call.output)

    measures = {
        "module_flops": flops,
        "macs": lambda: macs_attention(call.module, inputs, call.output),
        "dmas": lambda: dmas_attention(call.module, inputs, call.output),
    }
    estimates: dict[str, MetricResult] = {}
    for name, measure in measures.items():
        if name in requested:
            if _first_tensor(call.output) is None:
                _diagnostic(
                    diagnostics,
                    code="missing_metric_tensor",
                    metric=name,
                    path=path,
                    message="The native call did not return an activation tensor; its forward may have failed.",
                )
                result = metric_result(
                    status="unavailable", unit=_METRIC_UNITS[name], scope="subtree", method=_MODULE_METHOD
                )
            else:
                result = _measure_module_metric(name, _METRIC_UNITS[name], path, diagnostics, measure)
            result["scope"] = "subtree"
            estimates[name] = result
    for name in requested & _SPATIAL_METRICS:
        estimates[name] = metric_result(status="unavailable", unit="elements", scope="subtree", method="not_applicable")
    return cast(ModuleEstimates, estimates)


def _native_token_report(call: ModuleCall, layer: LayerReport, diagnostics: list[Diagnostic]) -> None:
    try:
        inputs = _ordered_inputs(inspect.signature(call.module.forward), call.args, call.kwargs)
        if _first_tensor(call.output) is None:
            raise NotImplementedError("The native call did not return an activation tensor.")
        layer["token_dependencies"] = module_token_dependencies(call.module, inputs, call.output)
    except Exception as error:  # ruff: ignore[blind-except] BLE001  # Optional report information.
        layer["token_dependencies"] = {
            "status": "unavailable",
            "scope": "module_call",
            "method": "torchscan_token_dependency_v1",
            "assumptions": [],
        }
        _diagnostic(
            diagnostics,
            code="unsupported_token_dependencies",
            metric="token_dependencies",
            path=layer["path"],
            message=f"{type(error).__name__}: {error}",
        )


@cache
def _package_version() -> str:
    try:
        return version("torchscan")
    except PackageNotFoundError:
        return "unknown"


def _describe(value: Any) -> dict[str, Any]:
    """Describe a Python value without retaining its contents."""
    if isinstance(value, torch.Tensor):
        return {
            **({"kind": "nested_tensor"} if value.is_nested else {"kind": "tensor", "shape": list(value.shape)}),
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
            "items": [
                {
                    **({"key": key} if isinstance(key, int) or (isinstance(key, str) and key.isidentifier()) else {}),
                    "key_type": type(key).__name__,
                    "value": _describe(item),
                }
                for key, item in value.items()
            ],
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


def _ordered_inputs(
    signature: inspect.Signature | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Return forward arguments in signature order for the legacy formula functions."""
    if signature is None:
        return (*args, *kwargs.values())
    # Most leaf modules have a single, fully supplied positional argument.
    # Avoid constructing BoundArguments while still binding masks and defaults.
    if (
        not kwargs
        and len(args) == len(signature.parameters)
        and all(
            parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        )
    ):
        return args
    try:
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
        return metric_result(status="unavailable", unit=unit, scope="module_call", method=_MODULE_METHOD)

    if caught:
        unsupported = next((warning for warning in caught if "Module type not supported" in str(warning.message)), None)
        _diagnostic(
            diagnostics,
            code="unsupported_module_metric" if unsupported is not None else "module_metric_warning",
            metric=metric,
            path=path,
            message=str(unsupported.message) if unsupported is not None else "; ".join(str(w.message) for w in caught),
        )
    return metric_result(
        status="partial" if caught else "complete",
        value=value,
        known_value=value,
        unit=unit,
        scope="module_call",
        method=_MODULE_METHOD,
    )


def _aggregate_metric(layers: list[LayerReport], name: str, unit: str) -> MetricResult:
    results = [layer["metrics"][name] for layer in layers if name in layer["metrics"]]
    if not results:
        return metric_result(status="unavailable", unit=unit, scope="forward", method=_MODULE_METHOD)

    known_values = [result["known_value"] for result in results if result["known_value"] is not None]
    known_total = sum(known_values)
    methods = sorted({result["method"] for result in results})
    method = _MODULE_METHOD if methods == [_MODULE_METHOD] else f"sum_module_estimates[{', '.join(methods)}]"
    if all(result["status"] == "complete" for result in results):
        status = "complete"
    elif known_values:
        status = "partial"
    else:
        status = "unavailable"
    return metric_result(
        status=status,
        value=known_total,
        known_value=known_total,
        unit=unit,
        scope="forward",
        method=method,
    )


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

    shape_source = cast(list[tuple[int, ...]] | tuple[int, ...], input_shape)
    shapes = shape_source if isinstance(shape_source, list) else [shape_source]
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

    def generate(shape: tuple[int, ...], current_dtype: torch.dtype) -> torch.Tensor:
        # Integer inputs were generated as FP32 random values then truncated to
        # zero. Allocate those zeros directly without a random/conversion pass.
        if current_dtype == torch.bool:
            return torch.ones(1, *shape, device=target_device, dtype=current_dtype)
        if not current_dtype.is_floating_point and not current_dtype.is_complex:
            return torch.zeros(1, *shape, device=target_device, dtype=current_dtype)
        return torch.rand(1, *shape, device=target_device).to(dtype=current_dtype)

    call_args = tuple(starmap(generate, zip(shapes, dtypes, strict=True)))
    call_kwargs: dict[str, Any] = {}
    return call_args, call_kwargs, {"source": "generated", **_describe_call(call_args, call_kwargs)}


def crawl_module(
    module: Module,
    input_shape: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    dtype: torch.dtype | Iterable[torch.dtype] | None = None,
    *,
    args: tuple[Any, ...] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    strict: bool = False,
    mode: Literal["full", "structure"] = "full",
    custom_modules: Mapping[type[Module], ModuleHandler] | None = None,
    custom_mapping: Mapping[Any, Callable[..., int | float]] | None = None,
) -> AnalysisReport:
    """Collect a truthful, machine-readable report from one inference forward pass.

    Calls sharing a module instance must be serialized because analysis temporarily
    changes its training state and installs forward hooks.

    ``mode="structure"`` collects shapes, calls, parameters, and buffers without
    module formulas or operator dispatch. Unrequested compute totals are unavailable
    with method ``not_requested``; ``strict`` checks only requested metrics.

    ``custom_modules`` maps module types to scoped ``ModuleHandler`` callbacks.
    The closest class in the module's MRO wins; caller handlers precede built-ins.
    Declared ``subtree_metrics`` are inclusive and suppress descendant estimates
    for those fields, even when a callback fails. Other fields are module-local.
    ``custom_mapping`` supplies separate operator FLOP overrides to ``measure_flops``.
    Neither mapping changes global registries; structure mode executes neither.
    """
    if mode not in ("full", "structure"):
        raise ValueError("mode must be 'full' or 'structure'.")
    call_args, call_kwargs, input_metadata = _prepare_inputs(module, input_shape, dtype, args, kwargs, device)
    custom_handlers = _validate_handlers(custom_modules)
    builtin_handlers = _builtin_module_handlers()
    diagnostics: list[Diagnostic] = []
    layers: list[LayerReport] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    pending: dict[int, list[int]] = {}
    call_counts: dict[int, int] = {}
    seen_tensor_ids: set[int] = set()
    training_flags = [(child, child.training) for child in module.modules()]
    signatures: dict[Callable[..., Any], inspect.Signature | None] = {}
    active_calls: list[int] = []
    metric_diagnostics: list[tuple[int, Diagnostic]] = []
    grouped_diagnostics: dict[int, tuple[str, ...]] = {}

    def is_metric_leaf(current: Module) -> bool:
        return (
            not any(current.children())
            or isinstance(current, nn.MultiheadAttention)
            or (current is module and isinstance(module, nn.Transformer))
        )

    def own_descendant_metrics(layer_index: int, names: Iterable[str]) -> None:
        layer = layers[layer_index]
        owner = {"path": layer["path"], "call_index": layer["call_index"]}
        for name in names:
            layer.setdefault("metric_ownership", {})[name] = "subtree"
            if name in layer["metrics"]:
                layer["metrics"][name]["scope"] = "subtree"
            for descendant in layers[layer_index + 1 :]:
                descendant["metrics"].pop(name, None)
                descendant.get("metric_ownership", {}).pop(name, None)
                descendant.setdefault("metric_owners", {})[name] = owner
                if name == "receptive_field":
                    descendant.pop("token_dependencies", None)

    def register(current: Module, path: str) -> None:
        forward_signature: inspect.Signature | None = None
        custom_handler = _resolve_handler(current, custom_handlers)
        builtin_handler = _resolve_handler(current, builtin_handlers)
        handler = custom_handler or builtin_handler
        # A registered composite defines its own boundary. Falling back to an
        # inclusive atomic formula while also observing children double-counts.
        metric_leaf = not any(current.children()) or (handler is None and is_metric_leaf(current))
        if metric_leaf and mode == "full":
            forward = current.forward
            # Bound methods of the same implementation have the same signature.
            signature_key = forward.__func__ if inspect.ismethod(forward) else None
            if signature_key is not None and signature_key in signatures:
                forward_signature = signatures[signature_key]
            else:
                with suppress(TypeError, ValueError):
                    forward_signature = inspect.signature(forward)
                if signature_key is not None:
                    signatures[signature_key] = forward_signature

        def pre_hook(hooked: Module, hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any]) -> None:
            call_index = call_counts.get(id(hooked), 0)
            call_counts[id(hooked)] = call_index + 1
            recurse = isinstance(hooked, _NATIVE_TRANSFORMERS) or (
                hooked is module and isinstance(module, nn.Transformer)
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

            layers.append({
                "path": path,
                "call_index": call_index,
                "name": path.rpartition(".")[-1] or hooked.__class__.__name__.lower(),
                "depth": 0 if not path else path.count(".") + 1,
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
            layer_index = len(layers) - 1
            layer = layers[layer_index]
            if mode == "full":
                covered: dict[str, dict[str, str | int]] = {}
                for ancestor_index in active_calls:
                    ancestor = layers[ancestor_index]
                    for metric, scope in ancestor.get("metric_ownership", {}).items():
                        if scope == "subtree":
                            covered.setdefault(metric, {"path": ancestor["path"], "call_index": ancestor["call_index"]})
                if covered:
                    layer["metric_owners"] = covered
                if handler is not None:
                    layer["metric_ownership"] = {
                        name: "subtree" for name in sorted(handler.subtree_metrics) if name not in covered
                    }
            pending.setdefault(id(hooked), []).append(layer_index)
            active_calls.append(layer_index)

        def post_hook(
            hooked: Module,
            hook_args: tuple[Any, ...],
            hook_kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            layer_index = pending[id(hooked)][-1]
            active_calls.pop()
            layer = layers[layer_index]
            layer["output"] = _describe(output)
            if mode == "structure" or (not metric_leaf and handler is None):
                return

            if (
                hooked is module
                and atomic_custom_paths
                and any(descendant["path"] in atomic_custom_paths for descendant in layers[layer_index + 1 :])
            ):
                # Descendant estimates replace the old inclusive boundary;
                # their sum cannot account for unknown root-local work.
                for name in ("module_flops", "macs", "dmas"):
                    layer["metrics"][name] = metric_result(
                        status="unavailable",
                        unit=_METRIC_UNITS[name],
                        scope="module_call",
                        method="expanded_atomic_boundary",
                    )
                    _diagnostic(
                        diagnostics,
                        code="expanded_atomic_boundary",
                        metric=name,
                        path=path,
                        message="Custom descendant estimates expand the atomic model boundary; root-local work is unestimated.",
                    )
                return

            ordered_inputs = _ordered_inputs(forward_signature, hook_args, hook_kwargs)
            # Formula work must not appear in the operator report. Suspend dispatch
            # only while calculating metadata-based estimates, then release all
            # activation references before the next module executes.
            # PyTorch exposes no public context for suspending dispatch modes.
            call_diagnostics: list[Diagnostic] = []

            def run_handler(selected: ModuleHandler, names: set[str], *, custom: bool) -> dict[str, MetricResult]:
                call = ModuleCall(hooked, hook_args, hook_kwargs, output)
                if not custom and selected.estimate is _native_module_estimates:
                    results = cast(
                        dict[str, MetricResult],
                        _native_module_estimates(call, requested=names, diagnostics=call_diagnostics, path=path),
                    )
                    if "receptive_field" in names:
                        _native_token_report(call, layer, call_diagnostics)
                    return results
                return _handler_metrics(selected, call, names, call_diagnostics, path, custom=custom)

            with torch._C._DisableTorchDispatch():
                requested = set(_METRIC_UNITS) - layer.get("metric_owners", {}).keys()
                if handler is not None and requested:
                    estimates = run_handler(handler, requested, custom=custom_handler is not None)
                    layer["metrics"].update(estimates)
                    layer.setdefault("metric_ownership", {}).update({
                        name: "subtree" if name in handler.subtree_metrics else "module_call" for name in estimates
                    })
                    requested -= estimates.keys()
                if custom_handler is not None and builtin_handler is not None and requested:
                    estimates = run_handler(builtin_handler, requested, custom=False)
                    layer["metrics"].update(estimates)
                    layer.setdefault("metric_ownership", {}).update({
                        name: "subtree" if name in builtin_handler.subtree_metrics else "module_call"
                        for name in estimates
                    })
                    # Caller fields are only known after forward. If an inclusive
                    # built-in supplies an omitted field, discard earlier child
                    # estimates for that field; never retain child activations.
                    own_descendant_metrics(layer_index, estimates.keys() & builtin_handler.subtree_metrics)
                    requested -= estimates.keys()
                if metric_leaf and requested:
                    populate_metrics(
                        layer_index,
                        hooked,
                        ordered_inputs,
                        output,
                        _first_tensor(ordered_inputs),
                        _first_tensor(output),
                        requested,
                        call_diagnostics,
                    )
            if hooked is module and atomic_custom_paths:
                # A matching registration may never execute. Keep the legacy
                # inclusive root formula and remove duplicate child estimates.
                own_descendant_metrics(layer_index, _METRIC_UNITS)
            metric_diagnostics.extend((layer_index, diagnostic) for diagnostic in call_diagnostics)

        def cleanup_hook(hooked: Module, _hook_args: tuple[Any, ...], _output: Any) -> None:
            pending_calls = pending.get(id(hooked))
            if not pending_calls:
                return
            layer_index = pending_calls.pop()
            layer = layers[layer_index]
            if not active_calls or active_calls[-1] != layer_index:
                return
            # Only cleanup runs on a failed forward, so a legitimate None output
            # still reaches the estimator through the ordinary post-hook.
            active_calls.pop()
            layer["output"] = {"kind": "failed"}
            _diagnostic(
                diagnostics,
                code="module_forward_error",
                metric="calls",
                path=path,
                message="The module forward did not complete; no estimation callback was executed.",
            )
            if mode == "full" and (metric_leaf or handler is not None):
                if type(hooked) in _NATIVE_TRANSFORMERS and "receptive_field" not in layer.get("metric_owners", {}):
                    layer["token_dependencies"] = {
                        "status": "unavailable",
                        "scope": "module_call",
                        "method": "torchscan_token_dependency_v1",
                        "assumptions": [],
                    }
                for name in set(_METRIC_UNITS) - layer.get("metric_owners", {}).keys():
                    layer["metrics"][name] = metric_result(
                        status="unavailable",
                        unit=_METRIC_UNITS[name],
                        scope="subtree" if handler is not None and name in handler.subtree_metrics else "module_call",
                        method="forward_failed",
                    )

        handles.append(current.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(current.register_forward_hook(post_hook, with_kwargs=True))
        # A parent can catch a child's forward error and continue. Always unwind
        # the dynamic ownership stack before observing later sibling calls.
        handles.append(current.register_forward_hook(cleanup_hook, always_call=True))

    def populate_metrics(
        layer_index: int,
        hooked: Module,
        ordered_inputs: tuple[Any, ...],
        output: Any,
        input_tensor: torch.Tensor | None,
        output_tensor: torch.Tensor | None,
        requested: set[str],
        call_diagnostics: list[Diagnostic],
    ) -> None:
        layer = layers[layer_index]
        if input_tensor is None or output_tensor is None:
            for metric, unit in _METRIC_UNITS.items():
                if metric not in requested:
                    continue
                layer["metrics"][metric] = metric_result(
                    status="unavailable",
                    unit=unit,
                    scope="module_call",
                    method=_MODULE_METHOD,
                )
                _diagnostic(
                    call_diagnostics,
                    code="missing_metric_tensor",
                    metric=metric,
                    path=layer["path"],
                    message="The module call did not expose both an input and output tensor.",
                )
            return

        flops_output = output if isinstance(hooked, nn.MultiheadAttention) else output_tensor
        measures = {
            "module_flops": lambda: module_flops(hooked, ordered_inputs, flops_output),
            "macs": lambda: module_macs(hooked, input_tensor, output_tensor),
            "dmas": lambda: module_dmas(hooked, input_tensor, output_tensor),
        }
        for metric, measure in measures.items():
            if metric in requested:
                layer["metrics"][metric] = _measure_module_metric(
                    metric, _METRIC_UNITS[metric], layer["path"], call_diagnostics, measure
                )
        receptive_metrics = tuple(
            name for name in ("receptive_field", "effective_stride", "effective_padding") if name in requested
        )
        if not receptive_metrics:
            return
        failure: tuple[str, str] | None = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                receptive_field, stride, padding = module_rf(hooked, input_tensor, output_tensor)
        except Exception as error:  # ruff: ignore[blind-except] BLE001  # Optional report metric.
            caught = []
            receptive_values: tuple[float, float, float] | None = None
            failure = ("module_metric_error", f"{type(error).__name__}: {error}")
        else:
            receptive_values = (receptive_field, stride, padding)
            if caught:
                failure = ("unsupported_module_metric", "; ".join(str(warning.message) for warning in caught))
        if failure is not None:
            _diagnostic(
                call_diagnostics,
                code=failure[0],
                metric=receptive_metrics[0],
                path=layer["path"],
                message=failure[1],
            )
            # Keep the legacy single diagnostic, but remember exactly which
            # omitted fields failed if inclusive ownership later covers some.
            grouped_diagnostics[id(call_diagnostics[-1])] = receptive_metrics
        status = "unavailable" if receptive_values is None or caught else "complete"
        for index, name in enumerate(("receptive_field", "effective_stride", "effective_padding")):
            if name not in requested:
                continue
            layer["metrics"][name] = metric_result(
                status=status,
                value=receptive_values[index] if receptive_values is not None else None,
                unit="elements",
                scope="module_call",
                method=_MODULE_METHOD,
            )

    legacy_atomic = (
        isinstance(module, nn.Transformer)
        and _resolve_handler(module, custom_handlers) is None
        and _resolve_handler(module, builtin_handlers) is None
    )
    atomic_custom_paths = (
        {
            path
            for path, child in module.named_modules()
            if path and _resolve_handler(child, custom_handlers) is not None
        }
        if legacy_atomic
        else set()
    )
    targets = [("", module)] if legacy_atomic and not atomic_custom_paths else list(module.named_modules())
    flop_report: FlopReport
    try:
        module.eval()
        for module_path, current in targets:
            register(current, module_path)
        with torch.no_grad():
            # PyTorch 2.1's explicit module tracker replaces caller tensors in hooks.
            # Omitting it preserves exact args; newer releases still attribute modules automatically.
            if mode == "full":
                flop_report = measure_flops(lambda: module(*call_args, **call_kwargs), custom_mapping=custom_mapping)
            else:
                module(*call_args, **call_kwargs)
                flop_report = {
                    "schema_version": 1,
                    "context": {"torch_version": torch.__version__, "method": "not_requested"},
                    "total": metric_result(
                        status="unavailable", unit="FLOPs", scope="workload", method="not_requested"
                    ),
                    "by_module": {},
                    "by_operator": {},
                    "ignored_operators": {},
                    "diagnostics": [],
                }
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
    for layer_index, diagnostic in metric_diagnostics:
        layer = layers[layer_index]
        affected_metrics = grouped_diagnostics.get(
            id(diagnostic),
            ("receptive_field" if diagnostic["metric"] == "token_dependencies" else diagnostic["metric"],),
        )
        affected = next((name for name in affected_metrics if name not in layer.get("metric_owners", {})), None)
        if affected is not None:
            diagnostics.append(
                diagnostic
                if diagnostic["metric"] == "token_dependencies" or affected == diagnostic["metric"]
                else {**diagnostic, "metric": affected}
            )
    diagnostics.extend(flop_report["diagnostics"])

    report: AnalysisReport = {
        "schema_version": 1,
        "context": {
            "torchscan_version": _package_version(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "model_type": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            "execution_mode": "evaluation_no_grad",
            "training_before": training_flags[0][1],
            "devices": sorted({str(tensor.device) for tensor in model_tensors}),
            "dtypes": sorted({str(tensor.dtype) for tensor in model_tensors}),
        },
        "inputs": input_metadata,
        "layers": layers,
        "operator_flops": flop_report,
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
            "operator_flops": flop_report["total"],
            "macs": _aggregate_metric(layers, "macs", "MACs"),
            "dmas": _aggregate_metric(layers, "dmas", "DMAs"),
        },
        "diagnostics": diagnostics,
    }
    if mode == "structure":
        report["context"]["analysis_mode"] = mode
        for name, unit in (("module_flops", "FLOPs"), ("macs", "MACs"), ("dmas", "DMAs")):
            report["totals"][name] = metric_result(
                status="unavailable", unit=unit, scope="forward", method="not_requested"
            )
    if strict and (
        report["diagnostics"]
        or any(
            result["status"] != "complete" and result["method"] != "not_requested"
            for result in report["totals"].values()
        )
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
    mode: Literal["full", "structure"] = "full",
    custom_modules: Mapping[type[Module], ModuleHandler] | None = None,
    custom_mapping: Mapping[Any, Callable[..., int | float]] | None = None,
) -> AnalysisReport:
    """Print and return a module report; use ``mode="structure"`` for shapes and counts only."""
    report = crawl_module(
        module,
        input_shape,
        dtype,
        args=args,
        kwargs=kwargs,
        device=device,
        strict=strict,
        mode=mode,
        custom_modules=custom_modules,
        custom_mapping=custom_mapping,
    )
    display_report = aggregate_info(report, max_depth) if isinstance(max_depth, int) else report
    print(format_info(display_report, wrap_mode, receptive_field, effective_rf_stats))  # ruff: ignore[print] T201
    return report
