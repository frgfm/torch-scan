# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Replay operator metadata without importing PyTorch or executing a workload."""

import json
from collections import Counter
from math import prod
from typing import Any

from ._flop_formulas import _IGNORED_OPERATOR_REASONS, FORMULAS
from .report import FlopReport, OperatorCapture, metric_result

__all__ = ["OperatorCapture", "analyze_capture"]

_METHOD = "ATen.RecordFunction/torchscan_flops_v1"
_SCOPE = "calling_thread_and_autograd_eager_ops"
_PRODUCTS = {"mm", "bmm", "addmm", "baddbmm", "_scaled_mm"}
_CONVOLUTIONS = {"convolution", "_convolution", "cudnn_convolution", "convolution_overrideable", "_slow_conv2d_forward"}
_ATTENTION = {
    f"_scaled_dot_product_{kind}_attention{suffix}"
    for kind in ("flash", "efficient", "cudnn")
    for suffix in ("", "_backward")
}
_SUPPORTED = FORMULAS.keys() | _PRODUCTS | _CONVOLUTIONS | _ATTENTION | {"convolution_backward"}


def _shapes(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"nonfinite"} and value["nonfinite"] in {"inf", "-inf", "nan"}:
            return float(value["nonfinite"])
        if "unknown" in value:
            raise NotImplementedError(str(value["unknown"]))
        shape = value.get("shape")
        if (
            not isinstance(shape, list)
            or any(type(size) is not int or not 0 <= size < 2**63 for size in shape)
            or any(type(value.get(key)) is not bool for key in ("floating", "complex"))
            or any(not isinstance(value.get(key), str) for key in ("dtype", "device"))
        ):
            raise ValueError("Invalid tensor metadata.")
        return tuple(shape)
    if isinstance(value, list):
        return tuple(_shapes(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError("Invalid argument metadata.")


def _tensors(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value] if "shape" in value else [t for item in value.values() for t in _tensors(item)]
    return [t for item in value for t in _tensors(item)] if isinstance(value, list) else []


def _product(*args: Any, out_shape: Any, left: int = 0, **_kwargs: Any) -> int:
    return 2 * prod(out_shape) * args[left][-1]


def _convolution(inp: Any, weight: Any, output: Any, transposed: bool = False) -> int:
    return 2 * inp[0] * prod(weight) * prod((inp if transposed else output)[2:])


def _count(packet: str, args: Any, kwargs: Any, output: Any) -> int:
    if packet in _PRODUCTS:
        return _product(*args, out_shape=output, left=int(packet in {"addmm", "baddbmm"}))
    if packet in _CONVOLUTIONS:
        transposed = args[6] if packet in {"convolution", "_convolution", "convolution_overrideable"} else False
        return _convolution(args[0], args[1], output, transposed)
    if packet == "convolution_backward":
        grad, inp, weight = args[:3]
        transposed, mask = args[7], args[10]
        # Each input/weight gradient visits the same dense products as forward,
        # including grouped channels. Fused bias arithmetic remains excluded.
        return _convolution(inp, weight, grad, transposed) * (mask[0] + mask[1])
    if packet in _ATTENTION:
        backward = packet.endswith("_backward")
        query, key, value = args[1:4] if backward else args[:3]
        if (
            len(query) != 4
            or len(key) != 4
            or len(value) != 4
            or query[0] != key[0]
            or key[:3] != value[:3]
            or query[3] != key[3]
            or key[1] <= 0
            or query[1] % key[1]
        ):
            raise NotImplementedError("Fused attention requires compatible dense batched Q/K/V shapes.")
        return 2 * prod(query[:3]) * key[2] * ((3 * query[3] + 2 * value[3]) if backward else (query[3] + value[3]))
    return FORMULAS[packet](*args, out_shape=output, **kwargs)


def analyze_capture(capture: OperatorCapture) -> "FlopReport":
    """Convert saved native events into a FLOP report using metadata only.

    Args:
        capture: JSON-loaded capture_workload result. Unknown operators, layouts, lost
            events, and incomplete arguments retain partial counts and diagnostics.

    Returns:
        A standard FlopReport at the calling-thread and autograd eager dispatcher boundary.
        Module counts are inclusive. Complex arithmetic retains an explicitly partial lower bound.

    Raises:
        ValueError: If the capture schema, method, event structure, or loss counters are invalid.
    """
    try:
        json.dumps(capture, allow_nan=False)
        if (
            type(capture["schema_version"]) is not int
            or capture["schema_version"] != 1
            or capture["report_type"] != "operator_capture"
            or capture["context"]["method"] != _METHOD
            or capture["context"]["scope"] != _SCOPE
            or capture["context"].get("boundary") != "dispatcher_operator"
            or not isinstance(capture["context"].get("torch_version"), str)
            or not isinstance(capture["inputs"], dict)
            or not isinstance(capture["operators"], list)
            or any(type(capture[key]) is not int or capture[key] < 0 for key in ("dropped_events", "errors"))
        ):
            raise ValueError("Unsupported operator capture.")
    except (KeyError, TypeError, RecursionError) as error:
        raise ValueError("Invalid operator capture.") from error
    counts: Counter[str] = Counter()
    modules: Counter[str] = Counter()
    ignored: dict[str, Any] = {}
    diagnostics = []

    def gap(operator: str, message: str, code: str = "unsupported_operator_formula") -> None:
        diagnostics.append({
            "code": code,
            "severity": "warning",
            "metric": "flops",
            "operator": operator,
            "message": message,
        })

    if capture["dropped_events"] or capture["errors"]:
        gap(
            "",
            f"Capture lost {capture['dropped_events']} events and recorded {capture['errors']} collector errors.",
            "incomplete_capture",
        )
    for event in capture["operators"]:
        if (
            not isinstance(event, dict)
            or any(not isinstance(event.get(key), str) for key in ("operator", "overload", "module"))
            or any(not isinstance(event.get(key), list) for key in ("inputs", "outputs"))
            or not isinstance(event.get("kwargs"), dict)
        ):
            raise ValueError("Invalid operator event.")
        name = event["operator"].replace("::", ".")
        custom = "custom_flops" in event or "custom_error" in event
        reason = None if custom else _IGNORED_OPERATOR_REASONS.get(name)
        if reason is None:
            try:
                packet = name.removeprefix("aten.") if name.startswith("aten.") else ""
                tensors = _tensors([event["inputs"], event["kwargs"], event["outputs"]])
                if packet in {"sum", "mean", "cumsum", "cumsum_", "linalg_vector_norm"}:
                    arithmetic = _tensors(event["outputs"])
                elif packet in {"pow", "pow_", "addcmul", "addcmul_", "addcdiv", "addcdiv_"}:
                    arithmetic = _tensors(event["inputs"])
                else:
                    arithmetic = tensors
                scalar_power = packet in {"pow", "pow_"} and any(
                    isinstance(arg, float) or (isinstance(arg, dict) and "nonfinite" in arg) for arg in event["inputs"]
                )
                complex_work = any(t["complex"] for t in arithmetic)
                if "custom_error" in event:
                    raise NotImplementedError(event["custom_error"])
                if custom:
                    count = event["custom_flops"]
                else:
                    if packet not in _SUPPORTED:
                        raise KeyError(packet)
                    args = _shapes(event["inputs"])
                    kwargs = {key: _shapes(value) for key, value in event["kwargs"].items()}
                    outputs = _shapes(event["outputs"][:1] if packet in _ATTENTION else event["outputs"])
                    if not outputs:
                        raise NotImplementedError("Missing operator outputs.")
                    out_shape = outputs[0] if len(outputs) == 1 else outputs
                    if arithmetic and not (scalar_power or any(t["floating"] or t["complex"] for t in arithmetic)):
                        count = 0
                    elif complex_work and packet not in _PRODUCTS | _CONVOLUTIONS | _ATTENTION:
                        raise NotImplementedError("Supplemental FLOP formulas cover real arithmetic only.")
                    elif kwargs.get("rounding_mode") is not None:
                        raise NotImplementedError("Rounded division is not covered by the scalar arithmetic formula.")
                    else:
                        count = _count(packet, args, kwargs, out_shape)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("Invalid formula result.")
                if reason is None:
                    counts[name] += count
                    path = event["module"]
                    if path:
                        parts = path.split(".")
                        for length in range(1, len(parts) + 1):
                            modules[".".join(parts[:length])] += count
                    if not custom and (complex_work or packet in _ATTENTION):
                        gap(
                            name,
                            "Shape counts cover only core arithmetic for complex inputs or fused attention.",
                            "incomplete_operator_formula",
                        )
            except LookupError:
                gap(name, f"{name} was observed 1 time(s), but no FLOP formula is registered.", "uncounted_operator")
            except (ValueError, TypeError, NotImplementedError, ZeroDivisionError) as error:
                gap(name, str(error))
        if reason is not None:
            item = ignored.setdefault(name, {"calls": 0, "reason": reason})
            item["calls"] += 1
    known = sum(counts.values())
    return {
        "schema_version": 1,
        "context": {
            "torch_version": capture["context"].get("torch_version", "unknown"),
            "method": _METHOD,
            "counting_convention": "torchscan_flops_v1",
            "boundary": "dispatcher_operator",
        },
        "total": metric_result(
            status="partial" if diagnostics else "complete",
            value=None if diagnostics else known,
            known_value=known if diagnostics else None,
            unit="FLOPs",
            scope=_SCOPE,
            method=_METHOD,
        ),
        "by_operator": dict(sorted(counts.items())),
        "by_module": dict(sorted(modules.items())),
        "ignored_operators": ignored,
        "diagnostics": diagnostics,
    }
