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
_SCOPE = "calling_thread_eager_ops"
_FORMULAS = {
    **FORMULAS,
    **{
        name: FORMULAS[target]
        for name, target in {
            "layer_norm": "native_layer_norm",
            "batch_norm": "native_batch_norm",
            "softmax": "_softmax",
            "log_softmax": "_log_softmax",
            "adaptive_avg_pool2d": "_adaptive_avg_pool2d",
            "adaptive_avg_pool3d": "_adaptive_avg_pool3d",
        }.items()
    },
}


def _shapes(value: Any) -> Any:
    if isinstance(value, dict):
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


def _convolution(inp: Any, weight: Any, *_args: Any, out_shape: Any, **_kwargs: Any) -> int:
    transposed = len(_args) > 4 and _args[4] is True
    return 2 * prod(inp if transposed else out_shape) * prod(weight[1:])


def analyze_capture(capture: OperatorCapture) -> "FlopReport":
    """Convert saved native events into a FLOP report using metadata only.

    Args:
        capture: JSON-loaded capture_workload result. Unknown operators, layouts, lost
            events, and incomplete arguments retain partial counts and diagnostics.

    Returns:
        A standard FlopReport at the calling-thread, outermost eager-operator boundary.
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
            or capture["context"].get("boundary") != "outermost_native_operator"
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

    def gap(operator: str, message: str) -> None:
        diagnostics.append({
            "code": "incomplete_capture",
            "severity": "warning",
            "metric": "flops",
            "operator": operator,
            "message": message,
        })

    if capture["dropped_events"] or capture["errors"]:
        gap("", f"Capture lost {capture['dropped_events']} events and recorded {capture['errors']} collector errors.")
    for event in capture["operators"]:
        if (
            not isinstance(event, dict)
            or any(not isinstance(event.get(key), str) for key in ("operator", "overload", "module"))
            or any(not isinstance(event.get(key), list) for key in ("inputs", "outputs"))
            or not isinstance(event.get("kwargs"), dict)
        ):
            raise ValueError("Invalid operator event.")
        name = event["operator"].replace("::", ".")
        reason = _IGNORED_OPERATOR_REASONS.get(name)
        if (
            name == "aten.dropout"
            and len(event["inputs"]) == 3
            and (event["inputs"][1] == 0 or event["inputs"][2] is False)
        ):
            reason = "Dropout evaluation or zero probability is identity."
        if reason is None:
            try:
                args = _shapes(event["inputs"])
                kwargs = {key: _shapes(value) for key, value in event["kwargs"].items()}
                outputs = _shapes(event["outputs"])
                if not outputs:
                    raise NotImplementedError("Missing operator outputs.")
                out_shape = outputs[0] if len(outputs) == 1 else outputs
                packet = name.removeprefix("aten.") if name.startswith("aten.") else ""
                formula = _FORMULAS.get(packet)
                if packet in {"mm", "bmm", "matmul", "linear", "addmm", "baddbmm"}:
                    count = _product(*args, out_shape=out_shape, left=int(packet in {"addmm", "baddbmm"}))
                elif packet in {"convolution", "_convolution", "conv1d", "conv2d", "conv3d"}:
                    count = _convolution(*args, out_shape=out_shape)
                elif packet in {"max_pool2d", "max_pool3d"}:
                    count = FORMULAS[packet + "_with_indices"](*args, out_shape=(out_shape, None), **kwargs)
                elif packet == "scaled_dot_product_attention":
                    query, key, value, mask, dropout, causal = args[:6]
                    count = FORMULAS["_scaled_dot_product_flash_attention_for_cpu"](
                        query,
                        key,
                        value,
                        dropout,
                        causal,
                        attn_mask=mask,
                    )
                elif formula is not None:
                    count = formula(*args, out_shape=out_shape, **kwargs)
                else:
                    raise NotImplementedError("No portable formula for this outermost native operator.")
                tensors = _tensors([event["inputs"], event["kwargs"], event["outputs"]])
                if tensors and not any(t["floating"] or t["complex"] for t in tensors):
                    reason = "Integer-only arithmetic is excluded from FLOPs."
                else:
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        raise ValueError("Invalid formula result.")
                    counts[name] += count
                    path = event["module"]
                    if path:
                        parts = path.split(".")
                        for length in range(1, len(parts) + 1):
                            modules[".".join(parts[:length])] += count
                    if any(t["complex"] for t in tensors):
                        gap(name, "Shape counts do not establish complete real-arithmetic counts for complex inputs.")
            except (ValueError, TypeError, NotImplementedError, IndexError, KeyError, ZeroDivisionError) as error:
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
            "boundary": "outermost_native_operator",
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
