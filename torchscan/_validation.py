# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Shared JSON report validation, independent of workload execution."""

import math
from typing import Any

_TIMING_METHOD = "torch.utils.benchmark.Timer"


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    # Keep both outputs valid, including XML 1.0. Reject invalid Unicode rather
    # than silently changing a stored report's identities or evidence.
    if any(
        (ord(c) < 32 and c not in "\t\r\n") or 0xD800 <= ord(c) <= 0xDFFF or ord(c) in (0xFFFE, 0xFFFF) for c in value
    ):
        raise ValueError(f"{where} contains invalid text")
    return value


def _number(value: Any, where: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    if integer and (not isinstance(value, int) or value < 0):
        raise ValueError(f"{where} must be a nonnegative integer")
    return value


def _json_tree(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _string(key, f"{where} key")
            _json_tree(item, f"{where}.{key}")
    elif isinstance(value, list):
        for item in value:
            _json_tree(item, where)
    elif isinstance(value, str):
        _string(value, where)
    elif value is not None and not isinstance(value, bool):
        _number(value, where)


def _metric(value: Any, where: str) -> None:
    value = _object(value, where)
    for field in ("unit", "scope", "method"):
        _string(value.get(field), f"{where}.{field}")
    status = value.get("status")
    if status == "complete":
        _number(value.get("value"), f"{where}.value")
        _number(value.get("known_value"), f"{where}.known_value")
        valid = value["value"] == value["known_value"]
    elif status == "partial":
        _number(value.get("known_value"), f"{where}.known_value")
        valid = "value" in value and value["value"] is None
    elif status == "unavailable":
        valid = all(field in value and value[field] is None for field in ("value", "known_value"))
    else:
        raise ValueError(f"{where} has invalid status")
    if not valid:
        raise ValueError(f"{where} violates the {status} measurement invariant")


def _metrics(value: Any, where: str) -> None:
    for name, result in _object(value, where).items():
        _metric(result, f"{where}.{name}")
