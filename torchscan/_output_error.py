# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Numerical output fidelity against the fixed original model's outputs."""

import math
from collections.abc import Mapping
from typing import Any

import torch

ERROR_STATISTICS = ("mean", "median", "p90", "p95", "p99", "max")


def compare_outputs(actual: Any, reference: Any, check: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Measure absolute errors per output tensor and enforce the owner's fixed limits."""
    outputs: dict[str, Any] = {}

    def reject(path: str, reason: str) -> bool:
        outputs[path or "/"] = {"status": "failed", "reason": reason}
        return False

    def visit(candidate: Any, expected: Any, path: str) -> bool:
        if type(candidate) is float and type(expected) is float:
            return visit(
                torch.tensor(candidate, dtype=torch.float64), torch.tensor(expected, dtype=torch.float64), path
            )
        if isinstance(candidate, torch.Tensor) and isinstance(expected, torch.Tensor):
            if candidate.shape != expected.shape:
                return reject(path, "Output shapes differ.")
            if not (candidate.is_floating_point() and expected.is_floating_point()):
                try:
                    torch.testing.assert_close(candidate, expected, rtol=0, atol=0, equal_nan=False)
                except AssertionError:
                    return reject(path, "Non-floating outputs must match exactly, including dtype.")
                return True
            left, right = candidate.to(torch.float64), expected.to(torch.float64)
            errors = (left - right).abs().reshape(-1)
            if not errors.numel() or not torch.isfinite(errors).all().item():
                return reject(path, "Empty or non-finite output errors cannot establish numerical fidelity.")
            ordered = errors.sort().values

            def quantile(probability: float) -> float:
                position = (ordered.numel() - 1) * probability
                low, high = math.floor(position), math.ceil(position)
                if low == high:
                    return ordered[low].item()
                weight = position - low
                return ordered[low].item() * (1 - weight) + ordered[high].item() * weight

            statistics = {"mean": errors.mean().item(), "max": ordered[-1].item()}
            statistics.update({
                name: quantile(probability)
                for name, probability in (("median", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))
            })
            if not all(math.isfinite(value) for value in statistics.values()):
                return reject(path, "Output error statistics overflowed.")
            outputs[path or "/"] = {
                "status": "complete",
                "count": errors.numel(),
                "reference_dtype": str(expected.dtype),
                "candidate_dtype": str(candidate.dtype),
                "failed_constraints": [],
                **statistics,
            }
            if not check:
                return True
            if "absolute_error" in check:
                exceeded = [name for name, limit in check["absolute_error"].items() if statistics[name] > limit]
                outputs[path or "/"]["failed_constraints"] = exceeded
                return not exceeded
            try:
                torch.testing.assert_close(left, right, **check, equal_nan=False)
            except AssertionError:
                outputs[path or "/"]["failed_constraints"] = ["pointwise_tolerance"]
                return False
            return True
        if isinstance(candidate, Mapping) and isinstance(expected, Mapping) and candidate.keys() == expected.keys():
            results = [
                visit(candidate[key], expected[key], path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                for key in expected
            ]
            return all(results)
        if (
            type(candidate) is type(expected)
            and isinstance(candidate, (tuple, list))
            and len(candidate) == len(expected)
        ):
            results = [
                visit(left, right, f"{path}/{index}")
                for index, (left, right) in enumerate(zip(candidate, expected, strict=True))
            ]
            return all(results)
        if type(candidate) is type(expected) and candidate is None:
            return True
        if type(candidate) is type(expected) and type(candidate) in (str, bool, int, float) and candidate == expected:
            return True
        return reject(path, "Output structure or scalar values differ.")

    return visit(actual, reference, ""), outputs
