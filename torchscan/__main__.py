# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .check import MemoryLimit, PerformanceCheck, check_performance


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Check saved JSON without executing workloads; return 0, 1, or 2.

    Args:
        argv: CLI arguments, defaulting to the process arguments.

    Returns:
        Zero for passed, one for failed, two for inconclusive or invalid input.
    """
    parser = argparse.ArgumentParser(prog="python -m torchscan")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check a saved BenchmarkComparison without running workloads.")
    check.add_argument("report", type=Path, help="Path to comparison JSON.")
    check.add_argument("--max-latency-regression", type=float, required=True, help="Fraction, e.g. 0.05 allows 5%%.")
    check.add_argument("--memory-metric", help="Exact candidate totals key, e.g. process_peak_rss.")
    check.add_argument("--memory-scope", help="Exact scope recorded for that memory metric.")
    check.add_argument("--max-memory-bytes", type=int, help="Inclusive candidate memory ceiling in bytes.")
    args = parser.parse_args(argv)
    result: PerformanceCheck
    try:
        memory = (args.memory_metric, args.memory_scope, args.max_memory_bytes)
        limit: MemoryLimit | None = None
        if any(value is not None for value in memory):
            if any(value is None for value in memory):
                raise ValueError(
                    "Memory limits require --memory-metric, --memory-scope, and --max-memory-bytes together."
                )
            limit = {"metric": args.memory_metric, "scope": args.memory_scope, "max_bytes": args.max_memory_bytes}
        comparison = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=_json_object)
        result = check_performance(comparison, max_latency_regression=args.max_latency_regression, memory_limit=limit)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        result = {"status": "inconclusive", "reasons": [f"Invalid input: {error}"]}
    print(json.dumps(result))  # ruff: ignore[print]
    return {"passed": 0, "failed": 1, "inconclusive": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
