# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Evaluate trusted, owner-defined inference experiments in fresh processes."""

import argparse
import hashlib
import json
import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from .benchmark import BenchmarkReport, _positive_finite, _validate_timing_settings
from .benchmark_compare import BenchmarkComparison, compare_benchmarks
from .render import render_report
from .workload import measure_workload

__all__ = ["evaluate_experiment"]


def _load(path: Path, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    # Caller-owned Python is executable code; bypass stale timestamp-based bytecode caches.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # ruff: ignore[exec-builtin]
    return module


def _settings(experiment: ModuleType) -> dict[str, Any]:
    supplied = getattr(experiment, "SETTINGS", None)
    if not isinstance(supplied, dict):
        raise ValueError("The experiment must define SETTINGS as a dictionary.")
    settings: dict[str, Any] = {
        "trials": 3,
        "timeout": 120,
        "warmup": 5,
        "min_run_time": 0.2,
        "min_repeats": 5,
        "work_units": 1,
        "work_unit": "calls",
    }
    allowed = settings.keys() | {
        "device",
        "objective",
        "min_improvement",
        "max_memory_bytes",
        "max_latency_seconds",
        "workload_id",
    }
    if supplied.keys() - allowed:
        raise ValueError(f"Unknown experiment settings: {sorted(supplied.keys() - allowed)}.")
    settings.update(supplied)
    if settings.get("objective") not in ("latency", "memory"):
        raise ValueError("objective must be 'latency' or 'memory'.")
    device = settings.get("device")
    if not isinstance(device, str) or not (
        device in ("cpu", "cuda") or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'.")
    if not isinstance(settings.get("workload_id"), str) or not settings["workload_id"].strip():
        raise ValueError("workload_id must identify the fixed work, inputs and weights.")
    if not _positive_finite(settings.get("min_improvement")) or settings["min_improvement"] >= 1:
        raise ValueError("min_improvement must be a finite fraction strictly between 0 and 1.")
    if type(settings["trials"]) is not int or settings["trials"] < 2:
        raise ValueError("trials must be an integer >= 2.")
    if not _positive_finite(settings["timeout"]):
        raise ValueError("timeout must be positive finite seconds per worker.")
    ceiling = "max_memory_bytes" if settings["objective"] == "latency" else "max_latency_seconds"
    unused = "max_latency_seconds" if settings["objective"] == "latency" else "max_memory_bytes"
    if unused in settings or not _positive_finite(settings.get(ceiling)):
        raise ValueError(f"This objective requires {ceiling} and does not accept {unused}.")
    if ceiling == "max_memory_bytes" and type(settings[ceiling]) is not int:
        raise ValueError("max_memory_bytes must be an integer.")
    _validate_timing_settings(
        lambda: None,
        settings["work_units"],
        settings["work_unit"],
        settings["warmup"],
        settings["min_run_time"],
        settings["min_repeats"],
    )
    if not all(callable(getattr(experiment, name, None)) for name in ("build", "check")):
        raise ValueError("The experiment must define build(implementation) and check(implementation).")
    return settings


def _identities(paths: list[Path]) -> dict[str, str]:
    identities = {}
    for path in paths:
        with path.open("rb") as source:
            identities[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return identities


def _verify_files(identities: dict[str, str]) -> None:
    if _identities([Path(path) for path in identities]) != identities:
        raise ValueError("Experiment, implementation or declared dependency changed during evaluation.")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = request_path.with_suffix(".result.json")
    try:
        _verify_files(request["identities"])
        experiment_path, implementation_path = Path(request["experiment"]), Path(request["implementation"])
        sys.path[:0] = [str(experiment_path.parent), str(implementation_path.parent)]
        experiment = _load(experiment_path, "_torchscan_experiment")
        settings = _settings(experiment)
        if request["mode"] == "inspect":
            files = getattr(experiment, "FILES", [])
            if not isinstance(files, (list, tuple)) or any(not isinstance(path, (str, Path)) for path in files):
                raise ValueError("FILES must be a list of dependency paths, relative to the experiment.")
            result = {
                "settings": settings,
                "files": [str((experiment_path.parent / path).resolve()) for path in files],
            }
        else:
            if settings != request["settings"]:
                raise ValueError("Experiment settings changed between workers.")
            implementation = _load(implementation_path, "_torchscan_implementation")
            if request["mode"] == "check":
                try:
                    outcome = experiment.check(implementation)
                except AssertionError:
                    outcome = False
                if outcome is not None and type(outcome) is not bool:
                    raise ValueError("check must return None/True or False, or raise AssertionError.")
                result = {"output_check": "failed" if outcome is False else "passed"}
            else:
                workload, inputs = experiment.build(implementation)
                if inputs is None:
                    raise ValueError("build must return a callable and non-None representative inputs.")
                result = measure_workload(
                    workload,
                    inputs=inputs,
                    metrics=("latency", "throughput", "memory"),
                    print_summary=False,
                    **{
                        key: settings[key]
                        for key in ("device", "work_units", "work_unit", "warmup", "min_run_time", "min_repeats")
                    },
                )
        _verify_files(request["identities"])
        _write(result_path, result)
    except Exception as error:  # ruff: ignore[blind-except] -- Preserve worker failures at the process boundary.
        traceback.print_exc()
        _write(result_path, {"error": f"{type(error).__name__}: {error}"})


def _run_worker(request: dict[str, Any], path: Path, timeout: float) -> dict[str, Any]:
    _write(path, request)
    with path.with_suffix(".log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-m", "torchscan.autoresearch", "--_worker", str(path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path(request["experiment"]).parent,
            start_new_session=os.name == "posix",
        ) as child:
            try:
                child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                        [
                            str(Path(os.environ["SYSTEMROOT"]) / "System32" / "taskkill.exe"),
                            "/PID",
                            str(child.pid),
                            "/T",
                            "/F",
                        ],
                        check=False,
                        stdout=log,
                        stderr=log,
                    )
                    child.kill()
                child.wait()
                raise
        if child.returncode:
            raise RuntimeError(f"Worker exited with status {child.returncode}; see {path.with_suffix('.log').name}.")
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    if "error" in result:
        raise RuntimeError(f"{result['error']} (see {path.with_suffix('.log').name}).")
    return result


def evaluate_experiment(
    experiment: str | Path, baseline: str | Path, candidate: str | Path, *, output: str | Path
) -> dict[str, Any]:
    """Evaluate one inference candidate, with checks and measurements in fresh workers.

    Args:
        experiment: Trusted Python file defining SETTINGS, build(module), check(module)
            and optional FILES dependencies. The owner controls fixed work and checks.
        baseline: Current incumbent implementation file, imported by fresh workers.
        candidate: Proposed implementation file. Neither implementation is modified.
        output: New artifact directory. Existing paths are never overwritten.

    Returns:
        A JSON-serializable report and keep/discard/inconclusive decision. Report JSON,
        worker logs, raw results and per-trial HTML are also saved in output.
        A tentative keep requires an additional fresh checked confirmation pair.

    Raises:
        FileExistsError: If output already exists.
        OSError: If required source files cannot be read or artifacts cannot be written.

    Notes:
        Files execute as trusted Python, without a security sandbox. SHA-256 checks
        detect changes to declared files; the owner must declare external weights/data
        in FILES or use immutable revisions in workload_id. Input metadata alone cannot
        prove constant work. Workers run sequentially, with a per-worker wall timeout
        (including imports/compilation); inspection has a 120-second timeout.
        CPU memory is tracked tensor peak; CUDA memory is reserved allocator peak
        after warmup, not device capacity. Timing spread is descriptive, not significance.
    """
    from ._experiment_decision import decide

    experiment, baseline, candidate = (Path(path).resolve() for path in (experiment, baseline, candidate))
    identities = _identities([experiment, baseline, candidate])
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "resource_experiment",
        "decision": "inconclusive",
        "reasons": [],
        "identities": identities,
        "experiment": str(experiment),
        "baseline": str(baseline),
        "candidate": str(candidate),
        "settings": {},
        "checks": {},
        "trials": [],
        "confirmation": None,
    }
    request: dict[str, Any] = {"experiment": str(experiment), "identities": identities}

    def run(mode: str, implementation: Path, label: str) -> dict[str, Any]:
        return _run_worker(
            {**request, "mode": mode, "implementation": str(implementation)},
            destination / f"{label}.json",
            report["settings"].get("timeout", 120),
        )

    def checks(prefix: str) -> bool:
        results = report["checks"][prefix] = {}
        for name, implementation in (("baseline", baseline), ("candidate", candidate)):
            results[name] = run("check", implementation, f"{prefix}-{name}")["output_check"]
        if results["baseline"] != "passed":
            report["decision"], report["reasons"] = "inconclusive", ["The incumbent failed the fixed reference check."]
            return False
        if results["candidate"] != "passed":
            report["decision"], report["reasons"] = (
                "discard",
                ["Candidate failed the fixed reference check; gains withheld."],
            )
            return False
        return True

    def pair(index: int, label: str) -> BenchmarkComparison:
        order = (("baseline", baseline), ("candidate", candidate))
        if index % 2:
            order = tuple(reversed(order))
        measurements = {name: run("measure", implementation, f"{label}-{name}") for name, implementation in order}
        comparison = compare_benchmarks(
            cast(BenchmarkReport, measurements["baseline"]),
            cast(BenchmarkReport, measurements["candidate"]),
            check=lambda: checks(label),
        )
        (destination / f"{label}.html").write_text(render_report(comparison, title=label), encoding="utf-8")
        return comparison

    try:
        inspected = run("inspect", baseline, "inspect")
        _verify_files(identities)
        dependencies = _identities([Path(path) for path in inspected["files"]])
        _verify_files(identities)
        identities.update({path: digest for path, digest in dependencies.items() if path not in identities})
        report["settings"] = request["settings"] = inspected["settings"]
        if checks("initial"):
            for index in range(report["settings"]["trials"]):
                report["trials"].append(pair(index, f"trial-{index + 1}"))
                if report["trials"][-1]["output_check"] == "failed":
                    break
            else:
                report["decision"], report["reasons"] = decide(report["trials"], report["settings"])
                if report["decision"] == "keep":
                    report["confirmation"] = pair(len(report["trials"]), "confirmation")
                    if report["confirmation"]["output_check"] == "passed":
                        report["decision"], report["reasons"] = decide(
                            [*report["trials"], report["confirmation"]], report["settings"]
                        )
        _verify_files(identities)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["decision"], report["reasons"] = "inconclusive", [str(error)]
    _write(destination / "report.json", report)
    return report


def main() -> None:
    """Run one experiment and print a compact machine-readable decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, nargs="?")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker is not None:
        _worker(args._worker)
        return
    if any(getattr(args, key) is None for key in ("experiment", "baseline", "candidate", "output")):
        parser.error("experiment, --baseline, --candidate and --output are required")
    result = evaluate_experiment(args.experiment, args.baseline, args.candidate, output=args.output)
    print(  # ruff: ignore[print]
        json.dumps({
            "decision": result["decision"],
            "reasons": result["reasons"],
            "report": str(args.output.resolve() / "report.json"),
        })
    )


if __name__ == "__main__":
    main()
