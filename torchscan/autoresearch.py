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
import tempfile
import traceback
from collections.abc import Mapping
from math import isclose
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from .benchmark import BenchmarkReport, _positive_finite, _validate_timing_settings
from .benchmark_compare import BenchmarkComparison, compare_benchmarks
from .process.rss import _SUPERVISOR
from .render import render_report
from .report import metric_result

__all__ = ["evaluate_model"]


def _load(path: Path, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    # Caller-owned Python is executable code; bypass stale timestamp-based bytecode caches.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # ruff: ignore[exec-builtin]
    return module


def _settings(contract: dict[str, Any]) -> dict[str, Any]:
    supplied = contract.get("settings", {})
    settings = {
        "trials": 3,
        "timeout": 120,
        "warmup": 5,
        "min_run_time": 0.2,
        "min_repeats": 5,
        "min_improvement": 0.03,
    }
    if not isinstance(supplied, dict) or supplied.keys() - (settings.keys() | {"max_memory_bytes", "memory_metric"}):
        raise ValueError("Unknown inference settings.")
    settings.update(supplied)
    settings.update(device=contract.get("device", "cpu"), work_units=1, work_unit="calls")
    device = settings["device"]
    if not isinstance(device, str) or not (
        device in ("cpu", "cuda") or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("device must be cpu, cuda or cuda:N.")
    if type(settings["trials"]) is not int or settings["trials"] < 2:
        raise ValueError("trials must be an integer >= 2.")
    if (
        not _positive_finite(settings["timeout"])
        or not _positive_finite(settings["min_improvement"])
        or settings["min_improvement"] >= 1
    ):
        raise ValueError("timeout must be positive; min_improvement must be between zero and one.")
    cap = settings.get("max_memory_bytes")
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError("max_memory_bytes must be a positive integer.")
    if settings.get("memory_metric", "peak_memory") not in ("peak_memory", "process_peak_rss"):
        raise ValueError("memory_metric must be peak_memory or process_peak_rss.")
    _validate_timing_settings(
        lambda: None, 1, "calls", settings["warmup"], settings["min_run_time"], settings["min_repeats"]
    )
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


def _worker(path: Path) -> None:
    from ._model_workload import model_worker

    request = json.loads(path.read_text(encoding="utf-8"))
    try:
        _verify_files(request["identities"])
        sys.path.insert(0, str(Path(request["experiment"]).parent))
        result = model_worker(request)
        _verify_files(request["identities"])
    except Exception as error:  # ruff: ignore[blind-except] -- Preserve failures at the process boundary.
        traceback.print_exc()
        result = {"error": f"{type(error).__name__}: {error}"}
    _write(path.with_suffix(".result.json"), result)


def _run_worker(request: dict[str, Any], path: Path, timeout: float) -> dict[str, Any]:
    _write(path, request)
    command = [sys.executable, "-m", "torchscan.autoresearch", "--_worker", str(path)]
    rss_path = path.with_suffix(".rss.json")
    rss_supported = sys.platform in {"darwin", "linux"} and hasattr(os, "wait4")
    if rss_supported:
        command = [sys.executable, "-I", "-S", "-c", _SUPERVISOR, json.dumps(command), str(rss_path)]
    with path.with_suffix(".log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path(request["experiment"]).parent,
            start_new_session=os.name == "posix",
            env={**os.environ, "PYTHONPYCACHEPREFIX": request.get("bytecode_cache", sys.pycache_prefix or "")},
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
    rss_result = json.loads(rss_path.read_text()) if rss_supported else None
    if rss_result is not None and rss_result["returncode"]:
        raise RuntimeError(
            f"Worker exited with status {rss_result['returncode']}; see {path.with_suffix('.log').name}."
        )
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    if "error" in result:
        raise RuntimeError(f"{result['error']} (see {path.with_suffix('.log').name}).")
    peak = rss_result["peak"] * (1 if sys.platform == "darwin" else 1024) if rss_result is not None else None
    result["_process_peak_rss"] = metric_result(
        status="complete" if rss_supported else "unavailable",
        value=peak,
        unit="bytes",
        scope="child_process_lifetime",
        method="os.wait4.ru_maxrss" if rss_supported else "unavailable",
    )
    return result


def _decide(trials: list[BenchmarkComparison], settings: dict[str, Any]) -> tuple[str, list[str]]:
    cap, metric = settings.get("max_memory_bytes"), settings.get("memory_metric", "peak_memory")
    memory = [trial["after"]["totals"][metric] for trial in trials]
    if cap is not None and any(item["status"] == "complete" and item["value"] > cap for item in memory):
        return "discard", ["Candidate exceeds the memory ceiling."]
    if all(trial["latency_change"] == "slower" for trial in trials):
        return "discard", ["Candidate is consistently slower beyond observed timing spread."]
    if any(trial["context_changes"].keys() - {"model_reference"} for trial in trials):
        return "inconclusive", ["Execution settings changed between implementations."]
    if cap is not None and any(item["status"] != "complete" for item in memory):
        return "inconclusive", ["The selected memory ceiling could not be checked."]
    gains = [
        -cast(float, trial["totals"]["latency"]["delta"]) / cast(float, trial["before"]["totals"]["latency"]["value"])
        for trial in trials
        if trial["latency_change"] == "faster"
    ]
    minimum = settings["min_improvement"]
    if (
        gains
        and len(gains) == len(trials)
        and all(gain > minimum and not isclose(gain, minimum, rel_tol=1e-12) for gain in gains)
    ):
        return "keep", ["Every pair clears the latency improvement and timing-spread thresholds."]
    return "inconclusive", ["Latency gains are mixed, too small or within observed timing spread."]


def evaluate_model(
    workload: str | Path, *, output: str | Path, candidate_model: str | None = None, baseline_model: str | None = None
) -> dict[str, Any]:
    """Measure batch-one FP32 inference or compare a candidate against an incumbent.

    The JSON contract names the original model, args/kwargs or saved inputs, optional
    absolute-error limits and timing settings. Numerical references stay fixed to the
    original model when baseline_model selects an accepted incumbent. Output must be
    a new directory; it retains JSON, HTML, hashes and worker logs. Models execute as
    trusted code. Only declared dependencies are hashed. Without error ceilings, keep
    confirms resource gains and output compatibility, not numerical equivalence.
    """
    experiment, destination = Path(workload).resolve(), Path(output).resolve()
    contract = json.loads(experiment.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not isinstance(contract.get("model"), str):
        raise ValueError("The workload must name an existing model/factory.")
    baseline = baseline_model or contract["model"]
    destination.mkdir(parents=True, exist_ok=False)
    identities = _identities([experiment])
    report: dict[str, Any] = {
        "schema_version": 1,
        "decision": "inconclusive",
        "reasons": [],
        "workload": contract,
        "baseline": baseline,
        "candidate": candidate_model,
        "identities": identities,
        "output_error_policy": "enforced" if contract.get("check") else "report_only",
        "settings": {},
        "trials": [],
        "confirmation": None,
    }
    request = {
        "experiment": str(experiment),
        "identities": identities,
        "baseline": baseline,
        "candidate": candidate_model or baseline,
        "check_outputs": candidate_model is not None,
        "reference_outputs": str(destination / "reference.pt"),
    }

    def run(mode: str, implementation: str, label: str) -> dict[str, Any]:
        result = _run_worker(
            {**request, "mode": mode, "implementation": implementation},
            destination / f"{label}.json",
            report["settings"].get("timeout", 120),
        )
        if mode == "measure":
            rss = run("rss", implementation, label + "-rss")
            result["totals"]["process_peak_rss"] = rss["_process_peak_rss"]
            result["context"]["rss_boundary"] = (
                "fresh process imports, model/input loading, warmup and one inference; no instrumentation"
            )
            _write(destination / f"{label}.result.json", result)
        return result

    def checked(results: dict[str, Any]) -> bool:
        for role in ("baseline", "candidate"):
            if results[role]["output_check"] != "passed":
                report["decision"] = "inconclusive" if role == "baseline" else "discard"
                report["reasons"] = [f"{role.capitalize()} failed the fixed reference check."]
                return False
        return True

    def pair(index: int, label: str) -> BenchmarkComparison:
        order = [("baseline", baseline), ("candidate", candidate_model)]
        results = {
            name: run("measure", cast(str, model), f"{label}-{name}")
            for name, model in (order if index % 2 == 0 else reversed(order))
        }
        comparison = compare_benchmarks(
            cast(BenchmarkReport, results["baseline"]),
            cast(BenchmarkReport, results["candidate"]),
            check=lambda: checked(results),
        )
        (destination / f"{label}.html").write_text(render_report(comparison, title=label))
        return comparison

    # Fresh bytecode avoids running stale imports after rapid, same-size agent edits.
    with tempfile.TemporaryDirectory(prefix="torchscan-imports-") as cache:
        request["bytecode_cache"] = cache
        try:
            inspected = run("inspect", baseline, "inspect")
            _verify_files(identities)
            identities.update(_identities([Path(path) for path in inspected["files"]]))
            report["settings"] = request["settings"] = inspected["settings"]
            if candidate_model is None:
                report["measurement"] = run("measure", baseline, "measurement")
                report["decision"] = "measured"
                report["feedback"] = _feedback(report["measurement"])
                (destination / "measurement.html").write_text(
                    render_report(cast(BenchmarkReport, report["measurement"]))
                )
            else:
                run("reference", contract["model"], "reference")
                identities.update(_identities([Path(request["reference_outputs"])]))
                initial = report["initial_checks"] = {
                    name: run("check", model, "initial-" + name)
                    for name, model in (("baseline", baseline), ("candidate", candidate_model))
                }
                if checked(initial):
                    for index in range(report["settings"]["trials"]):
                        report["trials"].append(pair(index, f"trial-{index + 1}"))
                        if report["trials"][-1]["output_check"] == "failed":
                            break
                    else:
                        report["decision"], report["reasons"] = _decide(report["trials"], report["settings"])
                        if report["decision"] == "keep":
                            report["confirmation"] = pair(len(report["trials"]), "confirmation")
                            if report["confirmation"]["output_check"] == "passed":
                                report["decision"], report["reasons"] = _decide(
                                    [*report["trials"], report["confirmation"]], report["settings"]
                                )
                last = report["confirmation"] or (report["trials"][-1] if report["trials"] else None)
                evidence = (
                    {"before": last["before"], "after": last["after"]}
                    if last
                    else {"before": initial["baseline"], "after": initial["candidate"]}
                )
                report["feedback"] = {name: _feedback(value) for name, value in evidence.items()}
            _verify_files(identities)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            report["decision"], report["reasons"] = "inconclusive", [str(error)]
    _write(destination / "report.json", report)
    return report


def _feedback(result: Mapping[str, Any]) -> dict[str, Any]:
    def value(name: str, scale: float = 1) -> float | None:
        metric = result.get("totals", {}).get(name, {})
        return metric["value"] * scale if metric.get("status") == "complete" else None

    return {
        "latency_ms": value("latency", 1000),
        "latency_iqr_ms": value("latency_iqr", 1000),
        "throughput": value("throughput"),
        "throughput_unit": "calls/s",
        "peak_rss_mib": value("process_peak_rss", 1 / 1024**2),
        "pytorch_peak_mib": value("peak_memory", 1 / 1024**2),
        "pytorch_memory_scope": result.get("totals", {}).get("peak_memory", {}).get("scope"),
        "output_errors": result.get("output_errors"),
        "output_check_phase": result.get("output_check_phase"),
    }


def main() -> None:
    """Print one compact JSON result for an existing coding agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", type=Path, nargs="?")
    parser.add_argument("--candidate-model")
    parser.add_argument("--baseline-model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker is not None:
        _worker(args._worker)
        return
    if args.workload is None or args.output is None:
        parser.error("workload and --output are required")
    result = evaluate_model(
        args.workload, output=args.output, candidate_model=args.candidate_model, baseline_model=args.baseline_model
    )
    print(  # ruff: ignore[print]
        json.dumps(
            {key: result.get(key) for key in ("decision", "reasons", "output_error_policy", "feedback")}
            | {"report": str(args.output.resolve() / "report.json")}
        )
    )


if __name__ == "__main__":
    main()
