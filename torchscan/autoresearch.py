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
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch

from .benchmark import BenchmarkReport, _positive_finite, _synchronizer, _validate_timing_settings
from .benchmark_compare import BenchmarkComparison, compare_benchmarks
from .process.rss import _SUPERVISOR
from .render import render_report
from .report import metric_result
from .workload import measure_workload

__all__ = ["evaluate_experiment", "evaluate_model"]


def _load(path: Path, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    # Caller-owned Python is executable code; bypass stale timestamp-based bytecode caches.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # ruff: ignore[exec-builtin]
    return module


def _settings(experiment: ModuleType | dict[str, Any], *, require_limits: bool = True) -> dict[str, Any]:
    supplied = experiment if isinstance(experiment, dict) else getattr(experiment, "SETTINGS", None)
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
        "memory_metric",
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
    if unused in settings or ((require_limits or ceiling in settings) and not _positive_finite(settings.get(ceiling))):
        raise ValueError(f"This objective requires {ceiling} and does not accept {unused}.")
    if ceiling in settings and ceiling == "max_memory_bytes" and type(settings[ceiling]) is not int:
        raise ValueError("max_memory_bytes must be an integer.")
    if settings.get("memory_metric", "peak_memory") not in ("peak_memory", "process_peak_rss"):
        raise ValueError("memory_metric must be peak_memory or process_peak_rss.")
    _validate_timing_settings(
        lambda: None,
        settings["work_units"],
        settings["work_unit"],
        settings["warmup"],
        settings["min_run_time"],
        settings["min_repeats"],
    )
    if isinstance(experiment, ModuleType) and not all(
        callable(getattr(experiment, name, None)) for name in ("build", "check")
    ):
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
        if request.get("model_contract"):
            from ._model_workload import model_worker

            sys.path.insert(0, str(Path(request["experiment"]).parent))
            result = model_worker(request)
            _verify_files(request["identities"])
            _write(result_path, result)
            return
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
                if request["mode"] == "rss":
                    _, synchronize = _synchronizer(torch.device(settings["device"]))
                    for _ in range(settings["warmup"] + 1):
                        workload()
                    synchronize()
                    result = {"completed": True}
                else:
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
    return _evaluate(experiment, str(baseline), str(candidate), output=output, model_contract=False)


def evaluate_model(
    workload: str | Path,
    *,
    output: str | Path,
    candidate_model: str | None = None,
    baseline_model: str | None = None,
) -> dict[str, Any]:
    """Measure a fixed-FP32 model/input contract, optionally comparing a candidate factory.

    Args:
        workload: JSON file defining model, args/kwargs (or saved inputs), and inference settings.
        output: New directory for measurements, checks, JSON, HTML and worker logs.
        candidate_model: Optional module:factory or file.py:factory to evaluate against the baseline.
        baseline_model: Optional incumbent factory; defaults to the contract's model.

    Returns:
        A measured baseline when no candidate is supplied; otherwise a checked keep/discard/inconclusive result.
        Resource feedback includes clean timing, tracked/allocator memory and separate fresh-process peak RSS.
        Output errors are reported against the original model. Explicit check limits gate acceptance;
        without limits, keep confirms resource gains and compatible outputs, not numerical equivalence.
    """
    contract = json.loads(Path(workload).read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("The workload must be a JSON object.")
    baseline = baseline_model or contract.get("model")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError("The workload must define model as an existing model/factory reference.")
    return _evaluate(
        workload,
        baseline,
        candidate_model,
        output=output,
        model_contract=True,
    )


def _evaluate(
    experiment: str | Path,
    baseline: str,
    candidate: str | None,
    *,
    output: str | Path,
    model_contract: bool,
) -> dict[str, Any]:
    from ._experiment_decision import decide

    experiment = Path(experiment).resolve()
    if not model_contract:
        baseline, candidate = str(Path(baseline).resolve()), str(Path(cast(str, candidate)).resolve())
    identities = _identities(
        [experiment] if model_contract else [experiment, Path(baseline), Path(cast(str, candidate))]
    )
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
        "candidate": candidate,
        "settings": {},
        "checks": {},
        "trials": [],
        "confirmation": None,
    }
    if model_contract:
        report["workload"] = json.loads(experiment.read_text(encoding="utf-8"))
        report["output_error_policy"] = "enforced" if report["workload"].get("check") else "report_only"
    request: dict[str, Any] = {
        "experiment": str(experiment),
        "identities": identities,
        "model_contract": model_contract,
        "baseline": baseline,
        "candidate": candidate or baseline,
        "reference_outputs": str(destination / "reference.pt"),
    }
    # A new import cache prevents fast, same-size agent edits from executing stale timestamp-based bytecode.
    import_cache = tempfile.TemporaryDirectory(prefix="torchscan-imports-")
    request["bytecode_cache"] = import_cache.name

    def run(mode: str, implementation: str, label: str) -> dict[str, Any]:
        result = _run_worker(
            {
                **request,
                "mode": mode,
                "implementation": str(implementation),
            },
            destination / f"{label}.json",
            report["settings"].get("timeout", 120),
        )
        if mode == "measure" and model_contract:
            rss = run("rss", implementation, f"{label}-rss")
            result["totals"]["process_peak_rss"] = rss["_process_peak_rss"]
            result["context"]["rss_boundary"] = (
                "fresh process imports, model/input loading, warmup and one inference; no instrumentation"
            )
            _write(destination / f"{label}.result.json", result)
        return result

    def checks(prefix: str) -> bool:
        results = report["checks"][prefix] = {}
        for name, implementation in (("baseline", baseline), ("candidate", candidate)):
            checked = run("check", cast(str, implementation), f"{prefix}-check-{name}")
            results[name] = checked["output_check"]
            if "output_errors" in checked:
                report.setdefault("output_errors", {}).setdefault(prefix, {})[name] = checked["output_errors"]
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
        measurements = {
            name: run("measure", cast(str, implementation), f"{label}-{name}") for name, implementation in order
        }
        comparison = compare_benchmarks(
            cast(BenchmarkReport, measurements["baseline"]),
            cast(BenchmarkReport, measurements["candidate"]),
            check=lambda: checks(label),
        )
        (destination / f"{label}.html").write_text(render_report(comparison, title=label), encoding="utf-8")
        return comparison

    def _compare() -> None:
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

    try:
        inspected = run("inspect", baseline, "inspect")
        _verify_files(identities)
        dependencies = _identities([Path(path) for path in inspected["files"]])
        _verify_files(identities)
        identities.update({path: digest for path, digest in dependencies.items() if path not in identities})
        report["settings"] = request["settings"] = inspected["settings"]
        if model_contract and candidate is None:
            report["measurement"] = run("measure", baseline, "measurement")
            report["decision"], report["reasons"] = (
                "measured",
                ["Baseline measured; supply a candidate model to evaluate a change."],
            )
            report["feedback"] = _feedback(report["measurement"])
            (destination / "measurement.html").write_text(
                render_report(cast(BenchmarkReport, report["measurement"])), encoding="utf-8"
            )
        else:
            if model_contract:
                run("reference", report["workload"]["model"], "reference")
                identities.update(_identities([Path(request["reference_outputs"])]))
            _compare()
            last = report["confirmation"] or (report["trials"][-1] if report["trials"] else None)
            if last is not None and model_contract:
                report["feedback"] = {name: _feedback(last[name]) for name in ("before", "after")}
        _verify_files(identities)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["decision"], report["reasons"] = "inconclusive", [str(error)]
    finally:
        import_cache.cleanup()
    if report.get("output_errors"):
        report["feedback"] = {
            **report.get("feedback", {}),
            "output_errors": list(report["output_errors"].values())[-1],
            "output_error_policy": report["output_error_policy"],
        }
    _write(destination / "report.json", report)
    return report


def _feedback(measurement: dict[str, Any]) -> dict[str, Any]:
    totals = measurement["totals"]

    def value(name: str, scale: float = 1) -> float | None:
        metric = totals[name]
        return metric["value"] * scale if metric["status"] == "complete" else None

    return {
        "latency_ms": value("latency", 1000),
        "latency_iqr_ms": value("latency_iqr", 1000),
        "throughput": value("throughput"),
        "throughput_unit": totals["throughput"]["unit"],
        "peak_rss_mib": value("process_peak_rss", 1 / 1024**2),
        "pytorch_peak_mib": value("peak_memory", 1 / 1024**2),
        "pytorch_memory_scope": totals["peak_memory"]["scope"],
        "execution_precision": measurement["context"]["execution_precision"],
    }


def main() -> None:
    """Run one experiment and print a compact machine-readable decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, nargs="?")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-model", help="Incumbent model/factory for a JSON workload.")
    parser.add_argument("--candidate-model", help="Candidate model/factory for a JSON workload.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker is not None:
        _worker(args._worker)
        return
    if args.experiment is None or args.output is None:
        parser.error("workload/experiment and --output are required")
    if args.experiment.suffix == ".json":
        if args.baseline is not None or args.candidate is not None:
            parser.error("JSON workloads use --baseline-model and --candidate-model references")
        result = evaluate_model(
            args.experiment,
            output=args.output,
            baseline_model=args.baseline_model,
            candidate_model=args.candidate_model,
        )
    else:
        if args.baseline is None or args.candidate is None or args.baseline_model or args.candidate_model:
            parser.error("Python experiments require --baseline FILE and --candidate FILE")
        result = evaluate_experiment(args.experiment, args.baseline, args.candidate, output=args.output)
    print(  # ruff: ignore[print]
        json.dumps({
            "decision": result["decision"],
            "reasons": result["reasons"],
            "report": str(args.output.resolve() / "report.json"),
            "feedback": result.get("feedback"),
        })
    )


if __name__ == "__main__":
    main()
