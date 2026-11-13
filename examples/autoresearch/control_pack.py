# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Bounded autoresearch controls; write all evidence outside the checkout."""

import argparse
import json
import os
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from pathlib import Path
from statistics import median

import torch
from torch.utils.benchmark import Timer

from torchscan._model_workload import _factory
from torchscan.autoresearch import evaluate_model

SETTINGS = {"trials": 3, "warmup": 5, "min_run_time": 0.2, "min_repeats": 5, "timeout": 300, "min_improvement": 0.03}


class Redundant(torch.nn.Module):
    """Repeat an idempotent FP32 operation without changing its output."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the same ReLU 64 times."""
        for _ in range(64):
            inputs = inputs.relu()
        return inputs


class Wrong(torch.nn.Module):
    """Produce the right shape with deliberately incorrect values."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Shift every output beyond the zero error ceiling."""
        return inputs.relu() + 1


def baseline() -> torch.nn.Module:
    """Build the deliberately redundant baseline."""
    return Redundant()


def faster() -> torch.nn.Module:
    """Remove all redundant ReLUs, preserving exact finite FP32 outputs."""
    return torch.nn.ReLU()


def compiled() -> torch.nn.Module:
    """Compile the redundant baseline with the default Inductor backend."""
    return torch.compile(baseline(), fullgraph=True)


def write_json(path: Path, value: object) -> None:
    """Save raw evidence without non-finite JSON values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def native_timing(contract: dict, reference: str) -> dict:
    """Time the same calls directly with native PyTorch and synchronized boundaries."""
    device = contract["device"]
    torch.set_num_threads(contract["threads"])
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(contract["seed"])
    factory, _ = _factory(reference, Path.cwd())
    start = time.perf_counter()
    model = factory().eval().to(device)
    construction = time.perf_counter() - start
    torch.manual_seed(contract["seed"])
    inputs = torch.randn(contract["args"][0]["shape"], dtype=torch.float32, device=device)

    @torch.inference_mode()
    def call() -> torch.Tensor:
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            return model(inputs)

    def clock() -> float:
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    start = clock()
    output = call()
    first_call = clock() - start
    for _ in range(SETTINGS["warmup"]):
        call()
    timer = Timer(stmt="call()", globals={"call": call}, timer=clock, num_threads=contract["threads"])
    measurement = timer.blocked_autorange(min_run_time=SETTINGS["min_run_time"])
    # Match the minimum sample count without using TorchScan's timing implementation.
    samples = list(measurement.raw_times)
    while len(samples) < SETTINGS["min_repeats"]:
        samples.extend(timer.timeit(measurement.number_per_run).raw_times)
    torch.manual_seed(contract["seed"])
    original, _ = _factory(contract["model"], Path.cwd())
    with torch.inference_mode():
        expected = original().eval().to(device)(inputs)
    torch.testing.assert_close(output, expected, rtol=0, atol=contract["check"]["absolute_error"]["max"])
    return {
        "model": reference,
        "construction_seconds": construction,
        "first_call_seconds": first_call,
        "compilation_metrics": torch._dynamo.utils.compile_times() if reference.endswith(":compiled") else None,
        "first_call_boundary": "synchronized model call; includes lazy compilation/cache loading when compiled",
        "raw_times_seconds": samples,
        "number_per_run": measurement.number_per_run,
        "median_seconds": median(samples) / measurement.number_per_run,
        "max_absolute_error": (output - expected).abs().max().item(),
        "output_check": "passed",
    }


def release_gates(reports: dict) -> dict:
    """Keep incomplete controls distinct from passed release gates."""
    controls = [reports[f"identical-{index:02}"] for index in range(10)]
    complete = all(
        len(report["trials"]) == SETTINGS["trials"]
        and report["output_error_policy"] == "enforced"
        and all(trial["output_check"] == "passed" for trial in report["trials"])
        for report in controls
    )
    wrong = reports["incorrect"]
    return {
        "zero_false_keeps": "failed"
        if any(report["decision"] == "keep" for report in controls)
        else "passed"
        if complete
        else "inconclusive",
        "incorrect_output_rejected": "passed"
        if wrong["decision"] == "discard"
        and wrong.get("initial_checks", {}).get("candidate", {}).get("output_check") == "failed"
        else "failed"
        if wrong["decision"] == "keep"
        else "inconclusive",
    }


def run_pack(destination: Path, device: str) -> dict:
    """Run ten identical controls and four candidates, preserving every verdict."""
    script = Path(__file__).resolve()
    checkout = script.parents[2]
    if destination.is_relative_to(checkout):
        raise ValueError("Evidence must be outside the checkout.")
    destination.mkdir(parents=True, exist_ok=False)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(destination / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(destination / "triton-cache")
    configuration = {
        "device": device,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name()
        if device == "cuda" and torch.cuda.is_available()
        else platform.processor(),
        "torch_config": torch.__config__.show(),
        "revision": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),  # ruff: ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
        "diff": subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD"], text=True),  # ruff: ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
        "settings": SETTINGS,
        "compiler_cache": "new directory shared by this pack; native compiled control runs before autoresearch",
    }
    write_json(destination / "configuration.json", configuration)
    if device == "cuda":
        if not torch.cuda.is_available():
            summary = {"cuda_gate": "unvalidated", "reason": "CUDA is unavailable; no CPU substitution."}
            write_json(destination / "summary.json", summary)
            return summary
        hardware = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, check=False)  # ruff: ignore[start-process-with-partial-path]
        (destination / "nvidia-smi.txt").write_text(hardware.stdout + hardware.stderr, encoding="utf-8")
    toy = {
        "model": f"{script}:baseline",
        "args": [{"shape": [1, 4096], "dtype": "float32"}],
        "device": device,
        "threads": 1,
        "seed": 0,
        "check": {"absolute_error": dict.fromkeys(("mean", "median", "p90", "p95", "p99", "max"), 0)},
        "settings": SETTINGS,
    }
    resnet = json.loads((script.parent / "workload.json").read_text())
    resnet.update(model=f"{script.parent / 'models.py'}:baseline", device=device, threads=1, seed=0)
    resnet["settings"] = {**SETTINGS, **resnet["settings"]}
    reports, native = {}, {}
    cases = [
        *((f"identical-{index:02}", toy, toy["model"]) for index in range(10)),
        ("faster", toy, f"{script}:faster"),
        ("incorrect", toy, f"{script}:Wrong"),
        ("compiled", toy, f"{script}:compiled"),
        ("resnet-fusion", resnet, f"{script.parent / 'models.py'}:candidate"),
    ]
    for name, contract, candidate in cases:
        contract_path = destination / f"{name}.json"
        write_json(contract_path, contract)
        if name in ("identical-00", "faster", "compiled", "resnet-fusion"):
            native[name] = {}
            for side, reference in (("before", contract["model"]), ("after", candidate)):
                try:
                    native[name][side] = native_timing(contract, reference)
                except Exception as error:  # ruff: ignore[blind-except] -- Retain failed compiler/device evidence.
                    native[name][side] = {"error": f"{type(error).__name__}: {error}"}
            write_json(destination / f"{name}-native.json", native[name])
        reports[name] = evaluate_model(contract_path, candidate_model=candidate, output=destination / name)
        for side, timing in native.get(name, {}).items():
            trials = reports[name]["trials"] + (
                [reports[name]["confirmation"]] if reports[name]["confirmation"] else []
            )
            if trials and "median_seconds" in timing:
                value = median(trial[side]["totals"]["latency"]["value"] for trial in trials)
                timing["autoresearch_median_seconds"] = value
                timing["relative_difference"] = abs(timing["median_seconds"] / value - 1)
                timing["agreement"] = "within_20_percent" if timing["relative_difference"] <= 0.2 else "inconclusive"
        write_json(destination / "native.json", native)
        print(  # ruff: ignore[print]
            json.dumps({"case": name, "decision": reports[name]["decision"], "reasons": reports[name]["reasons"]}),
            flush=True,
        )
    summary = {
        "release_gates": release_gates(reports),
        "cuda_gate": "unvalidated" if device != "cuda" else "measured",
        "decisions": {name: report["decision"] for name, report in reports.items()},
        "native": native,
    }
    write_json(destination / "summary.json", summary)
    return summary


def main() -> None:
    """Run the fixed control pack on the selected device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_pack(args.output.resolve(), args.device)
    print(json.dumps(summary))  # ruff: ignore[print]
    if any(value == "failed" for value in summary.get("release_gates", {}).values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
