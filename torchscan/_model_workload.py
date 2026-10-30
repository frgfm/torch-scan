# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""The declarative model/input contract used by autoresearch workers."""

import importlib
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .benchmark import _synchronizer
from .workload import measure_workload


def _factory(reference: str, directory: Path) -> tuple[Any, list[Path]]:
    module_name, separator, attribute = reference.rpartition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Models use 'package.module:factory' or 'file.py:factory'.")
    if module_name.endswith(".py"):
        from .autoresearch import _load

        model_path = (directory / module_name).resolve()
        sys.path.insert(0, str(model_path.parent))
        module = _load(model_path, "_torchscan_model")
    else:
        module = importlib.import_module(module_name)
    factory = module
    for part in attribute.split("."):
        factory = getattr(factory, part)
    source = inspect.getsourcefile(
        factory if inspect.isfunction(factory) or inspect.isclass(factory) else type(factory)
    )
    sources = {path for path in (source, module.__file__) if path is not None}
    if not sources:
        raise ValueError("Cannot locate the model factory's source for experiment identity.")
    return factory, [Path(path).resolve() for path in sorted(sources)]


def _tensor(value: Any, device: str, *, descriptors: bool = True) -> Any:
    if descriptors and isinstance(value, dict) and "shape" in value:
        if value.keys() - {"shape", "dtype", "fill", "low", "high"}:
            raise ValueError("Unknown tensor descriptor fields.")
        shape = value["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("Tensor shape must be a list of positive dimensions, including batch.")
        dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "bool": torch.bool,
        }.get(value.get("dtype", "float32"))
        if dtype is None:
            raise ValueError("Unsupported tensor dtype.")
        if "fill" in value:
            if "low" in value or "high" in value:
                raise ValueError("Choose fill or integer low/high bounds.")
            return torch.full(shape, value["fill"], dtype=dtype, device=device)
        if (dtype.is_floating_point or dtype == torch.bool) and ("low" in value or "high" in value):
            raise ValueError(
                "low/high bounds apply to integer descriptors; floating tensors use normal samples or fill."
            )
        if dtype == torch.bool:
            return torch.ones(shape, dtype=dtype, device=device)
        if not dtype.is_floating_point:
            low, high = value.get("low"), value.get("high")
            if type(low) is not int or type(high) is not int or low >= high:
                raise ValueError("Integer tensors require explicit integer low < high bounds, or fill.")
            return torch.randint(low, high, shape, dtype=dtype, device=device)
        return torch.randn(shape, dtype=dtype, device=device)
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _tensor(item, device, descriptors=descriptors) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_tensor(item, device, descriptors=descriptors) for item in value)
    return value


def _outputs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _outputs(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_outputs(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("Automatic checks support tensors and nested mappings/lists/tuples of tensors or scalar values.")


def model_worker(request: dict[str, Any]) -> dict[str, Any]:
    """Load a model contract, construct inputs and execute one worker phase."""
    from .autoresearch import _settings

    path = Path(request["experiment"])
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.keys() - {
        "model",
        "model_kwargs",
        "checkpoint",
        "args",
        "kwargs",
        "inputs",
        "workload",
        "device",
        "seed",
        "threads",
        "settings",
        "check",
        "files",
    }:
        raise ValueError("Unknown model workload fields; expected a JSON object.")
    if contract.get("workload", "inference") != "inference":
        raise ValueError(
            "The built-in workload is inference; custom training/stateful workloads use a Python experiment."
        )
    if not isinstance(contract.get("model"), str):
        raise ValueError("model must name an existing model or factory.")
    if "inputs" in contract and ("args" in contract or "kwargs" in contract):
        raise ValueError("Use args/kwargs descriptors or a saved inputs file, not both.")
    if not any(key in contract for key in ("inputs", "args", "kwargs")):
        raise ValueError("Supply args/kwargs descriptors or an inputs file.")
    seed, threads = contract.get("seed", 0), contract.get("threads", 1)
    if type(seed) is not int or type(threads) is not int or threads < 1:
        raise ValueError("seed must be an integer and threads a positive integer.")
    supplied = contract.get("settings", {})
    if not isinstance(supplied, dict) or "workload_id" in supplied or "device" in supplied:
        raise ValueError("settings must be an object; device and workload identity belong to the workload contract.")
    settings = _settings(
        {
            "device": contract.get("device", "cpu"),
            "objective": "latency",
            "min_improvement": 0.03,
            **supplied,
            "workload_id": request["identities"][str(path)],
        },
        require_limits=False,
    )
    tolerance = contract.get("check", {"rtol": 1e-4, "atol": 1e-5})
    if not isinstance(tolerance, dict) or tolerance.keys() - {"rtol", "atol"}:
        raise ValueError("check accepts only rtol and atol.")
    if any(type(value) not in (int, float) or not 0 <= value < float("inf") for value in tolerance.values()):
        raise ValueError("Output tolerances must be finite nonnegative numbers.")
    if request["mode"] != "inspect" and settings != request["settings"]:
        raise ValueError("Model workload settings changed between workers.")
    declared_files = contract.get("files", [])
    if not isinstance(declared_files, list) or any(not isinstance(item, str) for item in declared_files):
        raise ValueError("files must be a list of dependency paths.")
    files = [str((path.parent / item).resolve()) for item in declared_files]
    files.extend(str((path.parent / contract[name]).resolve()) for name in ("inputs", "checkpoint") if name in contract)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    factory, _ = _factory(request["implementation"], path.parent)
    if request["mode"] == "inspect":
        for reference in (contract["model"], request["baseline"], request["candidate"]):
            _, model_sources = _factory(reference, path.parent)
            files.extend(str(source) for source in model_sources)
        return {"settings": settings, "files": files}
    model_kwargs = contract.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError("model_kwargs must be an object.")
    model = factory if isinstance(factory, torch.nn.Module) else factory(**model_kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The model reference must be an nn.Module or a factory returning one.")
    if "checkpoint" in contract:
        model.load_state_dict(torch.load(path.parent / contract["checkpoint"], map_location="cpu", weights_only=True))
    device = settings["device"]
    model = model.eval().to(device)
    torch.manual_seed(seed)  # Input values must not depend on how many RNG draws a candidate's factory performs.
    if "inputs" in contract:
        inputs = torch.load(path.parent / contract["inputs"], map_location="cpu", weights_only=True)
        if not isinstance(inputs, dict) or inputs.keys() - {"args", "kwargs"}:
            raise ValueError("Saved inputs must contain args and/or kwargs, loaded with weights_only=True.")
    else:
        inputs = {name: contract.get(name, [] if name == "args" else {}) for name in ("args", "kwargs")}
    args = _tensor(inputs.get("args", []), device, descriptors="inputs" not in contract)
    kwargs = _tensor(inputs.get("kwargs", {}), device, descriptors="inputs" not in contract)
    if not isinstance(args, (tuple, list)) or not isinstance(kwargs, dict):
        raise ValueError("args must be a list/tuple and kwargs an object.")

    @torch.inference_mode()
    def workload() -> Any:
        return model(*args, **kwargs)

    if request["mode"] == "reference":
        torch.save(_outputs(workload()), request["reference_outputs"])
        return {"reference_outputs": request["reference_outputs"]}
    if request["mode"] == "check":
        reference = torch.load(request["reference_outputs"], map_location="cpu", weights_only=True)
        try:
            torch.testing.assert_close(_outputs(workload()), reference, **tolerance, equal_nan=False)
        except AssertionError:
            return {"output_check": "failed"}
        return {"output_check": "passed"}
    if request["mode"] == "rss":
        _, synchronize = _synchronizer(torch.device(device))
        for _ in range(settings["warmup"] + 1):
            workload()
        synchronize()
        return {"completed": True}
    result: dict[str, Any] = dict(
        measure_workload(
            workload,
            inputs={"args": args, "kwargs": kwargs},
            metrics=("latency", "throughput", "memory"),
            print_summary=False,
            **{
                key: settings[key]
                for key in ("device", "work_units", "work_unit", "warmup", "min_run_time", "min_repeats")
            },
        )
    )
    result["context"].update(
        workload="inference",
        inputs_origin="saved" if "inputs" in contract else "synthetic",
        seed=seed,
        model_reference=request["implementation"],
    )
    return result
