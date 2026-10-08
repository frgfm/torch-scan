# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Compare per-sample and batched inference on your CPU, CUDA, or MPS device without downloads."""

import argparse
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch

from torchscan import compare_benchmarks, measure_flops, measure_latency, profile_workload, render_report
from torchscan.process import measure_peak_memory, measure_peak_rss
from torchscan.report import metric_result


def build_model(name, device):
    # Reuse the no-download definitions in tests/test_model_zoo.py.
    if name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        inputs = {"input": torch.randn(4, 3, 32, 32)}
        config = {"architecture": name, "weights": None, "torchvision_version": version("torchvision")}
        work_units, work_unit = 4, "images"
    elif name == "bert":
        from transformers import BertConfig, BertModel

        bert_config = BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
        )
        model = BertModel(bert_config)
        input_ids = torch.arange(32).reshape(4, 8)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        # Native config dictionaries can have integer label keys; reports require JSON object keys.
        config = json.loads(bert_config.to_json_string(use_diff=False)) | {
            "transformers_version": version("transformers")
        }
        work_units, work_unit = input_ids.numel(), "input_tokens"
    else:
        model = torch.nn.Linear(128, 128)
        inputs = {"input": torch.randn(32, 128)}
        config = {"architecture": "Linear", "in_features": 128, "out_features": 128}
        work_units, work_unit = 32, "samples"
    # Initialize on CPU before placement so the seed describes identical tensors on every backend.
    return (
        model.eval().to(device),
        {key: value.to(device) for key, value in inputs.items()},
        config,
        work_units,
        work_unit,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=["linear", "resnet18", "bert"], default="linear")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument(
        "--reverse", action="store_true", help="Measure batched first to check execution-order effects."
    )
    parser.add_argument("--rss", action="store_true", help="Also run each variant in a fresh process (Linux/macOS).")
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "torchscan-comparison")
    parser.add_argument("--rss-child", choices=["per_sample", "batched"], help=argparse.SUPPRESS)
    options = parser.parse_args()
    device = torch.device(options.device)
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model, inputs, config, work_units, work_unit = build_model(options.model, device)
    batch_size = next(iter(inputs.values())).shape[0]

    def forward(batch):
        if options.model == "bert":
            return model(**batch, return_dict=False)  # Hidden states and pooled output are both checked.
        return (model(batch["input"]),)

    @torch.inference_mode()
    def per_sample():
        outputs = [
            forward({key: value[index : index + 1] for key, value in inputs.items()}) for index in range(batch_size)
        ]
        return tuple(torch.cat(values) for values in zip(*outputs, strict=True))

    @torch.inference_mode()
    def batched():
        return forward(inputs)

    workloads = {"per_sample": per_sample, "batched": batched}

    if options.rss_child:
        # Includes imports, model/input loading, and 100 completed calls; no profiler in this RSS trial.
        for _ in range(100):
            workloads[options.rss_child]()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        return

    order = ("batched", "per_sample") if options.reverse else ("per_sample", "batched")
    reports = {}
    # Finish all clean timing before any instrumentation or RSS children.
    for name in order:
        reports[name] = measure_latency(
            workloads[name],
            device=device,
            inputs=inputs,
            work_units=work_units,
            work_unit=work_unit,
            min_run_time=options.min_run_time,
        )
        reports[name]["context"].update(
            experiment=name,
            model=options.model,
            model_config=config,
            seed=0,
            weights="locally_initialized",
            precision="float32; no autocast",
            torch_build=torch.__config__.show(),
            mkldnn_enabled=torch.backends.mkldnn.enabled,
            cudnn_version=torch.backends.cudnn.version(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            execution_mode="eval/inference_mode",
            execution_order=list(order),
            timing_boundary="resident inputs -> forward(s) -> concatenated outputs; excludes loading/transfers/checks",
            rss_boundary="fresh process imports, model/input construction and placement, 100 calls; no instrumentation",
            memory_boundary="separate single call after timing; includes observed resident tensors/allocator state",
            output_tolerance={"rtol": 1e-4, "atol": 1e-5},
        )

    for name in order:
        report, workload = reports[name], workloads[name]
        compute = measure_flops(workload)
        report["totals"]["operator_flops"] = compute["total"]
        report["context"]["flop_pass"] = compute
        method = {
            "cpu": "torch.profiler.export_memory_timeline",
            "cuda": "torch.cuda.max_memory_reserved",
            "mps": "torch.accelerator.memory.max_memory_reserved",
        }[device.type]
        scope = "pytorch_tracked_tensor_peak" if device.type == "cpu" else "allocator_reserved_peak"
        try:
            memory = measure_peak_memory(workload, device=options.device)
        except NotImplementedError as error:
            report["totals"]["pytorch_peak_memory"] = metric_result(
                status="unavailable", unit="bytes", scope=scope, method=method
            )
            report["context"]["memory_unavailable_reason"] = str(error)
        else:
            report["totals"]["pytorch_peak_memory"] = metric_result(
                status="complete", value=memory["peak_bytes"], unit="bytes", scope=scope, method=method
            )
            report["context"]["memory_pass"] = memory
            if "allocated_peak_bytes" in memory:
                report["totals"]["pytorch_allocated_peak"] = metric_result(
                    status="complete",
                    value=memory["allocated_peak_bytes"],
                    unit="bytes",
                    scope="allocator_allocated_peak",
                    method=method.replace("reserved", "allocated"),
                )
        if options.rss:
            report["totals"]["process_peak_rss"] = measure_peak_rss([
                sys.executable,
                str(Path(__file__).resolve()),
                "--device",
                options.device,
                "--model",
                options.model,
                "--threads",
                str(options.threads),
                "--rss-child",
                name,
            ])
        report["profile"] = profile_workload(workload, device=options.device, limit=10)

    before, after = reports["per_sample"], reports["batched"]
    check_evidence = {}

    def check():
        reference, candidate = per_sample(), batched()
        check_evidence.update(
            shapes=[list(value.shape) for value in reference],
            max_abs_error=max(
                (left - right).abs().max().item() for left, right in zip(reference, candidate, strict=True)
            ),
            finite=all(value.isfinite().all().item() for value in (*reference, *candidate)),
        )
        assert check_evidence["finite"]
        torch.testing.assert_close(reference, candidate, rtol=1e-4, atol=1e-5)

    comparison = compare_benchmarks(before, after, check=check)
    for report in (comparison["before"], comparison["after"]):
        report["context"]["output_check_evidence"] = check_evidence
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (options.output / "comparison.html").write_text(
        render_report(comparison, title=f"{options.model}: per-sample versus batched inference"), encoding="utf-8"
    )
    print(
        f"PyTorch {torch.__version__}; {after['context']['device_name']}; device={options.device}; threads={options.threads}"
    )
    print(f"Variant | first call ms | warmed median ms | IQR ms | {work_unit}/s | peak process RSS MiB")
    for name, report in reports.items():
        totals = report["totals"]
        rss = totals.get("process_peak_rss")
        rss_text = (
            f"{rss['value'] / 1024**2:.2f}" if rss and rss["status"] == "complete" else "unavailable/not requested"
        )
        print(
            f"{name} | {totals['first_call_latency']['value'] * 1000:.4f} | "
            f"{totals['latency']['value'] * 1000:.4f} | {totals['latency_iqr']['value'] * 1000:.4f} | "
            f"{totals['throughput']['value']:.0f} | {rss_text}"
        )
    print(f"Output check: {comparison['output_check']}; latency change: {comparison['latency_change']}")
    print(f"Output evidence: {check_evidence}")
    print(f"Saved {options.output.resolve()}/comparison.json and comparison.html")
    if comparison["output_check"] != "passed":
        raise SystemExit("Output check failed; deltas withheld. Inspect the saved evidence.")


if __name__ == "__main__":
    main()
