# Batch-one inference autoresearch

Give an existing coding agent one JSON contract. TorchScan builds and compares FP32 inference calls, with
latency, memory and numerical output evidence. No experiment script is needed. Requires the development version.

```json
{
  "model": "torchvision.models:resnet18",
  "model_kwargs": {"weights": null},
  "args": [{"shape": [1, 3, 224, 224], "dtype": "float32"}],
  "check": {"absolute_error": {"mean": 0.0001, "p99": 0.001}}
}
```

```shell
python -m torchscan.autoresearch workload.json --output /tmp/baseline
python -m torchscan.autoresearch workload.json \
  --candidate-model your_package.models:candidate --output /tmp/attempt-001
```

The first command returns `measured`; the second compares a candidate. `model` accepts `package.module:factory`,
`models.py:factory` or an existing `nn.Module` instance. Supply constructor arguments in `model_kwargs`.
An optional `checkpoint` loads a local state dict with `weights_only=True`.

Use `args`/`kwargs` descriptors or `"inputs": "inputs.pt"`, saved with
`torch.save({"args": (sample,), "kwargs": {...}}, "inputs.pt")`. Paths are relative to the contract.
Descriptors accept `shape`, `dtype`, `fill` and integer `low`/`high` bounds; dtypes are float32, int64, int32 or bool.
Nested containers and scalar arguments are supported. Supply batch size one on your model's batch axis;
TorchScan does not infer axes. Saved tensors retain their data and are never interpreted as descriptors.

Calls use `eval()` and inference mode, with autocast and CUDA TF32 disabled. Defaults are CPU, seed 0 and one
PyTorch thread; override `device`, `seed` and `threads` at the top level. Floating inputs and model state must be FP32.

Each floating output reports mean, median, p90, p95, p99, maximum and count of absolute errors against the original
model. Selected `check.absolute_error` ceilings apply separately to every output tensor; p99 can permit outliers.
Without ceilings, `output_error_policy: report_only` means differences are measured without asserting equivalence.
Shapes, structures and non-floating outputs must match; empty or non-finite error samples cannot pass.
Checks cover first/warmed calls and each measured instance after timing. Feedback retains the selected phase.

The latency objective requires more than 3% improvement and a gain larger than observed timing spread in every
pair, followed by fresh confirmation. Timing spread is the interquartile range (IQR), the width of the middle 50%
of samples; it is descriptive, not a confidence interval. Decisions are `keep`, `discard` or `inconclusive`.
Completed commands exit zero; agents must read `decision`. JSON, HTML, raw measurements and logs stay in a new
output directory. `settings` accepts `trials`, `timeout`, `warmup`, `min_run_time`, `min_repeats`, `min_improvement`,
and an optional `max_memory_bytes` ceiling. `memory_metric` chooses `peak_memory` (default) or `process_peak_rss`.

For a bounded search, agree on an attempt limit, run an identical-model control, then edit only the candidate.
Preserve the contract, original model, inputs and limits; declare extra immutable dependencies in `files`.
Files are hashed and checked during evaluation. Use `--baseline-model` for an accepted incumbent; numerical
reference outputs remain tied to the contract's original model. Record each revision, hypothesis and report path.
Promote only on `keep`; rerun the final winner against the original model on an idle target.

Latency measures warmed model calls with inputs already on the device, not transfer, preprocessing, queueing or
request p95. RSS covers a separate child's imports, loading, warmup and inference on Linux/macOS. PyTorch memory
covers tracked CPU tensors or reserved CUDA allocator bytes; it is separate from RSS. Models execute trusted Python.
Results cover supplied inputs and hardware, not task accuracy, unseen data or a real-time deadline. Large-batch
training and precision optimization are outside this workflow; real CUDA remains unvalidated.

The checked-in `examples/autoresearch/workload.json` and `models.py` demonstrate ResNet18 convolution/BatchNorm
fusion without downloading weights. Install `.[model-test]`, then run the same command with
`--candidate-model models.py:candidate` and the example contract.

## Release control pack

From a checkout with `.[model-test]` installed, run the fixed control pack on an idle target:

```shell
python examples/autoresearch/control_pack.py --device cpu --output /tmp/torchscan-controls-cpu
python examples/autoresearch/control_pack.py --device cuda --output /tmp/torchscan-controls-cuda
```

Use a new output directory outside the checkout. The pack runs ten identical-model controls, a redundant FP32
toy against an equivalent single ReLU, a wrong-output candidate, a compiled toy, and the ResNet18 fusion example.
Toys require zero maximum error. ResNet limits mean error to 0.0001 and p99 and maximum error to 0.001.
Each comparison uses batch one, seed 0, one PyTorch compute thread, and three alternating process pairs plus
confirmation for tentative keeps.

`summary.json` records every verdict and both release gates: zero false keeps and rejection of incorrect values.
A failed gate exits nonzero. Review inconclusive gates before relying on a run. `native.json` compares
independent PyTorch timing with the median across TorchScan pairs, including confirmation. It saves raw blocks
and labels differences over 20% inconclusive. CUDA timing synchronizes each clock boundary. The compiled toy
saves construction, compiler diagnostics, and first-call time (including lazy compilation) separately from warmed timing.

Keep the contracts, configuration, worker logs, JSON/HTML, and compiler cache outside Git. Missing CUDA is
recorded as unvalidated. Wheel and source-distribution checks exercise baseline measurement and wrong-output
rejection through the installed CLI outside the checkout.

In two prior runs on a shared Apple M3 Pro CPU, both gates passed and the exact toy optimization was kept.
Native warmed latency fell from about 45–46 µs to 6 µs. ResNet fusion remained inconclusive in both runs,
and the compiled toy was inconclusive on the combined revision. See the
[PR results](https://github.com/frgfm/torch-scan/pull/213) for revision-specific measurements.

::: torchscan.autoresearch.evaluate_model
