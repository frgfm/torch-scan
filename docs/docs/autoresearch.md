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

## Fixed input cases

For a small set of caller-chosen inputs, replace top-level `args`/`kwargs` or `inputs` with a named `cases` object:

```json
{
  "model": "torchvision.models:resnet18",
  "model_kwargs": {"weights": null},
  "cases": {
    "zeros": {"args": [{"shape": [1, 3, 224, 224], "fill": 0}]},
    "saved-sample": {"inputs": "inputs.pt"}
  },
  "check": {"absolute_error": {"mean": 0.0001, "p99": 0.001}}
}
```

Run this manifest with the same Python API or CLI command. Each case accepts only `args`/`kwargs` descriptors or
one saved `inputs` file. Paths remain relative to the manifest. The original model, checkpoint, FP32 execution,
seed, threads, numerical limits and timing/memory settings are shared. Supply batch size one in every case.
The manifest must contain at least one case; cases cannot override models, settings or checks.

Every case runs independently in fresh workers, including cases after a rejection. Each compares against its own
fixed outputs from the original model, even with `--baseline-model`. Any `discard` discards the set. Otherwise any
`inconclusive` keeps the set inconclusive. Only all `keep` results produce `keep`; probes return `measured` only
when every case is measured. Gains on one case never offset a loss on another. An unchecked memory ceiling in any
case prevents acceptance. Without numerical ceilings, all cases retain report-only fidelity semantics.

The combined `report.json` contains `cases.<name>` with each full case result and its `report` path.
Each `case-N/` directory retains the existing JSON, HTML, raw trials, confirmation, output errors and logs.
Latency, scoped memory and errors remain separate for each case; unavailable or unmeasured evidence stays explicit.
Initial output failures stop that case before timing, as in single-case evaluation. CLI output includes per-case
decisions, feedback and report paths. Single-case contracts and their report layout remain supported.

One shared identity set covers the manifest, model sources, checkpoint, every case's saved inputs, declared `files`
and generated reference outputs. Files are checked between workers and cases; changed files cannot be rehashed
as a fresh experiment halfway through the set. Preserve these identities across attempts, except for the edited
candidate source. Only declared dependencies are covered. Dataset loaders, training, precision search and scheduling
are outside this interface.

## Checks and acceptance

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

::: torchscan.autoresearch.evaluate_model
