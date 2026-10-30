# Autoresearch for model resources

This workflow requires the development version of TorchScan.

Describe your **model, inputs and workload** once. TorchScan builds the measurement and comparison harness. The
coding agent edits the model implementation and uses the same contract for every experiment.

## Start without an experiment script

Create `workload.json` using an existing importable model or factory:

```json
{
  "model": "torchvision.models:resnet18",
  "args": [{"shape": [4, 3, 32, 32], "dtype": "float32"}],
  "workload": "inference"
}
```

Then run:

```shell
python -m torchscan.autoresearch workload.json --output /tmp/torchscan-baseline
```

No `build()`, `check()`, timing loop or RSS command is needed. The first run returns `measured` and feedback containing
latency and timing spread in milliseconds, throughput, fresh-process peak RSS and PyTorch tensor/allocator memory.
It saves full JSON, offline HTML and worker logs in a new directory. `measured` establishes a baseline;
it does not endorse a change.

The default workload is `model.eval()` under `torch.inference_mode()`, on CPU, with seed 0 and one PyTorch thread.
Floating parameters, buffers, inputs and checked outputs must be FP32. Integer and boolean inputs retain their types.
Autocast and CUDA TF32 are disabled. Compare architecture and implementation changes while keeping precision fixed;
precision optimization and quantization are separate work.
Shapes include the batch dimension. Throughput defaults to calls/s; declare `settings.work_units` and `work_unit`
when one call completes several images, tokens or other units. Synthetic inputs are labelled in the saved context.
They measure the declared call, not task accuracy or production traffic.

## Use your own model and real inputs

`model` accepts `your_package.models:build_model`, an existing `your_package.runtime:model` instance, or
`models.py:build_model`. The factory returns an `nn.Module`; `model_kwargs` supplies constructor keyword arguments.
An optional `checkpoint` loads a local state dict with `weights_only=True`. Factories can also load their own weights.
Use locally available weights; factories execute trusted application code and may have their own side effects.

Describe positional tensors with `args` and named tensors with `kwargs`. Descriptors accept `shape`, `dtype`, and
optional `fill`; integer inputs require explicit `low`/`high` bounds or a fill value. Boolean masks default to true.
Nested containers and literal scalars/`null` are forwarded, so token IDs, masks and optional arguments need no callback.

For representative tensors you already have, save them once:

```python
torch.save({"args": (batch,), "kwargs": {"attention_mask": mask}}, "inputs.pt")
```

Use `"inputs": "inputs.pt"` instead of descriptor `args`/`kwargs`. Tensor data and checkpoints load with
`weights_only=True`, and their files are hashed for experiment identity. Paths are relative to the JSON contract.
Declare additional immutable dependencies in `files`. Keep the contract, inputs and original baseline fixed.

## Give the agent an objective

Use the same workload while selecting an existing candidate factory:

```shell
python -m torchscan.autoresearch workload.json \
  --candidate-model your_package.models:build_candidate --output /tmp/torchscan-candidate-001
```

The baseline defaults to the contract's `model`; `--baseline-model` selects a previously accepted incumbent.
The numerical reference always remains the contract's original model, so tolerances do not drift with promotion.
TorchScan generates reference outputs in a fresh process, checks each implementation against them, alternates
repeated measurement pairs, and confirms a tentative winner with another fresh checked pair. Reference outputs are
saved locally with the experiment. They can contain application data.

## Numerical output error

Output-error statistics are always measured for comparisons. To make numerical fidelity a constraint, set fixed
error ceilings in the contract, for example:

```json
"check": {
  "absolute_error": {"mean": 0.0001, "p99": 0.001}
}
```

Each floating output produces `mean`, `median`, `p90`, `p95`, `p99`, `max` and element `count`, measured on
`abs(candidate - original_reference)`. Values are converted to float64 **after** FP32 inference to calculate error
statistics. These are output-value differences, not task accuracy. Floating scalars use the same rules.
Selected ceilings must hold independently for every tensor; differently scaled outputs are not pooled.
Percentiles use linear interpolation over the flattened tensor, including batch. Statistics, original output dtypes
and failed constraint names are retained in JSON/CLI feedback, including rejected candidates.

Only selected statistics are constraints. A p99 ceiling can permit outliers; add `max` to bound every error.
Shapes/containers and non-floating outputs must still match; empty/non-finite error samples cannot pass. Limits are
in owner-defined output units and are never silently relaxed for a candidate. Without `check`, finite floating
output differences are reported without an implicit error ceiling: feedback says `output_error_policy: report_only`.
A `keep` then confirms resource gains and output compatibility, not numerical equivalence. Different random
initializations can produce unrelated outputs; their distance does not establish model quality.
An explicit `check` can instead set pointwise `rtol`/`atol`; missing members default to `1e-4`/`1e-5`.
Do not combine pointwise tolerances and `absolute_error` in one `check`.
Error calculation and checks run outside clean latency and RSS passes.

## Resource objectives

The default objective is a latency reduction greater than 3% and the observed timing spread in every pair.
The spread is the interquartile range (IQR): the difference between the 75th and 25th percentiles of timing samples.
It describes the middle 50% of timings and helps avoid promoting gains smaller than measurement noise.
It is not a confidence interval. Add explicit settings when you want different requirements:

```json
{
  "objective": "memory",
  "memory_metric": "process_peak_rss",
  "min_improvement": 0.05,
  "max_latency_seconds": 0.02
}
```

Place these values under `settings` in the contract. `memory_metric` selects `peak_memory` (default) or
`process_peak_rss`; a memory objective minimizes that metric. A latency objective can set `max_memory_bytes` for the
selected metric. Resource ceilings are optional and never inferred from available hardware. Timing settings, paired
`trials` and a per-worker `timeout` can also be set there. `device`, `seed` and `threads` belong at the top level.

- **`keep`**: reference compatibility checks pass, the objective improves consistently, explicit limits hold, and fresh
  confirmation agrees.
- **`discard`**: a check fails, a resource limit is exceeded, or the objective consistently worsens.
- **`inconclusive`**: missing, incompatible, noisy or failed measurements cannot establish an improvement.

The command prints compact JSON with the decision, reasons, numeric feedback and report path. Completed evaluations
exit zero for all decisions; agents must read `decision`. Run an identical-model control before searching, preserve
each attempt, and rerun the final winner against the original baseline. Run one experiment at a time on an idle target.

## Measurement boundaries

Clean latency excludes loading and instrumentation and describes block averages, not request p95. Peak RSS comes
from a separate child doing imports, model/input loading, warmup and one inference without profiling. Linux/macOS
provide per-child OS accounting; other platforms report RSS unavailable. It includes the child's whole lifetime,
not an inference-only delta or summed process tree.

CPU PyTorch memory covers tracked tensors; CUDA memory covers reserved allocator bytes. Neither is total device
memory or process RSS, and the scopes must not be added. Output checks establish the selected numerical constraints
on the supplied inputs, not unseen inputs or mathematical ground truth. IQR is descriptive.

## Runnable example and custom workloads

Install the optional model dependencies, then run the checked-in model contract:

```shell
uv pip install -e ".[model-test]"
python -m torchscan.autoresearch examples/autoresearch/workload.json \
  --candidate-model models.py:candidate --output /tmp/torchscan-resnet
```

The ResNet18 factories initialize weights locally without downloads. The
[bounded agent recipe](https://github.com/frgfm/torch-scan/blob/main/examples/autoresearch/program.md) describes the
editable surface, incumbent promotion and stop condition.

For stateful workloads or task-quality evaluators beyond output equivalence, the advanced Python experiment API
still accepts explicit `SETTINGS`, `build(implementation)` and `check(implementation)` functions.

::: torchscan.autoresearch.evaluate_model

::: torchscan.autoresearch.evaluate_experiment
