# Autoresearch for model resources

This workflow requires the development version of TorchScan.

Use TorchScan as the fixed evaluator in a coding agent's edit / measure / keep loop. The owner supplies the workload,
reference, output check, device and resource limit. The agent changes a candidate implementation and keeps it only
when TorchScan returns `keep`.

## Run the ResNet18 example

In a TorchScan checkout, install the optional model example dependencies:

```shell
uv pip install -e ".[model-test]"
```

Review and configure `examples/autoresearch/experiment.py`, then compare the initial per-image implementation with
batched inference:

```shell
python -m torchscan.autoresearch examples/autoresearch/experiment.py \
  --baseline examples/autoresearch/baseline.py \
  --candidate examples/autoresearch/candidate.py \
  --output /tmp/torchscan-autoresearch-run
```

The output directory must not exist. It contains `report.json`, comparison reports and worker logs. Each report keeps
the environment, file identities, per-trial reference checks, measurement scopes and the final confirmation. Keep
each run directory separate.
The command prints compact JSON with `decision`, `reasons` and the report path. Completed evaluations exit zero for
all three decisions; automation must read `decision` before keeping a change.

The example initializes a local ResNet18 and a fixed batch of four 32 × 32 images. It downloads no weights or data.
`candidate_wrong.py` has the expected output shape with incorrect values and should fail the reference check.
`candidate_memory.py` adds a large temporary allocation and should fail a suitably low memory limit.

## Configure an experiment

The owner-controlled experiment module declares `SETTINGS`:

| Setting | Purpose |
| --- | --- |
| `workload_id` | Stable owner label for the weights, data, seed and preprocessing. |
| `device` | CPU or CUDA execution device. |
| `objective` | `latency` minimizes warmed latency under `max_memory_bytes`; `memory` minimizes peak memory under `max_latency_seconds`. |
| `min_improvement` | Fractional improvement required to keep a candidate. |
| `trials`, `timeout` | Paired trial count and per-worker wall-clock deadline. |
| `warmup`, `min_run_time`, `min_repeats` | Timing settings shared by both implementations. |
| `work_units`, `work_unit` | Work completed by each call and its throughput label. |

`FILES` lists immutable owner-controlled paths relative to the experiment, such as the reference implementation. The
evaluator records their SHA-256 identities along with the experiment, baseline and candidate. A file change during a
run makes its evidence inconclusive. Start a new run after changing the owner contract.

For a memory objective, change `objective` to `memory`, remove `max_memory_bytes`, and set `max_latency_seconds`.
This example uses a 256 MiB PyTorch memory limit for the latency objective.

The experiment supplies `build(implementation)` to return a repeatable zero-argument workload and representative inputs, and
`check(implementation)` to run the owner's reference check. Checks run in fresh workers outside timed measurement.
Build a fresh model and inputs for each worker, set seeds and threads explicitly, and freeze precision, execution mode
and preprocessing for the full run. A passing `None`/`True` keeps checking; `False` or an assertion failure rejects the
candidate. Checker and worker errors remain inconclusive.
Run experiments sequentially on an otherwise idle target. TorchScan serializes its own workers but does not reserve
the device against other processes.

## Interpret the decision

- **`keep`** means checks pass, requested measurements are complete, limits hold, and the repeated improvement clears
  both the configured threshold and variability rule. TorchScan remeasures a tentative winner in fresh workers.
- **`discard`** means a reference check fails, a complete measurement crosses a resource limit, or the objective consistently worsens.
- **`inconclusive`** means evidence is missing, a worker fails or times out, or improvement is too small or noisy.

Latency is warmed block-average time rather than request p95. CPU peak memory is PyTorch-tracked tensors; CUDA peak
memory is the PyTorch reserved allocator. These metrics are not process RSS or total device capacity. A reference
check establishes its declared tolerance; it does not measure task accuracy unless the owner supplies a task-quality
evaluator.

See [`examples/autoresearch/program.md`](https://github.com/frgfm/torch-scan/blob/main/examples/autoresearch/program.md)
for the bounded agent recipe, baseline control and final rerun instructions.

## Python API

::: torchscan.autoresearch.evaluate_experiment
