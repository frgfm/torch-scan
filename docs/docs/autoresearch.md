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

The output directory must be new and outside the checkout. The pack runs ten identical-model controls,
a redundant FP32 toy (64 ReLUs) against an exactly equivalent single ReLU, a same-shape wrong-output candidate,
an Inductor-compiled toy, and the existing ResNet18 fusion example. It downloads no weights. All toy error ceilings
are zero; ResNet uses mean ≤ 0.0001, p99 ≤ 0.001 and max ≤ 0.001. Batch size, FP32 precision, seed 0 and one
PyTorch compute thread stay fixed. Three alternating process pairs use five warmups, at least five timing blocks,
at least 0.2 seconds per measurement and the existing >3% improvement rule; a tentative keep needs confirmation.

`summary.json` records two release gates: **zero false keeps in all ten completed controls** and **rejection of
incorrect output**. A failed gate exits nonzero. An unfinished control leaves its gate inconclusive; exit zero alone
does not mean release approval. Inspect all positive controls and native cross-checks too. Preserve every discard
and inconclusive verdict; do not rerun selectively until a candidate passes.

Native PyTorch controls time the identical toy, faster toy, compiled toy and ResNet independently using
`torch.utils.benchmark.Timer`, with CUDA synchronization before each clock boundary. `native.json` retains raw block
times and compares native medians with the median of autoresearch pair medians, including confirmation when present.
A difference over 20% is a diagnostic inconclusive result, not a significance test. Native checks use the explicit
maximum absolute-error ceiling. Construction and the first call are separate from warmed timing. The compiled
candidate also saves PyTorch compiler timing diagnostics; its first-call wall time includes lazy compilation/cache
loading and execution, rather than claiming pure compiler time. A new compiler cache is shared only within each pack.

Keep `configuration.json`, generated contracts, `summary.json`, `native.json`, worker JSON/logs, reference tensors,
and comparison HTML outside Git. Configuration includes the revision, tracked diff, Python/PyTorch build and device;
CUDA also records `nvidia-smi -q`. Missing CUDA writes an explicit unvalidated result without substituting CPU.
`cuda_gate: measured` means hardware was exercised, not that all results passed review. Ten clean controls do not
establish a general false-positive rate. These synthetic inputs do not establish task accuracy.

Wheel and source-distribution verification also runs the CLI and its child workers outside the checkout,
checking exact identical outputs, wrong-value rejection, JSON feedback and saved HTML from the installed package.

### CPU observations, 2026-11-14

Revision `f5b59c6` (based on `be11b7b`) ran on an Apple M3 Pro with 36 GiB RAM, macOS 15.8.3,
Python 3.11.13, PyTorch 2.13.0 and torchvision 0.28.0. The host was shared and under variable load;
these are observations, not stable performance estimates or CUDA validation. Use the commands above to reproduce.

All ten identical-model controls were **inconclusive**, with **zero false keeps**. The same-shape wrong-output
candidate was **discarded** at its first reference check. Both release control gates passed on CPU.

| Candidate | TorchScan before → after (ms) | Native before → after (ms) | Autoresearch decision |
| --- | ---: | ---: | --- |
| Single ReLU | 0.04115 → 0.00425 | 0.04646 → 0.00601 | keep, confirmed |
| Compiled redundant toy | 0.04232 → 0.02625 | 0.04237 → 0.02689 | keep, confirmed |
| ResNet18 fusion | 18.116 → 16.689 | 22.294 → 17.282 | inconclusive |

TorchScan values are medians of pair medians, including confirmation for keeps. Native values come from separate
warmed blocks. Two of eight native side comparisons exceeded the 20% diagnostic band: the single-ReLU candidate
(41.6%) and the ResNet baseline (23.1%). Those cross-checks remain inconclusive. ResNet had two pairs within timing
spread and one faster pair, so its lower aggregate median does not establish a keep. No selective reruns were made.
Toy outputs had zero absolute error. Native fused ResNet maximum error was 0.0000023544, within its explicit ceiling.

The compiled native first call took 5.842 seconds, including compilation and execution, after 0.729 seconds of
construction/wrapping; warmed calls took 0.02689 ms. Compiler phase diagnostics were saved separately from warmed
samples. The full contracts, hardware/configuration, JSON, reference tensors, HTML, logs and compiler cache were
retained outside Git. **CUDA remains unvalidated**; this run supplies no accelerator speed or accuracy claim.

::: torchscan.autoresearch.evaluate_model
