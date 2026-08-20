# Understanding results

TorchScan measures one concrete execution. Results are useful only with the input metadata, execution context,
method, status, and diagnostics that accompany them.

## Metric result states

Every measured metric uses the same trust contract:

| Status | Meaning | Numeric field |
| --- | --- | --- |
| `complete` | The method covered the requested execution scope. | `value` |
| `partial` | Some work was counted and some was not. | `known_value`, a lower bound |
| `unavailable` | The method could not produce a result. | Neither field is authoritative |

A complete value of zero is different from an incomplete result. TorchScan does not report a coverage percentage:
one uncounted operator can dominate a workload, so the fraction of recognized operator kinds would be misleading.

Use `strict=True` with `crawl_module` or `summary` when partial or unavailable module metrics must raise
`IncompleteAnalysisError` instead of returning a report.

## Parameters and model storage

Parameter and buffer counts come from tensors registered on the model. Storage size is not peak memory: it excludes
or separates activations, gradients, optimizer state, allocator behavior, Python objects, and third-party allocations.

## Module FLOPs, MACs, and DMAs

Module metrics use TorchScan formulas for recognized module families:

- FLOPs count formula-defined arithmetic for the forward pass.
- MACs count multiply-accumulate work for supported modules.
- DMAs estimate formula-defined tensor and parameter data movement.

These are theoretical counts, not FLOP/s, memory bandwidth, or latency. Diagnostics identify unsupported module work.

## Operator FLOPs

Operator FLOPs use PyTorch's `torch.utils.flop_counter.FlopCounterMode`. This observes dispatcher operations, including
functional calls and operations inside custom modules, but can count only operators with registered or caller-provided
formulas. Counts are grouped globally, by module, and by operator.

Operator and module FLOPs can differ because their formulas, boundaries, and decomposition differ. Report both with
their method labels; do not average, add, or substitute one silently for the other.

## Receptive field

Receptive-field values follow module execution order. Sequential convolutional paths can be described, including
dilation, but hook order does not reconstruct arbitrary branch topology. Residual and other skip-connected models can
therefore yield partial or unavailable results.

## Peak memory

`measure_peak_memory` measures one owner-provided workload:

- CPU uses PyTorch profiler memory categories.
- CUDA and supported MPS versions use public PyTorch allocator peak statistics.

It does not report process RSS, total device use, driver memory, or third-party allocations. Compare memory only with
matching hardware, PyTorch version, model state, inputs, dtype, optimizer state, allocator warmup, and workload.

## Latency and throughput

TorchScan does not wrap latency measurement. Use PyTorch's
[`torch.utils.benchmark.Timer`](https://docs.pytorch.org/docs/stable/benchmark_utils.html), which already handles warmup,
replicates, and accelerator synchronization. Keep latency results separate from TorchScan's theoretical counts.

## Reproducible reporting

For research or regression analysis, retain:

- The complete report, including `schema_version` and execution context.
- Exact model revision and configuration.
- Input shapes, dtypes, devices, and non-sensitive call structure.
- TorchScan, PyTorch, and Python versions.
- Every diagnostic and custom formula definition.
- Hardware and workload preparation for memory or latency measurements.

Use [Report comparison](report-schema.md#reportdiff) only when both reports use the same schema and compatible
methods.
