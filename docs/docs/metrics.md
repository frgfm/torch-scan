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
- DMAs estimate formula-defined logical element reads and writes, including parameters and intermediates.

These are theoretical counts, not FLOP/s, memory bandwidth, or latency. Diagnostics identify unsupported module work.
Native Transformer MACs are independently derived from matrix dimensions, and their DMAs describe a staged logical
algorithm rather than cache or fused-kernel traffic. See [Native Transformer estimates](transformers.md) for counts,
mask handling, returned attention weights, and supported boundaries.

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

Native attention and Transformer stacks instead provide optional module-local `token_dependencies` records. These
distinguish query/target and key-value/source axes and represent all-token or position-dependent causal dependencies.
Feature-only normalization and feed-forward operations are token-local. These records do not claim graph-wide
effective receptive fields; legacy spatial receptive-field, stride, and padding fields are unavailable. See
[token dependencies](transformers.md#module-local-token-dependencies) for relation semantics and mask limitations.

## Peak memory

`measure_peak_memory` measures one owner-provided workload:

- CPU uses PyTorch profiler memory categories.
- CUDA and supported MPS versions use public PyTorch allocator peak statistics.

It does not report process RSS, total device use, driver memory, or third-party allocations. Compare memory only with
matching hardware, PyTorch version, model state, inputs, dtype, optimizer state, allocator warmup, and workload.
Accelerator statistics are process-global, so unrelated concurrent allocations can affect the result.

## Latency and throughput

The unreleased `measure_latency` API uses
[`torch.utils.benchmark.Timer`](https://docs.pytorch.org/docs/stable/benchmark_utils.html) to measure one
caller-controlled workload. It synchronizes the selected CPU/CUDA/MPS device explicitly, including on older
supported PyTorch versions. It returns JSON-serializable first-call time, warmed latency, variability, and throughput:

Prefer a current PyTorch release for timing. PyTorch 2.1's native benchmark imports require compatible legacy build
dependencies (`setuptools<70`). TorchScan imports these tools only when timing is requested, so this requirement does
not affect the model inspection APIs. Missing benchmark dependencies raise an actionable import error before execution.

```python
import json

import torch
from torchscan import measure_latency

model = torch.nn.Linear(64, 16).eval()
inputs = torch.ones(32, 64)


def workload():
    with torch.inference_mode():
        return model(inputs)


report = measure_latency(
    workload,
    device=inputs.device,
    inputs=inputs,
    work_units=inputs.shape[0],
    work_unit="samples",
    num_threads=1,
)
print(json.dumps(report, indent=2))
```

| `totals` metric | Meaning |
| --- | --- |
| `first_call_latency` | One completed call before warmup, in seconds. Not model loading or fresh-process startup. |
| `latency` | Median seconds per call, computed from warmed block averages. |
| `latency_iqr` | Interquartile range of the same block averages, in seconds. |
| `throughput` | Declared work units completed across timed blocks, divided by their total elapsed seconds. |

`work_units` describes what **one call** completes; `work_unit` gives the unit name. For example, a call processing
32 samples uses `work_units=32, work_unit="samples"`. Tokens must count the work actually completed, not a requested
maximum. The default unit is calls per second. These are local workload measurements, not queued-service throughput.

The callable owns evaluation/training state, gradient mode, precision, transfers, preprocessing, and other side
effects. Only work inside that callable is included. It is invoked once for the first-call measurement, explicitly
warmed up, and then invoked many times by native timer calibration and timed blocks. Training workloads must manage
gradients and changing optimizer state themselves. No weights are downloaded or inputs moved automatically.

`num_threads` defaults to the current PyTorch intra-op thread count and is restored after success or failure. Timing
calls are serialized because this setting is process-global; unrelated work can still affect measurements.
Native-threadpool builds cannot restore a changed count, so TorchScan rejects such overrides before executing the
workload. On these builds, configure threads before constructing the model and omit `num_threads` when measuring.
`warmup`, `min_run_time`, and `min_repeats` control explicit warmup and minimum block measurements. PyTorch performs
additional warmup/calibration, so `min_run_time` is not a wall-clock timeout. Timed blocks use timeit's default garbage
collection behavior; first-call timing uses an ordinary synchronized clock.

Inputs are optional caller-supplied metadata, not arguments forwarded to the callable. The report labels whether they
were supplied and stores shapes/dtypes rather than tensor contents. Hardware, software, thread settings, work units,
timing options, raw block durations, calls per block, and timed call count accompany the metrics. See
[`BenchmarkReport`](torchscan.md#workload-timing).

Keep block-average latency separate from individual request percentiles. First-call time also includes a completion
synchronization that warmed blocks amortize over several calls; its difference from warmed latency is not a pure
startup-cost measurement. Use one device per workload. Multi-device/distributed synchronization, memory measurement,
FLOP counting, profiling, and output-quality checks are separate tasks. `compare_reports` and `render_report` currently
accept model analysis reports, not benchmark reports. Counting can be partial without preventing timing.

## Reproducible reporting

For research or regression analysis, retain:

- The complete report, including `schema_version` and execution context.
- Exact model revision and configuration.
- Input shapes, dtypes, devices, and non-sensitive call structure.
- TorchScan, PyTorch, and Python versions.
- Every diagnostic and custom formula definition.
- Hardware, warmup, allocator, model, gradient, autocast, and optimizer state for memory or latency measurements.

Use [Report comparison](report-schema.md#reportdiff) only when both reports use the same schema and compatible
methods.
