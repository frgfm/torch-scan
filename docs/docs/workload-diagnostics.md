# Workload memory and bottlenecks

These APIs require the development version.

## Measure a workload in one call

[`measure_workload`](torchscan.md#torchscan.measure_workload) collects FLOPs, latency, throughput, and PyTorch peak
memory. It prints ms, samples/s, and MiB and returns a `BenchmarkReport` for `json.dumps` and `render_report`.
See the [runnable example](index.md#start-with-your-workload).

Set `work_units` to samples per call; batch size is never inferred. Use `work_unit="tokens"` for tokens/s,
`metrics=("latency", "throughput")` for timing only, or `print_summary=False` for quiet collection.
Unrequested metrics are unavailable with method `not_requested`. Collector limitations retain diagnostics;
partial FLOPs retain a known lower bound. Workload errors propagate.

Timing runs first; FLOPs, memory, and optional `profile=True` each invoke the same callable once more.
All passes share caller state. You control evaluation mode, placement, precision, gradients, and threads.
Use a repeatable callable and keep threads fixed. Latency describes block averages, not request percentiles;
first-call time includes all callable work. Use the individual collectors for custom formulas or trace export.

RSS requires `rss_command=[sys.executable, "my_workload.py"]`. That fresh command owns its configuration and must
exit; no deadline is added. RSS covers its whole lifetime, including imports and loading. It has a different scope
from backend-specific PyTorch peak memory. The command's settings cannot be inferred from the callable.

## Whole-process peak RSS

Run a model script in a new process so a previous trial cannot supply its high-water mark:

```python
import sys
from torchscan.process import measure_peak_rss

rss = measure_peak_rss([sys.executable, "my_workload.py"])
print(rss["value"] / 1024**2, "MiB peak resident RAM")
```

The command executes without a shell and owns model construction, inputs, and device placement. Its standard streams
are inherited; failures propagate. Linux/macOS provide per-child OS accounting. Other platforms raise
`NotImplementedError` before launching a command. RSS includes Python/PyTorch imports, model loading, and every phase
inside the child. It is not an inference-only delta, a summed process-tree peak, or accelerator memory.
The command must exit; TorchScan adds no deadline.

For a benchmark report from the same child/configuration, the caller can append the returned metric under
`report["totals"]["process_peak_rss"]`. Retain the model revision and workload configuration with that report.

Keep existing `measure_peak_memory` results separate: CPU tensor bytes and accelerator allocated/reserved bytes have
different scopes. CUDA/MPS allocator peaks are not process RSS or total device memory; Apple unified memory can overlap.

See the [`measure_peak_rss`](process.md#torchscan.process.measure_peak_rss) API reference.

## Find expensive operations

```python
import torch
from torchscan import profile_workload

inputs = torch.ones(32, 64)
weights = torch.ones(64, 64)
report = profile_workload(lambda: inputs @ weights, device="cpu", trace_path="profile.json")
for row in report["operators"][:5]:
    print(row["operator"], row["cpu_self_seconds"], row["cpu_net_bytes"])
```

The callable executes once. Rows group operators by input shape and list call count, CPU/device self time in seconds,
and CPU net allocation bytes. Rows are ranked by available CUDA self time or CPU self time. Net allocations can be
negative and are not peaks. Self times may overlap and cannot be summed into model latency. Trace export is optional;
the output belongs to the experiment and can contain execution details and input shapes.
Rows also retain runtime events and caller annotations, which can expose copy or dispatch overhead.

MPS rows describe CPU dispatch only and carry an explicit diagnostic; use `torch.mps.profiler` with Instruments for
GPU traces. CUDA time is unavailable when the installed profiler lacks CUDA activity support. Missing GPU time is
represented by `None`, never a claimed zero. The context records row limits and the total grouped operator count.

Use the profile to choose an experiment, then remeasure it without profiling and check output quality. A profiler
hotspot is evidence about that instrumented call, not a promised optimization gain.

See the [`profile_workload`](torchscan.md#torchscan.profile_workload) and
[`ProfileReport`](torchscan.md#torchscan.ProfileReport) API reference.
