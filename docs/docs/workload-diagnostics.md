# Workload memory and bottlenecks

These APIs require the development version. Keep diagnostic runs separate from clean
[`measure_latency`](metrics.md#latency-and-throughput) measurements.

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

For a benchmark report from the same child/configuration, the caller can append the returned metric under
`report["totals"]["process_peak_rss"]`. Retain the model revision and workload configuration with that report.

Keep existing `measure_peak_memory` results separate: CPU tensor bytes and accelerator allocated/reserved bytes have
different scopes. CUDA/MPS allocator peaks are not process RSS or total device memory; Apple unified memory can overlap.

::: torchscan.process.measure_peak_rss

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

MPS rows describe CPU dispatch only and carry an explicit diagnostic; use `torch.mps.profiler` with Instruments for
GPU traces. CUDA time is unavailable when the installed profiler lacks CUDA activity support. Missing GPU time is
represented by `None`, never a claimed zero. The context records row limits and the total grouped operator count.

Use the profile to choose an experiment, then remeasure it without profiling and check output quality. A profiler
hotspot is evidence about that instrumented call, not a promised optimization gain.

::: torchscan.profile_workload

::: torchscan.ProfileReport
