# Workload memory and bottlenecks

These APIs require the development version.

## Measure a workload in one call

`measure_workload` collects FLOPs, latency, throughput, and PyTorch peak memory. It prints a short summary and returns
the same evidence as a `BenchmarkReport`. Choose which measurements to run with `metrics`.

```python
import json
from pathlib import Path

import torch
from torchscan import measure_workload, render_report

model = torch.nn.Linear(64, 16).eval()
inputs = torch.ones(8, 64)


def workload():
    with torch.inference_mode():
        return model(inputs)


report = measure_workload(workload, device="cpu", inputs=inputs, work_units=8)
Path("workload.json").write_text(json.dumps(report))
Path("workload.html").write_text(render_report(report))
```

The summary uses milliseconds, samples/s, and MiB. Set `work_units` to the number of samples per call. TorchScan does
not guess the batch size. Set `work_unit="tokens"` for tokens/s. Set `print_summary=False` to return evidence quietly.

For timing only, use `metrics=("latency", "throughput")`. For FLOPs and memory only, use
`metrics=("flops", "memory")`. Unrequested metrics have `status="unavailable"` and `method="not_requested"`.
Unsupported collector results also stay unavailable and have a diagnostic. Partial FLOPs retain a known lower bound
and the uncounted operator names. Workload errors propagate; they are not converted into missing measurements.

Timing runs first. FLOPs and memory each invoke the callable once more. Add `profile=True` for one further operator
pass; use `trace_path="profile.json"` to save its trace. Profiler times are kept separate from clean latency.
All passes share caller state. Use a repeatable callable and manage gradients, caches, and random state yourself.
TorchScan does not set evaluation mode, move tensors, select precision, or choose thread counts. Configure threads
before building the model and keep them fixed. Latency is a median of block averages, not a request percentile.
First-call time excludes imports and model loading. PyTorch peak memory retains its backend-specific scope.

RSS is collected only when you supply a fresh-process command:

```python
import sys

report = measure_workload(
    workload,
    device="cpu",
    inputs=inputs,
    work_units=8,
    rss_command=[sys.executable, "my_workload.py"],
)
```

That script must use the configuration you want to measure. Its RSS covers its whole process lifetime. The command
must exit; TorchScan adds no deadline. RSS and PyTorch memory have different scopes. The separate command's inputs
and settings cannot be inferred from the in-process callable.

See the [`measure_workload`](torchscan.md#torchscan.measure_workload) API reference. The individual collectors remain
available when you need a single measurement.

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
