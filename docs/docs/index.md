# Inspect cost. Measure your workload. Check the change.

TorchScan helps you understand a PyTorch model's cost, measure actual resource use on your hardware, check an
optimization, and save a report you can inspect or automate. Every metric carries its status, method, and scope.
Unsupported work stays visible as `partial` or `unavailable`.

## Start with your workload

The one-call measurement API is on `main` and unreleased. Install the development version:

```shell
python -m pip install "torchscan @ git+https://github.com/frgfm/torch-scan.git@main"
```

Use a callable that performs the same work as your application. This small CPU example uses the locally initialized
linear model from the [checked comparison example](benchmark-comparison.md); it downloads no weights:

```python
import json
from pathlib import Path

import torch
from torchscan import measure_workload, render_report

torch.set_num_threads(1)
torch.manual_seed(0)
model = torch.nn.Linear(128, 128).eval()
inputs = torch.randn(32, 128)


@torch.inference_mode()
def workload():
    return model(inputs)


report = measure_workload(workload, device="cpu", inputs=inputs, work_units=32, work_unit="samples")
Path("workload.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
Path("workload.html").write_text(render_report(report), encoding="utf-8")
```

One observed run (Apple M3 Pro CPU, macOS 15.7, Python 3.11.13, PyTorch 2.14.1, TorchScan `main`, FP32, one PyTorch thread):

```text
Workload on cpu (1 PyTorch threads)
  First call: 0.174 ms
  Latency (block median): 0.006 ms
  Latency IQR: 0.001 ms
  Throughput: 4,191,866.300 samples/s
  Operator FLOPs: 1,048,576.000 FLOPs
  PyTorch peak memory: 0.094 MiB
    Scope: PyTorch tracked CPU tensors
  Process peak RSS: unavailable (not requested)
  Profiler: not requested (separate instrumented pass)
```

Open `workload.html` in a browser or load the JSON; never scrape terminal text. `work_units=32` declares the samples
completed by one call. `inputs` records metadata, not tensor values or arguments forwarded to the callable.
Your numbers depend on hardware, software, inputs, precision, and threads.

## Inspect cost and check a change

`summary(model, args=(inputs,))` prints a table and returns an `AnalysisReport` for an evaluation forward: shapes,
parameters, storage, module estimates, and separate operator FLOPs. It disables gradients temporarily and restores
training flags. `crawl_module` returns the same report quietly; `mode="structure"` skips compute estimates.
Model storage is not peak memory, and theoretical FLOPs do not predict latency or RSS. Never add module/operator FLOPs.

`measure_workload` returns a `BenchmarkReport` for your callable. You own model state, precision, gradients, placement,
transfers, and preprocessing. Timing repeats it; diagnostic passes share its state. Configure threads before building
models and manage changing training state yourself. Use `metrics` to select passes and `print_summary=False` for automation.

Use `value` only for `complete` metrics, `known_value` as a lower bound for `partial`, and preserve diagnostics.
`unavailable` is missing evidence, not zero. `strict=True` rejects incomplete requested **model** metrics.

Follow [checked experiments](benchmark-comparison.md) to compare a baseline with one controlled change using
`compare_benchmarks(..., check=...)`. Failed output checks withhold deltas; passing checks do not establish task accuracy.
Use `compare_reports` for model estimates. Serialize the returned mappings and render saved evidence without rerunning it.

Warmed latency/IQR describes block averages, not request p95; IQR labels are descriptive. First-call time includes
all callable work. RSS covers a fresh child's lifetime; CPU tensor and accelerator allocator peaks have separate scopes
and must not be summed with RSS. Profiler times include instrumentation. CUDA/MPS claims require real matching hardware.
See [measurement boundaries](workload-diagnostics.md) and [metric meanings](metrics.md).

Continue with [installation](installing.md), [agent quickstart](agent-quickstart.md),
[model/input support](model-support.md), [report schema](report-schema.md), and [API reference](torchscan.md).
