# Inspect cost. Measure your workload. Check the change.

TorchScan helps you understand a PyTorch model's cost, measure actual resource use on your hardware, check an
optimization, and save a report you can inspect or automate. Every metric carries its status, method, and scope.
Unsupported work stays visible as `partial` or `unavailable`.

## Start with your workload

The one-call measurement API is prepared in [PR #176](https://github.com/frgfm/torch-scan/pull/176).
Until it is merged, install this development preview:

```shell
python -m pip install "torchscan @ git+https://github.com/frgfm/torch-scan.git@codex/workload-measurement"
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

One observed run (AMD EPYC 9V74 CPU, Linux, Python 3.11.16, PyTorch 2.13.0+cpu, TorchScan 0.2.0.dev0 (PR #176), FP32, one PyTorch thread):

```text
Workload on cpu (1 PyTorch threads)
  First call: 0.449 ms
  Latency (block median): 0.018 ms
  Latency IQR: 0.000 ms
  Throughput: 1,734,801.967 samples/s
  Operator FLOPs: partial (known lower bound: 1,048,576.000 FLOPs)
  PyTorch peak memory: 0.094 MiB
    Scope: PyTorch tracked CPU tensors
  Process peak RSS: unavailable (not requested)
  Profiler: not requested (separate instrumented pass)
  flops: aten.linear was observed 1 time(s), but no FLOP formula is registered.
```

Read `workload.html` in a browser or load the JSON directly. Shapes/dtypes are retained as metadata, not tensor values.
`work_units=32` says one invocation completes 32 samples; tokens or other units must count actual completed work.
Numbers depend on your machine, inputs, software, precision, and thread settings.

## How this relates to `summary()`

`summary(model, args=(inputs,))` prints a module table and returns an `AnalysisReport`: parameters, storage, shapes,
module-formula counts, and a separate operator FLOP view for an evaluation forward. It disables gradients temporarily
and restores each module's original training flag. Use `mode="structure"` for shapes and parameters alone, and
`strict=True` when incomplete requested model metrics must stop automation.

Workload collectors measure the callable you supply and return `BenchmarkReport` evidence. You own model state,
gradient mode, precision, inputs, device placement, preprocessing, and transfers. Timing repeats the callable;
training steps must manage changing state themselves. Configure threads before model construction.

Model storage is not peak memory, and theoretical FLOPs do not predict latency or RSS. Keep model inspection and
workload measurements together, with their distinct methods and boundaries. Never add module and operator FLOPs.

## Choose the next action

| Outcome | Use |
| --- | --- |
| Inspect model cost | `summary` for a table; `crawl_module` for JSON only. Use real `args`/`kwargs` for masks or nested inputs. |
| Measure actual resource use | Workload timing, separate operator FLOPs and scoped memory, explicit fresh-process RSS, optional profiler evidence. |
| Check one optimization | `compare_benchmarks(before, after, check=...)`; your callback defines output tolerance. Use `compare_reports` for model estimates. |
| Consume the evidence | Serialize the mapping as JSON; `render_report` creates offline HTML. Model reports also support SVG. |

Use `value` only for `complete` metrics. A `partial` metric exposes `known_value` as a lower bound; preserve its
diagnostics. An `unavailable` result is missing evidence, not zero. Do not scrape terminal text.

Clean warmed latency/IQR describes block averages, not request p95. First-call timing excludes imports and loading.
Process RSS includes the fresh child's whole lifetime. CPU tensor peaks and accelerator allocator peaks cover
separate scopes; do not sum them with RSS. Profiler timings include instrumentation. A passing output check is not
task accuracy, and IQR labels do not establish statistical significance. CUDA/MPS claims require real matching hardware.

## Continue with your model

- [Installation](installing.md): stable 0.2.0, development features, and supported compatibility checks.
- [Checked performance experiments](benchmark-comparison.md): run a baseline and controlled change, check outputs, save evidence.
- [Workload memory and bottlenecks](workload-diagnostics.md): fresh-process RSS and separate profiler passes.
- [Agent quickstart](agent-quickstart.md): automate the same workflow using metric status and owner-supplied budgets.
- [Model and input support](model-support.md), [Understanding results](metrics.md), and [API reference](torchscan.md).
