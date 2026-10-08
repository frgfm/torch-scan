<p align="center">
  <img src="https://github.com/frgfm/torch-scan/releases/download/v0.1.1/logo_text.png" width="30%">
</p>

<p align="center">
  <a href="https://github.com/frgfm/torch-scan/actions/workflows/package.yml">
    <img alt="CI Status" src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-scan/package.yml?branch=main&label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://codecov.io/gh/frgfm/torch-scan">
    <img src="https://img.shields.io/codecov/c/github/frgfm/torch-scan.svg?logo=codecov&style=flat-square&label=Coverage" alt="Test coverage percentage">
  </a>
  <a href="https://pypi.org/project/torchscan/">
    <img src="https://img.shields.io/pypi/v/torchscan.svg?logo=PyPI&logoColor=fff&style=flat-square&label=PyPI" alt="PyPI version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/torchscan.svg?logo=Python&label=Python&logoColor=fff&style=flat-square" alt="Supported Python versions">
  <a href="https://github.com/frgfm/torch-scan/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/frgfm/torch-scan.svg?label=License&logoColor=fff&style=flat-square" alt="License">
  </a>
</p>

TorchScan helps you **inspect model cost, measure actual resource use, check an optimization, and consume the report**.
Use your own model and inputs. Save JSON or offline HTML; metrics carry a method, scope, and completeness status.

## Measure your workload

The workload timing and offline report APIs require the development version. The current PyPI release is **0.2.0**;
**0.3.0 is proposed and unreleased**. The one-call API is prepared in [PR #176](https://github.com/frgfm/torch-scan/pull/176).
Until it is merged, install its preview branch after installing PyTorch for your hardware:

```shell
python -m pip install "torchscan @ git+https://github.com/frgfm/torch-scan.git@codex/workload-measurement"
```

This short CPU example reuses the model from [the checked comparison example](docs/docs/benchmark-comparison.md).
It initializes weights locally and downloads nothing:

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

One observed run on an AMD EPYC 9V74 CPU, Linux, Python 3.11.16, PyTorch 2.13.0+cpu, TorchScan 0.2.0.dev0 (PR #176), FP32, one PyTorch thread:

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

Your numbers will differ. One call completes 32 samples, so the report uses `samples/s`. Timing repeats the callable;
it owns model state, gradients, precision, transfers, and preprocessing. `inputs` records metadata rather than supplying
arguments. Place your model/tensors and configure threads before measuring. Open `workload.html` in a browser or
consume the JSON directly; terminal text is a view, not a data format.

## How this relates to `summary()`

Use `summary(model, args=(inputs,))` to inspect the same model call.

`summary` prints the familiar module table and returns an `AnalysisReport`: shapes, parameters, storage,
module-formula counts, and separate operator FLOPs for an evaluation forward. It disables gradients temporarily and
restores each module's original training flag. `crawl_module` returns the same report without printing;
`mode="structure"` skips compute estimates, and `strict=True` rejects incomplete requested model metrics.

Workload measurement produces `BenchmarkReport` evidence for the callable you supply. Model storage is not peak
memory, and theoretical FLOPs do not predict latency or process RSS. Keep the two views together with their distinct
methods and boundaries. Pass real `args`/`kwargs` to inspect masks or nested inputs; `input_shape` makes a synthetic
batch of one and excludes the batch dimension.

## Check a change and read the report

Measure a baseline and one controlled change on the same hardware. Use
`compare_benchmarks(before, after, check=...)` with your output tolerance; a failed check preserves evidence and
withholds numeric deltas. Save JSON and offline HTML with `render_report`. Use `compare_reports` for model estimates.
The [checked experiment guide](docs/docs/benchmark-comparison.md) reuses the existing linear, CNN, and Transformer
examples, including fresh-process RSS and separate diagnostic passes.

## Read the limits with the numbers

- `complete`: use `value` with its documented method and scope. A complete zero is meaningful.
- `partial`: `known_value` is a lower bound; preserve diagnostics and never treat missing work as zero.
- `unavailable`: no measurement was produced. Unrequested evidence is also explicit.
- Keep module and operator FLOPs separate. Do not add or average them.
- Warmed latency/IQR describes block averages, not individual-request p95. First-call time excludes imports/loading.
- Process RSS covers a fresh child's whole lifetime on Linux/macOS. CPU tracked-tensor and accelerator allocator
  peaks have different scopes; do not add them together or equate them to total device use.
- Profiler time includes instrumentation and can overlap. MPS profiles describe CPU dispatch, not GPU execution.
- CUDA/MPS claims need real matching hardware. Mocks and skips provide no hardware evidence.
- Passing an output check does not establish task accuracy; IQR labels are descriptive, not statistical significance.

## Install and continue

Stable model inspection: `python -m pip install torchscan==0.2.0`. Requirements are Python ≥3.11,<4 and PyTorch ≥2.1,<3.
See [installation](https://frgfm.github.io/torch-scan/installing.html) for backend selection, development APIs,
compatibility checks, and timing dependencies on PyTorch 2.1.

- [Getting started](https://frgfm.github.io/torch-scan/)
- [Agent quickstart](https://frgfm.github.io/torch-scan/agent-quickstart.html) and the repository
  [agent skill](.agents/skills/torchscan/SKILL.md)
- [Model and input support](https://frgfm.github.io/torch-scan/model-support.html) and
  [custom extensions](https://frgfm.github.io/torch-scan/extensions.html)
- [Report schema](https://frgfm.github.io/torch-scan/report-schema.html),
  [API reference](https://frgfm.github.io/torch-scan/torchscan.html), and
  [changelog](https://frgfm.github.io/torch-scan/changelog.html)
- [v0.2 migration guide](https://frgfm.github.io/torch-scan/migration-v02.html) and [Contributing](CONTRIBUTING.md)

Citation metadata is in [CITATION.cff](CITATION.cff). TorchScan uses the [Apache License 2.0](LICENSE).
