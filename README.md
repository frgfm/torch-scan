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

TorchScan inspects a PyTorch model and returns a JSON-serializable report of its structure, parameters, inputs,
module estimates, and operator FLOPs. Every metric says whether it is complete, partial, or unavailable, so an
unsupported operation cannot masquerade as zero.

## Quickstart

```python
import torch.nn as nn
from torchscan import crawl_module, summary

model = nn.Conv2d(3, 8, 3)

# Print the human-readable table and receive the same structured report.
report = summary(model, (3, 32, 32))

# Or collect the report without printing the table.
report = crawl_module(model, (3, 32, 32), strict=True)
```

`summary` keeps the familiar terminal UX while returning the structured report:

```text
__________________________________________________________
Layer     Type      Output Shape      Param #    Trainable
==========================================================
conv2d    Conv2d    (1, 8, 30, 30)    224        True
==========================================================
Trainable params: 224
Non-trainable params: 0
Total params: 224
----------------------------------------------------------
Model size (params + buffers): 0.00 Mb
----------------------------------------------------------
Module-formula forward FLOPs: 388.80 kFLOPs
Multiply-Accumulations: 194.40 kMACs
Direct memory accesses: 201.82 kDMAs
Operator forward FLOPs: 388.80 kFLOPs
__________________________________________________________
```

`input_shape` excludes the batch dimension. For realistic calls—including masks, scalars, `None`, and nested
containers—pass complete `args` and `kwargs` instead:

```python
import json

import torch
from torch import nn
from torchscan import crawl_module


class MaskedModel(nn.Module):
    def forward(self, input_ids, *, attention_mask):
        return input_ids * attention_mask


transformer_model = MaskedModel()
input_ids = torch.ones(1, 4)
attention_mask = torch.tensor([[True, True, False, False]])
report = crawl_module(
    transformer_model,
    args=(input_ids,),
    kwargs={"attention_mask": attention_mask},
)
print(json.dumps(report["inputs"]["kwargs"]["attention_mask"], indent=2))
```

Only metadata is retained:

```json
{
  "kind": "tensor",
  "shape": [1, 4],
  "dtype": "torch.bool",
  "device": "cpu",
  "requires_grad": false
}
```

TorchScan temporarily evaluates the model with gradients disabled and restores every module's original training
state. It records input metadata, never tensor values.

For shapes and parameter counts, skip compute analysis with one option:

```python
report = summary(model, (3, 32, 32), mode="structure")
```

Structure mode collects the same hierarchy, calls, input/output metadata, parameters, and buffers without FLOP
dispatch or module formulas. Unrequested compute totals have `status="unavailable"` and `method="not_requested"`.
`strict=True` checks the requested metrics. Full analysis remains the default. Both modes release intermediate
activations as execution progresses.

Add your own module/model estimates with
`custom_modules={YourModule: ModuleHandler(your_callback)}` on `crawl_module` or `summary`. Callbacks receive the
complete actual call and supply FLOPs, MACs, DMAs, or receptive-field fields independently. Explicit subtree ownership
prevents inclusive parent estimates from double-counting children. Both APIs also accept `custom_mapping` for separate
operator FLOP overrides. Registrations belong to one analysis and require no TorchScan dependency on your model library.
See the [copyable extension tutorial](docs/docs/extensions.md), including a complex-valued example and counting conventions.

## Workload measurements

Measure a workload in one call with the development version:

```python
import torch
from torchscan import measure_workload

model = torch.nn.Linear(64, 16).eval()
inputs = torch.ones(8, 64)


def workload():
    with torch.inference_mode():
        return model(inputs)


workload_report = measure_workload(workload, device="cpu", inputs=inputs, work_units=8)
```

Get FLOPs, latency in ms, throughput in samples/s, and PyTorch peak memory in MiB. Set `work_units` to samples per
call. Select measurements with `metrics=("latency", "throughput")`, or add `profile=True` for operator evidence.
The report works with `json.dumps` and `render_report`; incomplete values stay explicit. You control model state,
placement, precision, gradients, and threads. See the [workload guide](docs/docs/workload-diagnostics.md) for pass
order and fresh-process RSS through an explicit `rss_command`.

The individual FLOP, timing, memory, and profiler collectors remain available in the [API reference](docs/docs/torchscan.md).

## Before/after comparison

```python
import torch.nn as nn
from torchscan import compare_reports, crawl_module

before = crawl_module(nn.Conv2d(3, 8, 3), (3, 32, 32))
after = crawl_module(nn.Conv2d(3, 12, 3), (3, 32, 32))
diff = compare_reports(before, after)
parameters = diff["totals"]["parameters"]
print(parameters["status"], parameters["delta"])
```

```text
complete 112
```

`compare_reports` propagates incomplete metrics. It does not store baselines or decide whether a model fits a budget;
the model owner supplies those policies.

## Offline visual report

```python
from pathlib import Path
import webbrowser
from torchscan import render_report

path = Path("torchscan-report.html").resolve()
path.write_text(render_report(report), encoding="utf-8")
webbrowser.open(path.as_uri())

# A static SVG, or an HTML comparison using compare_reports internally:
Path("torchscan-report.svg").write_text(render_report(report, format="svg"), encoding="utf-8")
Path("comparison.html").write_text(render_report(after, before=before), encoding="utf-8")
```

HTML opens a module cost explorer: nested rectangles show the hierarchy and the concentration of recorded compute or
first-attributed parameters. Select a module for tensor shapes, repeated-call evidence, methods, and diagnostics.
An unscaled rail keeps unknown work and tiny/zero contributions visible; comparisons share one hierarchy and scale.
SVG exports the same visual composition. Reports work offline with no server, CDN, or extra dependencies.
See [the guide](docs/docs/visual-report.md) for interpretation, comparison rules, and keyboard controls.
Generate the [local-model examples](examples/visual-report/README.md) to explore incomplete work and a channel comparison.

## Trust the status, not only the number

- `complete`: the requested scope was counted; `value` is authoritative for the documented method.
- `partial`: `known_value` is a lower bound and diagnostics identify missing work.
- `unavailable`: TorchScan cannot produce the metric for this execution.

Use `strict=True` when any incomplete analysis must stop automation. See the
[report schema](https://frgfm.github.io/torch-scan/report-schema.html) and
[methodology](https://frgfm.github.io/torch-scan/methodology.html) before comparing results.

## Installation

The stable release is **v0.2.0**. It requires Python ≥3.11,<4 and PyTorch ≥2.1,<3:

```shell
pip install torchscan
```

The unreleased version on `main` adds `render_report`, the `custom_modules` extension API, `custom_mapping` on
`crawl_module` and `summary`, and native Transformer MAC, DMA, and token-dependency estimates.
Install `main` to use these features:

```shell
pip install git+https://github.com/frgfm/torch-scan.git
```

See the [installation guide](https://frgfm.github.io/torch-scan/installing.html) and
[v0.2 migration guide](https://frgfm.github.io/torch-scan/migration-v02.html).
For a local development checkout, follow [Contributing](CONTRIBUTING.md).

## Documentation

- [Agent quickstart](https://frgfm.github.io/torch-scan/agent-quickstart.html)
- [Model and input support](https://frgfm.github.io/torch-scan/model-support.html)
- [Custom module extensions](https://frgfm.github.io/torch-scan/extensions.html)
- [v0.2 migration guide](https://frgfm.github.io/torch-scan/migration-v02.html)
- [API reference](https://frgfm.github.io/torch-scan/torchscan.html)

Agents can also load the repository skill at [`.agents/skills/torchscan/SKILL.md`](.agents/skills/torchscan/SKILL.md).

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Contributing and license

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). TorchScan is distributed under the
[Apache License 2.0](LICENSE).
