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
import json

import torch.nn as nn
from torchscan import crawl_module, summary

model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())

# Print the human-readable table and receive the same structured report.
report = summary(model, (3, 32, 32))
json.dumps(report)

# Or collect the report without printing the table.
report = crawl_module(model, (3, 32, 32), strict=True)
```

`input_shape` excludes the batch dimension. For realistic calls—including masks, scalars, `None`, and nested
containers—pass complete `args` and `kwargs` instead:

```python
report = crawl_module(
    model,
    args=(pixel_values,),
    kwargs={"attention_mask": attention_mask, "return_dict": True},
)
```

TorchScan temporarily evaluates the model under inference mode and restores every module's original training state.
It records input metadata, never tensor values.

## Workload measurements

Use zero-argument callables when the owner needs full control over execution:

```python
from torchscan import measure_flops
from torchscan.process import measure_peak_memory

flops = measure_flops(lambda: model(inputs))
memory = measure_peak_memory(lambda: model(inputs), device=inputs.device)
```

`measure_flops` uses PyTorch's operator dispatch. `measure_peak_memory` invokes the workload exactly once and reports
backend-specific PyTorch memory—not process RSS or total device memory.

## Before/after comparison

```python
from torchscan import compare_reports

diff = compare_reports(before, after)
```

`compare_reports` propagates incomplete metrics. It does not store baselines or decide whether a model fits a budget;
the model owner supplies those policies.

## Trust the status, not only the number

- `complete`: the requested scope was counted; `value` is authoritative for the documented method.
- `partial`: `known_value` is a lower bound and diagnostics identify missing work.
- `unavailable`: TorchScan cannot produce the metric for this execution.

Use `strict=True` when incomplete module analysis must stop automation. See the
[report schema](https://frgfm.github.io/torch-scan/report-schema.html) and
[methodology](https://frgfm.github.io/torch-scan/methodology.html) before comparing results.

## Installation

TorchScan v0.2 requires Python 3.11+ and PyTorch 2.1+:

```shell
pip install torchscan
```

Development installation:

```shell
git clone https://github.com/frgfm/torch-scan.git
cd torch-scan
uv venv --python 3.11
uv pip install -e .
```

## Documentation

- [Agent quickstart](https://frgfm.github.io/torch-scan/agent-quickstart.html)
- [Examples](https://frgfm.github.io/torch-scan/examples.html)
- [Model and input support](https://frgfm.github.io/torch-scan/model-support.html)
- [v0.2 migration guide](https://frgfm.github.io/torch-scan/migration-v02.html)
- [API reference](https://frgfm.github.io/torch-scan/torchscan.html)

Agents can also load the repository skill at [`.agents/skills/torchscan/SKILL.md`](.agents/skills/torchscan/SKILL.md).

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Contributing and license

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). TorchScan is distributed under the
[Apache License 2.0](LICENSE).
