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

## Workload measurements

Use zero-argument callables when the owner needs full control over execution:

```python
import json

import torch
from torchscan import measure_flops
from torchscan.process import measure_peak_memory

inputs = torch.ones(8)
flops = measure_flops(lambda: torch.sin(inputs))
print(json.dumps(flops["total"], indent=2))
print("uncounted operator:", flops["diagnostics"][0]["operator"])

memory = measure_peak_memory(lambda: torch.cos(inputs), device=inputs.device)
print(memory["device"], memory["metric"])
```

`measure_flops` uses PyTorch's operator dispatch. `measure_peak_memory` invokes the workload exactly once and reports
backend-specific PyTorch memory—not process RSS or total device memory.

Here, PyTorch has no built-in `aten.sin` formula, so TorchScan shows a lower bound instead of a false zero:

```text
{
  "status": "partial",
  "value": null,
  "known_value": 0,
  "unit": "FLOPs",
  "scope": "workload",
  "method": "torch.utils.flop_counter.FlopCounterMode"
}
uncounted operator: aten.sin
cpu pytorch_tensor_bytes
```

Peak byte values are intentionally omitted because they depend on the workload, allocator, PyTorch version, and
hardware; the returned mapping includes `baseline_bytes`, `peak_bytes`, and `delta_bytes`.

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

## Trust the status, not only the number

- `complete`: the requested scope was counted; `value` is authoritative for the documented method.
- `partial`: `known_value` is a lower bound and diagnostics identify missing work.
- `unavailable`: TorchScan cannot produce the metric for this execution.

Use `strict=True` when any incomplete analysis must stop automation. See the
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
- [Model and input support](https://frgfm.github.io/torch-scan/model-support.html)
- [v0.2 migration guide](https://frgfm.github.io/torch-scan/migration-v02.html)
- [API reference](https://frgfm.github.io/torch-scan/torchscan.html)

Agents can also load the repository skill at [`.agents/skills/torchscan/SKILL.md`](.agents/skills/torchscan/SKILL.md).

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Contributing and license

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). TorchScan is distributed under the
[Apache License 2.0](LICENSE).
