# TorchScan: truthful PyTorch model analysis

TorchScan produces human-readable summaries and JSON-serializable reports for PyTorch models. Reports include model
structure, parameters, input metadata, module estimates, and operator FLOPs. Each metric is explicitly marked
`complete`, `partial`, or `unavailable`.

## 60-second quickstart

Install the development version documented here:

```shell
pip install git+https://github.com/frgfm/torch-scan.git
```

Inspect a model on CPU:

```python
import json

import torch.nn as nn
from torchscan import crawl_module, summary

model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())

report = summary(model, (3, 32, 32))
json.dumps(report)

# Fail instead if any requested module metric is incomplete.
strict_report = crawl_module(model, (3, 32, 32), strict=True)
```

`input_shape` excludes the batch dimension. TorchScan creates a synthetic batch of one, temporarily switches all
modules to evaluation mode, runs with gradients disabled, and restores each module's original training state.

## Use real calls when shape is not enough

Pass complete positional and keyword arguments for realistic inputs:

```python
report = crawl_module(
    model,
    args=(input_ids,),
    kwargs={"attention_mask": attention_mask, "return_dict": True},
)
```

TorchScan forwards values unchanged and records only recursive metadata. Tensor values and local paths are never
stored in the report.

## Choose the right entry point

| Need | API |
| --- | --- |
| Printable model table plus structured result | `summary(...)` |
| Structured module report only | `crawl_module(...)` |
| Operator FLOPs for an arbitrary forward or training workload | `measure_flops(workload)` |
| Peak PyTorch memory for one owner-controlled workload | `measure_peak_memory(workload, device=...)` |
| Pure before/after report comparison | `compare_reports(before, after)` |

## Read results safely

Do not consume a number without checking its status:

- `complete` means the documented method covered the requested scope.
- `partial` exposes only a known lower bound and diagnostics for uncounted work.
- `unavailable` means the method could not produce the metric for this execution.

These are theoretical measurements for one execution. They are not latency, throughput, process RSS, energy use, or
proof that a model fits a target device.

## Next steps

- Automating inspection? Start with the [Agent quickstart](agent-quickstart.md).
- Supplying masks, scalars, or nested inputs? Read [Model and input support](model-support.md).
- Comparing research results? Read [Methodology](methodology.md) and [Understanding results](metrics.md).
- Upgrading from 0.1? Follow the [v0.2 migration guide](migration-v02.md).
- Need exact fields? Use the [Report schema](report-schema.md) and [API reference](torchscan.md).
