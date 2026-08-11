# Examples

These examples keep policy with the model owner and preserve incomplete results instead of inventing values.

## Hugging Face-style keyword inputs

The model can be any callable `torch.nn.Module`; no Transformers dependency is required by TorchScan:

```python
import torch
from torch import nn
from torchscan import crawl_module


class TokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask=None, return_dict=False):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        logits = self.projection(hidden.mean(dim=1))
        return {"logits": logits} if return_dict else logits


model = TokenModel()
input_ids = torch.tensor([[1, 2, 0, 0]])
attention_mask = input_ids.ne(0)

report = crawl_module(
    model,
    args=(input_ids,),
    kwargs={"attention_mask": attention_mask, "return_dict": True},
)
```

Tensor values are used by the model but excluded from report metadata.

## Count one owner-controlled workload

```python
import torch
from torchscan import measure_flops

model = torch.nn.Linear(64, 16)
inputs = torch.randn(32, 64)

report = measure_flops(lambda: model(inputs))
```

The callable is invoked exactly once. It owns inference or training mode, autocast, gradients, and side effects.

## Supply a custom operator formula

PyTorch shape formulas receive shapes rather than live tensor values. This illustrative formula assigns one operation
per sine output; replace it with the convention justified by your experiment:

```python
import math

import torch
from torchscan import measure_flops


def owner_sin_formula(input_shape, *, out_shape):
    return math.prod(out_shape)


inputs = torch.randn(8, 16)
report = measure_flops(
    lambda: torch.sin(inputs),
    custom_mapping={torch.ops.aten.sin: owner_sin_formula},
)
```

Mappings are scoped to the call and do not mutate PyTorch or TorchScan's global formula registries. Check the exact
shape-formula signature for the installed PyTorch release when targeting an operator with additional arguments.

## Compare two reports

```python
from torch import nn
from torchscan import compare_reports, crawl_module

before = crawl_module(nn.Sequential(nn.Linear(16, 8), nn.ReLU()), (16,))
after = crawl_module(nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)), (16,))

diff = compare_reports(before, after)
```

The result contains `schema_version`, metric diffs under `totals`, and `layers.added`, `layers.removed`, and
`layers.changed`. Layer calls match by full path and call index. A metric diff contains `before`, `after`, propagated
`status`, and a numeric `delta` only when both inputs are complete; otherwise `delta` is `None`.

## Enforce an owner-provided memory budget

```python
import os

import torch
from torchscan.process import measure_peak_memory

model = torch.nn.Linear(64, 16)
inputs = torch.randn(32, 64)


def workload() -> None:
    with torch.inference_mode():
        model(inputs)


stats = measure_peak_memory(workload, device=inputs.device)
budget_bytes = int(os.environ["MODEL_MEMORY_BUDGET_BYTES"])

if stats["peak_bytes"] > budget_bytes:
    raise RuntimeError(
        f"PyTorch peak {stats['peak_bytes']} bytes exceeds owner budget {budget_bytes} bytes"
    )
```

TorchScan does not select the threshold or claim target-device readiness from a CPU run.
