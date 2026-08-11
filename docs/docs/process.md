# `torchscan.process`

The process subpackage exposes explicit workload memory measurement. Legacy `get_process_gpu_ram` was removed in v0.2
because process snapshots and allocator deltas could not provide a truthful model measurement.

## Measure one workload's peak memory

`measure_peak_memory` runs an owner-provided callable exactly once. The callable owns the model, tensors, optimizer,
gradient mode, device placement, warmup, and cache state.

::: torchscan.process.measure_peak_memory

### Inference

```python
import os

import torch
from torchscan.process import measure_peak_memory

model = torch.nn.Linear(64, 16)
inputs = torch.randn(32, 64)


def run_inference() -> None:
    with torch.inference_mode():
        model(inputs)


stats = measure_peak_memory(run_inference, device=inputs.device)
owner_budget = int(os.environ["MODEL_MEMORY_BUDGET_BYTES"])
if stats["peak_bytes"] > owner_budget:
    raise RuntimeError("Owner-approved PyTorch memory budget exceeded")
```

### Training step

```python
import torch
from torchscan.process import measure_peak_memory

model = torch.nn.Linear(64, 16)
inputs = torch.randn(32, 64)
targets = torch.randn(32, 16)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def run_training_step() -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


stats = measure_peak_memory(run_training_step, device=inputs.device)
```

### Limitations

- CPU `pytorch_tensor_bytes` covers PyTorch-tracked tensor and operator allocations, not Python heap or process RSS.
- Accelerator `pytorch_reserved_bytes` uses PyTorch caching-allocator state and is not total device or driver memory.
- TorchScan does not warm up, repeat, move, reset, or roll back the workload.
- Unrelated concurrent allocations can affect process-global metrics.
- A mock or skipped accelerator test is not hardware evidence.
