# `torchscan.process`

The process subpackage contains tools for inspecting active Python processes.

::: torchscan.process.get_process_gpu_ram

## Measure one workload's peak memory

`measure_peak_memory` runs an owner-provided callable exactly once. The callable keeps responsibility for the model,
tensors, optimizer, gradient mode, and device placement; TorchScan only measures the requested backend.

::: torchscan.process.measure_peak_memory

### Inference

```python
import os

import torch

from torchscan.process import measure_peak_memory

model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 16))
inputs = torch.randn(32, 64)


def run_inference() -> None:
    with torch.inference_mode():
        model(inputs)


stats = measure_peak_memory(run_inference, device=inputs.device)
owner_approved_budget_bytes = int(os.environ["MODEL_MEMORY_BUDGET_BYTES"])
assert stats["peak_bytes"] <= owner_approved_budget_bytes
```

### Training step

```python
import torch

from torchscan.process import measure_peak_memory

device = torch.device("cpu")  # Use the device where your workload already lives.
model = torch.nn.Linear(64, 16).to(device)
inputs = torch.randn(32, 64, device=device)
targets = torch.randn(32, 16, device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def run_training_step() -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


stats = measure_peak_memory(run_training_step, device=device)
```

### Metrics and limitations

- CPU reports `pytorch_tensor_bytes`, the sum of PyTorch profiler memory categories. It covers PyTorch-tracked CPU
  tensor and operator allocations, not Python heap, process RSS, or arbitrary third-party native allocations.
- CUDA and supported MPS versions report `pytorch_reserved_bytes`, the caching allocator's reserved memory. The result
  also includes `allocated_peak_bytes`, the peak bytes occupied by live tensors.
- `baseline_bytes` is measured before the workload, `peak_bytes` is the backend-specific peak, and `delta_bytes` is
  always their direct difference.
- MPS requires a PyTorch version exposing public resettable peak statistics through `torch.accelerator.memory`; older
  versions raise `NotImplementedError`.

!!! warning

    Accelerator reserved memory is not total device, driver, or process memory. Compare results only on matching
    hardware and workload state, including input shape and dtype, model and gradient modes, optimizer state, and
    allocator warmup. TorchScan does not warm up, repeat, move, or reset the workload. Its lock prevents overlapping
    `measure_peak_memory` calls, but unrelated concurrent PyTorch allocations can still affect process-global metrics.
