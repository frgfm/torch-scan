# Collect once, analyze offline

TorchScan runs the caller's Python workload once. Its C++ collector records operator metadata; Python
applies FLOP formulas after execution. This is the only operator collector, used automatically by
`measure_flops`, `measure_workload`, `crawl_module`, and `summary`.

## Capture and replay

```python
import json
from pathlib import Path
from torchscan import capture_workload, analyze_capture

capture = capture_workload(workload, inputs=inputs, modules=model)
Path("capture.json").write_text(json.dumps(capture), encoding="utf-8")
report = analyze_capture(capture)
```

The workflow is:

1. Python runs the eager model and labels module entry/exit.
2. C++ copies shapes, dtypes/devices and formula arguments at operator boundaries. Composite wrappers
   are opened so their underlying operations are counted once. Ordinary operators never call Python.
3. Collection stops and releases temporary references. Offline formulas compute global, operator and
   inclusive module totals, with explicit diagnostics for incomplete evidence.

Later, another process can load `capture.json` and call `analyze_capture` without the model, native
extension or PyTorch. Install a TorchScan wheel with `--no-deps` for an analysis-only environment.
`measure_flops(workload)` combines collection and replay and retains the `capture` in its report.
Timing, memory and profiling use separate passes.

## Installation and custom formulas

The native extension is built with TorchScan; there is no backend selector or runtime compilation.
Wheels require the PyTorch release they were built against. To use another supported PyTorch version,
[build from source](installing.md#development-version) against that installation. Rebuild after changing PyTorch.

`custom_mapping` remains per call. Shape callbacks receive `out_shape`; callbacks marked `_get_raw = True`
receive live tensors and `out_val`, including nested tensors. Only explicitly overridden operators call
Python during collection. Return a nonnegative whole FLOP count. The capture stores the result or error, so replay needs no callback.
Module estimation callbacks run with collection paused.

## Boundaries and completeness

- Scope is **eager operators on the calling thread and its autograd workers**, not hardware instructions. The method and scope labels
  change from the old Python counter; reports with different labels are incompatible for automatic comparison.
- Captures retain metadata, not tensor contents or activations. Custom callbacks temporarily access tensors
  until their operator returns. Module labels are discovered automatically; pass roots explicitly, or `modules=[]`
  to disable labels. Backward operators inherit labels from their forward autograd sequence when available.
- Defaults retain at most 100,000 events and 16 MiB of metadata payload, excluding container overhead.
  Dropped events, collector errors, unknown operators, unsupported layouts and incomplete arguments make replay partial.
  Known counts remain lower bounds. Sparse/nested arithmetic requires a justified custom formula for completeness.
- Compiled kernels and unrelated threads are outside the scope. Module calls on other threads are detected and
  make the report partial; unlabelled foreign-thread work is unobserved. CUDA/MPS claims require real hardware.
- Module totals are inclusive. Clean latency, CPU tracked tensors, accelerator allocator peaks and process RSS
  retain their separate methods and scopes.

## Measure overhead

With the optional model-test dependencies, run:

```shell
python -m scripts.native_capture_benchmark --model pointwise --output /tmp/native.json
```

Also supports `resnet18` and `bert`, with locally initialized weights. Run the same script with
`PYTHONPATH=/path/to/main-checkout python scripts/native_capture_benchmark.py --output /tmp/main.json`
for the reference counter (use direct script execution, rather than `-m`). JSON records counts, all timing samples,
environment details and capture sizes. Warmed, randomized measurements describe instrumentation overhead,
not an improvement in model inference speed.

On Linux, AMD EPYC 9V45, Python 3.11.16, PyTorch 2.13.0+cpu, two threads and 15 samples:

| Workload | FLOPs, both versions | Main counting (ms) | Native collection + replay (ms) |
| --- | ---: | ---: | ---: |
| 600 pointwise ops, 1,024 elements | 614,400 | 19.07 | 7.45 |
| ResNet18, 4 × 3 × 32 × 32 | 301,391,232 | 13.58 | 6.78 |
| Tiny BERT, 1 layer, hidden 16, 4 × 8 tokens | 172,480 | 4.00 | 1.28 |

Totals and per-operator counts match `main` at `e861252`; counting time falls 50–68% on these workloads.
Grouped convolution backward now respects groups when counting weight gradients; the old upstream formula overcounted them.
These measurements do not establish accelerator performance.
