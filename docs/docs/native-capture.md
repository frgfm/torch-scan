# Collect once, count after Python exits

Python runs the model. C++ captures operator metadata and calculates operator FLOPs.
`measure_flops`, `measure_workload`, `crawl_module`, and `summary` use this automatically.
The counting engine is shared by the Python API and a native executable; there is no Python counter.

## Two-process workflow

Save the capture in `collect.py`:

```python
import json
from pathlib import Path

import torch
from torchscan import capture_workload

model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU()).eval()
inputs = torch.ones(2, 4)
with torch.inference_mode():
    capture = capture_workload(lambda: model(inputs), inputs=inputs, modules=model)
Path("capture.json").write_text(json.dumps(capture), encoding="utf-8")
```

Then run:

```shell
python collect.py && torchscan-count capture.json -o flops.json
```

1. Python runs the eager model and labels module entry/exit. C++ records shapes, dtypes/devices and
   formula arguments at operator boundaries. Known composite wrappers are opened to count their operations once.
2. Collection stops and releases temporary references. `collect.py` saves metadata and exits, releasing
   Python, PyTorch and the model before counting starts.
3. `torchscan-count` reads the JSON and computes global, per-operator and inclusive module counts in C++.
   It neither runs the model nor loads Python or PyTorch. The example produces **78 FLOPs**.

The executable writes a standard `FlopReport` to stdout, or to `-o`. Exit code **0** means a complete count;
**2** means a partial count or invalid input. Partial reports retain known counts and diagnostics.
A failed collection stops the shell command before counting. Saved custom results are replayed without callbacks.

For an existing Python process, `analyze_capture(capture)` calls the same C++ engine without PyTorch.
`measure_flops(workload)` combines collection and counting in that process and retains the capture in its report;
it does not exit the caller. Timing, memory and profiling use separate passes. Module estimates remain a separate view.

## Installation and custom formulas

The collector, counting library and `torchscan-count` executable are built with TorchScan; there is no backend selector or runtime compilation.
Wheels require the PyTorch release they were built against. To use another supported PyTorch version,
[build from source](installing.md#development-version) against that installation. Rebuild after changing PyTorch. Source builds require a C++ compiler compatible with the installed PyTorch; wheels need no compiler.

To build only the standalone counter, install CMake ≥3.19 and a C++17 compiler, then run:

```shell
cmake -S native/count -B /tmp/torchscan-count-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/torchscan-count-build --config Release
```

This build verifies and prepares the same pinned upstream headers, using `TORCHSCAN_DEPS_CACHE` when set.
It uses neither Python nor PyTorch. The executable is in the build directory (`Release/torchscan-count.exe` on Windows).

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

## Measured cost

Linux, AMD EPYC 9V45, Python 3.11.16, PyTorch 2.13.0+cpu, two threads, locally initialized models.
Against the previous C++ capture + Python counting implementation (`d2b03d4`):

| Workload | FLOPs, both versions | Full pass, Python counting | Full pass, C++ counting |
| --- | ---: | ---: | ---: |
| 600 pointwise ops, 1,024 elements | 614,400 | 7.44 ms | 5.58 ms |
| ResNet18, 4 × 3 × 32 × 32 | 301,391,232 | 6.83 ms | 6.36 ms |
| Tiny BERT, 1 layer, hidden 16, 4 × 8 tokens | 172,480 | 1.31 ms | 1.21 ms |

The full pass includes model execution, collection, metadata transfer and counting. Medians of five fresh
processes, each with five warmups and 31 timing samples, pinned to CPUs 0–1; process jobs were randomized.
Peak process RSS stays approximately unchanged: pointwise 193→193 MiB, ResNet18 330→330 MiB, BERT 311→311 MiB.
These are whole-process lifetime peaks, including imports, model/inputs and repeated measurements.

Counting saved JSON after Python exits uses much less memory:

| Saved workload | Python time / peak RSS | C++ executable time / peak RSS |
| --- | ---: | ---: |
| Pointwise | 35.98 ms / 18.09 MiB | 3.95 ms / 3.80 MiB |
| ResNet18 | 31.71 ms / 16.25 MiB | 1.84 ms / 2.55 MiB |
| Tiny BERT | 31.65 ms / 16.00 MiB | 1.49 ms / 2.30 MiB |

Medians of nine randomized fresh-process runs, including startup, JSON input and full report output,
with cached files and CPU affinity 0–1. Peak RSS is each counting process's lifetime high-water mark, measured
with a native `wait4` supervisor; neither saved-capture counter imports PyTorch. All three full reports match exactly.
Totals and per-operator counts also match `main` at `e861252`; grouped convolution backward corrects the old formula's overcount.
This reduces memory **after collection**, not the peak needed to run the model. Clean inference is unchanged;
these CPU measurements do not establish accelerator performance.

With the optional model-test dependencies, collect your own timing samples and capture:

```shell
python -m scripts.native_capture_benchmark --model pointwise --output /tmp/timing.json --capture-output /tmp/capture.json
/usr/bin/time -v torchscan-count /tmp/capture.json -o /tmp/flops.json
```

Also supports `resnet18` and `bert` without downloading weights. Run the same script against `d2b03d4` without
`--capture-output` to compare the full pass. The benchmark records counts, timing samples and environment details;
Linux `time -v` records counting-process RSS.
