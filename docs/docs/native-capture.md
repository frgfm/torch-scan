# Collect once, analyze offline

The optional C++ collector records eager PyTorch operator metadata while the original Python workload runs.
`analyze_capture` applies counting formulas afterwards, including in a process without PyTorch.
The Python dispatch counter remains the default.

## Install the collector

In a checkout with PyTorch installed, install `setuptools` and `wheel`, then build the optional package:

```shell
uv pip install "setuptools<84" wheel
uv pip install --no-build-isolation --no-cache --no-deps ./native
```

A C++ compiler is required for this explicit build; imports never compile code. Rebuild after changing PyTorch.
The native wheel records the exact PyTorch release it was built against. CI builds and tests minimum/current
PyTorch wheels on Linux, macOS, and Windows; ordinary TorchScan installs remain Python-only.
Setuptools 84 removed the Windows compiler hook used by PyTorch's build helper.
PyTorch 2.1 benchmark utilities additionally need `setuptools<70` and `numpy<2`.

## Capture and replay

```python
import json
from pathlib import Path
from torchscan import capture_workload

capture = capture_workload(workload, inputs=inputs, modules=model)
Path("capture.json").write_text(json.dumps(capture), encoding="utf-8")
```

In a separate process, no model, inputs, native extension, or PyTorch is needed:

```python
import json
from pathlib import Path
from torchscan import analyze_capture

capture = json.loads(Path("capture.json").read_text(encoding="utf-8"))
report = analyze_capture(capture)  # Standard FlopReport: counts, completeness, and diagnostics.
```

An analysis-only environment can install the main wheel with `--no-deps`. Package exports load lazily;
requesting collection/model APIs still requires PyTorch. For the existing workload report API, use
`measure_workload(..., flops_backend="native")` or `measure_flops(..., backend="native")`.
The FLOP report retains its raw `capture`; timing, memory, and profiling remain separate measurement passes.

## Boundaries and completeness

- Native capture observes **outermost eager operators on the calling thread**, not hardware instructions.
  Internal calls belong to their outer operator and are not counted twice. Its scope/method differ from Python
  dispatch reports; compatible numerical totals do not make the observation boundaries interchangeable.
- Only shapes, dtypes/devices, relevant scalar arguments, and optional module labels are copied. Tensor contents,
  tensor objects, and Python outputs are not retained. Python hooks run at module boundaries when labels are requested.
- Defaults retain at most 100,000 events and 16 MiB of metadata payload. Limits are configurable. Lost events and
  collector errors make replay partial; unknown operators, layouts/subclasses, or missing metadata retain diagnostics.
- Compiled kernels and operations on other threads are outside the declared scope. Live custom operator callbacks
  keep using the Python backend. Sparse/nested inputs and unsupported arithmetic never become fabricated zero counts.
  Module labels detect calls on other threads and make the result partial; unlabelled foreign-thread work is unobserved.
- Module totals are inclusive. Clean latency is measured without the collector; CPU tracked tensors, accelerator
  allocated/reserved peaks, and process RSS retain their existing, distinct scopes.

## Reproduce the overhead comparison

Use the existing optional model-test dependencies; models use locally initialized weights:

```shell
python -m scripts.native_capture_benchmark --model pointwise --output /tmp/native-pointwise.json
python -m scripts.native_capture_benchmark --model resnet18 --output /tmp/native-resnet18.json
python -m scripts.native_capture_benchmark --model bert --output /tmp/native-bert.json
```

The script checks count parity, warms each path, and randomizes measurement order. JSON contains all samples,
environment details, capture sizes, and separate medians for clean execution, Python counting, native capture,
offline replay, and native capture plus replay. Instrumented durations describe collection overhead, not clean model latency.

Measured on Linux, AMD EPYC 9V74, Python 3.11.16, PyTorch 2.13.0+cpu, two PyTorch threads, 15 samples per path:

| Workload | Clean (ms) | Python counter (ms) | Native capture (ms) | Replay (ms) | Native total (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 600 pointwise ops, 1,024 elements | 1.59 | 34.78 | 5.21 | 6.32 | 11.67 |
| ResNet18, 4 × 3 × 32 × 32 | 9.24 | 27.05 | 12.68 | 1.64 | 13.76 |
| Tiny BERT, 1 layer, hidden 16, 4 × 8 tokens | 0.58 | 7.67 | 1.46 | 0.55 | 2.12 |

FLOP counts match the Python counter in all three workloads: 614,400; 301,391,232; and 172,480 respectively.
These CPU measurements do not establish accelerator performance. Separately, fresh-process API import medians
(five runs) were 25 ms / 15.6 MiB peak process RSS for `analyze_capture`, versus 857 ms / 186.7 MiB for
`measure_flops`. RSS includes the Python process and imports; it is not tracked-tensor or allocator memory.
