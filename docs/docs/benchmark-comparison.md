# Check an optimization on your hardware

These APIs require the development version. Start with a representative workload, measure it, try one change, and
check the outputs. TorchScan records the evidence; your check defines the accepted tolerance and task requirements.

## Compare two runs

```python
from pathlib import Path
import json
import torch
from torchscan import compare_benchmarks, measure_latency, profile_workload, render_report

torch.set_num_threads(1)  # Configure before constructing the model.
torch.manual_seed(0)
model = torch.nn.Linear(128, 128).eval()
inputs = torch.randn(32, 128)


@torch.inference_mode()
def baseline():
    return torch.stack([model(row) for row in inputs])


@torch.inference_mode()
def candidate():
    return model(inputs)


settings = dict(device="cpu", inputs=inputs, work_units=32, work_unit="samples")
before = measure_latency(baseline, **settings)
after = measure_latency(candidate, **settings)
before["profile"] = profile_workload(baseline, device="cpu")  # Separate from timing.
after["profile"] = profile_workload(candidate, device="cpu")
comparison = compare_benchmarks(
    before,
    after,
    check=lambda: torch.testing.assert_close(baseline(), candidate(), rtol=1e-4, atol=1e-5),
)
Path("comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
Path("comparison.html").write_text(render_report(comparison), encoding="utf-8")
```

The check runs outside timing. Returning `None` or `True` passes; returning `False` or raising `AssertionError` fails.
Other exceptions propagate. `output_check` records this callback's result. A failed check preserves both measurements but withholds every numeric delta and the
speed label. Check representative outputs and task accuracy yourself. Matching a few tensors is not a task-level
accuracy guarantee, and TorchScan does not retain outputs or choose a tolerance.

Comparisons require matching hardware, PyTorch/Python/TorchScan versions, CUDA runtime, caller-supplied input metadata,
and work-unit definitions. Shapes/dtypes do not establish equal tensor contents. Preserve input seeds or dataset
revisions, model revisions, precision, execution mode, transfers, and state in the experiment record. Changed thread
counts and timing settings remain visible so a thread-count experiment is possible. Only complete metrics with the
same method, scope, and unit receive `after - before` deltas. Missing or incomplete metrics keep unknown deltas.

`latency_change` is `faster`, `slower`, `within_variability`, or `unavailable`. A median change no larger than the larger
run's interquartile range (IQR) is labelled `within_variability`. This rule describes observed spread; it is not a
statistical significance test or protection against thermal drift, execution order, or background load. Repeat runs
and alternate their order before claiming a small gain. Raw block durations remain in both reports.

See the [`compare_benchmarks`](torchscan.md#torchscan.compare_benchmarks) and
[`BenchmarkComparison`](torchscan.md#torchscan.BenchmarkComparison) API reference.

## Save the complete experiment

`render_report` accepts one `BenchmarkReport` or the checked comparison above. HTML contains timing, any appended
scoped memory metrics, a separate optional operator profile, hardware/input metadata, raw timing blocks, and changed
settings. It works offline with native tables and expandable details. Benchmark SVGs and `before=` are unsupported;
use `compare_benchmarks` to supply checked before/after evidence. A loaded comparison renders its saved check status
without rerunning the check. Inspect the evidence before sharing it.

Use [`measure_peak_rss`](workload-diagnostics.md#whole-process-peak-rss) in a fresh child and append its `MetricResult`
to the matching report's `totals`. Convert `measure_peak_memory` results to separate byte metrics with explicit
methods and scopes. CPU tensor peaks include observed tensors already resident before the call; `delta_bytes` is
the increment from the recorded baseline. Do not add RSS to allocator or tensor peaks, especially on unified-memory hardware. Profiler
operator times include instrumentation and are not clean workload timing.

Run the complete example on each available device:

```shell
python scripts/benchmark_comparison.py --device cpu --rss --output /tmp/torchscan-cpu
python scripts/benchmark_comparison.py --device mps --rss --output /tmp/torchscan-mps
python scripts/benchmark_comparison.py --device cuda:0 --rss --output /tmp/torchscan-cuda
# Reuse the CNN and Transformer definitions from the model integration tests:
HF_HUB_OFFLINE=1 python scripts/benchmark_comparison.py --model resnet18 --device cpu --threads 2 --min-run-time 1 --rss --output /tmp/torchscan-resnet18
HF_HUB_OFFLINE=1 python scripts/benchmark_comparison.py --model bert --device cpu --threads 2 --min-run-time 1 --rss --output /tmp/torchscan-bert
```

This no-download example compares a locally initialized linear layer, one sample at a time versus one batch. It
records clean timing, separate operator FLOPs and tensor/allocator memory passes, an optional fresh-process RSS trial, an operator
profile, and an output check. Each RSS child includes imports, model/input construction, and 100 completed calls;
it does not include profiling. MPS GPU operator time or allocator peaks can be unavailable in the installed PyTorch.
The report preserves that limit. Omit `--rss` on Windows. Each device needs real matching hardware; results cannot
predict another machine's latency or service throughput. The example records a microbenchmark, not a production
speedup promise. Incomplete FLOP counts keep their lower bounds and diagnostics; they do not prevent timing.

The optional models use the existing `model-test` dependencies (`uv pip install -e ".[model-test]"`). ResNet18 uses
`weights=None` with four `3 × 32 × 32` images; BERT uses the small configuration in `tests/test_model_zoo.py` with
four eight-token sequences and an all-visible attention mask. Both use seed 0, float32, eval/inference mode, and
locally initialized weights. Throughput means images/s for ResNet18 and **input** tokens/s for the BERT encoder;
it is not autoregressive generation throughput. The output check requires finite CNN logits, or BERT hidden states
and pooled outputs, at `rtol=1e-4`, `atol=1e-5`. This checks runtime equivalence, not task accuracy.

The controlled change replaces per-sample forwards and output concatenation with one batch forward on identical
resident inputs. Timing excludes loading, transfers, and output checks and finishes before diagnostic passes.
Configuration and measurement boundaries are saved in the comparison. Use `--reverse` in a fresh process to check
order effects, and keep generated JSON/HTML outside the checkout, as in the commands above.

## Recorded CPU validation

[Validation PR #175](https://github.com/frgfm/torch-scan/pull/175) records this experiment.

One recorded CPU run used an AMD EPYC 9V74 VM, Linux, Python 3.11.16, PyTorch 2.13.0+cpu,
torchvision 0.28.0+cpu, transformers 5.15.1, FP32 without autocast, two intra-op threads and one inter-op thread.
Weights were locally initialized with seed 0; both variants used the same resident inputs in eval/inference mode.
Both timings complete all four samples; the baseline loops over individual forwards and concatenates their outputs.
The candidate ran first to check order sensitivity; both orders favored batching.

| Workload | Loop median / IQR ms | Batched median / IQR ms | Throughput before → after | RSS MiB before → after | CPU tensor peak MiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18, four `[3,32,32]` images | 24.7072 / 1.2594 | 7.6609 / 0.7181 | 156.99 → 519.59 images/s | 318.69 → 329.12 | 44.89 → 53.72 |
| BERT, four sequences of eight input tokens | 1.5963 / 0.2735 | 0.4485 / 0.0184 | 18,876.47 → 67,674.99 input tokens/s | 310.47 → 310.07 | 0.0187 → 0.0273 |

The output checks passed at `rtol=1e-4`, `atol=1e-5` with finite outputs. BERT counts **input tokens**, not generated
tokens. Its roughly 310 MiB RSS includes imports/loading, whereas its tensor peak covers a separate instrumented call.
ResNet18's batched tensor peak rose with the oneDNN convolution path. Operator FLOPs remained partial for both models,
so their FLOP deltas stayed unknown. Faster timing does not imply smaller memory use or complete compute coverage.

Latency excludes imports, loading, transfers, and checks. Each RSS child includes initialization and 100 completed
calls without instrumentation. Tensor peaks include observed resident tensors; profiler passes are separate from
timing. MPS and real CUDA were unavailable. These results establish CPU runtime behavior and the declared output
tolerance, not task accuracy, statistical significance, service throughput, or another machine's performance.
The linked PR preserves exact configurations, raw measurement boundaries, discrepancy findings, and reproduction details.
