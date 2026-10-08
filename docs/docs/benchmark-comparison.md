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
it is not autoregressive generation throughput. The output check covers CNN logits, or BERT hidden states and
pooled outputs, at `rtol=1e-4`, `atol=1e-5`, and records shapes, finiteness, and maximum absolute error. This checks
runtime equivalence, not task accuracy.

The controlled change replaces per-sample forwards and output concatenation with one batch forward on identical
resident inputs. Timing excludes loading, transfers, and output checks. Both timings finish before the separate
FLOP, memory, RSS, and profiler passes. CPU tensor peaks include observed resident tensors; allocator peaks retain
the cache state after timing. Compare only like scopes. Threads, model/library versions, precision, input metadata,
measurement boundaries, raw blocks, and output evidence are saved in the comparison. Use `--reverse` in a fresh
process to check order effects. Use `--device mps` or `--device cuda:0` only on available real hardware, and keep
generated JSON/HTML outside the checkout, as in the commands above.
