# Agent quickstart

Use TorchScan to inspect model cost, measure the owner's workload, check one optimization, and save the evidence.
Install [the development version](installing.md) for timing, process RSS, profiler diagnostics, and workload reports.
The owner defines acceptable output quality and resource budgets.

## Default workflow

1. Import the existing model and construct representative inputs without downloading weights unless authorized.
2. Use `crawl_module(..., args=..., kwargs=...)` for a real call or `input_shape` for a simple tensor input.
3. Define a zero-argument workload that owns evaluation/training state, gradient mode, precision, transfers, and inputs.
   Configure device placement and threads before measurement. Timing repeats the callable; training must manage its state.
4. Use `measure_workload` to assemble selected FLOPs, clean timing, tensor/allocator memory, and optional profiler passes. Measure RSS with an
   explicit fresh-process command that reconstructs the same configuration.
5. Compare baseline and candidate with `compare_benchmarks(..., check=...)`; preserve a failed check and withheld deltas.
6. Serialize reports as JSON and use `render_report` for offline HTML. Check status and diagnostics before numbers;
   never scrape `summary` or workload terminal text.
7. Preserve model/input revisions, seeds, hardware/software, execution mode, precision, threads, and measurement boundaries.
   Use owner-supplied thresholds for budgets or pass/fail decisions.

Use `strict=True` when incomplete model metrics must stop the task; it raises `IncompleteAnalysisError`.

## Pick one API

| Task | Use |
| --- | --- |
| Inspect module structure and formula metrics | `crawl_module` |
| Show a table to a person and retain the report | `summary` |
| Inspect shapes and parameters with less overhead | `crawl_module(..., mode="structure")` or `summary(..., mode="structure")` |
| Assemble selected workload resources and print a readable summary | `measure_workload` |
| Count operator FLOPs for arbitrary code | `measure_flops` |
| Measure one workload's PyTorch peak memory | `measure_peak_memory` |
| Measure warmed latency/IQR and declared work-unit throughput | `measure_latency` |
| Measure imports, loading, and execution in a fresh child | `measure_peak_rss(command)` |
| Investigate operators in a separate instrumented pass | `profile_workload` |
| Compare model cost estimates | `compare_reports` |
| Check workload performance with an output callback | `compare_benchmarks` |
| Consume a saved model or workload report offline | `render_report` |

`summary` returns an `AnalysisReport` from an evaluation forward with gradients disabled, restoring original training
flags. `measure_workload` and `measure_latency` return `BenchmarkReport` evidence and preserve callable side effects.
Model storage and formula counts do not predict latency or process RSS. Keep the two reports together;
do not add module and operator FLOPs or use `compare_reports` for timing.

## Trust rules

- `complete`: use `value` with the report's method and context.
- `partial`: `known_value` is only a lower bound. Preserve diagnostics and do not extrapolate.
- `unavailable`: report that no measurement was produced.
- Numeric zero is meaningful only when status is `complete`.
- Module FLOPs and operator FLOPs are separate methods; never add or average them.
- Peak PyTorch memory is not process RSS or total accelerator use.
- A skipped or mocked CUDA/MPS check is not device validation.
- First-call latency includes all callable work; warmed block-average latency is not individual-request p95.
- Throughput uses the work completed by one call, such as 32 samples. It is not queued-service throughput.
- RSS, CPU tracked-tensor peaks, and accelerator allocated/reserved peaks have distinct scopes. Do not sum them.
- Profiler self times include instrumentation and may overlap. They are not clean latency.
- An output check establishes only its declared tolerance, not task accuracy. IQR labels are descriptive.

Reuse the [checked experiment](benchmark-comparison.md) and its repository scripts. Locally initialized CNN/Transformer
weights can validate runtime behavior without downloads; they cannot validate task accuracy or another device's speed.

## Owner-controlled budgets

The owner supplies memory budgets and quality thresholds. Compare them with complete metrics of the matching scope;
record hardware and workload state with accelerator results. Do not invent a default budget.

## When an operator is uncounted

1. Preserve the partial result and diagnostic.
2. Confirm the operator and its exact overload in the installed PyTorch version.
3. Rerun with `custom_mapping` on `crawl_module`, `summary`, or `measure_flops` only when the counting method is known
   and reviewable. Operator overrides stay scoped to that analysis.
4. Keep the formula with the experiment or project that owns the assumption.

For custom module/model estimates, pass `custom_modules={ModuleType: ModuleHandler(callback)}`. The callback receives
the actual complete call context and may supply each metric independently. Declare inclusive subtree ownership for
any estimate that includes child work; preserve diagnostics when fields are incomplete or callbacks fail.

See the copyable [extension tutorial](extensions.md), [Model and input support](model-support.md#custom-formulas),
and [Methodology](methodology.md).
