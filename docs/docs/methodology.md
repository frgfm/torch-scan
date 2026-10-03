# Methodology

TorchScan describes one executed model call or workload. It combines two observation mechanisms while keeping their
results separate:

1. Module hooks capture hierarchy, call order, tensor metadata, parameters, buffers, and module formulas.
2. PyTorch operator dispatch captures executed operations with registered FLOP formulas.

Neither mechanism is a hardware benchmark or a complete graph export.

## Module analysis

`crawl_module` and `summary` make one forward call. Generated inputs add a batch dimension of one; caller-provided
`args` and `kwargs` are forwarded unchanged. Analysis runs in evaluation mode with gradients disabled, then restores each
module's previous training flag.

Use `args` and `kwargs` when the model does not accept a leading batch dimension, including `batch_first=False`
sequence modules. Calls sharing the same module instance must be serialized because crawling temporarily changes its
training state and installs hooks.

Layer identity is the full module path plus a call index. This distinguishes repeated calls through a shared module
without inventing duplicate parameters.

Supported module formulas calculate theoretical FLOPs, MACs, DMAs, and receptive field. Unsupported work changes
the affected metric state instead of contributing zero. Formula definitions and their tested boundaries live in the
package source; diagnostics expose unsupported paths.

## Operator FLOPs

`measure_flops` uses PyTorch's `torch.utils.flop_counter.FlopCounterMode` around one owner-provided workload. The
dispatcher can observe functional operations and work inside custom modules that hooks cannot assign to a supported
leaf formula.

For structure, shapes, parameters, and buffers alone, use `mode="structure"` in `crawl_module` or `summary`. This
skips operator counting and module formulas while preserving one evaluation forward pass. Skipped compute totals
are explicitly unavailable with method `not_requested`; strict checks apply only to requested metrics. Module
formula work in full mode runs in post-hooks with dispatch suspended, allowing activations to be released during
execution instead of retaining them for deferred analysis.

Only operators with registered or caller-provided formulas contribute to the known count. TorchScan records executed
but uncounted operators and marks the result partial. Caller formulas are scoped to one invocation and use the
installed PyTorch version's shape-formula contract.

The workload owns model state, gradient mode, autocast, device placement, warmup, and side effects. Exceptions are
propagated unchanged.

## Why the two FLOP views can differ

Module formulas and operator formulas may choose different boundaries or arithmetic conventions. Composite modules
can decompose into several dispatcher operations, while fused operators can combine work that a module formula
describes separately. Custom formulas can also use experiment-specific conventions.

For this reason TorchScan labels both methods and does not reconcile them into one number. A paper or regression
report should state which view it uses.

## Partial is a lower bound, not coverage

For a partial result, `known_value` is the sum of counted work. It is a lower bound only. TorchScan does not calculate
a coverage percentage because operator kinds or call counts do not reveal the cost of the missing work.

Diagnostics are part of the measurement. Store them with the numeric result and resolve them before making a
completeness claim.

## Peak memory

`measure_peak_memory` invokes a zero-argument workload once. CPU uses profiler memory categories; supported
accelerators use allocator statistics. These are PyTorch measurements, and concurrent allocations can affect them.
See [Peak memory](metrics.md#peak-memory) for backend boundaries and comparison requirements.

## Latency

Use [`torch.utils.benchmark.Timer`](https://docs.pytorch.org/docs/stable/benchmark_utils.html) for warmup, replicates,
and synchronization. Keep latency separate from theoretical operation counts.

## Minimum reproducibility record

Follow the [reproducible reporting checklist](metrics.md#reproducible-reporting): retain the report, diagnostics,
software versions, model revision, input metadata, and custom formulas. Workload measurements also need hardware and
execution-state details. Target-device acceptance requires real checks on that device.
