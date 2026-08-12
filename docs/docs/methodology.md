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

`measure_peak_memory` invokes a zero-argument workload once:

- CPU uses PyTorch profiler memory categories.
- CUDA and supported MPS releases use public allocator statistics.

The method does not include all Python, process, driver, device, or third-party memory. Accelerator results are
process-global and can be affected by unrelated allocations. Compare only equivalent hardware and prepared workload
state.

## Latency

Use [`torch.utils.benchmark.Timer`](https://docs.pytorch.org/docs/stable/benchmark_utils.html) for latency. The native
PyTorch tool already supplies warmup, replicates, and synchronization. TorchScan deliberately does not convert
theoretical operation counts into time.

## Minimum reproducibility record

- Full JSON report and every diagnostic.
- TorchScan, PyTorch, and Python versions from report context.
- Model source revision and configuration.
- Input metadata and non-sensitive call structure.
- Custom formula code and rationale.
- Hardware, warmup, allocator, model, gradient, autocast, and optimizer state for workload measurements.

TorchScan does not claim camera, accelerator, production, or target-device acceptance without those real checks.
