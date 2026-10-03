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

## FLOP conventions

Counts describe real scalar arithmetic. They do not describe kernel instructions or latency.
Keep module and operator counts separate. The operator convention is `torchscan_flops_v1`.
Native PyTorch formulas take priority. TorchScan fills missing formulas for one call only.
Caller overrides take priority over both. No global registry changes.

| Work | Module count | Operator count |
| --- | --- | --- |
| Dot product with K>0 terms | K multiplies + K-1 adds | Native: K multiply-adds x 2 |
| Bias | One add per output | Native fused matrix/convolution bias is omitted; a separate add counts |
| Grouped convolution | Each output uses `Cin/groups * kernel_volume` terms | Native grouped weight shape sets the dense MAC count |
| Transposed convolution | `input_elements * Cout/groups * kernel_volume * 2`, plus output bias | Native uses the same input-based MAC count; bias is omitted |
| Fixed pooling with K values per window | Max: K-1 comparisons; average: K-1 adds and one divide | Unregistered pooling operators stay partial |
| Stable softmax: R rows, S values per row | R(5S-2) | Same; safe softmax adds 2RS for comparison and selection |
| Dropout | Eval/p=0: zero; training: 2N for mask/rescale, or N when p=1 | Visible arithmetic counts; unknown dropout/RNG work stays partial |

Multiply-adds count as two native FLOPs. Module dot products omit the first accumulator add.
Empty dot products cost zero; bias still counts separately.
A scalar add, multiply, divide, exp, sqrt, comparison, or selection counts as one operation.
Broadcast operations count output elements. Sum uses K-1 adds; mean adds one divide per output.
Views, copies, fills, allocation, integer counters, boolean control operations, and Python shape/constant work are excluded.
Reduction arithmetic uses the requested `dtype`, then the `out` buffer dtype, then the input dtype.
Supplemental integer arithmetic is excluded; native matrix shape counts remain dtype-agnostic, including integer matrices.
Native matrix formulas also omit alpha/beta scaling. Complex module/supplemental arithmetic is unsupported;
native complex shape counts remain partial lower bounds unless the caller supplies a formula.
Shape formulas require dense strided tensors. Sparse/nested calls remain incomplete before shape extraction;
their dense estimates cannot become lower bounds. Caller overrides retain control.

Convolution counts use dense padded arithmetic. They include zero-padding products and nominal transposed
scatter products that can be cropped. Stride, dilation, and output padding set the output shape and bias count.
They do not add input-based scatter MACs. For input `(2,4,3)`, six output channels, two groups, and three taps,
24 input values each scatter to 3 channels through 3 taps: 216 MACs, or 432 native FLOPs.
Stride two, padding one, and output padding one give 72 output biases: 504 module FLOPs.

Normalization uses a two-pass mathematical count. For N values in R rows: mean N, biased variance 3N,
epsilon/sqrt 2R, and normalization 2N. LayerNorm and GroupNorm cost `6N + 2R`, plus N for each affine weight/bias.
LayerNorm rows span `normalized_shape`; GroupNorm has `batch_size * num_groups` rows.
BatchNorm with saved statistics costs `2N + 2C`, plus affine work. Batch statistics add 4N in training
or evaluation without saved statistics. Tracked training adds 8C for unbiased variance and running averages.
Actual saved buffers determine BatchNorm's statistics path, even if the tracking flag changes after construction.
Running updates require the saved buffers. Kernel algorithms such as Welford can use different instruction counts.
Empty LayerNorm rows and GroupNorm groups stay unsupported; a zero batch with nonempty rows can count zero.

Batched MultiheadAttention includes projections and bias, query scaling, score/value dot products, stable softmax,
visible masks, training dropout, and optional head averaging. Both layouts and unequal sequence/input widths are supported.
Each mask adds one selection/add per score. Causal/masked positions do not reduce dense matrix work.
Unbatched/empty sequences, `add_bias_kv`, and `add_zero_attn` remain unsupported.
Transformer module estimates require native encoder/decoder stacks, ReLU feed-forward blocks, and final
normalization that is LayerNorm, Identity, or absent. Other final modules remain unavailable.

The CPU fused SDPA fallback supports dense 4D tensors, matching batch/head counts and Q/K widths, equal K/V lengths,
and no dropout. Unsupported broadcast batches remain partial; functional SDPA can select a supported math path.
It reuses native matrix formulas, then counts score scaling, stable softmax, and one mask operation per score.
Boolean mask conversion counts separately at the stored mask shape. Explicit scale changes the value, not the count.
When the backend accepts causal and explicit masks together, each adds its own operation per score.
Saved-state outputs and data movement are excluded. The math path can scale Q and K separately and use safe softmax.
Tests derive each path's count independently. Native fused flash/efficient/cuDNN attention counts only matrix work,
so hidden scaling, softmax, masks, and dropout produce `incomplete_operator_formula`. PyTorch 2.1 uses this native
core-only formula on CPU too. Fused MHA/encoder operations without formulas remain uncounted.

`crawl_module` counts an evaluation forward under `no_grad`. `measure_flops` counts the supplied forward/backward
workload; it does not multiply the forward count to estimate backward. Unknown normalization/softmax backward,
RNG, optimizer, embedding/gather, and activation/pooling operations stay diagnostic gaps. Specialized nested/sparse
attention and GPU ancillary work are not covered. Fixed pooling counts nominal dense windows, including padding
and ceil boundaries. Adaptive pooling and the other pooling metrics retain legacy approximations.
CPU and meta checks do not validate CUDA/MPS execution. Unsupported calls stay incomplete even when another call
of the same operator packet was counted. Strict mode and independent MAC/DMA/receptive-field status are preserved.

Run `python scripts/benchmark.py --json /tmp/torchscan-matrix.json` for three CNNs with seed zero,
float32 `(1,3,32,32)` inputs, CPU execution, and one thread. Cells show complete, partial (`>=`), or unavailable.
Full JSON retains diagnostics. `tests/test_flops.py` uses deterministic workloads and hand-derived counts.
The seven-model matrix in `tests/test_model_zoo.py` adds timm ViT and tiny BERT/T5 self/cross attention.
These optional smokes download no weights and check integration, not formula correctness.
