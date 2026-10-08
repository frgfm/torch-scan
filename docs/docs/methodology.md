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

Native attention/Transformer formulas use complete forward arguments and own compute for their subtree, preventing
parent/child double-counting when fast paths bypass child hooks. Parameter accounting remains independent. Their
MACs come from matrix dimensions, DMAs count documented logical stage reads/writes, and receptive-field information
uses module-local token relations rather than spatial scalars. See [Native Transformer estimates](transformers.md)
for independent derivations, tiny examples, normalization assumptions, and mask boundaries.

Per-analysis `custom_modules` handlers can supply estimates for custom leaves and composite modules/models. They
receive complete actual calls and override supplied fields independently. Inclusive parent formulas declare subtree
ownership per metric; only the owner contributes that metric to totals, while child structure and parameter counts
remain visible. An incomplete owner remains incomplete rather than falling back to child estimates. See
[Custom module extensions](extensions.md) for matching, ownership, and callback contracts.

## Operator FLOPs

`measure_flops` uses PyTorch's `torch.utils.flop_counter.FlopCounterMode` around one owner-provided workload. The
dispatcher can observe functional operations and work inside custom modules that hooks cannot assign to a supported
leaf formula.

For structure, shapes, parameters, and buffers alone, use `mode="structure"` in `crawl_module` or `summary`. This
skips operator counting and module formulas while preserving one evaluation forward pass. Skipped compute totals
are explicitly unavailable with method `not_requested`; strict checks apply only to requested metrics. Module
formula and custom-handler work in full mode runs in post-hooks with dispatch suspended, allowing activations to be
released during execution instead of retaining them for deferred analysis.

Only operators with registered or caller-provided formulas contribute to the known count. TorchScan records executed
but uncounted operators and marks the result partial. Caller formulas are scoped to one invocation and use the
installed PyTorch version's shape-formula contract. Pass `custom_mapping` to `crawl_module` or `summary` to use the
same capability for their single-forward operator report. Module handlers do not supply operator estimates.

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

Use [`measure_latency`](metrics.md#latency-and-throughput) for first-call time, warmed timing, and throughput. It uses
PyTorch's native timer for warmup and replicates, with explicit selected-device synchronization. Keep timing separate
from theoretical operation counts.

## Minimum reproducibility record

Follow the [reproducible reporting checklist](metrics.md#reproducible-reporting): retain the report, diagnostics,
software versions, model revision, input metadata, and custom formulas. Workload measurements also need hardware and
execution-state details. Target-device acceptance requires real checks on that device.

## FLOP conventions

`torchscan_flops_v1` keeps module/operator counts separate. Priority: caller override > native PyTorch > invocation-only
fallback; no global mutation. N = elements, R = rows, C = channels, A = affine tensors present (0–2).
Scalar arithmetic, exp/sqrt/erf, comparison, and selection each cost one operation. Sigmoid expands to four operations
and tanh to six, as in the existing module formulas. Constant-only arithmetic is excluded.

| Work | Count / boundary |
| --- | --- |
| K-term dot, K>0 | Module: K multiplies + K-1 adds; native: 2K. Empty dots: zero. |
| Bias/scaling | Module: one bias add/output. Native fused bias and alpha/beta scaling omitted; separate arithmetic counts. |
| Grouped convolution | Outputs × `Cin/groups × kernel_volume` terms; dense padded work. |
| Transposed convolution | `2 × input_elements × Cout/groups × kernel_volume`, plus module bias; cropped scatter products included. |
| Fixed pooling, K values/window | Max: K-1; average: K. Nominal dense padding/ceil windows. |
| Adaptive pooling | Exact overlapping floor/ceil windows: each spatial axis visits `L+O-gcd(L,O)` values over all output positions. Multiply axis sums and batch/channels. Max subtracts output elements; average adds one divide/window. Logical DMAs add output writes and optional max indices. Legacy pooling MAC conventions are retained with corrected geometry. |
| Broadcast/reduction | Per output; sum: max(N-R,0); mean adds R divides. |
| Stable/safe softmax | `5N-2R`: max, subtract, exp, sum, divide. Safe adds 2N comparisons/selections. |
| LogSoftmax / Softmin | Stable LogSoftmax: `4N-R`; Softmin: negate N, then stable softmax. |
| Native activations | Symbolic expansions per element: Hardtanh/ReLU6/Threshold 2; Hardshrink/Softsign 3; LeakyReLU/PReLU/RReLU(eval)/Hardsigmoid 4; Hardswish/Softshrink 5; ELU 6; CELU/SELU/LogSigmoid/Softplus/Tanhshrink 7; Mish 10. Counts expand both branches; they do not depend on tensor values. |
| LayerNorm/GroupNorm | `6N+2R+AN`: mean N, variance 3N, normalize 2N, eps/sqrt 2R, affine AN. |
| GELU | Exact: `5N`; tanh approximation: `14N`, including two multiplies for the cube and the six-operation tanh expansion. |
| SiLU / GLU | `5N`: four sigmoid operations plus a multiply. Here N is **output** elements; GLU halves its split dimension. |
| SwiGLU arithmetic | SiLU plus a separate multiply: `6N` output operations, excluding projections. |
| RMSNorm | `(3+W)N+2R`: square N, mean N, normalize N, eps/rsqrt 2R, affine WN; W is 1 when a weight is present. |
| Rows | LayerNorm: `N/prod(normalized_shape)`; GroupNorm: batch × groups. Empty rows/groups unsupported; zero batches supported. |
| BatchNorm | Saved stats: `2N+2C+AN`; batch stats add 4N. Actual buffers select the path; passed mean/variance updates add 3C/5C. Empty modules: zero; empty native calls unsupported. |
| Dropout | Eval/p=0: zero; training mask/rescale: 2N, or N at p=1. RNG excluded. |

Transpose example: `(2,4,3)`, Cout=6, groups=2, kernel=3, stride=2, padding=output_padding=1:
`24×3×3×2 + 72 biases = 504 module FLOPs`. Shape options add no scatter MACs.

| Attention | Included work / limits |
| --- | --- |
| Batched MHA | Both layouts, unequal widths: projections/bias, Q scale, dense products, softmax, masks, dropout, optional head averaging. |
| CPU SDPA fallback | Dense 4D, matching batches, query heads divisible by KV heads, matching Q/K widths and K/V lengths, no dropout: two dense products + score scale + softmax + masks. GQA uses the query head count. |
| Native fused MHA | Dense batched self-attention: four native 2K projections, Q scaling, two products, stable softmax, masks and optional head averaging. Module MAC/DMA conventions stay separate. |
| Native fused encoder | Self-attention plus two feed-forward projections, ReLU/exact GELU, two affine LayerNorm calls and two residual additions. |
| Masks/math | Each explicit/causal mask costs one operation/score; boolean conversion counts stored entries. Dense products unchanged. Math can scale Q/K separately and use safe softmax; explicit scale changes values only. |
| Native fused attention | Matrix core only; ancillary gaps keep counts partial, including PyTorch 2.1 CPU. |

Reduction dtype: `dtype` > `out.dtype` > input. Fallback integer/boolean arithmetic, views/copies/fills/allocation, and
Python constants are excluded. For `addcmul` and `addcdiv`, tensor operands set the arithmetic dtype;
the scalar coefficient and output buffer do not. Native matrix counts include integers. Complex module/fallback arithmetic and sparse/nested work
stay incomplete without explicit caller overrides; native complex counts remain partial. The
[extension tutorial](extensions.md#a-complete-custom-call) demonstrates a caller convention of six real FLOPs per
complex multiply and two per complex add, with separate MAC and logical DMA conventions. Such overrides are scoped to
one analysis and are the caller's responsibility. Mixed-call diagnostics and strict mode are preserved.

Exact native activation/softmax classes, GroupNorm, and optional RMSNorm require unchanged forwards, dense real inputs, and native
parameter shapes. Empty batches count zero; empty normalized rows/groups are unsupported. PyTorch 2.1 omits RMSNorm.
Scalar-power operator formulas cover only exponent two. Sine/cosine/log/reciprocal cost one operation per value.
Unary math and cumulative sums include in-place calls. Cumulative sums cost N minus row count.
Norm orders 1/2/+inf/-inf cover absolute values or squares, reductions,
and a square root for order 2; other orders remain partial. Packed-row grouped matmul counts `2 × input_elements × output_width`.

Activation MACs are zero; norm MACs are `N(1+W)` for square-sum and optional affine terms. P=parameter elements.
DMAs count staged logical reads/writes, including in-place writes, not hardware traffic. Nonempty calls use:

| Primitive | Logical DMAs |
| --- | --- |
| GELU / SiLU / GLU | Input + output elements, including output writes for in-place SiLU. |
| Other pointwise activations | Input + output elements; PReLU adds reads of its learned weights. |
| Softmax / LogSoftmax / Softmin | Staged logical accesses: `8N+4R` / `8N+6R` / `10N+4R`. Activation MACs remain zero. |
| RMSNorm | `3N+4R+1+W(2N+P)`: mean-square reads N/writes R; eps/rsqrt reads R plus epsilon/writes R; normalize reads N+R/writes N; optional affine reads N+P/writes N. |
| GroupNorm | `4N+5R+1+P+2N` when affine tensors are present; omit the final 2N without affine. R is batch × groups. This uses the existing LayerNorm staged convention. |

GLU and these norms mark spatial metrics `not_applicable`, which alone does not fail strict analysis. GELU/SiLU are
pointwise. Model-wide dependency graphs are not inferred; custom composites need handlers. SwiGLU has operator coverage.

Remaining FLOP gaps: MHA unbatched/empty sequences, `add_bias_kv`/`add_zero_attn`, specialized attention,
unknown normalization/softmax backward, RNG/optimizer/embedding/gather, and unregistered operators.
Transformer requires native stacks, `activation="relu"` / `"gelu"` or exact `F.relu` / `F.gelu`, and final LayerNorm/Identity/None. Adaptive/other pooling metrics retain legacy
spatial approximations for receptive fields. Normalization kernel algorithms can differ; CPU/meta checks do not validate CUDA/MPS or latency.

Native Transformer MAC/DMA boundaries include ReLU/GELU and dense masks, while token dependencies support the
documented canonical mask patterns. Fused module execution uses subtree formulas without claiming a fused hardware
memory model. Custom/einops model graphs still need supported module formulas or independently justified callbacks
through the invocation-scoped [`custom_modules` extension](extensions.md) shared with Issue 41. Token relations
remain native-module information; recognizing an operator does not complete the other module metrics.

`crawl_module`: eval/no_grad forward. `measure_flops`: supplied forward/backward work, without a backward multiplier.
Run `python scripts/benchmark.py --json /tmp/torchscan-matrix.json`: CPU, seed=0, one thread, float32 `(1,3,32,32)`.
Cells retain complete/partial (`>=`)/unavailable states and JSON diagnostics. Derivations: `tests/test_flops.py`;
seven no-weight-download integration smokes: `tests/test_model_zoo.py`.
