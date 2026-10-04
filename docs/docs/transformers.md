# Native Transformer estimates

TorchScan estimates one evaluation call to the exact native PyTorch types `nn.MultiheadAttention`,
`nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`, `nn.TransformerEncoder`, `nn.TransformerDecoder`, and
`nn.Transformer`. MACs, logical DMAs, and module-local token dependencies use the complete call, including query,
key, value, encoder memory, masks, causal hints, and attention-weight options. Their formulas apply even when a
PyTorch fast path bypasses child hooks.

Use explicit `args` and `kwargs`; a first-input-only description cannot describe cross-attention or its masks:

```python
import torch
from torch import nn

from torchscan import crawl_module

attention = nn.MultiheadAttention(4, 2, batch_first=True)
query = torch.ones(1, 2, 4)
memory = torch.ones(1, 3, 4)
report = crawl_module(
    attention,
    args=(query, memory, memory),
    kwargs={"need_weights": False},
)
print(report["totals"]["macs"])  # complete: 208
print(report["totals"]["dmas"])  # complete: 352 logical element accesses
print(report["layers"][0]["token_dependencies"])
```

Run `python scripts/transformer_estimates.py --json /tmp/transformer-estimates.json` from the repository for
standalone attention, encoder/decoder layers, native stacks, both batch layouts, and explicit masked calls. The
script uses local CPU tensors and saves full reports, including diagnostics and token dependencies.

## Supported boundaries

The input tensors must be real floating-point, dense strided, batched 3D tensors with positive batch and token
dimensions. Both `(batch, tokens, features)` and `(tokens, batch, features)` layouts are supported. Query width is
`embed_dim`; key/value widths may use `kdim` and `vdim`. Biases may be present or absent. Attention must use native
`add_bias_kv=False` and `add_zero_attn=False` behavior.

Native layers use their native Linear feed-forward stages, ReLU or GELU, feature-only `LayerNorm`, and evaluation
dropout. Pre-norm and post-norm layers have the same MAC/DMA stage totals. Normalization affine weight and bias may
be independently present or absent; native `Identity` or no final normalization adds no stage. Encoder and decoder
stacks must be nonempty and contain native layer types with matching embedding widths and batch layouts.
`nn.Transformer` must contain native encoder/decoder stacks matching its embedding width and layout.
Invoked native `forward` methods must be unchanged; instance replacements require explicit custom estimates.
The attention output projection uses its parameter tensors directly, so replacing its unused `forward` has no effect.
For an encoder call with a padding mask, construct `nn.TransformerEncoder(..., enable_nested_tensor=False)` so the
estimate describes dense work rather than padding-based packing.

PyTorch 2.1.0 has an encoder fast-path bug: a batch-first evaluation call can fail when a bias or LayerNorm affine
tensor is absent. Use sequence-first layers/stacks for those configurations on that release. On releases that expose
`torch.backends.mha.set_fastpath_enabled`, disabling the fast path is another option. TorchScan propagates native
execution errors; these are separate from unsupported estimation formulas.

MAC/DMA formulas accept native dense boolean and floating masks of the documented PyTorch shapes. Masks do not
reduce dense matrix products. Token dependencies have a narrower mask boundary, described below. Compute estimates
require an explicit mask with a causal hint; dependency estimates additionally require matching canonical causal
exclusions. A hint by itself has ambiguous native fast-path behavior and is unavailable.
These formulas describe evaluation calls; `crawl_module` temporarily selects evaluation mode and restores the
original training flags.

Existing FLOP formulas remain separate: native layer/stack module FLOPs retain the ReLU-only boundary. GELU can
therefore have complete MACs/DMAs with unavailable module FLOPs. Operator FLOPs follow the actual dispatcher path
and may be partial for fused operations. MACs do not come from dividing either FLOP view by two.

Unsupported types or configurations produce diagnostics and unavailable affected metrics. A supported dense MAC
count is exact under this convention; an unsupported sparse/nested call is not assigned its dense upper bound as
`known_value`. Partial values remain lower bounds. An unavailable token relation never contains a fabricated
all-token relation. `strict=True` checks requested metric diagnostics, including token-dependency failures; unavailable
spatial scalars on a supported token module are expected. `mode="structure"` skips all formulas and token analysis.

## Independent MAC derivation

Let batch size be `B`, target/query length `T`, source/key-value length `S`, embedding width `E`, head count `H`,
head width `D = E/H`, and key/value input widths `K` and `V`. Each matrix contraction counts one MAC per product
term contributing to an output coordinate, including a one-term product. Bias addition adds no MAC.

| Stage | Matrix dimensions per batch/head | MACs |
| --- | --- | --- |
| Query projection | `(T, E) @ (E, E)` | `B T E²` |
| Key projection | `(S, K) @ (K, E)` | `B S K E` |
| Value projection | `(S, V) @ (V, E)` | `B S V E` |
| Scores | `H` products `(T, D) @ (D, S)` | `B H T S D = B T S E` |
| Weighted values | `H` products `(T, S) @ (S, D)` | `B H T D S = B T S E` |
| Output projection | `(T, E) @ (E, E)` | `B T E²` |

Thus attention MACs are `2 B T E² + B S E(K + V) + 2 B T S E`. For equal-width self-attention with `T = S = L`,
this becomes `4 B L E² + 2 B L² E`. Head count cancels because splitting the width preserves the number of matrix
terms. Masked attention still executes the same dense products, including causal attention.

A feed-forward block with hidden width `F` has two independent matrix products: `(T, E) @ (E, F)` and
`(T, F) @ (F, E)`, giving `2 B T E F` MACs. Feature-only LayerNorm counts `N` variance square-sum terms and another
`N` affine products when its weight is present: `N (1 + weight_present)`. A bias-only norm adds no affine MAC.
Mean reduction, subtract/divide, softmax, query scaling, residual addition, ReLU, and GELU arithmetic add no MACs
under this convention. This is a count of the documented operations, independent of normalization kernel choices.

An encoder layer adds self-attention, feed-forward, and two norm counts. A decoder layer adds target
self-attention, query/memory cross-attention, feed-forward, and three norm counts. Each stack sums its layers and
optional final norm. A full Transformer sums the encoder and decoder stacks; cross-attention uses source length
`S`, while target stages use length `T`.

## Logical DMA derivation

One DMA is one logical element read or write, not a byte, a DMA-engine transaction, or measured hardware traffic.
The estimate uses a staged dense algorithm. A matrix stage reads each input operand and parameter tensor once and
writes its result once. The same aliased tensor used as query, key, and value is read once by each projection stage.
Parameter reuse across repeated calls is counted per logical stage. Cache reuse, kernel fusion, tiling, allocations,
layout copies, and hardware-specific algorithms are outside this model. Evaluation dropout and views add no stage.

Define `Q = B T E`, `J = B S E`, `A = B H T S`, `R = B H T`, and
`I = B T E + B S K + B S V`. Let `P` be the number of attention parameter elements: projection and output matrices,
plus each bias tensor that is present.

| Stage | Logical element accesses |
| --- | --- |
| Three input projections | `I + input_projection_parameters + Q + 2 J` |
| Query scale | `2 Q` |
| Score matrix product | `Q + J + A` |
| Staged stable softmax | `8 A + 4 R` |
| Weighted-value matrix product | `A + J + Q` |
| Output projection | `2 Q + output_projection_parameters` |

Softmax first reads scores and writes row maxima (`A + R`); subtraction reads scores and maxima and writes shifted
scores (`2 A + R`); exponentiation reads shifted scores and writes exponentials (`2 A`); summation reads
exponentials and writes row sums (`A + R`); division reads exponentials and sums and writes probabilities
(`2 A + R`). These separate stages give `8 A + 4 R`. This is a logical reference algorithm rather than an assertion
about a fused implementation's intermediates.

The attention total is `I + P + 7 Q + 4 J + 10 A + 4 R`, before masks or returned-weight averaging:

- Each explicit attention or key-padding mask adds `mask.numel() + 2 A`: one read of its stored entries, then a
  read/write pass over scores. Boolean conversion buffers and broadcast materialization are excluded. A causal hint
  accompanying a mask does not add a second mask stage.
- `need_weights=False` adds no returned-weight stage. `need_weights=True, average_attn_weights=False` returns the
  already counted probability tensor by alias and adds no accesses. Head-averaged weights add `A + B T S`, a read
  across heads and a write of the averaged result.

Feature-only LayerNorm with `N` elements and `r = N/E` rows adds `4 N + 5 r + 1 + norm_parameters`, plus `2 N` when
weight or bias is present. The mean stage reads `N` and writes `r`; variance reads `N` and the `r` means and writes
`r`; normalization reads `N` and both `r` statistics, reads epsilon once, and writes `N`. An affine stage reads and
writes `N` and reads its parameters once. `Identity`/no norm adds zero.

With `N = B T E` and hidden elements `U = B T F`, feed-forward stages add `2 N + 4 U + linear_parameters`: each
Linear reads its input/parameters and writes its output; ReLU/GELU reads and writes `U`. Each residual addition
adds `3 N` for two reads and one write. Encoder layers have two residual additions and decoder layers have three.
Stage order, `norm_first`, and PyTorch's fused fast paths do not change this logical estimate.

## Hand-derived tiny examples

All examples below use `B = 1`, `E = 4`, `H = 2`, biases, affine feature-only LayerNorm, ReLU, and no masks or
returned weights. Stack layers use hidden width `F = 8`. Attention examples omit normalization.

| Call | Independent MAC calculation | MACs | Logical DMAs |
| --- | --- | --- | --- |
| Self-attention, `L = 3` | `4 × 3 × 4² + 2 × 3² × 4` | 264 | 452 |
| Cross-attention, `T = 2`, `S = 3`, `K = V = 4` | `2 × 2 × 4² + 3 × 4 × (4+4) + 2 × 2 × 3 × 4` | 208 | 352 |
| Cross-attention, `T = 2`, `S = 3`, `K = 6`, `V = 5` | `64 + 3 × 4 × (6+5) + 48` | 244 | 373 |
| Encoder layer, `S = 3` | `264 + 192 + 2 × 24` | 504 | `452 + 196 + 2 × 96 + 2 × 36 = 912` |
| Decoder layer, `T = 2`, `S = 3` | `160 + 208 + 128 + 3 × 16` | 544 | `288 + 352 + 156 + 3 × 67 + 3 × 24 = 1069` |
| One-layer encoder and decoder, both final norms | `504 + 24 + 544 + 16` | 1088 | `912 + 96 + 1069 + 67 = 2144` |

For self-attention, `I=36`, `P=80`, `Q=J=12`, `A=18`, and `R=6`, so DMAs are
`36 + 80 + 84 + 48 + 180 + 24 = 452`. Averaged returned weights add `18+9=27`, giving 479.
For equal-width cross-attention, `I=32`, `P=80`, `Q=8`, `J=12`, `A=12`, and `R=4`, so DMAs are
`32 + 80 + 56 + 48 + 120 + 16 = 352`. Wider key/value inputs add 9 input reads and 12 parameter reads, giving 373.
The full Transformer retains its existing 2694 module FLOPs; its independently derived 1088 MACs demonstrate why
division of FLOPs by two would give the wrong result.

## Module-local token dependencies

Attention has a token axis, rather than a convolutional spatial receptive field. Supported calls add an optional
`token_dependencies` object to their layer record. Its `scope` is `module_call`; it describes potential structural
dependencies of the main output tensor on that call's input tokens. It does not describe numerical gradients,
attention-weight outputs, or a graph-wide effective receptive field. Legacy scalar `receptive_field`, `stride`, and
`padding` metrics remain unavailable on token modules.

Relations use zero-based positions:

| Relation | Input token positions that may affect output position `i` |
| --- | --- |
| `all` | Positions `0` through `limit-1`; `limit` defaults to the input length |
| `same_position` | Position `i` |
| `prefix` | Positions `0` through `min(i+1, limit)-1`; `limit` defaults to the input length |
| `none` | No input token positions |

Optional `first_position` is the first **output** position with a dependency; earlier output positions have none.
For example, the first row of causal attention has a one-key softmax, so it depends on values but cannot depend on
query/key scores. Separate query/key relations then have `first_position=1`; a shared key/value relation includes
the value dependency from output position zero. With a single source token, query/key relations are `none` at all
positions. These cases distinguish real dependency semantics from merely listing tensor operands.

The report records input argument names, axis indices, lengths, output axis metadata, and explicit assumptions.
Self-attention with shared query/key/value tensors groups those argument names on one source relation. Unmasked
self-attention has an `all` relation; canonical causal self-attention has a position-dependent `prefix` relation.
Cross-attention records query dependencies separately from key/value dependencies: the query is token-local and
unmasked source dependencies span the source axis. Feature-only normalization, feed-forward stages, residual paths,
and final normalization preserve token-local relations. Native stacks compose these relations within their own
call boundary, including source dependencies through the encoder in `nn.Transformer`.

Composition can broaden a relation inside a native stack. For example, an unmasked encoder makes a full
Transformer's source dependency span every source token even when decoder memory attention is causal. In a
two-layer decoder with unmasked target self-attention and causal memory attention, later target mixing combines
the first layer's memory prefixes: each output may then depend on every memory position below `min(T, S)`.
This is an `all` relation with `limit=T` when `S>T`, rather than an unbounded source span.

Dependency masks support unmasked/all-allowed attention, finite additive biases, and canonical causal exclusions.
Floating masks use negative infinity to exclude an entry; finite entries preserve structural dependencies.
Boolean `True` excludes an entry, following native Transformer mask semantics. Every batch/head must have the same
canonical causal exclusions when using a 3D causal mask. Key-padding masks may contain all `False` or finite
additive values but must not exclude tokens. Genuine padding, arbitrary exclusion patterns, NaN/positive infinity,
or inconsistent causal hints are unavailable with diagnostics; their dense MAC/DMA estimates can still be complete.
Meta masks are unavailable because their values cannot be inspected. All-masked attention is unavailable because a
dense softmax row is not guaranteed to define a meaningful dependency. A mask's restrictions change dependency
relations without pretending that dense arithmetic or logical score intermediates disappear.

Token dependencies also require nondegenerate feature-only normalization: width-one LayerNorm is unavailable for
this relation model because its output is intrinsically constant. Its MAC/DMA stage counts remain supported. The
relations otherwise describe potential dependencies at generic finite parameter values, without assuming particular
weights, accidental cancellation, or numerical underflow.

The optional field is JSON-serializable and preserved by text/visual reporting and report comparison. Older reports
without it remain valid. Readers must check its status rather than interpreting an unavailable relation as empty.

## Ownership, compatibility, and remaining gaps

A native attention or stack formula owns compute for its subtree. Its children may still appear in execution
structure, but their work is not added again; parameters remain accounted for exactly once. This makes standalone
modules, wrappers containing native modules, and fused native paths consistent. Operator FLOPs stay in their own
report and are never added to module MACs, DMAs, or token relations.

Native `nn.Transformer` reports now retain its observed child calls rather than forcing a single root row. A fast
path may bypass some child forwards, so structural row counts can differ between PyTorch versions or configurations.
Use the ownership metadata and retained metric fields, rather than layer counts, to identify counted work.

This support does not infer an arbitrary custom attention implementation or an einops/custom model's dependency
graph. Unsupported custom leaf work continues to produce diagnostics; operator FLOPs may recognize its functional
matrix operations without completing module MACs/DMAs or dependency information. The invocation-scoped
[`custom_modules` extension](extensions.md) shared with Issue 41 provides `ModuleHandler` callbacks with the actual
complete `ModuleCall` and per-metric `subtree_metrics` ownership. Native handlers use that same contract. Custom
callbacks can supply independently justified numerical estimates for custom/einops modules, with explicit incomplete
states; registering a model alone does not fill those gaps. Token-dependency metadata currently comes from the
native handlers; arbitrary custom token relations and graph-wide propagation remain unsupported. Specialized
attention, training/backward estimates, unbatched or empty inputs, sparse/nested tensors, and irregular dependency
masks remain outside these native formulas.
