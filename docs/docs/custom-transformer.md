# Analyze a custom Transformer

This example needs the [development version (`main`)](installing.md#development-version). It uses the module
extension API and native Transformer estimates added after v0.2.0. It runs on CPU and downloads no models.

Think of the model as two boxes. TorchScan already knows how to estimate the encoder. You supply a formula for the
custom head. Each formula counts its box and its children once.

```text
Input: 3 tokens, 4 features each
                |
                v
     Native encoder          Built-in formula owns this box
                |
                v
     Custom head             Your formula owns this box
       Linear -> sine -> scale
                |
                v
Output: 3 tokens, 2 values each
```

## Run the example

From your development checkout, run:

```shell
python scripts/custom_transformer.py --json /tmp/custom-transformer.json
```

The [full script](https://github.com/frgfm/torch-scan/blob/main/scripts/custom_transformer.py) defines the model and
the head formula. It uses one encoder layer: feature width 4, two attention heads, hidden width 8, and no dropout.
The head uses a bias-free `Linear(4, 2)`, sine, and a scalar multiplier. The scan uses evaluation mode with no gradients.

The analysis call is:

```python
import torch
from torchscan import crawl_module
from scripts.custom_transformer import CustomTransformer, HEAD_HANDLER, SineHead

model = CustomTransformer()
report = crawl_module(
    model,
    args=(torch.ones(1, 3, 4),),
    kwargs={"scale": 0.5, "tag": "demo"},
    custom_modules={SineHead: HEAD_HANDLER},
)
```

The callback receives the actual input, keyword arguments, and nested output. The head returns
`{"logits": (tensor,), "tag": tag}`. The report retains tensor metadata and container structure, not tensor or tag values.
Treat callback inputs as read-only. Do not keep their tensors after the callback.

## Read the counts

These module counts are complete under the stated formulas. One MAC is one multiply-accumulation term. One DMA is
one logical tensor-element read or write. DMAs do not measure bytes, hardware traffic, or peak memory.

| Block | Module FLOPs | MACs | Logical DMAs | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Encoder, including its children | 1,224 | 504 | 912 | 172 |
| Head, including its projection | 54 | 24 | 50 | 8 |
| Model total | 1,278 | 528 | 962 | 180 |

The head has six output elements. Each element needs four multiplies and three adds for its projection, one sine,
and one scale multiplication. Sine costs one FLOP here. Scaling adds no MAC because it has no accumulation.

| Head metric | Calculation |
| --- | --- |
| FLOPs | `6 × (4 + 3 + 1 + 1) = 54` |
| MACs | `6 × 4 = 24` |
| DMAs | `12 input reads + 8 weight reads + 6 projection writes + 12 sine accesses + 12 scale accesses = 50` |

The DMA formula excludes the Python scalar and tag. It counts separate stages and does not assume cache reuse or
kernel fusion. See [native Transformer derivations](transformers.md) for the encoder counts.

The handler declares all six fields in `subtree_metrics`. Thus, `head.projection` stays in the structural report
and retains its eight parameters. Its compute estimates are owned by `head` and are not added again. See
[subtree ownership](extensions.md#own-an-inclusive-subtree-explicitly) before you write an inclusive formula.

## Check each result's status

The module view and operator view answer different questions. Do not add their FLOP counts together.

| View | Result in this example | How to use it |
| --- | --- | --- |
| Module estimates | `complete` | Read `value` under the documented formulas. |
| Operator FLOPs | `complete` on covered dense CPU paths | Read `value` under the operator convention. |
| Encoder token dependencies | `complete`, relation `all` | Each encoder output token can depend on all three input tokens. |
| Encoder spatial receptive field | `unavailable`, not applicable | Use token dependencies instead. |

TorchScan counts `aten.sin` and dense CPU attention independently of the head handler. Other backends or
uncounted operations can still make operator FLOPs partial. Check diagnostics; `strict=True` raises
`IncompleteAnalysisError` and preserves the report when applicable work is incomplete.

The head's receptive-field value of 1 describes its own token-local call. It does not mean that the final output
depends on only one original input token. The encoder has already mixed information across tokens.

Structure mode skips both estimation views and custom callbacks. It still reports shapes, calls, and 180 parameters:

```python
report = crawl_module(
    model,
    args=(torch.ones(1, 3, 4),),
    custom_modules={SineHead: HEAD_HANDLER},
    mode="structure",
    strict=True,
)
```

The integration tests also pass a causal mask through the wrapper. It changes encoder token dependencies to `prefix`
and adds mask accesses. Dense MACs stay at 528. Custom padding or arbitrary masks can make token dependencies
unavailable; see [Transformer limits](transformers.md) before interpreting them.
