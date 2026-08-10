# Model and input support

TorchScan attaches forward hooks to `torch.nn.Module` objects and executes one forward pass. This makes module-based models easy to inspect, but it does not trace arbitrary tensor operations.

## Inputs

For one tensor input, pass its shape without the batch dimension:

```python
summary(model.eval(), (3, 224, 224))
```

For multiple independent positional tensor inputs, pass a list of shapes:

```python
import torch.nn as nn
from torchscan import summary


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 2)
        self.right = nn.Linear(6, 2)

    def forward(self, left, right):
        return self.left(left) + self.right(right)


summary(TwoInputs().eval(), [(4,), (6,)])
```

The list maps to positional arguments in order. `crawl_module` also accepts one `dtype` for all generated inputs or an iterable with one dtype per input.

For correlated, integer, masked, or otherwise data-dependent inputs, pass the tensors themselves:

```python
import torch
import torch.nn as nn
from torchscan import summary


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def forward(self, xs, xs_len):
        positions = torch.arange(xs.shape[1], device=xs.device)
        mask = positions < xs_len[:, None]
        return self.embedding(xs) * mask.unsqueeze(-1)


xs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
xs_len = torch.tensor([2, 4])
summary(Encoder().eval(), input_data=(xs, xs_len))
```

A single tensor is one positional model argument. A non-empty list or tuple supplies positional arguments in order. Each tensor already includes every dimension, and TorchScan forwards it without cloning, casting, moving, detaching, or adding a batch dimension. Exactly one of `input_shape` and `input_data` is required; `dtype` is only valid with generated `input_shape` inputs. Keyword/dictionary inputs and non-tensor leaves are not supported.

## Outputs

Hooked modules may return tensors nested in tuples, lists, or dictionaries. Output shapes preserve that structure, dictionary keys and insertion order, and `None` leaves. Unsupported leaves raise a `TypeError` that identifies their path.

Metric calculators still receive one numerical output: the first tensor found in deterministic depth-first container order. Secondary tensors remain in the displayed shape structure but are not folded into FLOP, MAC, DMA, or receptive-field totals. Long structures are abbreviated with `[...]` in the summary table; `crawl_module()` retains the complete value. An output containing no tensor raises a `TypeError`.

This makes tuple-returning modules such as `torch.nn.MultiheadAttention` safe to inspect. Its FLOP calculator uses the complete output to distinguish averaged attention weights from `need_weights=False`.

## Synthetic execution

- A batch dimension of one is added to every generated input shape.
- Generated inputs use the first model parameter's device and, by default, its dtype.
- Caller-provided tensors retain their complete shape, value, device, and dtype.
- The forward pass runs under `torch.no_grad()` but preserves the model's current training mode.
- Calling `eval()` first is recommended because training-mode modules can update buffers.
- The model must contain at least one parameter.
- Exceptions from the model's forward pass are propagated.

## Metric capability

“Supported” means the calculator recognizes the module family. Some supported operations legitimately contribute zero or leave the receptive field unchanged.

| Module family | FLOPs | MACs | DMAs | Receptive field |
| --- | :---: | :---: | :---: | :---: |
| Identity and Flatten | ✓ | ✓ | ✓ | ✓ |
| Linear | ✓ | ✓ | ✓ | ✓ |
| ReLU, ELU, LeakyReLU, ReLU6, Tanh, Sigmoid | ✓ | ✓ | ✓ | ✓ |
| Conv1d/2d/3d and transposed convolutions | ✓ | ✓ | ✓ | ✓ |
| BatchNorm1d/2d/3d | ✓ | ✓ | ✓ | ✓ |
| Max, average, and adaptive pooling | ✓ | ✓ | ✓ | ✓ |
| Dropout | ✓ | ✓ | ✓ | ✓ |
| `torch.nn.LayerNorm` | ✓ | — | — | — |
| Batched `torch.nn.MultiheadAttention` | ✓* | — | — | — |
| `torch.nn.Transformer` | ✓* | — | — | — |

*`MultiheadAttention` supports 3D self- and cross-attention in both layouts, standard projection bias, and masks passed positionally. Keyword masks are invisible to hooks; unbatched inputs, `add_bias_kv`, and `add_zero_attn` are rejected.*

*A native `torch.nn.Transformer` passed directly to `summary()` is counted as one composite FLOP operation. The verified path covers native encoder/decoder stacks with ReLU, batched sequence-first or batch-first tensors, and positional attention and key-padding masks; masks hidden in keyword arguments cannot be counted. Wrappers and standalone native Transformer components are rejected rather than partially counted. Child-level FLOP detail, custom stacks or activations, arbitrary Transformer families, Hugging Face internals, and `einops` operations remain unsupported.*

## Custom and functional operations

A custom composite module is inspected through its supported child modules. A custom leaf module is not understood automatically: it emits unsupported-module warnings and contributes zero or a neutral receptive-field value.

Operations called through `torch.nn.functional`, tensor methods, or Python operators do not create module-hook events. They are therefore absent from the totals. For example, the addition in the multiple-input example runs normally but is not counted.

See [Understanding results](metrics.md) for the consequences when comparing estimates.
