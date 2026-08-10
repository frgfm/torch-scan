# Model and input support

TorchScan attaches forward hooks to `torch.nn.Module` objects and executes one synthetic forward pass. This makes module-based models easy to inspect, but it does not trace arbitrary tensor operations.

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

The list maps to positional arguments in order. `crawl_module` also accepts one `dtype` for all inputs or an iterable with one dtype per input.

TorchScan accepts shapes, not real input tensors. Inputs are generated independently with `torch.rand`, so correlated or data-dependent inputs cannot be represented. An `input_data` argument is not currently supported.

## Outputs

Hooked modules may return tensors nested in tuples, lists, or dictionaries. Output shapes preserve that structure, dictionary keys and insertion order, and `None` leaves. Unsupported leaves raise a `TypeError` that identifies their path.

Metric calculators still receive one numerical output: the first tensor found in deterministic depth-first container order. Secondary tensors remain in the displayed shape structure but are not folded into FLOP, MAC, DMA, or receptive-field totals. Long structures are abbreviated with `[...]` in the summary table; `crawl_module()` retains the complete value. An output containing no tensor raises a `TypeError`.

This makes tuple-returning modules such as `torch.nn.MultiheadAttention` safe to inspect. Native attention and Transformer metric coverage remains out of scope; tuple safety alone does not make those totals meaningful.

## Synthetic execution

- A batch dimension of one is added to every input shape.
- Inputs use the first model parameter's device and, by default, its dtype.
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
| `torch.nn.Transformer` | ✓* | — | — | — |

*The low-level FLOP calculator recognizes `torch.nn.Transformer`, but native attention and Transformer coverage in `summary()` is not complete. This does not imply meaningful totals for arbitrary Transformer or `einops` implementations.*

## Custom and functional operations

A custom composite module is inspected through its supported child modules. A custom leaf module is not understood automatically: it emits unsupported-module warnings and contributes zero or a neutral receptive-field value.

Operations called through `torch.nn.functional`, tensor methods, or Python operators do not create module-hook events. They are therefore absent from the totals. For example, the addition in the multiple-input example runs normally but is not counted.

See [Understanding results](metrics.md) for the consequences when comparing estimates.
