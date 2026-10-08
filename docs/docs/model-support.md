# Model and input support

`crawl_module` and `summary` attach hooks to `torch.nn.Module` objects and execute one forward pass. The same pass also
uses PyTorch operator dispatch for operator FLOPs. Module metrics and operator FLOPs remain separate because they use
different methods and coverage.

## Generated inputs

For one tensor input, pass its shape without the batch dimension:

```python
report = crawl_module(model, (3, 224, 224))
```

For multiple positional tensors, pass a list of shapes:

```python
report = crawl_module(model, [(4,), (6,)])
```

One `dtype` applies to every generated input; an iterable supplies one dtype per shape. Lengths must match exactly.
Use `device=` to select a generated-input device. Without explicit values, TorchScan infers dtype and device from the
first parameter, then the first buffer, then defaults to CPU and `torch.float32`. Parameterless models are supported.
Generated integer inputs are zero and boolean inputs are true, allocated without a random FP32 temporary. These
inputs do not advance the random generator; provide `args`/`kwargs` for representative IDs and masks.

## Complete Python calls

Use `args` and `kwargs` for masks, integer IDs, correlated inputs, scalars, `None`, and nested containers:

```python
import torch

input_ids = torch.tensor([[1, 2, 0, 0]])
attention_mask = input_ids.ne(0)

report = crawl_module(
    model,
    args=(input_ids,),
    kwargs={"attention_mask": attention_mask, "return_dict": True},
)
```

Values are forwarded without cloning, casting, moving, detaching, or adding a batch dimension. Recursive input
metadata preserves container structure and records tensor shape, dtype, device, and gradient requirement, but never
tensor values.

Generated `input_shape` and caller-provided `args`/`kwargs` are mutually exclusive. `dtype` and `device` only configure
generated inputs. A kwargs-only call is valid.

## Execution state

TorchScan:

1. Records every module's training flag.
2. Switches modules to evaluation mode.
3. Runs one forward pass under `torch.no_grad()`.
4. Removes hooks and restores every original training flag, including after an exception.

The model owns every other side effect. TorchScan does not move the model, seed random generators, warm up kernels,
clear accelerator caches, or suppress forward errors. Use `measure_flops` or `measure_peak_memory` for training and
other owner-controlled workloads.

## Outputs and repeated modules

Hooked modules may return tensors nested in tuples, lists, or mappings. Output metadata preserves the recursive
structure. Stable layer identities use the full module path plus a call index, so a shared module invoked more than
once produces distinct layer-call records.

## Two metric views

### Module estimates

Hooks provide layer structure, parameters, buffers, MACs, DMAs, receptive field, and formula-based module FLOPs for
supported module families. An unsupported leaf makes affected metrics `partial` or `unavailable`; it never contributes
a fabricated zero.

### Operator FLOPs

PyTorch dispatch observes functional calls, tensor methods, Python operators backed by PyTorch operations, and
operations inside custom modules. Registered formulas produce counts. Executed operators without formulas are listed
in diagnostics and make the operator result partial.

Operator FLOPs do not make MAC, DMA, or receptive-field module formulas complete. TorchScan never merges the two FLOP
methods into one total.

### Native attention and Transformer stacks

Exact native `MultiheadAttention`, encoder/decoder layers and stacks, and `Transformer` support independent MACs,
logical DMAs, and module-local token dependencies for documented dense evaluation configurations. Pass all query,
key/value, memory, masks, and relevant options through `args`/`kwargs`, including when native modules are wrapped in
custom containers. Both batch layouts and unequal attention sequence lengths are supported. Native subtree formulas
prevent double-counting and cover fused paths that bypass child hooks. See [Native Transformer estimates](transformers.md)
for complete boundaries and a copyable example. Arbitrary custom/einops attention graphs remain outside this native
support; recognized operator FLOPs alone do not supply their MAC/DMA or dependency estimates.

### Modern activations and normalization

The development version supports GELU (exact/tanh), SiLU, GLU, optional native RMSNorm, and GroupNorm MAC/DMA
estimates. Use native classes with unchanged forwards; custom composites need handlers. See the
[counting conventions](methodology.md#flop-conventions) for formulas and spatial limits.

## Custom formulas

Supply custom operator formulas through `custom_mapping` on `crawl_module`, `summary`, or standalone `measure_flops`.
Overrides apply to that analysis and take precedence over built-in formulas without modifying a global registry.
Formulas use PyTorch's shape-based `FlopCounterMode` contract:

```python
from math import prod

report = measure_flops(
    lambda: torch.sin(inputs),
    custom_mapping={torch.ops.aten.sin: lambda input_shape, *, out_shape: prod(out_shape)},
)
```

See [FLOP conventions and limits](methodology.md#flop-conventions) before supplying overrides.

Use operator packets such as `torch.ops.aten.sin`, not overloads such as `.default`, and check the installed PyTorch
version's documentation when defining formulas.

For custom leaves or composite modules/models, pass `custom_modules={ModuleType: ModuleHandler(callback)}`. The
callback receives a `ModuleCall` with the module, actual complete `args`/`kwargs`, and complete output, including nested
structures and non-tensor values. Supplied metric fields override built-in estimates independently. Type matching
selects the closest registered class in the Python MRO, regardless of mapping order.

Use `subtree_metrics` to declare inclusive estimates that own child work for specific metrics. Covered child calls
remain in the structure and parameter report, while their covered estimates are omitted to prevent double-counting.
Both extension mappings are ignored for compute in `mode="structure"`: no handler or operator formula executes.
See the [extension tutorial](extensions.md) for callbacks, ownership, incomplete states, complex counting, and limits.
