# Extend module and operator analysis

Pass `custom_modules` to `crawl_module` or `summary` to analyze your own `torch.nn.Module` types. Registrations belong
to that analysis only: they do not change TorchScan's formulas or require your model library as a TorchScan dependency.
Use `custom_mapping` on the same call when you also need operator FLOP overrides. The two views remain separate.

## A complete custom call

A `ModuleHandler` wraps one callback taking a `ModuleCall` and returning `ModuleEstimates`. The frozen `ModuleCall`
contains the actual `module`, positional `args`, keyword `kwargs`, and complete `output`. Nested containers,
non-tensor values, and kwargs-only calls are preserved. These are the objects from that invocation, rather than tensor
metadata or a flattened input list; omitted default arguments are not inserted into `args` or `kwargs`.

The following example needs only PyTorch and TorchScan. It counts complex scalar multiplication using four real
multiplies and two real additions per complex element: **six real FLOPs**. A complex addition would count as two real
FLOPs. This conventional algorithm does not assume fused instructions or the three-multiply complex algorithm.

```python
from math import prod

import torch
from torch import nn
from torchscan import ModuleCall, ModuleEstimates, ModuleHandler, crawl_module


class ComplexScale(nn.Module):
    def forward(self, signal, *, gain: complex, tag: str = "example"):
        return {"samples": (signal * gain,), "tag": tag}


def estimate_complex_scale(call: ModuleCall) -> ModuleEstimates:
    # Complete output and kwargs are available, including the non-tensor tag.
    signal = call.args[0] if call.args else call.kwargs["signal"]
    output = call.output["samples"][0]
    assert signal.dtype.is_complex and output.shape == signal.shape
    assert call.output["tag"] == call.kwargs.get("tag", "example")
    assert isinstance(call.kwargs["gain"], complex)
    elements = output.numel()
    return {
        "module_flops": 6 * elements,
        "macs": 0,
        "dmas": 2 * elements,
        "receptive_field": 1,
        "effective_stride": 1,
        "effective_padding": 0,
    }


def complex_mul_flops(*input_shapes, out_shape, **kwargs):
    return 6 * prod(out_shape)


signal = torch.ones(2, 4, dtype=torch.complex64)
report = crawl_module(
    ComplexScale(),
    args=(signal,),
    kwargs={"gain": 1 + 2j, "tag": "demo"},
    custom_modules={ComplexScale: ModuleHandler(estimate_complex_scale)},
    custom_mapping={torch.ops.aten.mul: complex_mul_flops},
    strict=True,
)
assert report["totals"]["module_flops"]["value"] == 48
assert report["totals"]["operator_flops"]["value"] == 48
```

Here `macs=0` is intentional: standalone multiplication does not perform an accumulation. A complex dot product needs
its own stated MAC convention; one complex multiply-accumulate is not automatically one real MAC. The DMA formula
counts one logical tensor-element read and one write per element, excluding the scalar gain. DMAs are neither bytes
nor measured memory transactions, and this example does not model caching, vectorization, or intermediate storage.
Receptive-field values describe an elementwise operation.

The operator override applies to **every `aten.mul` call in this analysis**. Its shape formula assumes those calls are
complex multiplications. Do not reuse it unchanged for a model containing real, integer, or boolean multiplications.
PyTorch shape formulas do not generally expose dtype. A module callback can inspect actual tensor dtypes and choose a
formula for each call. Complex support is supplied explicitly here; registering a model does not make every complex
operator complete.

Replace `crawl_module` with `summary` to print the table and receive the same report; both accept `custom_modules` and
`custom_mapping`. Retain the formulas and their conventions with your report.

## Match types and override fields

`custom_modules` maps module classes to `ModuleHandler` instances. Matching uses the closest registered class in the
module's Python method resolution order: an exact class wins, then its nearest registered base class. Mapping insertion
order does not affect selection. This also defines precedence for multiple inheritance. Virtual subclasses outside
the MRO do not match. A caller handler takes precedence over a built-in formula for every field it supplies.

The callback may return any subset of these independent fields:

| Field | Unit | Meaning |
| --- | --- | --- |
| `module_flops` | `FLOPs` | Formula-based floating-point operations. |
| `macs` | `MACs` | Multiply-accumulations under your stated convention. |
| `dmas` | `DMAs` | Logical direct memory accesses under your stated convention. |
| `receptive_field` | `elements` | Receptive-field extent for the module's operation. |
| `effective_stride` | `elements` | Effective input stride. |
| `effective_padding` | `elements` | Effective input padding. |

A numeric field is a complete estimate, including a legitimate zero. A `MetricResult` preserves complete, partial, or
unavailable state; `None` explicitly means unavailable. Omission is different: on a formula leaf, an omitted field uses
the built-in formula, including its unsupported-work diagnostics. On an ordinary registered composite, an omitted,
unowned field delegates to descendants. When a built-in complete-call handler exists, omitted fields use that handler's
estimates and ownership scope before legacy leaf fallback or child delegation. Inclusive built-in fallbacks cover
descendants only for the fields they supply; a caller's module-local field still adds to those children.
An explicitly supplied `None` or unavailable result never falls back to a built-in zero or to children.

Callbacks run at most once per successfully completed matching invocation; an ancestor owning all fields skips a
descendant callback. A partially covered callback receives the full context, but its covered fields are ignored.
A reused module receives a new call context each time it is estimated. Registration is
for module types, not individual instances; branch on `call.module` attributes if instances need different formulas.

## Own an inclusive subtree explicitly

By default, each supplied estimate covers only the registered module's own call. Children are still counted. If your
formula includes children, name its fields in `ModuleHandler(..., subtree_metrics=frozenset({...}))`. Ownership is
independent for each metric, including receptive-field fields.

This example extends the definitions above. It owns inclusive FLOPs and DMAs while delegating MACs and
receptive-field estimates to the two children:

```python
class ComplexPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = ComplexScale()
        self.second = ComplexScale()

    def forward(self, signal):
        first = self.first(signal, gain=1 + 2j)["samples"][0]
        return self.second(first, gain=2 + 3j)["samples"][0]


def estimate_pair(call: ModuleCall) -> ModuleEstimates:
    elements = call.output.numel()
    return {"module_flops": 12 * elements, "dmas": 4 * elements}


pair_report = crawl_module(
    ComplexPair(),
    args=(signal,),
    custom_modules={
        ComplexScale: ModuleHandler(estimate_complex_scale),
        ComplexPair: ModuleHandler(
            estimate_pair,
            subtree_metrics=frozenset({"module_flops", "dmas"}),
        ),
    },
    custom_mapping={torch.ops.aten.mul: complex_mul_flops},
)
assert pair_report["totals"]["module_flops"]["value"] == 96
assert pair_report["totals"]["macs"]["value"] == 0
```

Ownership follows executed calls: it covers descendants invoked during that parent invocation. If a shared child is
called elsewhere, that other call remains independently counted. Descendant layer rows, their inputs/outputs, and
parameter statistics remain in the report. Covered metric fields are omitted from those rows; `metric_owners`
identifies the owning ancestor's `path` and `call_index`. The owning row records `metric_ownership` as `"subtree"`;
ordinary supplied estimates use `"module_call"`. Sum the retained estimates once, without inventing zeros for covered
fields. Model parameter totals still count shared parameters only once.

If a caller overrides some fields while other fields fall back to an inclusive built-in handler, descendant estimates
may run before that parent callback returns. The selected parent fields then replace those contributions and their
estimation diagnostics. This needs no retained child activations or extra model execution; it can perform additional
estimation bookkeeping. Explicit caller subtree ownership suppresses child estimates before they execute.

Declaring ownership is a commitment to that scope. If the parent omits an owned field, returns an unavailable value,
or its callback fails, the children remain covered and the owned result is incomplete. TorchScan does not silently
substitute child estimates for a failed inclusive formula. Likewise, a parent formula covering only its own work must
not declare subtree ownership.

Delegation only describes descendant estimates. Functional work performed directly by the composite needs its own
estimate for each affected metric; operator FLOPs may observe that work separately, but cannot supply missing MACs,
DMAs, or receptive-field estimates.

## Partial results, validation, and failure

Use `MetricResult` when you know only a lower bound. Continuing from the examples above:

```python
def lower_bound_scale(call: ModuleCall) -> ModuleEstimates:
    elements = call.output["samples"][0].numel()
    return {
        "module_flops": {
            "status": "partial",
            "value": None,
            "known_value": 4 * elements,
            "unit": "FLOPs",
            "scope": "module_call",
            "method": "complex_real_multiplies_only",
        },
        "macs": 0,
        "dmas": None,
    }
```

The four real multiplies are a lower bound, not a complete complex multiply count or a coverage percentage. Missing
additions keep the result partial. TorchScan supplies diagnostics for explicitly incomplete estimates and labels the
method `custom_module_handler:<callback module>.<callback qualname>`, appending your result's method when supplied.
The normalized scope is `module_call` or `subtree`, as declared by the handler.

Results must be mappings with supported field names; a non-mapping result or unsupported name invalidates the whole
callback result. A structured estimate must contain exactly the six `MetricResult` fields, with nonempty `scope` and
`method` strings. Counts and lower bounds must be finite, nonnegative real numbers; booleans are invalid. A complete
`MetricResult` requires equal numeric `value` and `known_value`. A partial
result requires `value=None` and numeric `known_value`; an unavailable result requires both numeric fields to be
`None`. The unit must match the field. Invalid fields receive diagnostics and unavailable results instead of
unexplained zeros. Callback exceptions also produce diagnostics and incomplete estimates. Other valid fields can
remain complete when one estimate is invalid.

`strict=True` raises `IncompleteAnalysisError` for incomplete requested analysis, including handler failures and
uncounted operator work. The exception's `report` retains the evidence. Supplying module estimates alone does not
guarantee strict success: inspect the separate operator report too.

## Callback scope and limitations

- Callbacks execute immediately in the module's post-hook, under `torch.no_grad()` with operator counting suspended.
  Tensor operations used for estimation are excluded from measured workload FLOPs. The model still executes once.
- Treat the module, inputs, containers, and output as read-only. The frozen call wrapper does not make the referenced
  objects immutable. Do not call `forward`, mutate tensors or model state, or install hooks from a callback.
- The report retains metadata, not tensor values. TorchScan releases temporary call references after estimation;
  retaining an activation in your callback's own state can retain its storage and is your responsibility.
- TorchScan removes its hooks and restores every original training flag after analysis, including after errors.
  Calls sharing a model instance must remain serialized.
- A failed forward does not invoke its callback. If the enclosing model catches that failure and continues, its row
  records output kind `failed`, unavailable estimates, and a `module_forward_error` diagnostic. Later sibling calls
  are counted independently. A successfully returned `None` remains a complete callback output.
- `mode="structure"` runs the forward pass for structural reporting but executes neither module handlers nor custom
  operator formulas. Compute metrics remain explicitly unrequested.
- Operator callbacks follow the installed PyTorch counter's contract. PyTorch 2.1 accepts shape callbacks only;
  native raw-tensor callbacks, including callbacks for nested tensors, require a newer counter.
- These are theoretical formulas for one observed call. They do not establish latency, hardware traffic, autograd
  cost, or correctness for unseen shapes and branches. Only declare complete coverage you can justify.
- Handlers run only for observed module calls. Functional operations and fused kernels may use child parameters
  without invoking a child's `forward`; a child handler cannot intercept that work. Native attention and Transformer
  handlers own their documented estimates across their subtree, while observed child calls remain in the structure.
  Register the enclosing model and declare ownership for the metrics whose formulas include its children. See
  [Native Transformer estimates](transformers.md) for complete-call fallback boundaries and token dependencies.
- An observed descendant with a matching registration opens an otherwise atomic legacy root when the root has no
  registered enclosing handler. The inclusive root formula is
  excluded to prevent double counting. Missing root work produces `expanded_atomic_boundary` diagnostics and partial
  totals with descendant counts as lower bounds. Register the root too to supply its own or inclusive estimates.
  Without a matching caller registration, the original atomic boundary remains.
- Atomic implementations can use child parameters without invoking their modules. Parameter totals stay unique,
  but storage can be attributed to the enclosing atomic row instead of its child rows.
- Third-party complex-model packages are optional. Register their module classes in your application or experiment;
  TorchScan does not import or install them.

For the wire contract see [Report schema](report-schema.md). For the built-in conventions and remaining coverage
limits see [Methodology](methodology.md#flop-conventions).
