# Agent quickstart

TorchScan gives coding agents a bounded, machine-readable way to inspect a PyTorch model. It does not decide whether
a model is acceptable, fast, or deployable.

## Default workflow

1. Import the existing model and construct representative inputs without downloading weights unless authorized.
2. Use `crawl_module(..., args=..., kwargs=...)` for a real call or `input_shape` for a simple tensor input.
3. Serialize the returned report as JSON; do not scrape `summary` text.
4. Check metric status and diagnostics before using any number.
5. Ask the owner for a threshold when the task involves a budget or pass/fail decision.
6. Preserve the report, model revision, and input configuration with the conclusion.

Prefer strict analysis when incomplete metrics must stop the task:

```python
import json

from torchscan import IncompleteAnalysisError, crawl_module

try:
    report = crawl_module(model, args=(inputs,), kwargs=model_kwargs, strict=True)
except IncompleteAnalysisError as error:
    raise RuntimeError(f"TorchScan could not complete the requested analysis: {error}") from error

print(json.dumps(report, sort_keys=True))
```

## Pick one API

| Task | Use |
| --- | --- |
| Inspect module structure and formula metrics | `crawl_module` |
| Show a table to a person and retain the report | `summary` |
| Count operator FLOPs for arbitrary code | `measure_flops` |
| Measure one workload's PyTorch peak memory | `measure_peak_memory` |
| Compare two compatible reports | `compare_reports` |

Do not create a parser around terminal output, a second report schema, a baseline database, or a project-specific
wrapper unless the project already requires one.

## Trust rules

- `complete`: use `value` with the report's method and context.
- `partial`: `known_value` is only a lower bound. Preserve diagnostics and do not extrapolate.
- `unavailable`: report that no measurement was produced.
- Numeric zero is meaningful only when status is `complete`.
- Module FLOPs and operator FLOPs are separate methods; never add or average them.
- Peak PyTorch memory is not process RSS or total accelerator use.
- A skipped or mocked CUDA/MPS check is not device validation.

## Owner-controlled budgets

TorchScan reports measurements. The owner supplies policy:

```python
memory = measure_peak_memory(workload, device=device)

if owner_budget_bytes is None:
    raise ValueError("Ask the model owner for a memory budget")
if memory["peak_bytes"] > owner_budget_bytes:
    raise RuntimeError("Owner-approved memory budget exceeded")
```

Record the hardware and workload state with accelerator results. Do not invent a default budget.

## When an operator is uncounted

1. Preserve the partial result and diagnostic.
2. Confirm the operator and its exact overload in the installed PyTorch version.
3. Rerun the equivalent workload with `measure_flops(..., custom_mapping=...)` only when the counting method is known
   and reviewable. `crawl_module` does not accept custom mappings.
4. Keep the formula with the experiment or project that owns the assumption.

See [Model and input support](model-support.md#custom-formulas) and [Methodology](methodology.md).
