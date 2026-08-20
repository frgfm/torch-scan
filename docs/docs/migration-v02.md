# Migrating to v0.2

Version 0.2 intentionally replaces the 0.1 crawler contract. It removes ambiguous memory snapshots and numeric zeroes
for unsupported work.

## Requirements

- Python 3.11 or newer.
- PyTorch 2.1 or newer and earlier than 3.0.

## API replacements

| Before v0.2 | v0.2 |
| --- | --- |
| `crawl_module(..., input_data=tensor)` | `crawl_module(..., args=(tensor,))` |
| `input_data=(left, right)` | `args=(left, right)` |
| No keyword/non-tensor input support | `args=(...)`, `kwargs={...}` |
| Unversioned `dict[str, Any]` | Versioned `AnalysisReport` |
| `report["overall"]["flops"]` | Structured metric in `report["totals"]`; check status before its numeric field. |
| Unsupported module contributes zero | Affected metric is partial/unavailable with diagnostics. |
| `get_process_gpu_ram(pid)` | `measure_peak_memory(workload, device=...)` |
| `report["overheads"]` | Removed; no replacement because the heuristic did not isolate model memory. |
| Manual before/after arithmetic | `compare_reports(before, after)` |

`summary` still prints a table, but now returns the same `AnalysisReport` used for automation.

## Input migration

```python
# v0.1 development API
report = crawl_module(model, input_data=(input_ids, attention_mask))

# v0.2
report = crawl_module(
    model,
    args=(input_ids,),
    kwargs={"attention_mask": attention_mask},
)
```

Do not pass `dtype` or `device` with `args`/`kwargs`; real call values are forwarded unchanged. Use those options only
with generated `input_shape` inputs.

## Result migration

Replace direct arithmetic on raw totals with an explicit status branch:

```python
metric = report["totals"]["module_flops"]

if metric["status"] == "complete":
    use(metric["value"])
elif metric["status"] == "partial":
    report_lower_bound(metric["known_value"], report["diagnostics"])
else:
    report_unavailable(report["diagnostics"])
```

For automation that cannot accept lower bounds, call `crawl_module(..., strict=True)` and handle
`IncompleteAnalysisError`.

## FLOP migration

Module-formula FLOPs and operator-dispatch FLOPs are separate metrics in v0.2. Use `measure_flops` when functional
operations, tensor methods, or custom modules matter. `crawl_module` also exposes the same-forward native report under
`operator_flops` and its total under `totals.operator_flops`. Do not add the two totals or treat agreement as proof
that every operation is covered.

## Memory migration

```python
from torchscan.process import measure_peak_memory

stats = measure_peak_memory(lambda: model(inputs), device=inputs.device)
```

The result is backend-specific PyTorch memory. It is not process RSS or total device memory. There is no automatic
replacement for `overheads` because its allocator/process subtraction was not a reliable model measurement.
