# Report schema

TorchScan reports are JSON-serializable public `TypedDict` values. `schema_version` versions their wire shape
independently of the package version.

## `AnalysisReport`

The model report contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer wire-schema version. Version 0.2 starts at `1`. |
| `context` | Version, model, execution-mode, prior training-state, device, and dtype metadata. |
| `inputs` | Input `source` plus recursive `args` and `kwargs` metadata; never tensor values. |
| `layers` | Ordered module-call records with stable full path and call index. |
| `totals` | Structured model-wide metric results, separated by method. |
| `operator_flops` | Complete standalone `FlopReport` collected during the same forward pass. |
| `diagnostics` | Stable machine codes and human-readable details for incomplete or notable work. |

`context` contains `torchscan_version`, `torch_version`, `python_version`, `model_type`, `execution_mode`,
`training_before`, `devices`, and `dtypes`.

With `mode="structure"`, `context.analysis_mode` is `"structure"`. Layers contain the `calls` metric, and compute
totals (`module_flops`, `operator_flops`, `macs`, `dmas`) are unavailable with method `not_requested`. The
`operator_flops` report has empty operator/module mappings and diagnostics. `strict=True` checks requested metrics,
so intentionally unrequested compute totals do not raise. Compare reports collected in the same mode; mixed modes
have incompatible compute methods.

`totals` contains `parameters`, `trainable_parameters`, `frozen_parameters`, `parameter_bytes`, `buffer_elements`,
`buffer_bytes`, `module_flops`, `macs`, `dmas`, and the separate `operator_flops` result.

The report is deterministic for the same model, call, software versions, and execution behavior. JSON object key
ordering is not a compatibility contract; consumers should address keys by name.

## `MetricResult`

Each metric result contains `status`, `value`, and `known_value`, plus its unit, scope, and method:

```json
{"status": "complete", "value": 42, "known_value": 42, "unit": "FLOPs", "scope": "forward", "method": "operator_dispatch"}
```

```json
{"status": "partial", "value": null, "known_value": 42, "unit": "FLOPs", "scope": "forward", "method": "operator_dispatch"}
```

An unavailable result has both numeric fields set to `null`. Consumers must branch on `status` before reading a
numeric field. A complete value of zero remains `{"status": "complete", "value": 0, "known_value": 0, ...}`.

## `Diagnostic`

A diagnostic contains `code`, `severity` (`warning` or `error`), `metric`, and `message`, plus `path` or `operator`
when applicable. Codes are suitable for program branches; messages are for people and may gain detail.

Diagnostics are not limited to errors. They can record unsupported operations, method limitations, or execution
context that changes interpretation.

## Input metadata

Recursive metadata preserves tuples, lists, mappings, scalars, and `None`. Tensor metadata includes shape, dtype,
device, and `requires_grad`. It excludes tensor contents, filenames, source paths, and object representations that can
leak private values.

## Layer calls

Each layer record contains:

- `path` and a zero-based `call_index` for stable identity.
- `name`, `depth`, and `type` for display.
- Recursive `input` and `output` metadata.
- `parameters` and `buffers` statistics.
- Structured `module_flops`, `macs`, `dmas`, `receptive_field`, `effective_stride`, and `effective_padding` results
  under `metrics`.

Shared parameters are not duplicated in model totals merely because a module is called more than once.

Custom handler rows may also contain optional ownership metadata:

| Field | Meaning |
| --- | --- |
| `metric_ownership` | Metric name to `"module_call"` or `"subtree"` for the scope supplied by a registered handler. |
| `metric_owners` | Covered metric name to the ancestor owner's `{ "path": ..., "call_index": ... }` identity. |

A covered child's metric is absent from `metrics`, because the ancestor supplies that inclusive estimate. Its layer
row, inputs/outputs, and parameter/buffer statistics remain present. Absence with `metric_owners` means covered work,
not zero or unsupported work. Ownership follows dynamic invocations, so a shared child called outside the owning
parent is counted independently. Sum retained estimates once; parameter totals are unaffected.

Custom results use method `custom_module_handler:<callback module>.<callback qualname>`, optionally followed by the
callback's supplied method, and normalized scope `module_call` or `subtree`. Partial and unavailable custom estimates,
invalid results, and callback failures have diagnostics. Ownership remains in force when its inclusive estimate is
incomplete; children do not silently become a substitute estimate. See [Custom module extensions](extensions.md).

| Extension diagnostic code | Meaning |
| --- | --- |
| `custom_handler_error` | The callback raised; requested estimates are unavailable. |
| `custom_handler_invalid` | The callback result is not a mapping of supported metric names. |
| `custom_metric_invalid` | One supplied metric violates its numeric or structured-result contract. |
| `custom_metric_partial` | A supplied estimate is a lower bound. |
| `custom_metric_unavailable` | An estimate is explicitly unavailable, including an omitted owned metric. |

## `FlopReport`

`measure_flops` returns:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` for the v0.2 wire shape. |
| `context` | `torch_version`, the counting `method`, and `counting_convention` for measured FLOPs. |
| `total` | Complete or partial FLOP `MetricResult`. |
| `by_module` | Best-effort known FLOPs keyed by PyTorch's upstream module labels when available. |
| `by_operator` | Known FLOPs keyed by normalized operator packet such as `aten.mm`. |
| `ignored_operators` | Explicit zero-FLOP metadata, movement, or allocation operators with call count and reason. |
| `diagnostics` | `uncounted_operator`, `unsupported_operator_formula`, `incomplete_operator_formula`, and other method limitations. |

Every non-ignored call needs a supported formula for completeness. Fused or unsupported calls retain diagnostics,
even when the same packet has counted work.

`crawl_module` stores the complete report under `operator_flops` and the same `total` result under
`totals.operator_flops`. Module hooks and the native operator counter observe the same forward call; module formulas
run in post-hooks with dispatch suspended so their own bookkeeping and custom callback tensor operations are not
counted. Intermediate activations are released as the forward pass progresses; the report retains only metadata.

`custom_mapping` on `crawl_module` or `summary` configures this operator report independently from module handlers.
Neither extension changes a global registry. `mode="structure"` executes no formulas or handlers.

`Global`/`total` is authoritative. `crawl_module` does not request PyTorch 2.1's explicit module tracker because that
tracker replaces tensors passed through its hooks; preserving the caller's exact `args` and `kwargs` takes priority.
Consequently, `by_module` can be empty on older supported PyTorch versions. Standalone `measure_flops(modules=...)`
requests native hierarchical attribution explicitly.
Upstream `by_module` labels are not stable layer identities, parent entries include their children, and rows must not be
joined to `layers` or summed. Use the global `total` for comparison.

## `ReportDiff`

`compare_reports(before: object, after: object)` requires matching schema versions and returns:

```json
{
  "schema_version": 1,
  "totals": {
    "module_flops": {
      "status": "complete",
      "delta": 12,
      "before": {"status": "complete", "value": 30, "known_value": 30},
      "after": {"status": "complete", "value": 42, "known_value": 42}
    }
  },
  "layers": {
    "added": [],
    "removed": [],
    "changed": []
  }
}
```

Layer calls match by full path plus call index. Added and removed entries contain `path`, `call_index`, and `metrics`;
changed entries contain the identity and only changed metrics. Each metric diff includes `status`, `delta`, `before`,
and `after`. `delta` is numeric only when both inputs are complete. Partial, unavailable, or missing metrics produce a
`null` delta and propagate incomplete state.

## Compatibility rule

Reject an unknown `schema_version` rather than guessing its meaning. Use [`compare_reports`](#reportdiff)
for same-schema reports and migrate stored reports explicitly when a future schema changes.
