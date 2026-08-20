---
name: torchscan
description: Inspect and compare PyTorch models with TorchScan reports, operator FLOPs, and peak-memory workloads. Use when an agent must analyze model structure, parameters, compute, memory, regressions, unsupported operations, or an owner-provided model budget without inventing completeness.
license: Apache-2.0
compatibility: Requires Python 3.11+, PyTorch 2.1+, and the torchscan package. Accelerator claims require matching real hardware.
metadata:
  author: frgfm
  version: "0.2"
---

# TorchScan

Use the smallest API that answers the request:

- `crawl_module(...)`: JSON-serializable module report.
- `summary(...)`: printed table plus the same report.
- `measure_flops(workload)`: operator FLOPs for one zero-argument workload call.
- `measure_peak_memory(workload, device=...)`: backend-specific PyTorch peak memory.
- `compare_reports(before, after)`: pure same-schema comparison.

## Workflow

1. Reuse the project's model and representative inputs. Do not download weights without permission.
2. Prefer `args` and `kwargs` for real calls; use `input_shape` only for simple synthetic tensors.
3. Use `strict=True` when incomplete module metrics must stop automation.
4. Serialize the report directly. Never parse the `summary` table.
5. Check every metric's `status` and preserve diagnostics.
6. Ask the owner for thresholds. TorchScan measures; it does not decide whether a model fits.

## Truth rules

- `complete`: use `value` with its method, unit, scope, and context.
- `partial`: `known_value` is only a lower bound; do not extrapolate.
- `unavailable`: report that no measurement was produced.
- Zero is valid only with `status == "complete"`.
- Keep module FLOPs and operator FLOPs separate.
- Peak PyTorch memory is not process RSS or total device memory.
- Mocked or skipped CUDA/MPS checks are not hardware evidence.

For an uncounted operator, preserve the partial result. Rerun the equivalent workload with
`measure_flops(..., custom_mapping=...)` only when the owner can justify that operator's counting convention;
`crawl_module` does not accept custom mappings. Do not create a global registry, baseline store, wrapper service, or
automatic budget policy.

In a repository checkout, read `../../../docs/docs/agent-quickstart.md` for the full workflow and
`../../../docs/docs/report-schema.md` for the report contract. Outside a checkout, use the published documentation at
`https://frgfm.github.io/torch-scan/`.
