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
- `mode="structure"` on either API: hierarchy, shapes, calls, parameters, and buffers with less overhead.
- `measure_flops(workload)`: operator FLOPs for one zero-argument workload call.
- `measure_peak_memory(workload, device=...)`: backend-specific PyTorch peak memory.
- `measure_peak_rss(command)`: Linux/macOS child-process lifetime RSS, including loading and imports.
- `profile_workload(workload, device=...)`: one instrumented operator diagnostic pass; not clean latency.
- `measure_latency(workload, device=..., inputs=...)`: first-call time, warmed block-average timing, and explicit
  work-unit throughput. Unreleased; install `main`. The callable is invoked repeatedly and owns its state.
- `compare_reports(before, after)`: pure same-schema comparison.
- `compare_benchmarks(before, after, check=...)`: compatible workload comparison with an owner-supplied output check.
- `render_report(report)`: offline model HTML/SVG or benchmark/comparison HTML, without remeasurement.

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
- Structure mode's compute totals have method `not_requested`; strict checks cover requested metrics only.
- Keep module FLOPs and operator FLOPs separate.
- Peak PyTorch memory is not process RSS or total device memory.
- Mocked or skipped CUDA/MPS checks are not hardware evidence.
- Timing inputs are caller-supplied metadata. Block-average latency is not request p95, and first-call time is not
  model loading or fresh-process startup. Use `compare_benchmarks` for timing and `compare_reports` for model estimates.
- A passed output check does not establish task accuracy. Failed checks withhold benchmark deltas; IQR labels are
  descriptive, not statistical significance. Preserve methods, memory scopes, hardware, and raw timing evidence.

For an uncounted operator, preserve the partial result. Supply `custom_mapping` to `crawl_module`, `summary`, or
`measure_flops` only when the owner can justify that operator's counting convention. For custom module estimates,
use per-analysis `custom_modules={ModuleType: ModuleHandler(callback)}`. Callbacks receive a complete `ModuleCall`;
declare inclusive subtree ownership per metric to avoid double-counting children. Keep the module and operator views
separate. Do not create a global registry, baseline store, wrapper service, or automatic budget policy.

In a repository checkout, read `../../../docs/docs/agent-quickstart.md` for the full workflow and
`../../../docs/docs/report-schema.md` for the report contract, and `../../../docs/docs/extensions.md` for copyable
extension examples. Outside a checkout, use the published documentation at
`https://frgfm.github.io/torch-scan/`.
