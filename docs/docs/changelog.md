# Changelog

## Unreleased (0.2.0)

Version 0.2 is a clean contract break focused on truthful, machine-readable analysis.

### Added

- Versioned `AnalysisReport`, `MetricResult`, and `Diagnostic` contracts with complete, partial, and unavailable states.
- Stable full module paths and per-path call indexes.
- Complete `args` and `kwargs` forwarding, including nested containers and non-tensor leaves.
- `strict=True` and `IncompleteAnalysisError` for automation that rejects incomplete metrics.
- PyTorch-native operator FLOPs through `measure_flops`, with per-call custom formulas and uncounted-op diagnostics.
- Invocation-scoped formulas for arithmetic, reductions, normalization, softmax, and dense CPU attention; GroupNorm
  module FLOPs; independent formula checks and a no-download model matrix.
- Pure `compare_reports` before/after comparison.
- Explicit workload peak-memory measurement through `measure_peak_memory` (#149).
- Opt-in `mode="structure"` for inexpensive shapes, call metadata, parameters, and buffers.
- Structured crawler output (#143), a `Trainable` summary column (#144), caller-provided tensors (#145), native
  Transformer FLOP formulas (#146), and raw structured compute totals (#148) during the v0.2 development cycle.

### Changed

- Require Python 3.11+ and PyTorch 2.1+.
- Run model crawling under evaluation mode with gradients disabled, then restore every original module training flag.
- Make crawler bookkeeping linear in layer-call count and cache forward signatures per module (#147, #152).
- Separate module-formula metrics from operator-dispatch FLOPs.
- Correct transposed-convolution FLOP/MAC geometry, unbatched convolution channels, BatchNorm statistics work,
  dropout execution state, and stable attention softmax arithmetic. Keep native fused attention core counts partial.
- Respect reduction arithmetic dtype, BatchNorm buffers, zero-width linear bias, and equivalent pooling kernels.
  Keep sparse/nested shape counts, empty normalization groups, and custom Transformer final norms incomplete.
- Label partial benchmark cells as lower bounds and unavailable cells explicitly; display both FLOP methods.
- Compute module estimates in post-hooks and release activations during the forward pass.
- Share forward signatures across identical implementations and avoid binding fully supplied positional inputs.
- Generate integer/boolean inputs without an FP32 temporary and normalize operator names once per distinct packet.
- Cache installed package-version metadata across scans.
- Modernize packaging, CI, and MkDocs Material documentation.

### Removed

- `input_data`; use `args` and `kwargs`.
- `get_process_gpu_ram`, report `overheads`, and automatic accelerator cache clearing.
- The unversioned legacy crawler report.

See the [v0.2 migration guide](migration-v02.md) for code changes and trust semantics.

## v0.1.2 (2022-08-03)

Release note: [v0.1.2](https://github.com/frgfm/torch-scan/releases/tag/v0.1.2)

## v0.1.1 (2020-08-04)

Release note: [v0.1.1](https://github.com/frgfm/torch-scan/releases/tag/v0.1.1)

## v0.1.0 (2020-05-21)

Release note: [v0.1.0](https://github.com/frgfm/torch-scan/releases/tag/v0.1.0)
