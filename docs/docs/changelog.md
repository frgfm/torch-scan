# Changelog

## Unreleased

These changes require the development version on `main`. They are not in the 0.2.0 package.

### Added

- Scoped arithmetic, normalization, and attention FLOP formulas, plus GroupNorm. Independent formula checks and
  model tests that do not download weights (#165).
- Offline HTML and SVG module cost reports with `render_report` (#166).
- Per-analysis custom module handlers with complete call context, independent metric estimates, and explicit subtree
  ownership (#167).
- `custom_mapping` operator overrides on `crawl_module` and `summary`, separate from module estimates (#167).
- Native Transformer MACs, logical DMAs, and module-local token dependencies. Spatial receptive-field scalars remain
  unavailable on token modules (#168).

### Changed

- Correct convolution, linear, pooling, normalization, and attention counts. Keep incomplete counts and diagnostics
  visible (#165).

## v0.2.0 (2026-10-03)

Release note: [v0.2.0](https://github.com/frgfm/torch-scan/releases/tag/v0.2.0)

Version 0.2 is a clean contract break focused on truthful, machine-readable analysis.

### Added

- Versioned `AnalysisReport`, `MetricResult`, and `Diagnostic` contracts with complete, partial, and unavailable states.
- Stable full module paths and per-path call indexes.
- Complete `args` and `kwargs` forwarding, including nested containers and non-tensor leaves.
- `strict=True` and `IncompleteAnalysisError` for automation that rejects incomplete metrics.
- PyTorch-native operator FLOPs through `measure_flops`, with per-call custom formulas and uncounted-op diagnostics.
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
