# Installation

Install PyTorch for your CPU, CUDA, or MPS system first, following the
[PyTorch installation guide](https://pytorch.org/get-started/locally/). TorchScan uses the installed backend;
it does not move your model or provision hardware. Choose the stable release or the development version:

| Track | Version | Python | PyTorch |
| --- | --- | --- | --- |
| Stable | 0.2.0 | ≥3.11,<4 | ≥2.1,<3 |
| Development (`main`) | Unreleased changes after 0.2.0 | ≥3.11,<4 | ≥2.1,<3 |

Version 0.2.0 was released on October 3, 2026. Use the [migration guide](migration-v02.md) when moving from 0.1.
The API reference covers unreleased development APIs, including the #176 preview. **0.3.0** is proposed and unpublished.

| Feature | Available in |
| --- | --- |
| Structured reports, `args`/`kwargs`, nested outputs, strict checks, structure mode, and report comparison | Stable 0.2.0 and `main` |
| Native Transformer module FLOPs and callable peak-memory measurement | Stable 0.2.0 and `main` |
| Offline HTML/SVG reports with `render_report` | `main` only; unreleased |
| First-call and warmed workload timing with `measure_latency` | `main` only; unreleased |
| Child process RSS with `measure_peak_rss` and operator diagnostics with `profile_workload` | `main` only; unreleased |
| Checked benchmark comparisons and offline workload HTML reports | `main` only; unreleased |
| One-call resource report and terminal summary with `measure_workload` | Preview PR #176; unreleased |
| The `custom_modules` extension API and `custom_mapping` on `crawl_module`/`summary` | `main` only; unreleased |
| New native Transformer MAC, DMA, and token-dependency estimates | `main` only; unreleased |

See the [changelog](changelog.md) for other changes after 0.2.0.

## Stable release

```shell
python -m pip install torchscan==0.2.0
```

## Development version

The one-call `measure_workload` API is prepared in [PR #176](https://github.com/frgfm/torch-scan/pull/176).
Until it is merged, install this preview without cloning the repository:

```shell
python -m pip install "torchscan @ git+https://github.com/frgfm/torch-scan.git@codex/workload-measurement"
```

After #176 merges, use `@main`. For repeatable experiments, use the commit you tested and record it with the report. Start with
the [quickstart](index.md), then [check an optimization](benchmark-comparison.md).

Python 3.11–3.14 is covered by the repository's installation CI on Linux, macOS, and Windows. PyTorch compatibility
checks cover 2.1.0 on Python 3.11 and the latest supported PyTorch on Python 3.14. Prefer a current PyTorch version:
its native timing utilities have current dependencies. For timing on PyTorch 2.1, also install:

```shell
python -m pip install "setuptools<70" "numpy<2"
```

CPU measurement needs no accelerator. CUDA/MPS timing and memory require real matching hardware and an appropriate
PyTorch build. MPS allocator peaks depend on the installed PyTorch APIs; MPS operator profiles show CPU dispatch,
not GPU execution. Fresh-process RSS is supported on Linux/macOS. See the
[measurement boundaries](workload-diagnostics.md).

## Local development checkout

Clone `main` and install it with [uv](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/frgfm/torch-scan.git
cd torch-scan
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

Install documentation or test dependencies only when needed:

```shell
uv pip install -e ".[docs]"
uv pip install -e ".[test]"
```
