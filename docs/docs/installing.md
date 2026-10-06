# Installation

Choose the stable release or the unreleased version on `main`:

| Track | Version | Python | PyTorch |
| --- | --- | --- | --- |
| Stable | 0.2.0 | ≥3.11,<4 | ≥2.1,<3 |
| Development (`main`) | Unreleased changes after 0.2.0 | ≥3.11,<4 | ≥2.1,<3 |

Version 0.2.0 was released on October 3, 2026. Use the [migration guide](migration-v02.md) when moving from 0.1.
The API reference on this site follows `main` and includes unreleased features.

| Feature | Available in |
| --- | --- |
| Structured reports, `args`/`kwargs`, nested outputs, strict checks, structure mode, and report comparison | Stable 0.2.0 and `main` |
| Native Transformer module FLOPs and callable peak-memory measurement | Stable 0.2.0 and `main` |
| Offline HTML/SVG reports with `render_report` | `main` only; unreleased |
| First-call and warmed workload timing with `measure_latency` | `main` only; unreleased |
| Child process RSS with `measure_peak_rss` and operator diagnostics with `profile_workload` | `main` only; unreleased |
| Checked benchmark comparisons and offline workload HTML reports | `main` only; unreleased |
| The `custom_modules` extension API and `custom_mapping` on `crawl_module`/`summary` | `main` only; unreleased |
| New native Transformer MAC, DMA, and token-dependency estimates | `main` only; unreleased |

See the [changelog](changelog.md) for other changes after 0.2.0.

## Stable release

Install the current PyPI release:

```shell
pip install torchscan
```

This installs v0.2.0 with the structured report contract. Install `main` for the unreleased features listed above.

## Development version

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
