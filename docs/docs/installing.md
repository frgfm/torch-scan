# Installation

TorchScan has a published stable release and an unreleased development version:

| Track | Version | Python | PyTorch |
| --- | --- | --- | --- |
| Stable | 0.1.2 | ≥3.6,<4 | ≥1.5,<2 |
| Development (`main`) | 0.2.0.dev0 | ≥3.11,<4 | ≥2.1,<3 |

The API reference on this site follows the development version. Version 0.2 is a deliberate breaking release; use the
[migration guide](migration-v02.md) when moving from 0.1.

## Stable release

Install the current PyPI release:

```shell
pip install torchscan
```

The 0.1.2 package does not contain the v0.2 report contract documented on this site.

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
