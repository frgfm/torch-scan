# Installation

TorchScan has a published stable release and an unreleased development version:

| Track | Version | Python | PyTorch |
| --- | --- | --- | --- |
| Stable | 0.1.2 | ≥3.6,<4 | ≥1.5,<2 |
| Development (`main`) | 0.2.0.dev0 | ≥3.11,<4 | ≥2,<3 |

The API reference on this site follows the development version. Use the stable release unless you specifically need to test current development.

## Stable release

Install from [PyPI](https://pypi.org/project/torchscan/) with pip:

```shell
pip install torchscan
```

Or install from [Anaconda.org](https://anaconda.org/frgfm/torchscan):

```shell
conda install -c frgfm torchscan
```

## Development version

Clone `main` and install it with [uv](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/frgfm/torch-scan.git
cd torch-scan
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```
