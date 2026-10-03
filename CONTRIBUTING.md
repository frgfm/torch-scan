# Contributing to TorchScan

Please follow the [code of conduct](CODE_OF_CONDUCT.md). Report bugs and feature requests through
[GitHub issues](https://github.com/frgfm/torch-scan/issues), checking existing issues first and using the templates.
For usage questions, visit [GitHub discussions](https://github.com/frgfm/torch-scan/discussions).

## Codebase structure

- [torchscan/](torchscan): library source.
- [tests/](tests): unit and model integration tests.
- [docs/](docs): MkDocs documentation.
- [scripts/](scripts): examples and utilities.

## Development setup

[Fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo), then clone your fork and create a
branch for the change:

```shell
git clone git@github.com:<YOUR_GITHUB_ACCOUNT>/torch-scan.git
cd torch-scan
git remote add upstream https://github.com/frgfm/torch-scan.git
git checkout -b a-short-description
uv venv --python 3.11
```

Activate `.venv` with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\activate` on Windows, then install
the contributor dependencies and hooks:

```shell
make install-quality install-test install-docs
prek install
```

## Checks before submitting

Add focused tests for changed behavior and use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
for public Python functions and classes. Follow the [commit message guide](http://udacity.github.io/git-styleguide/).

Run the checks used by CI:

```shell
make test
make quality
make precommit
make build-docs
```

`make quality` checks Ruff formatting/lint, ty types, and copyright headers. `make precommit` runs repository hooks.
Use `make style` to apply Ruff fixes and `make headers-fix` to refresh headers. CI also verifies installation,
compatibility, distributions, and model integrations; [Codecov](https://codecov.io/gh/frgfm/torch-scan) reports coverage.
See [the documentation guide](docs/README.md) for local preview instructions.

## Submitting a pull request

Push your branch with `git push -u origin a-short-description`, then
[open a pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
and complete the repository's PR template.
