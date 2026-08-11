# Changing the documentation

The documentation is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) using Python 3.11.

Create and activate a virtual environment from the repository root:

```shell
uv venv --python 3.11
source .venv/bin/activate
make install-docs
```

## Preview the documentation

Start the development server:

```shell
make serve-docs
```

## Build the documentation

Run the same strict build used by CI:

```shell
make build-docs
```

The generated site is written to `docs/site/`. Pull requests build the site automatically, and pushes to `main` deploy it to GitHub Pages.

Keep `llms.txt` and `docs/docs/llms.txt` identical: the first is discoverable in the repository and the second is
copied to the published site's root.
