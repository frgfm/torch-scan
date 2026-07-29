.PHONY: install install-quality install-test install-docs lint-check lint-format typing-check precommit quality style test serve-docs build-docs set-version build publish

install:
	uv pip install -e .

install-quality:
	uv pip install -e ".[quality]"

install-test:
	uv pip install -e ".[test]"

install-docs:
	uv pip install -e ".[docs]"

lint-check:
	ruff format --check .
	ruff check .

lint-format:
	ruff format .
	ruff check --fix .

typing-check:
	ty check

precommit:
	prek run --all-files

quality: lint-check typing-check

style: lint-format

test:
	pytest --cov=torchscan --cov-report=xml tests/

# Serve documentation locally
serve-docs:
	mkdocs serve -f docs/mkdocs.yml

# Build documentation
build-docs:
	mkdocs build --strict -f docs/mkdocs.yml

set-version:
	uv version --frozen --no-build "$(BUILD_VERSION)"

build:
	uv build

publish:
	uv publish --trusted-publishing always
