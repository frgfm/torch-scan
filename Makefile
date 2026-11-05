.PHONY: install install-quality install-test install-docs deps-check ruff-lint-fix ruff-format-fix lint-check lint-format typing-check headers-check headers-fix precommit native-check quality style test serve-docs build-docs set-version build publish

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

deps-check:
	python .github/verify_deps_sync.py

ruff-lint-fix:
	ruff check --fix .

ruff-format-fix:
	ruff format .

lint-format: ruff-lint-fix ruff-format-fix

typing-check:
	ty check

headers-check:
	lmh check

headers-fix:
	lmh fix --folders . --ignore-folders tests

precommit:
	prek run --all-files

native-check:
	cargo fmt --manifest-path native/count/Cargo.toml --check
	cargo clippy --manifest-path native/count/Cargo.toml --locked --all-targets --features python -- -D warnings

quality: deps-check lint-check typing-check headers-check native-check

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
