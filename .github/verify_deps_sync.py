# Copyright (C) 2026, François-Guillaume Fernandez.
"""Keep the project uv constraints and GitHub Actions uv pins in sync."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

UV_VERSION = "0.13.4"
UV_REQUIREMENT = ">=0.13.4,<1.0.0"


def fail(message: str) -> None:
    raise SystemExit(f"uv dependency sync check failed: {message}")


def check_project_constraints() -> None:
    with Path("pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    required_version = pyproject.get("tool", {}).get("uv", {}).get("required-version")
    if required_version != UV_REQUIREMENT:
        fail(f"pyproject.toml: expected tool.uv.required-version = {UV_REQUIREMENT!r}, found {required_version!r}")

    build_requires = pyproject.get("build-system", {}).get("requires", [])
    if f"uv_build{UV_REQUIREMENT}" not in build_requires:
        fail(f"pyproject.toml: build-system.requires must include uv_build{UV_REQUIREMENT}")


def check_action_versions() -> None:
    config_files = [
        path
        for path in Path(".github").rglob("*")
        if path.suffix in {".yml", ".yaml"} and "astral-sh/setup-uv@" in path.read_text()
    ]
    if not config_files:
        fail("no astral-sh/setup-uv actions found under .github")

    required_files = {
        Path(".github/actions/setup/action.yml"),
        Path(".github/workflows/update-copyright-years.yml"),
    }
    if not required_files.issubset(config_files):
        fail("the shared setup action and annual workflow must both install uv")

    for path in config_files:
        text = path.read_text()
        actions = list(re.finditer(r"(?m)^([ ]*)-\s+uses:\s+astral-sh/setup-uv@", text))
        for action in actions:
            indent = len(action.group(1))
            end = len(text)
            for next_step in re.finditer(r"(?m)^([ ]*)-\s+", text[action.end() :]):
                if len(next_step.group(1)) == indent:
                    end = action.end() + next_step.start()
                    break
            step = text[action.end() : end]
            version = re.search(r"(?m)^\s+version:\s*[\"']?([^\"'\s]+)", step)
            if version is None or version.group(1) != UV_VERSION:
                found = version.group(1) if version else "missing"
                fail(f"{path}: setup-uv version must be {UV_VERSION}, found {found}")


def main() -> None:
    check_project_constraints()
    check_action_versions()
    print("All uv versions are in sync!")


if __name__ == "__main__":
    main()
