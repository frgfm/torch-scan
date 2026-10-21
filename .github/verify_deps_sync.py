# Copyright (C) 2026, François-Guillaume Fernandez.
"""Keep the project uv constraints and GitHub Actions uv pins in sync."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"uv dependency sync check failed: {message}")


def lower_bound(specifier: str, source: str) -> str:
    versions = re.findall(r"(?:^|,)\s*>=\s*(\d+(?:\.\d+){1,2})(?=\s*(?:,|$))", specifier)
    if len(versions) != 1:
        fail(f"{source}: expected one >= lower bound, found {specifier!r}")
    return versions[0]


def check_project_constraints() -> str:
    with Path("pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    required_version = pyproject.get("tool", {}).get("uv", {}).get("required-version")
    if not isinstance(required_version, str):
        fail("pyproject.toml: tool.uv.required-version is missing")
    uv_minimum = lower_bound(required_version, "pyproject.toml tool.uv.required-version")

    build_requires = pyproject.get("build-system", {}).get("requires", [])
    uv_build_requires = [requirement for requirement in build_requires if requirement.startswith("uv_build")]
    if len(uv_build_requires) != 1:
        fail(f"pyproject.toml: expected one uv_build requirement, found {uv_build_requires!r}")
    uv_build_minimum = lower_bound(uv_build_requires[0].removeprefix("uv_build"), "pyproject.toml uv_build requirement")
    if uv_build_minimum != uv_minimum:
        fail(f"pyproject.toml: uv_build lower bound {uv_build_minimum} differs from uv lower bound {uv_minimum}")
    return uv_minimum


def check_action_versions(uv_minimum: str) -> None:
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

    versions: list[str] = []
    for path in config_files:
        text = path.read_text()
        actions = list(re.finditer(r"(?m)^([ ]*)-\s+uses:\s+astral-sh/setup-uv@", text))
        if not actions:
            fail(f"{path}: could not parse setup-uv action")
        for action in actions:
            indent = len(action.group(1))
            end = len(text)
            for next_step in re.finditer(r"(?m)^([ ]*)-\s+", text[action.end() :]):
                if len(next_step.group(1)) == indent:
                    end = action.end() + next_step.start()
                    break
            step = text[action.end() : end]
            version = re.search(r"(?m)^\s+version:\s*[\"']?([^\"'\s]+)", step)
            if version is None:
                fail(f"{path}: setup-uv is missing an explicit version")
            versions.append(version.group(1))

    unique_versions = set(versions)
    if len(unique_versions) != 1:
        fail(f"setup-uv pins must use one version, found {sorted(unique_versions)!r}")
    uv_version = next(iter(unique_versions))
    if uv_version != uv_minimum:
        fail(f"setup-uv pin {uv_version} must equal tool.uv.required-version lower bound {uv_minimum}")


def main() -> None:
    uv_minimum = check_project_constraints()
    check_action_versions(uv_minimum)
    print("All uv versions are in sync!")


if __name__ == "__main__":
    main()
