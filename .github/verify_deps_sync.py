# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Keep the project uv requirement and GitHub Actions pins in sync."""

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
    return lower_bound(required_version, "pyproject.toml tool.uv.required-version")


def setup_uv_versions(text: str, source: str) -> list[str]:
    lines = text.splitlines()
    versions = []
    action_lines = 0

    for index, line in enumerate(lines):
        action = re.match(r"^([ ]*)(-\s+)?uses:\s*[\"']?astral-sh/setup-uv@", line)
        if action is None:
            continue
        action_lines += 1
        step_indent = len(action.group(1))
        if action.group(2) is None:
            for previous in range(index - 1, -1, -1):
                item = re.match(r"^([ ]*)-\s+", lines[previous])
                if item is not None and len(item.group(1)) < step_indent:
                    step_indent = len(item.group(1))
                    step_start = previous
                    break
            else:
                fail(f"{source}: setup-uv step has no parent list item")
        else:
            step_start = index

        step_end = len(lines)
        for following in range(step_start + 1, len(lines)):
            item = re.match(r"^([ ]*)-\s+", lines[following])
            if item is not None and len(item.group(1)) == step_indent:
                step_end = following
                break

        step = "\n".join(lines[step_start:step_end])
        pins = re.findall(r"(?m)^[ ]+version:\s*[\"']?([^\"'\s]+)", step)
        if len(pins) != 1:
            fail(f"{source}: setup-uv step must have one explicit version, found {pins!r}")
        versions.append(pins[0])

    if "astral-sh/setup-uv@" in text and action_lines != text.count("astral-sh/setup-uv@"):
        fail(f"{source}: could not parse every setup-uv action")
    return versions


def version_sync_error(versions: list[str], uv_minimum: str) -> str | None:
    unique_versions = set(versions)
    if len(unique_versions) != 1:
        return f"setup-uv pins must use one version, found {sorted(unique_versions)!r}"
    uv_version = next(iter(unique_versions))
    if uv_version != uv_minimum:
        return f"setup-uv pin {uv_version} must equal tool.uv.required-version lower bound {uv_minimum}"
    return None


def check_version_sync(versions: list[str], uv_minimum: str) -> None:
    error = version_sync_error(versions, uv_minimum)
    if error is not None:
        fail(error)


def check_action_versions(uv_minimum: str) -> None:
    versions = []
    for path in Path(".github").rglob("*"):
        if path.suffix in {".yml", ".yaml"}:
            versions.extend(setup_uv_versions(path.read_text(encoding="utf-8"), str(path)))

    if not versions:
        fail("no astral-sh/setup-uv actions found under .github")

    check_version_sync(versions, uv_minimum)


def main() -> None:
    uv_minimum = check_project_constraints()
    check_action_versions(uv_minimum)
    print("All uv versions are in sync!")


if __name__ == "__main__":
    main()
