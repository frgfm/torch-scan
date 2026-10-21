from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from runpy import run_path
from typing import cast

checker = run_path(str(Path(__file__).parents[1] / ".github" / "verify_deps_sync.py"))
lower_bound = cast(Callable[[str, str], str], checker["lower_bound"])
setup_uv_versions = cast(Callable[[str, str], list[str]], checker["setup_uv_versions"])
version_sync_error = cast(Callable[[list[str], str], str | None], checker["version_sync_error"])


def test_named_and_unnamed_setup_uv_steps_are_parsed() -> None:
    workflow = """
steps:
  - uses: astral-sh/setup-uv@v10
    with:
      version: "3.4.5"
  - name: Setup uv
    uses: astral-sh/setup-uv@v10
    with:
      version: "3.4.4"
"""
    versions = setup_uv_versions(workflow, "workflow.yml")

    assert versions == ["3.4.5", "3.4.4"]
    error = version_sync_error(versions, "3.4.5")
    assert error is not None
    assert error.startswith("setup-uv pins must use one version")


def test_pin_matches_requirement_lower_bound() -> None:
    error = version_sync_error(["3.4.4"], "3.4.5")
    assert error == "setup-uv pin 3.4.4 must equal tool.uv.required-version lower bound 3.4.5"


def test_lower_bound_is_derived_from_requirement() -> None:
    minimum = lower_bound(">=3.4.5,<4.0.0", "pyproject.toml")
    assert minimum == "3.4.5"
    assert version_sync_error([minimum], minimum) is None
