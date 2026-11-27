# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Exercise installed autoresearch and its workers from outside the checkout."""

import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path

import torchscan

# An artifact can be installed in a virtual environment inside the checkout.
installed = Path(distribution("torchscan").locate_file("torchscan/__init__.py")).resolve()
assert Path(torchscan.__file__).resolve() == installed, "Expected the installed distribution."
environment = os.environ.copy()
environment.pop("PYTHONPATH", None)
with tempfile.TemporaryDirectory(prefix="torchscan-artifact-") as directory:
    root = Path(directory)
    (root / "models.py").write_text("from torch.nn import ReLU, Sigmoid\n", encoding="utf-8")
    (root / "workload.json").write_text(
        json.dumps({
            "model": "models.py:ReLU",
            "args": [{"shape": [1, 4], "fill": 0}],
            "check": {"absolute_error": {"max": 0}},
            "settings": {"warmup": 1, "min_run_time": 0.01, "min_repeats": 2},
        }),
        encoding="utf-8",
    )
    for name, candidate in (("baseline", []), ("incorrect", ["--candidate-model", "models.py:Sigmoid"])):
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-m", "torchscan.autoresearch", "workload.json", "--output", name, *candidate],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        assert json.loads(completed.stdout)["output_error_policy"] == "enforced"
        report = json.loads((root / name / "report.json").read_text())
        if name == "baseline":
            assert report["decision"] == "measured", report["reasons"]
            assert report["feedback"]["latency_ms"] > 0
            assert (root / name / "measurement.html").is_file()
        else:
            assert report["decision"] == "discard", report["reasons"]
            assert report["initial_checks"]["candidate"]["output_check"] == "failed"
print("Installed autoresearch subprocesses, output rejection and HTML passed.")
