# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Exercise installed autoresearch and its workers from outside the checkout."""

import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tempfile
from pathlib import Path

import torchscan

checkout = Path(__file__).resolve().parents[1]
assert not Path(torchscan.__file__).resolve().is_relative_to(checkout), (
    "Expected a distribution, not an editable install."
)
with tempfile.TemporaryDirectory(prefix="torchscan-artifact-") as directory:
    root = Path(directory)
    (root / "models.py").write_text(
        "import torch\ndef baseline(): return torch.nn.ReLU()\n"
        "class Wrong(torch.nn.Module):\n"
        "    def forward(self, inputs): return inputs.relu() + 1\n",
        encoding="utf-8",
    )
    (root / "workload.json").write_text(
        json.dumps({
            "model": "models.py:baseline",
            "args": [{"shape": [1, 4]}],
            "check": {"absolute_error": {"max": 0}},
            "settings": {"trials": 2, "warmup": 1, "min_run_time": 0.01, "min_repeats": 2, "min_improvement": 0.99},
        }),
        encoding="utf-8",
    )
    for name, candidate in (("identical", "baseline"), ("incorrect", "Wrong")):
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                sys.executable,
                "-m",
                "torchscan.autoresearch",
                "workload.json",
                "--candidate-model",
                f"models.py:{candidate}",
                "--output",
                name,
            ],
            cwd=root,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        assert json.loads(completed.stdout)["output_error_policy"] == "enforced"
        report = json.loads((root / name / "report.json").read_text())
        if name == "identical":
            assert report["decision"] in {"discard", "inconclusive"}, report["reasons"]
            assert len(report["trials"]) == 2
            assert all(trial["output_check"] == "passed" for trial in report["trials"])
            assert (root / name / "trial-2.html").is_file()
            assert report["feedback"]["after"]["output_errors"]["/"]["max"] == 0
        else:
            assert report["decision"] == "discard", report["reasons"]
            assert report["initial_checks"]["candidate"]["output_check"] == "failed"
print("Installed autoresearch subprocesses, output rejection and HTML passed.")
