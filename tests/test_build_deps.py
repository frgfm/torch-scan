import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path

import pytest


def test_corrupt_cached_source_is_rejected(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the build-integrity check")
    source = Path(__file__).resolve().parents[1] / "native/count"
    pin = json.loads((source / "dependencies.json").read_text())["boost"]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / pin["filename"]).write_bytes(b"altered source archive")
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [cmake, f"-DTORCHSCAN_DEPS_CACHE={cache}", "-P", str(source / "dependencies.cmake")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert not (tmp_path / "headers").exists()
