# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Prepare checksum-pinned upstream headers only when compiling native code."""

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen


def _checked_source(url: str, sha256: str, path: Path) -> Path:
    if urlsplit(url).scheme != "https":
        raise ValueError("Native dependencies require HTTPS.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        candidate = path
        if not path.exists():
            candidate = Path(temporary) / "download"
            # The manifest uses HTTPS sources; the digest is checked before promotion.
            with urlopen(url, timeout=60) as response, candidate.open("wb") as output:  # ruff: ignore[suspicious-url-open-usage]
                shutil.copyfileobj(response, output)
        with candidate.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != sha256:
            raise ValueError(f"SHA-256 mismatch for {path.name}; refusing to use cached or downloaded headers.")
        if candidate != path:
            candidate.replace(path)
    return path


def _extract_boost(archive: Path, destination: Path, prefix: str) -> None:
    with tarfile.open(archive) as source:
        for member in source:
            if not member.name.startswith(prefix) or member.isdir():
                continue
            relative = PurePosixPath(member.name.removeprefix(prefix))
            target = destination / relative
            if (
                not member.isfile()
                or "\\" in member.name
                or ":" in member.name
                or not target.resolve().is_relative_to(destination.resolve())
            ):
                raise ValueError(f"Unsafe Boost header archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as contents, target.open("wb") as output:
                shutil.copyfileobj(contents, output)


def prepare_headers(build_dir: Path) -> Path:
    """Fetch verified upstream headers, or verify the same files in an offline cache."""
    manifest = json.loads((Path(__file__).parent / "native/count/dependencies.json").read_text(encoding="utf-8"))
    cache = Path(os.environ.get("TORCHSCAN_DEPS_CACHE", build_dir / "downloads"))
    sources = {
        name: _checked_source(spec["url"], spec["sha256"], cache / spec["filename"]) for name, spec in manifest.items()
    }
    include = build_dir / "include"
    _extract_boost(sources["boost"], include / "boost", manifest["boost"]["prefix"])
    (include / "nlohmann").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sources["json"], include / "nlohmann/json.hpp")
    if not (include / "boost/multiprecision/cpp_int.hpp").is_file():
        raise ValueError("Verified Boost archive contains no multiprecision headers.")
    return include
