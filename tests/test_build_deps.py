import hashlib
import io
import tarfile

import pytest

import _build_deps
from _build_deps import _checked_source, _extract_boost


@pytest.mark.parametrize("cached", [True, False])
def test_corrupt_source_is_rejected(tmp_path, monkeypatch, cached):
    path = tmp_path / "headers.hpp"
    path.write_bytes(b"upstream header")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _checked_source("https://example.invalid/headers.hpp", digest, path) == path
    if cached:
        path.write_bytes(b"altered header")
    else:
        path.unlink()
        monkeypatch.setattr(_build_deps, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"altered header"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _checked_source("https://example.invalid/headers.hpp", digest, path)
    if not cached:
        assert not path.exists()
    with pytest.raises(ValueError, match="HTTPS"):
        _checked_source("http://example.invalid/headers.hpp", digest, path)


@pytest.mark.parametrize("name", ["integer.hpp", "../escape.hpp", "symlink.hpp"])
def test_only_regular_headers_are_extracted(tmp_path, name):
    archive = tmp_path / "headers.tar.xz"
    with tarfile.open(archive, "w:xz") as output:
        member = tarfile.TarInfo("boost-release/boost/" + name)
        member.size = len(b"header")
        if name == "symlink.hpp":
            member.type, member.linkname = tarfile.SYMTYPE, "../../escape.hpp"
        output.addfile(member, io.BytesIO(b"header"))
    destination = tmp_path / "include/boost"
    if name == "integer.hpp":
        _extract_boost(archive, destination, "boost-release/boost/")
        assert (destination / name).read_bytes() == b"header"
    else:
        with pytest.raises(ValueError, match="Unsafe Boost header archive entry"):
            _extract_boost(archive, destination, "boost-release/boost/")
    assert not (tmp_path / "include/escape.hpp").exists()
