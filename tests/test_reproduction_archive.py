"""Reproduction restoration preserves evidence and rejects partial corruption."""

import hashlib
import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "restore_reproduction_sources", ROOT / "scripts/restore_reproduction_sources.py"
)
restoration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restoration)


def package(tmp_path, files):
    archive = tmp_path / "evidence.tar.gz"
    records = {}
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
            records[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return archive, {"bytes": archive.stat().st_size, "sha256": restoration.sha(archive), "files": records}


def test_exact_restore_is_idempotent(tmp_path):
    archive, manifest = package(tmp_path, {"research/record.json": b'{"successes":0}'})
    destination = tmp_path / "restored"
    restoration.restore(archive, destination, manifest)
    restoration.restore(archive, destination, manifest)
    assert (destination / "research/record.json").read_bytes() == b'{"successes":0}'


def test_conflicting_later_member_prevents_all_writes(tmp_path):
    archive, manifest = package(tmp_path, {"research/first.json": b"first", "research/second.json": b"second"})
    destination = tmp_path / "restored"
    (destination / "research").mkdir(parents=True)
    existing = destination / "research/second.json"
    existing.write_bytes(b"valuable evidence")
    with pytest.raises(ValueError, match="Existing evidence differs"):
        restoration.restore(archive, destination, manifest)
    assert existing.read_bytes() == b"valuable evidence"
    assert not (destination / "research/first.json").exists()


def test_corrupt_member_prevents_all_writes(tmp_path):
    archive, manifest = package(tmp_path, {"research/first.json": b"first", "research/second.json": b"second"})
    manifest["files"]["research/second.json"]["sha256"] = "0" * 64
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="Archive member digest differs"):
        restoration.restore(archive, destination, manifest)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_unsafe_members_never_escape(tmp_path, name):
    archive, manifest = package(tmp_path, {name: b"unsafe"})
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="Unsafe destination"):
        restoration.restore(archive, destination, manifest)
    assert not destination.exists()
