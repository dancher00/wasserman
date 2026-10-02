import importlib.util
import shutil
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("archive", Path(__file__).parents[1] / "scripts/archive_revision_rgb.py")
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


@pytest.mark.unit
@pytest.mark.skipif(not shutil.which("zstd"), reason="zstd executable required")
def test_prune_requires_lossless_roundtrip(tmp_path):
    source, target = tmp_path / "wrist.rgb", tmp_path / "wrist.rgb.zst"
    source.write_bytes(bytes(range(256)) * 1000)
    result = archive.archive_one(source, target, prune=True)
    assert not source.exists()
    assert archive.decoded_digest(target) == (result["raw_sha256"], 256000)
    source.write_bytes(b"different recording")
    with pytest.raises(ValueError, match="mismatch"):
        archive.archive_one(source, target, prune=True)
    assert source.read_bytes() == b"different recording"


@pytest.mark.unit
def test_lossless_rgb_video_keeps_every_byte(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    source, target = tmp_path / "wrist.rgb", tmp_path / "wrist.rgb.mkv"
    source.write_bytes(bytes(range(256)) * 12)
    result = archive.archive_one(source, target, prune=True, codec="libx264rgb", image_shape=[16, 16, 3])
    assert not source.exists()
    assert archive.decoded_digest(target, "libx264rgb") == (result["raw_sha256"], 3072)


@pytest.mark.unit
@pytest.mark.parametrize("failed_write", [1, 2])
def test_manifest_failure_cannot_orphan_pruned_rgb(tmp_path, monkeypatch, failed_write):
    import json

    source = tmp_path / "wrist.rgb"
    source.write_bytes(bytes(range(256)) * 12)
    destination = tmp_path / "archive/seed_1/wrist.rgb.zst"
    entry = archive.archive_one(source, destination)
    entry["archive_path"] = "seed_1/wrist.rgb.zst"
    state = dict(files={"seed_1/wrist.rgb": entry})
    manifest = tmp_path / "archive/rgb-manifest.json"
    real_store, calls = archive.store_manifest, []

    def fail_at_boundary(path, payload):
        calls.append(1)
        if len(calls) == failed_write:
            raise OSError("Injected manifest write failure")
        real_store(path, payload)

    monkeypatch.setattr(archive, "store_manifest", fail_at_boundary)
    with pytest.raises(OSError, match="Injected"):
        archive.publish_before_prune(manifest, state, source, entry, prune=True)
    if source.exists():
        assert archive.digest_file(source) == entry["raw_sha256"]
    else:
        retained_state = json.loads(manifest.read_text())
        retained = retained_state["files"]["seed_1/wrist.rgb"]
        assert archive.decoded_digest(destination) == (retained["raw_sha256"], retained["raw_bytes"])
        monkeypatch.setattr(archive, "store_manifest", real_store)
        archive.reconcile_pruned(manifest, retained_state, tmp_path / "missing-source")
        assert json.loads(manifest.read_text())["files"]["seed_1/wrist.rgb"]["raw_pruned"]


@pytest.mark.unit
@pytest.mark.parametrize("codec", ["zstd", "libx264rgb"])
def test_restore_checks_archive_and_original_bytes(tmp_path, monkeypatch, codec):
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    sys.modules.pop("restore_revision_rgb", None)
    import restore_revision_rgb as restore

    source = tmp_path / "wrist.rgb"
    original = bytes(range(256)) * 12
    source.write_bytes(original)
    filename = "seed_1/wrist.rgb.zst" if codec == "zstd" else "seed_1/wrist.rgb.mkv"
    target = tmp_path / "archive" / filename
    record = archive.archive_one(source, target, prune=True, codec=codec, image_shape=[16, 16, 3])
    record["archive_path"] = filename
    result = restore.restore_one(
        tmp_path / "archive", tmp_path / "restored", "seed_1/wrist.rgb", record, reserve_bytes=0
    )
    assert result.read_bytes() == original
    result.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="destination differs"):
        restore.restore_one(tmp_path / "archive", tmp_path / "restored", "seed_1/wrist.rgb", record, reserve_bytes=0)
    assert result.read_bytes() == b"do not overwrite"


@pytest.mark.unit
@pytest.mark.parametrize("prune", [False, True])
def test_streaming_transcode_preserves_rgb_before_optional_pruning(tmp_path, prune):
    import json
    import subprocess
    import sys

    source = tmp_path / "wrist.rgb"
    source.write_bytes(bytes(range(256)) * 12)
    compressed = tmp_path / "seed_1/wrist.rgb.zst"
    entry = archive.archive_one(source, compressed, prune=True, image_shape=[16, 16, 3])
    entry["archive_path"] = "seed_1/wrist.rgb.zst"
    if prune:
        entry.pop("image_shape")
        (compressed.parent / "metadata.json").write_text(json.dumps(dict(image_shape=[16, 16, 3])))
    manifest = tmp_path / "rgb-manifest.json"
    manifest.write_text(json.dumps(dict(files={"seed_1/wrist.rgb": entry})))
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/transcode_revision_archive.py"),
            "--archive",
            str(tmp_path),
            *(["--prune-superseded"] if prune else []),
        ],
        check=True,
    )
    updated = json.loads(manifest.read_text())["files"]["seed_1/wrist.rgb"]
    assert archive.decoded_digest(tmp_path / updated["archive_path"], "libx264rgb") == (entry["raw_sha256"], 3072)
    assert compressed.exists() != prune
    receipt = json.loads((tmp_path / "transcode-receipts.jsonl").read_text())
    assert receipt["previous"]["archive_sha256"] == entry["archive_sha256"]
