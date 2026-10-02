import hashlib
import io
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from fetch_revision_task import fetch  # noqa: E402
from package_revision_task import pack  # noqa: E402
from restore_revision_task import extract_verified  # noqa: E402

pytestmark = pytest.mark.unit


def test_package_roundtrip_relocation_and_overwrite_protection(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b'{"experiment": "retained"}\n')
    record = pack(tmp_path / "task.tar.zst", {"data/task/metadata.json": source})
    output = tmp_path / "relocated"
    extract_verified(tmp_path, record, output)
    target = output / "data/task/metadata.json"
    assert target.read_bytes() == source.read_bytes()
    extract_verified(tmp_path, record, output)
    target.write_bytes(b"unique local evidence")
    with pytest.raises(ValueError, match="Existing file differs"):
        extract_verified(tmp_path, record, output)
    assert target.read_bytes() == b"unique local evidence"


def test_package_rejects_tampered_and_escaping_members(tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"original")
    record = pack(tmp_path / "escape.tar.zst", {"../escaped": source})
    with pytest.raises(ValueError, match="stay within"):
        extract_verified(tmp_path, record, tmp_path / "destination")
    assert not (tmp_path / "escaped").exists()
    record = pack(tmp_path / "normal.tar.zst", {"original": source})
    (tmp_path / "normal.tar.zst").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        extract_verified(tmp_path, record, tmp_path / "destination")


def test_download_verifies_before_installing_and_preserves_existing_data(tmp_path):
    payload = b"verified release bytes"

    @contextmanager
    def opener():
        yield io.BytesIO(payload)

    target = tmp_path / "download"
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        fetch(target, "0" * 64, len(payload), opener)
    assert not target.exists()
    assert not target.with_name("download.download-partial").exists()
    with pytest.raises(ValueError, match="exceeds"):
        fetch(target, digest, len(payload) - 1, opener)
    assert not target.exists()
    fetch(target, digest, len(payload), opener)
    assert target.read_bytes() == payload
    target.write_bytes(b"unique local data")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        fetch(target, digest, len(payload), opener)
    assert target.read_bytes() == b"unique local data"
