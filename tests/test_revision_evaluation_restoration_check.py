import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from check_revision_evaluation_restore import verify_restored_members  # noqa: E402

pytestmark = pytest.mark.unit


def test_restored_evidence_is_rehashed_and_duplicates_are_refused(tmp_path):
    p = tmp_path / "analysis/table.json"
    p.parent.mkdir()
    p.write_bytes(b"synthetic test fixture")
    expected = {"analysis/table.json": hashlib.sha256(p.read_bytes()).hexdigest()}
    assets = [dict(members=expected)]
    assert verify_restored_members(tmp_path, assets) == expected
    with pytest.raises(ValueError, match="duplicate"):
        verify_restored_members(tmp_path, assets * 2)
    p.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="changed"):
        verify_restored_members(tmp_path, assets)


def test_restored_evidence_rejects_escape_and_symlink_parents(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    p = data / "trace.bin"
    p.write_bytes(b"fixture")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("../trace.bin", "/trace.bin"):
        with pytest.raises(ValueError, match="Unsafe"):
            verify_restored_members(tmp_path, [dict(members={name: digest})])
    (tmp_path / "linked").symlink_to(data, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verify_restored_members(tmp_path, [dict(members={"linked/trace.bin": digest})])
