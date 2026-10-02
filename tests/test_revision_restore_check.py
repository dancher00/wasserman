import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import check_revision_task_restore as check  # noqa: E402

pytestmark = pytest.mark.unit


def test_pruning_checks_all_files_and_preserves_evidence_on_corruption(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"verified")
    second.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        check.prune_duplicates(
            [(first, check.sha256(first)), (second, "0" * 64)], tmp_path / "receipt.json", dict(pruned_files=[])
        )
    assert first.exists() and second.exists()


def test_pruning_requires_durable_receipt_before_deletion(tmp_path, monkeypatch):
    first = tmp_path / "first"
    first.write_bytes(b"verified")

    def failed_save(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(check, "save_receipt", failed_save)
    with pytest.raises(OSError):
        check.prune_duplicates([(first, check.sha256(first))], tmp_path / "receipt.json", dict(pruned_files=[]))
    assert first.exists()


def test_pruning_records_bytes_and_rejects_symlink(tmp_path):
    first = tmp_path / "first"
    first.write_bytes(b"verified")
    link = tmp_path / "link"
    link.symlink_to(first)
    with pytest.raises(ValueError, match="Unsafe"):
        check.prune_duplicates([(link, check.sha256(first))], tmp_path / "receipt.json", dict(pruned_files=[]))
    assert first.exists()
    receipt = tmp_path / "receipt.json"
    check.prune_duplicates([(first, check.sha256(first))], receipt, dict(pruned_files=[], cleanup_complete=False))
    result = json.loads(receipt.read_text())
    assert result["cleanup_complete"] and result["cleanup_planned_bytes"] == 8
    assert not first.exists()
