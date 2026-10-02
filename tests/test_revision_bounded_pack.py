import os
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pack_revision_bounded import pack
from package_revision_task import verify_members


def test_complete_archive_roundtrip(tmp_path):
    source = tmp_path / 'input'; source.write_bytes(b'unchanged evidence\0' * 10000)
    target = tmp_path / 'result.tar.zst'
    record = pack(target, {'nested/evidence.bin': source})
    verify_members(target, record['members'])
    assert not target.with_suffix('.zst.partial').exists()


def test_output_cap_preserves_partial_without_success(tmp_path):
    source = tmp_path / 'input'; source.write_bytes(os.urandom(2 * 1024**2))
    target = tmp_path / 'result.tar.zst'
    with pytest.raises(RuntimeError, match='byte limit or disk reserve'):
        pack(target, {'evidence.bin': source}, max_output_bytes=100)
    assert not target.exists()
    assert target.with_suffix('.zst.partial').stat().st_size <= 100


def test_existing_partial_is_never_overwritten(tmp_path):
    source = tmp_path / 'input'; source.write_bytes(b'evidence')
    target = tmp_path / 'result.tar.zst'
    partial = target.with_suffix('.zst.partial'); partial.write_bytes(b'previous attempt')
    with pytest.raises(ValueError, match='Preserve'):
        pack(target, {'evidence': source})
    assert partial.read_bytes() == b'previous attempt'


def test_direct_multipart_restores_without_assembled_archive(tmp_path):
    from restore_revision_task import extract_verified
    source = tmp_path / 'input'; source.write_bytes(os.urandom(200000))
    target = tmp_path / 'result.tar.zst'
    record = pack(target, {'nested/evidence.bin': source}, part_bytes=65536)
    assert not target.exists()
    assert len(record['parts']) > 1
    assert all(p['bytes'] <= 65536 for p in record['parts'])
    output = tmp_path / 'restored'
    extract_verified(tmp_path, record, output)
    assert (output / 'nested/evidence.bin').read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match='Preserve'):
        pack(target, {'nested/evidence.bin': source}, part_bytes=65536)


def test_direct_multipart_corruption_is_rejected(tmp_path):
    from revision_package_assets import verified_asset_paths
    source = tmp_path / 'input'; source.write_bytes(os.urandom(10000))
    record = pack(tmp_path / 'result.tar.zst', {'evidence.bin': source}, part_bytes=4096)
    part = tmp_path / record['parts'][0]['path']
    data = bytearray(part.read_bytes()); data[0] ^= 1; part.write_bytes(data)
    with pytest.raises(ValueError, match='checksum'):
        verified_asset_paths(tmp_path, record)
