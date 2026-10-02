import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stage_revision_supplement import package_records  # noqa: E402

pytestmark = pytest.mark.unit


def make_package(folder):
    archive = folder / "source.tar.zst"
    archive.write_bytes(b"fixture archive bytes")
    index = dict(
        schema="wm-open-source-package-v1",
        protocol="wm-open-v2-20260929",
        assets=[
            dict(
                path=archive.name, bytes=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest()
            )
        ],
    )
    (folder / "source-index.json").write_text(json.dumps(index))
    return archive, index


def test_stage_checks_bytes_before_any_network_request(tmp_path):
    archive, _ = make_package(tmp_path)
    assert len(package_records(tmp_path, "source-index.json")) == 2
    archive.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Changed"):
        package_records(tmp_path, "source-index.json")


def test_stage_rejects_duplicate_archive_names(tmp_path):
    _, index = make_package(tmp_path)
    index["assets"].append(index["assets"][0])
    (tmp_path / "source-index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Duplicate"):
        package_records(tmp_path, "source-index.json")


def test_stage_rejects_source_symlink(tmp_path):
    archive, _ = make_package(tmp_path)
    stored = tmp_path / "other"
    archive.rename(stored)
    archive.symlink_to(stored)
    with pytest.raises(ValueError, match="regular"):
        package_records(tmp_path, "source-index.json")


@pytest.mark.parametrize('corrupt', [False, True])
def test_resume_reuses_verified_downloads_and_rejects_changed_bytes(tmp_path, monkeypatch, corrupt):
    import stage_revision_supplement as stage
    package, download = tmp_path / 'package', tmp_path / 'download'
    package.mkdir()
    download.mkdir()
    make_package(package)
    records = stage.package_records(package, 'source-index.json')
    for row in records:
        (download / row['path']).write_bytes((package / row['path']).read_bytes())
    if corrupt:
        (download / records[0]['path']).write_bytes(b'changed')
    monkeypatch.setattr(stage, 'api', lambda _: {'private': True})
    monkeypatch.setattr(stage, 'api_pages', lambda url: (
        [dict(tag_name='draft', draft=True, id=1)] if 'releases?' in url else
        [dict(name=r['path'], size=r['bytes'], id=i + 2) for i, r in enumerate(records)]
    ))
    monkeypatch.setattr(stage.subprocess, 'run', lambda *a, **kw: pytest.fail('Verified cached files must not download again'))
    monkeypatch.setattr(stage.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 100 * 2**30})())
    monkeypatch.setattr(sys, 'argv', ['stage', '--package', str(package), '--index', 'source-index.json', '--repository', 'owner/repo', '--release', 'draft', '--download', str(download)])
    if corrupt:
        with pytest.raises(ValueError, match='Preserve changed'):
            stage.main()
    else:
        stage.main()
        receipt = json.loads((download / 'private-stage-verified.json').read_text())
        assert len(receipt['assets']) == 2
