"""Package only the new current-study evidence; retain the existing core releases."""
import json,subprocess
from pathlib import Path
from current_contact_study import ROOT,digest
from pack_revision_bounded import pack
P=ROOT/'research/current-contact-5090';A=ROOT/'artifacts/current-contact-5090'

def main():
    audit=json.loads((P/'results/audit.json').read_text())
    assert audit['physical_rescoring_passed'] and audit['cohorts']==18 and audit['episodes']==540
    video=json.loads((P/'video-manifest.json').read_text());assert len(video['clips'])==4
    destination=A/'release';destination.mkdir(exist_ok=True)
    if (destination/'index.json').exists():raise RuntimeError('Preserve existing package')
    files={}
    for directory in ['evaluation','video','benchmark','pilot','jobs']:
        for f in sorted((A/directory).rglob('*')):
            if f.is_file() and f.suffix!='.rgb':files[str(f.relative_to(ROOT))]=f
    for f in sorted(P.rglob('*')):
        if f.is_file() and f.name!='release.json':files[str(f.relative_to(ROOT))]=f
    for f in sorted((ROOT/'scripts').glob('*current_contact*.py')):files[str(f.relative_to(ROOT))]=f
    f=ROOT/'tests/test_current_contact_metrics.py';files[str(f.relative_to(ROOT))]=f
    record=pack(destination/'current-contact-evidence.tar.zst',files,max_output_bytes=2*2**30)
    index=dict(schema='wm-current-contact-evidence-v1',study='wm-current-contact-5090-v1',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        assets=[record],excluded='Expert-pilot raw RGB buffers; six pre-existing model weights and full base runtime are referenced by protocol hashes and available in the existing private core releases.',
        restoration='From the project root: zstd -dc current-contact-evidence.tar.zst | tar -xf -. Require fresh destinations; verify each member SHA-256 against this index before analysis.',
        archived_bytes=sum(p.stat().st_size for p in files.values()))
    (destination/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),compressed_bytes=record['bytes'],sha256=record['sha256'])))

if __name__=='__main__':main()
