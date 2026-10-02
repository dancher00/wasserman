"""Upload a new private draft release and stream-verify every downloaded byte."""
import json,subprocess
from pathlib import Path
from current_contact_study import ROOT,digest
from upload_revision_task import api,api_pages,verify_download
P=ROOT/'research/current-contact-5090';D=ROOT/'artifacts/current-contact-5090/release'
REPO='dancher00/WasserMan-Bench';TAG='wm-current-contact-5090-v1-20260930'

def main():
    assert api(f'repos/{REPO}')['private'],'Private repository required; never change visibility'
    index=json.loads((D/'index.json').read_text())
    files=[dict(path=D/r['path'],sha256=r['sha256'],bytes=r['bytes']) for r in index['assets']]
    files.append(dict(path=D/'index.json',sha256=digest(D/'index.json'),bytes=(D/'index.json').stat().st_size))
    for item in files:assert item['path'].is_file() and digest(item['path'])==item['sha256']
    releases=[r for r in api_pages(f'repos/{REPO}/releases?per_page=100') if r['tag_name']==TAG]
    if not releases:
        note=D/'release-notes.md'
        note.write_text('Controlled current/contact study: 540 physically rescored DP episodes across OpenHatch and RotateValve, three fixed training seeds and currents 0/0.10/0.20 m/s. Includes full scoring/physical trajectories, paired analysis, blue figures, selected video reruns, provenance and retained preparation failures. No new training, hardware calibration or public-access claim. Existing core models/runtime are referenced by SHA-256; prior releases are unchanged.\n')
        subprocess.run(['gh','release','create',TAG,'--repo',REPO,'--draft','--target',index['source_commit'],'--title','WasserMan-Bench current/contact study v1 — private evidence','--notes-file',str(note)],check=True)
        releases=[r for r in api_pages(f'repos/{REPO}/releases?per_page=100') if r['tag_name']==TAG]
    assert len(releases)==1 and releases[0]['draft']
    release=releases[0];receipt=dict(repository=REPO,private=True,draft=True,tag=TAG,release_id=release['id'],release_url=release['html_url'],source_commit=index['source_commit'],index_sha256=digest(D/'index.json'),assets=[])
    for item in files:
        assets=api_pages(f"repos/{REPO}/releases/{release['id']}/assets?per_page=100")
        matches=[r for r in assets if r['name']==item['path'].name]
        if not matches:
            subprocess.run(['gh','release','upload',TAG,str(item['path']),'--repo',REPO],check=True)
            matches=[r for r in api_pages(f"repos/{REPO}/releases/{release['id']}/assets?per_page=100") if r['name']==item['path'].name]
        assert len(matches)==1 and matches[0]['size']==item['bytes']
        asset=matches[0];verify_download(REPO,asset['id'],item['sha256'],item['bytes'])
        receipt['assets'].append(dict(name=item['path'].name,asset_id=asset['id'],bytes=item['bytes'],sha256=item['sha256'],authenticated_download_verified=True))
        (P/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
