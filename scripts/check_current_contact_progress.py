"""Bounded progress and incremental physical QC; final analysis rechecks everything."""
import json
from pathlib import Path
import torch
from current_contact_study import ROOT,digest
from revision_scoring_evidence import check_embedded_report,checked_tensor_outcomes
P=ROOT/'research/current-contact-5090';A=ROOT/'artifacts/current-contact-5090'
path=P/'interim-rescoring.json'
record=json.loads(path.read_text()) if path.exists() else {'scope':'Incremental QC of completed cells; not an interim stopping/tuning rule','cohorts':{}}
running=[];new=[]
for job in sorted((A/'jobs').glob('evaluation*.json')):
    j=json.loads(job.read_text());label=job.stem
    if 'returncode' not in j:
        steps=[l.split(' angle=')[0] for l in job.with_suffix('.log').read_text().splitlines() if l.startswith('step=')]
        running.append(dict(label=label,last_progress=steps[-1] if steps else 'startup'))
        continue
    assert j['returncode']==0,(label,'failed job')
    folder=A/'evaluation'/label
    if label in record['cohorts']:
        assert record['cohorts'][label]['report_sha256']==digest(folder/'summary.json')
        continue
    r=json.loads((folder/'summary.json').read_text());d=torch.load(folder/'rollout.pt',map_location='cpu',weights_only=True)
    task='OpenHatch' if 'OpenHatch' in label else 'RotateValve'
    check_embedded_report(d['summary'],r);checked_tensor_outcomes(task,r,d['trace'])
    record['cohorts'][label]=dict(physical_rescoring_passed=True,successes=r['successes'],episodes=r['episodes'],report_sha256=digest(folder/'summary.json'),trace_sha256=digest(folder/'rollout.pt'))
    new.append(dict(label=label,successes=r['successes'],episodes=r['episodes']))
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(physically_verified=len(record['cohorts']),total_cohorts=18,new=new,running=running)))
