"""Bind six existing models and audit the proposed study's seed separation."""
import json, subprocess
from pathlib import Path
from current_contact_study import ROOT, digest
from wasman.learning.revision_protocol import validate_runtime_snapshot

out=ROOT/'research/current-contact-5090'
if (out/'protocol.json').exists():
    raise RuntimeError('Do not overwrite a registered protocol')
models={}
for task in ['RotateValve','OpenHatch']:
    for seed in [17,43,101]:
        folder=ROOT/f'artifacts/revision-v2/core/models/{task}-DP-{seed}'
        config=json.loads((folder/'config.json').read_text()); completed=json.loads((folder/'completed.json').read_text())
        assert config['asset_profile']=='open-procedural-v1' and config['protocol']=='wm-open-v2-20260929'
        assert config['model']=='DP' and config['seed']==seed and not config['pilot'] and completed['samples']==320000
        h=digest(folder/'final.pt');assert h==completed['final_sha256']
        models[f'{task}-DP-{seed}']={k:str((folder/v).relative_to(ROOT)) for k,v in [('checkpoint_path','final.pt'),('config_path','config.json')]}
        models[f'{task}-DP-{seed}'].update(checkpoint_sha256=h,config_sha256=digest(folder/'config.json'),completed_sha256=digest(folder/'completed.json'),rollout_seed=config['rollout_seed'])
resets={t:dict(expert_pilot=list(range(9100000+100*i,9100002+100*i)),benchmark=list(range(9110000+100*i,9110030+100*i)),evaluation=list(range(9120000+100*i,9120030+100*i))) for i,t in enumerate(['RotateValve','OpenHatch'],1)}
proposed={v for d in resets.values() for ids in d.values() for v in ids}
seen=set(); sources=[]
def collect(v, seed_context=False):
    if isinstance(v,dict):
        for k,x in v.items():collect(x,seed_context or 'seed' in k.lower())
    elif isinstance(v,list):
        for x in v:collect(x,seed_context)
    elif seed_context and type(v)==int:
        seen.add(v)
paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','*.json','-g','!.git/**','research','artifacts'],cwd=ROOT,text=True).splitlines()
for name in paths:
    p=ROOT/name
    if p.stat().st_size>40*1024**2:continue
    if p.name not in ('summary.json','report.json','metadata.json','config.json','protocol.json','campaign.json') and not name.startswith('research/'):
        continue
    d=json.loads(p.read_text());collect(d);sources.append(name)
# Cover reserved but unexecuted core streams as well as actual recorded IDs.
reserved=set(range(30000,100000))
overlap=sorted(proposed&(seen|reserved));assert not overlap,overlap
assert len(proposed)==sum(len(ids) for d in resets.values() for ids in d.values())
audit=dict(scanned_files=len(sources),scanned_paths=sources,recorded_seed_values=len(seen),proposed=resets,overlap=overlap,reserved_conservative_range=[30000,99999],scope='All JSON seed fields in recorded metadata/reports/configs and research records; includes unrelated training/bootstrap seeds conservatively. Reserved whole 30000..99999 range additionally excludes unused core development/research slots.')
(out/'seed-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
protocol=dict(study='wm-current-contact-5090-v1',status='pilot_pending',parent_protocol='wm-open-v2-20260929',parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),asset_profile='open-procedural-v1',speeds_m_s=[0.,.1,.2],current_axis_world=[0,1,0],models=models,resets=resets,main_episodes=540,runtime_manifest_sha256=validate_runtime_snapshot(),seed_audit_sha256=digest(out/'seed-audit.json'))
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps({'models':models,'resets':resets,'overlap':overlap,'scanned_files':len(sources)},indent=2))
