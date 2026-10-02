"""Check current assignment and matched states before opening DP outcomes."""
import json
from pathlib import Path
import numpy as np
import torch
from current_contact_study import ROOT, digest

P=ROOT/'research/current-contact-5090'; A=ROOT/'artifacts/current-contact-5090'
rows=[]
for task in ['OpenHatch','RotateValve']:
    reference=None; config=None
    for speed in [0.,.1,.2]:
        suffix='-configfix' if task=='RotateValve' else ''
        folder=A/'pilot'/f'{task}-{speed:g}{suffix}'
        job=json.loads((A/'jobs'/f'expert-{task}-{speed:g}{suffix}.json').read_text());assert job['returncode']==0
        data=torch.load(folder/'current-physical.pt',map_location='cpu',weights_only=True)
        condition=json.loads((folder/'current-condition.json').read_text())
        state=data['initial']['before_current_and_warmup']
        c=json.loads((folder/'resolved-config.json').read_text())
        if reference is None:reference=state;config=c
        assert config==c
        assert state.keys()==reference.keys()
        assert all(torch.equal(v,reference[k]) for k,v in state.items()),(task,speed,'pre-current mismatch')
        assert len(data['trace'])==150
        for r in [data['initial']['after_warmup'],*data['trace']]:
            assert all(torch.isfinite(v).all() for v in r.values())
            assert not r['terminated'].any()
            expected=torch.tensor([0.,speed,0.],dtype=r['current'].dtype).expand_as(r['current'])
            assert torch.equal(r['current'],expected)
        post=data['initial']['after_warmup']
        delta=post['base_position']-state['base_position']
        rows.append(dict(task=task,speed_m_s=speed,elapsed_s=job['elapsed_s'],scored_steps=150,
            before_current_equal=True,config_equal=True,current_samples_verified=151,
            mean_warmup_displacement_xyz_m=delta.mean(0).tolist(),
            max_realized_motor_force_N=float(torch.stack([r['motor_force'].abs().max() for r in data['trace']]).max()),
            trace_sha256=digest(folder/'current-physical.pt'),config_sha256=condition['config_sha256'],
            folder=str(folder.relative_to(ROOT)),launch_wrapper_sha256=job['source_sha256']['scripts/current_contact_study.py']))
record=dict(passed=True,scope='Short unchanged feedback experts; implemented vector, finite response, matched initial physical fields and complete config. Not full-horizon expert feasibility.',rows=rows)
(P/'pilot-audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
