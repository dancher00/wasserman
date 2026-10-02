"""Inspect short-process checks and seal the study before main inference."""
import json,time,subprocess
from pathlib import Path
import torch
from current_contact_study import ROOT,digest
P=ROOT/'research/current-contact-5090';A=ROOT/'artifacts/current-contact-5090'

assert json.loads((P/'pilot-audit.json').read_text())['passed']
checks=[]
for task in ['OpenHatch','RotateValve']:
    run={}
    for mode in ['sequential','concurrent']:
        label=f'benchmark-{task}-DP-17-u0-{mode}';folder=A/'benchmark'/label
        job=json.loads((A/'jobs'/f'{label}.json').read_text());assert job['returncode']==0
        c=json.loads((folder/'current-condition.json').read_text());assert c['scored_steps']==180
        data=torch.load(folder/'rollout.pt',map_location='cpu',weights_only=True)
        phys=torch.load(folder/'current-physical.pt',map_location='cpu',weights_only=True)
        assert len(data['trace'])==180
        for row in data['trace']:
            assert all(torch.isfinite(v).all() for v in row.values() if isinstance(v,torch.Tensor))
        for prediction in data['predictions']:assert torch.isfinite(prediction['chunk']).all()
        samples=[json.loads(x)['sample'] for x in (A/'jobs'/f'{label}.gpu.jsonl').read_text().splitlines()]
        peak=max(float(x.split(',')[1]) for x in samples if x)
        run[mode]=(job,c,data,phys,peak,json.loads((folder/'resolved-config.json').read_text()))
    seq,con=run['sequential'],run['concurrent']
    same_config=seq[5]==con[5]
    keys=seq[3]['initial']['before_current_and_warmup']
    same_initial=all(torch.equal(v,con[3]['initial']['before_current_and_warmup'][k]) for k,v in keys.items())
    first=(seq[2]['predictions'][0]['chunk']-con[2]['predictions'][0]['chunk']).abs().max().item()
    checks.append(dict(task=task,config_equal=same_config,initial_equal=same_initial,first_prediction_max_abs=first,
        sequential_seconds=seq[0]['elapsed_s'],concurrent_seconds=con[0]['elapsed_s'],
        sequential_rollout_seconds=seq[1]['wrapped_runtime_seconds'],concurrent_rollout_seconds=con[1]['wrapped_runtime_seconds'],
        concurrent_peak_total_gpu_mib=con[4],sequential_peak_total_gpu_mib=seq[4],
        short_horizon_steps=180,envs=30))
sequential=sum(c['sequential_seconds'] for c in checks);concurrent=max(c['concurrent_seconds'] for c in checks)
workers=2 if concurrent<sequential and all(c['config_equal'] and c['initial_equal'] and c['first_prediction_max_abs']<1e-4 and c['concurrent_peak_total_gpu_mib']<32607-4096 for c in checks) else 1
performance=dict(checks=checks,process_workers=workers,sequential_pair_seconds=sequential,concurrent_pair_seconds=concurrent,
    scope='Short performance and configuration/initial-state comparability check; not a determinism study or an assertion of identical full trajectories.')
(P/'performance.json').write_text(json.dumps(performance,indent=2)+'\n')
plan=P/'PROTOCOL.md'
text=plan.read_text().replace('Status: preparation/expert pilot; the machine-readable protocol will be frozen\nand committed before the 540-episode evaluation.', 'Status: frozen after the expert pilot and short performance checks, before\nthe 540-episode evaluation. Exact commitment: `freeze.json`.')
plan.write_text(text)
p=P/'protocol.json';protocol=json.loads(p.read_text());assert protocol['status']=='pilot_pending'
protocol.update(status='frozen',frozen_unix=time.time(),process_workers=workers,analysis_plan_sha256=digest(P/'PROTOCOL.md'),
    pilot_audit_sha256=digest(P/'pilot-audit.json'),performance_sha256=digest(P/'performance.json'),
    control_steps={'OpenHatch':1790,'RotateValve':2240},prefix_steps=600,bootstrap_draws=20000,bootstrap_rng_seed=20261001)
p.write_text(json.dumps(protocol,indent=2)+'\n')
files=['scripts/current_contact_study.py','scripts/run_current_contact.py','scripts/current_contact_metrics.py','scripts/analyze_current_contact.py']
freeze=dict(protocol_sha256=digest(p),analysis_plan_sha256=digest(P/'PROTOCOL.md'),study_sources_sha256={f:digest(ROOT/f) for f in files},
    parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),frozen_unix=protocol['frozen_unix'],main_inference_started=False)
(P/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print(json.dumps(performance,indent=2))
