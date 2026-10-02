"""Physically rescore all 540 episodes and reproduce the registered paired analysis."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from current_contact_study import ROOT,digest
from current_contact_metrics import arrays,episode_metrics,paired_interval,PREFIX,DT
from revision_scoring_evidence import checked_tensor_outcomes,check_embedded_report

P=ROOT/'research/current-contact-5090';A=ROOT/'artifacts/current-contact-5090'
SEEDS=[17,43,101];TASKS=['OpenHatch','RotateValve'];SPEEDS=[0.,.1,.2]

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def mean_sd(values):
    a=np.asarray([x for x in values if x is not None],float)
    return dict(mean=float(a.mean()) if len(a) else None,sd=float(a.std(ddof=1)) if len(a)>1 else None,n=len(a))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=P/'results');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((P/'protocol.json').read_text());assert protocol['status']=='frozen'
    freeze=json.loads((P/'freeze.json').read_text());assert digest(P/'protocol.json')==freeze['protocol_sha256']
    loaded={};evidence=[];episodes=[];pairing=[]
    for task in TASKS:
        for seed in SEEDS:
            for speed in SPEEDS:
                label=f'evaluation-{task}-DP-{seed}-u{speed:g}';folder=A/'evaluation'/label
                r=json.loads((folder/'summary.json').read_text());condition=json.loads((folder/'current-condition.json').read_text())
                job=json.loads((A/'jobs'/f'{label}.json').read_text());assert job['returncode']==0
                data=torch.load(folder/'rollout.pt',map_location='cpu',weights_only=True)
                phys=torch.load(folder/'current-physical.pt',map_location='cpu',weights_only=True)
                check_embedded_report(data['summary'],r);checked_tensor_outcomes(task,r,data['trace'])
                assert r['seeds']==protocol['resets'][task]['evaluation'] and r['policy_random_seed']==42
                model=protocol['models'][f'{task}-DP-{seed}']
                assert r['checkpoint_sha256']==model['checkpoint_sha256']
                assert r['inference_precision']=='fp32' and not r['expert_at_inference']
                assert condition['protocol_sha256']==freeze['protocol_sha256']
                assert condition['wrapper_sha256']==freeze['study_sources_sha256']['scripts/current_contact_study.py']
                assert condition['speed_m_s']==speed and condition['control_dt']==DT
                config=json.loads((folder/'resolved-config.json').read_text())
                assert digest(folder/'resolved-config.json')==condition['config_sha256']
                for row in [phys['initial']['before_current_and_warmup'],phys['initial']['after_warmup'],*phys['trace']]:
                    assert all(torch.isfinite(v).all() for v in row.values()),(label,'nonfinite physical evidence')
                a=arrays(data,phys)
                assert len(a['valid'])==condition['scored_steps']
                for step,row in enumerate(phys['trace']):
                    valid=a['valid'][step]
                    expected=torch.tensor([0.,speed,0.],dtype=row['current'].dtype).expand_as(row['current'])
                    assert torch.equal(row['current'][valid],expected[valid]),(label,step,'current changed')
                    assert torch.equal(row['terminated'],data['trace'][step]['reset'])
                    assert torch.equal(row['motor_force'],data['trace'][step]['motor_force'])
                limits=condition['motor_force_limits_N']
                per=[]
                for j,reset in enumerate(r['seeds']):
                    n=min(PREFIX,len(a['valid']));metrics=episode_metrics(a,j,a['valid'][:n,j],limits)
                    success=bool(r['success_per_seed'][j]);time=r['first_success_step'][j]*DT if success else None
                    mask=a['valid'][:,j];angle=a['angle'][mask,j]
                    row=dict(task=task,training_seed=seed,speed_m_s=speed,reset_id=reset,success=success,
                        success_time_s=time,capped_completion_time_s=time if success else r['evaluated_steps']*DT,
                        first_reset_step=r['first_reset_step'][j],observed_full_duration_s=float(mask.sum()*DT),
                        final_angle_deg=float(np.rad2deg(angle[-1])) if len(angle) else None,**metrics)
                    episodes.append(row);per.append(row)
                loaded[task,seed,speed]=(r,a,phys,config,per,limits)
                evidence.append(dict(label=label,physical_rescoring_passed=True,episodes=30,
                    source_files={str((folder/name).relative_to(ROOT)):digest(folder/name) for name in ['summary.json','rollout.pt','current-physical.pt','current-condition.json','resolved-config.json']},
                    job_sha256=digest(A/'jobs'/f'{label}.json')))
            baseline=loaded[task,seed,0.]
            for speed in [.1,.2]:
                other=loaded[task,seed,speed]
                assert baseline[3]==other[3],(task,seed,speed,'full config mismatch')
                left=baseline[2]['initial']['before_current_and_warmup'];right=other[2]['initial']['before_current_and_warmup']
                assert left.keys()==right.keys()
                differences={k:float((v.to(torch.float64)-right[k].to(torch.float64)).abs().max()) for k,v in left.items()}
                assert all(v==0 for v in differences.values()),(task,seed,speed,differences)
                x=baseline[2]['initial']['before_first_command'];y=other[2]['initial']['before_first_command']
                pairing.append(dict(task=task,training_seed=seed,speed_m_s=speed,pre_current_exact=True,config_exact=True,
                    pre_current_max_abs_by_field=differences,
                    pre_first_command_position_max_abs_m=float((x['base_position']-y['base_position']).abs().max()),
                    pre_first_command_quaternion_max_abs=float((x['base_quaternion']-y['base_quaternion']).abs().max())))
    seed_rows=[];groups=[]
    early_keys=[k for k in episodes[0] if k not in ['task','training_seed','speed_m_s','reset_id','success','success_time_s','first_reset_step','metrics_available']]
    for task in TASKS:
        for speed in SPEEDS:
            group=[]
            for seed in SEEDS:
                rows=loaded[task,seed,speed][4];acquired=sum(x['contact_acquired'] for x in rows);lost=sum(x['contact_lost'] for x in rows)
                times=[r['success_time_s'] for r in rows if r['success']]
                row=dict(task=task,training_seed=seed,speed_m_s=speed,successes=sum(r['success'] for r in rows),episodes=30,
                    success_rate=sum(r['success'] for r in rows)/30,success_time_mean_s=float(np.mean(times)) if times else None,
                    success_time_n=len(times),acquired_episodes=acquired,lost_episodes=lost,
                    loss_given_acquisition=lost/acquired if acquired else None)
                for k in early_keys:row[k]=mean_sd([r.get(k) for r in rows])['mean']
                seed_rows.append(row);group.append(row)
            groups.append(dict(task=task,speed_m_s=speed,training_runs=3,
                per_training_successes=[r['successes'] for r in group],
                **{k:mean_sd([r.get(k) for r in group]) for k in ['success_rate','success_time_mean_s','loss_given_acquisition',*early_keys]}))
    comparisons=[];paired_episodes=[]
    for task in TASKS:
        for reference,speed in [(0.,.1),(0.,.2),(.1,.2)]:
            left=[];right=[]
            for seed in SEEDS:
                rr,ra,_,_,rp,limits=loaded[task,seed,reference]
                sr,sa,_,_,sp,_=loaded[task,seed,speed]
                n=min(PREFIX,len(ra['valid']),len(sa['valid']));valid=ra['valid'][:n]&sa['valid'][:n]
                lrows=[];rrows=[]
                for j,reset in enumerate(rr['seeds']):
                    rm=episode_metrics(ra,j,valid[:,j],limits);sm=episode_metrics(sa,j,valid[:,j],limits)
                    for d,r in [(rm,rp[j]),(sm,sp[j])]:
                        d.update(success_rate=float(r['success']),capped_completion_time_s=r['capped_completion_time_s'])
                    lrows.append(sm);rrows.append(rm)
                    paired_episodes.append(dict(task=task,training_seed=seed,reference_speed_m_s=reference,speed_m_s=speed,
                        reset_id=reset,joint_exposure_s=rm['exposure_s'],reference=rm,variant=sm))
                left.append(lrows);right.append(rrows)
            keys=set.intersection(*[set(r) for rows in left+right for r in rows]) - {'metrics_available'}
            contrasts={k:paired_interval([[r.get(k) for r in rows] for rows in left],[[r.get(k) for r in rows] for rows in right]) for k in sorted(keys)}
            comparisons.append(dict(task=task,reference_speed_m_s=reference,speed_m_s=speed,
                primary=reference==0.,contrasts=contrasts,
                joint_exposure_s=[[r['exposure_s'] for r in rows] for rows in left]))
    audit=dict(study=protocol['study'],physical_rescoring_passed=True,cohorts=len(evidence),episodes=len(episodes),
        all_episode_outcomes_retained=True,early_resets=[{k:r[k] for k in ['task','training_seed','speed_m_s','reset_id','first_reset_step']} for r in episodes if r['first_reset_step']>=0],
        pairing=pairing,evidence=evidence,analysis_source_sha256=digest(__file__),metrics_source_sha256=digest(ROOT/'scripts/current_contact_metrics.py'))
    result=dict(study=protocol['study'],groups=groups,comparisons=comparisons,
        uncertainty='Paired crossed 20,000-draw percentile bootstrap, RNG 20261001; 3 paired training draws, 30 shared reset draws. Marginal descriptive intervals, no multiplicity adjustment.',
        limits='Simulation current sensitivity; no physical calibration or hardware transfer. Conditional early censoring and 3 training runs limit interpretation.')
    for name,data in [('summary.json',result),('audit.json',audit),('paired-episodes.json',paired_episodes),('episodes.json',episodes)]:
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    write_csv(out/'episodes.csv',episodes);write_csv(out/'training-seeds.csv',seed_rows)
    clips=[]
    for task in TASKS:
        base=loaded[task,17,0.][0];strong=loaded[task,17,.2][0]
        choices=[i for i,(x,y) in enumerate(zip(base['success_per_seed'],strong['success_per_seed'])) if x and not y]
        rule='smallest reset with zero success and stronger-current failure'
        if not choices:
            choices=[i for i,(x,y) in enumerate(zip(base['success_per_seed'],strong['success_per_seed'])) if x!=y];rule='smallest reset with differing outcome'
        if not choices:choices=[0];rule='smallest reset; no outcome contrast'
        j=choices[0]
        for speed,r in [(0.,base),(.2,strong)]:clips.append(dict(task=task,training_seed=17,speed_m_s=speed,reset_id=r['seeds'][j],batch_index=j,selection_rule=rule,scored_success=r['success_per_seed'][j]))
    (P/'video-selection.json').write_text(json.dumps(dict(clips=clips),indent=2)+'\n')
    print(json.dumps(dict(cohorts=len(evidence),episodes=len(episodes),early_resets=len(audit['early_resets']),successes=[(g['task'],g['speed_m_s'],g['per_training_successes']) for g in groups]),indent=2))

if __name__=='__main__':main()
