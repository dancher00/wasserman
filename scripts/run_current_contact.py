"""Finite current-study jobs, preserving complete ordered policy batches."""
import argparse, concurrent.futures, json, subprocess, time
from pathlib import Path
from current_contact_study import ROOT, digest

PYTHON=str(ROOT/'.venv/bin/python')
P=ROOT/'research/current-contact-5090'
A=ROOT/'artifacts/current-contact-5090'

def launch(stage, task, speed, training_seed=17, *, suffix='', video_index=None):
    protocol=json.loads((P/'protocol.json').read_text())
    expert=stage=='expert-pilot'
    label=f"expert-{task}-{speed:g}{suffix}" if expert else f'{stage}-{task}-DP-{training_seed}-u{speed:g}{suffix}'
    folder=A/('pilot' if expert else stage)/(f'{task}-{speed:g}{suffix}' if expert else label)
    receipt=A/'jobs'/f'{label}.json'
    if receipt.exists():
        d=json.loads(receipt.read_text())
        if d.get('returncode')==0 and (folder/'current-condition.json').exists():
            return
        raise RuntimeError(f'Inspect incomplete job before any retry: {label}')
    key='expert_pilot' if expert else 'benchmark' if stage=='benchmark' else 'evaluation'
    steps=150 if expert else 180 if stage=='benchmark' else 1790 if task=='OpenHatch' else 2240
    command=[PYTHON,'scripts/current_contact_study.py','--task',task,'--stage',stage,'--speed',str(speed),
        '--output-dir',str(folder),'--steps',str(steps),'--seeds',*map(str,protocol['resets'][task][key])]
    if not expert:
        command+=['--checkpoint',str(ROOT/protocol['models'][f'{task}-DP-{training_seed}']['checkpoint_path']),
            '--purpose','development' if stage=='benchmark' else 'research']
    if video_index is not None:command+=['--video-index',str(video_index)]
    subprocess.run([PYTHON,'scripts/run_revision_job.py','--record',str(receipt),'--output-dir',str(folder),'--timeout','2400','--',*command],cwd=ROOT,check=True)
    for name in ['current-condition.json','current-physical.pt','summary.json']:
        if not (folder/name).exists():
            raise RuntimeError(f'Process exit alone is insufficient: missing {folder/name}')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['pilot','benchmark','concurrency','evaluation','video'],required=True);a=p.parse_args()
    if a.stage=='pilot':
        for task in ['OpenHatch','RotateValve']:
            for speed in [0.,.1,.2]:launch('expert-pilot',task,speed)
    elif a.stage=='benchmark':
        assert json.loads((P/'pilot-audit.json').read_text())['passed']
        for task in ['OpenHatch','RotateValve']:launch('benchmark',task,0.,suffix='-sequential')
    elif a.stage=='concurrency':
        assert json.loads((P/'pilot-audit.json').read_text())['passed']
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(launch,'benchmark',task,0.,suffix='-concurrent') for task in ['OpenHatch','RotateValve']]
            for f in futures:f.result()
    elif a.stage=='evaluation':
        protocol=json.loads((P/'protocol.json').read_text());assert protocol['status']=='frozen'
        frozen=json.loads((P/'freeze.json').read_text());assert digest(P/'protocol.json')==frozen['protocol_sha256']
        assert all(digest(ROOT/path)==h for path,h in frozen['study_sources_sha256'].items())
        # Rotate current order across trainings to avoid one constant order.
        for seed,order in [(17,[0.,.1,.2]),(43,[.1,.2,0.]),(101,[.2,0.,.1])]:
            for speed in order:
                tasks=['OpenHatch','RotateValve']
                if protocol['process_workers']==2:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        futures=[pool.submit(launch,'evaluation',task,speed,seed) for task in tasks]
                        for f in futures:f.result()
                else:
                    for task in tasks:launch('evaluation',task,speed,seed)
    else:
        selection=json.loads((P/'video-selection.json').read_text())
        for row in selection['clips']:
            launch('video',row['task'],row['speed_m_s'],row['training_seed'],video_index=row['batch_index'])
    (A/f'completed-{a.stage}.json').write_text(json.dumps(dict(stage=a.stage,finished_unix=time.time()))+'\n')

if __name__=='__main__':main()
