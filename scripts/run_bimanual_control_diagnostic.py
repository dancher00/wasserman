"""One predefined control counterfactual; no training or final-state evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--wait-for-pid',type=int)
p.add_argument('--grasp-effort',type=float)
p.add_argument('--feedback-frame',choices=['world','level'],default='world')
p.add_argument('--phase',choices=['initialization','first-turn'],default='first-turn')
p.add_argument('--solver-position-iterations',type=int,default=12)
p.add_argument('--solver-velocity-iterations',type=int,default=2)
p.add_argument('--balance-initial-load',action='store_true')
p.add_argument('--spawn-fixture-at-reset-pose',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
(a.output/'diagnostic_runner.py').write_bytes(Path(__file__).read_bytes())
sources=['scripts/probe_bimanual_valve.py','src/wasman/physics/contact_wrench.py',*[str(f.relative_to(root)) for f in sorted((root/'src/wasman/controllers').glob('bimanual*.py'))]]
hashes={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources}
state={'status':'queued','scope':a.phase+' diagnostic only','seconds':15 if a.phase=='initialization' else 45,'solver_iterations':[a.solver_position_iterations,a.solver_velocity_iterations],'grasp_effort_Nm':a.grasp_effort,'feedback_frame':a.feedback_frame,'balance_initial_load':a.balance_initial_load,'seeds':[91000,91001,91002],'source_sha256':hashes,'completed':[]}
state['fixtures_spawned_at_reset_pose']=a.spawn_fixture_at_reset_pose
def save():
 tmp=a.output/'status-next.json';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'status.json')
def call(cmd,log):
 with log.open('w') as stream: subprocess.run([sys.executable,*cmd],cwd=root,env={**os.environ,'OMP_NUM_THREADS':'4'},stdout=stream,stderr=subprocess.STDOUT,check=True)
save()
try:
 if a.wait_for_pid:
  while True:
   try:os.kill(a.wait_for_pid,0)
   except ProcessLookupError:break
   time.sleep(5)
 for hz in [240,480]:
  assert all(hashlib.sha256((root/s).read_bytes()).hexdigest()==h for s,h in hashes.items()),'Source changed'
  run=a.output/str(hz);state.update(status='running',active=str(run));save()
  call(['scripts/probe_bimanual_valve.py','--fixture','large','--mode','two-hands','--seeds','91000','91001','91002','--seconds',str(state['seconds']),'--solver-position-iterations',str(a.solver_position_iterations),'--solver-velocity-iterations',str(a.solver_velocity_iterations),'--dt',str(1/hz),'--feedback-frame',a.feedback_frame,*(['--spawn-fixture-at-reset-pose'] if a.spawn_fixture_at_reset_pose else []),*(['--balance-initial-load'] if a.balance_initial_load else []),*(['--grasp-effort',str(a.grasp_effort)] if a.grasp_effort is not None else []),'--output-dir',str(run)],a.output/f'{hz}.log')
  call(['scripts/analyze_bimanual_trace.py',str(run)],a.output/f'{hz}-replay.log')
  call(['scripts/audit_bimanual_trace.py',str(run)],a.output/f'{hz}-geometry.log')
  replay=json.loads((run/'independent-replay.json').read_text());geometry=json.loads((run/'geometry-audit.json').read_text())
  assert replay['finite'] and replay['recorded_success_agrees'] and not replay['episodes'][-1]['success']
  for row in geometry['episodes']:
   assert not row['cad_collisions'],f"CAD contact: {row['seed']}"
   assert row['max_joint_limit_excess_rad']<=.002
   assert max(row['tcp_fk_max_error_m'].values())<1e-4
  for row in (replay['episodes'][:-1] if a.phase=='first-turn' else []):
   turn=row['phases'].get('turn1')
   assert turn and row['max_angle_deg']>=80,'First turn not completed'
   assert min(turn['bilateral_contact_fraction'][s+'_wheel'] for s in ['left','right'])>=.99,'First-turn contact loss'
  state['completed'].append(str(run));save()
 gate=a.output/'numerical-gate.json'
 call(['scripts/audit_bimanual_numerics.py',str(a.output/'240'),str(a.output/'480'),'--output',str(gate)],a.output/'numerical-gate.log')
 assert json.loads(gate.read_text())['passed'],'Numerical gate failed'
 state.update(status='short diagnostic passed; full cycle not tested',active=None);save()
except BaseException as exc:
 state.update(status='stopped',error=repr(exc));save();raise
