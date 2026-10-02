"""Locate loss of contact and tracking in a saved development trace."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('--progress',action='store_true')
a=p.parse_args()
d=dict(np.load(io.BytesIO((a.run/('progress.npz' if a.progress else 'trace.npz')).read_bytes())))
spec=importlib.util.spec_from_file_location('saved_plan',a.run/'bimanual_turn_plan.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)
rows=[]
for n in range(d['valve_angle'].shape[1]-1):
 pt=d['plan_time'][:,n]
 phases=np.array([plan.two_hand_plan(float(t))[3] for t in pt])
 expected=np.minimum(np.pi/2,np.maximum(0,pt-18)*.07)
 expected+=np.minimum(np.pi/2,np.maximum(0,pt-plan.TURN_END-2*plan.REGRASP_DURATION)*.07)
 def first(mask):
  indices=np.flatnonzero(mask)
  return None if not len(indices) else float(d['time'][indices[0]])
 bilateral={}
 for side,groups in {'left':([8,9],[10,11]),'right':([19,20],[21,22])}.items():
  force=d['normal_contact_by_body'][:,n,:,0]
  bilateral[side]=np.logical_and.reduce([np.linalg.norm(force[:,g].sum(1),axis=-1)>.5 for g in groups])
 turning=np.isin(phases,['turn1','turn2'])
 row={'environment':n,'first_170_s':first(d['valve_angle'][:,n]>=np.deg2rad(170)),
      'first_turning_contact_loss_s':{s:first(turning & ~v) for s,v in bilateral.items()},
      'first_turning_shaft_error_over_10deg_s':first(turning & (abs(d['valve_angle'][:,n]-expected)>np.deg2rad(10))),
      'first_ik_residual_over_1cm_s':first(d['ik_residual'][:,n].max(-1)>.01),
      'phases':{}}
 for phase in dict.fromkeys(phases):
  mask=phases==phase
  row['phases'][phase]={'start_s':first(mask),'end_s':float(d['time'][np.flatnonzero(mask)[-1]]),
   'shaft_tracking_max_deg':float(np.rad2deg(abs(d['valve_angle'][mask,n]-expected[mask])).max()),
   'attitude_max_deg':float(np.rad2deg(d['attitude_error'][mask,n]).max()),
   'ik_residual_max_m':float(d['ik_residual'][mask,n].max()),
   'tcp_error_max_m':{s:float(np.linalg.norm(d[k][mask,n]-d[g][mask,n],axis=-1).max()) for s,k,g in [('left','left_tool_position','left_target_tool'),('right','tool_position','target_tool')]},
   'bilateral_fraction':{s:float(v[mask].mean()) for s,v in bilateral.items()}}
 rows.append(row)
result={'complete':not a.progress,'last_time_s':float(d['time'][-1]),'episodes':rows}
if not a.progress:(a.run/'execution-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
