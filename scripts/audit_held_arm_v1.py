"""Recompute why the immutable held-arm PressButton trials failed, without simulation."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def audit(root):
 result={'scope':'Recorded trajectories only; no unique causal attribution to morphology or gain tuning','robots':{}}
 for robot in ['blue','rex']:
  folder=root/(robot+'_final');trace=np.load(folder/'trace.npz');report=json.loads((folder/'report.json').read_text());ids=[i for i,s in enumerate(report['seeds']) if s!=42];n=len(ids);travel=trace['button_travel'][:,ids];att=trace['attitude' if robot=='blue' else 'attitude_error'][:,ids]
  predicates={'travel':travel>=.004,'distance':trace['distance'][:,ids]<.13,'alignment':trace['tool_alignment'][:,ids]>.7,'attitude':att<.25,'angular_speed':trace['angular_speed'][:,ids]<.35}
  valid=np.logical_and.reduce(list(predicates.values()));counter=np.zeros(n,dtype=int);success=np.zeros(n,dtype=bool)
  for valid_step in valid:counter=np.where(valid_step,counter+1,0);success|=counter>=4
  late=trace['time']>=15;delta=trace['target_tool'][:,ids]-trace['tool_position'][:,ids]
  record={'source_trace_sha256':hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest(),'source_script_sha256':hashlib.sha256((folder/'run_script.py').read_bytes()).hexdigest(),'episodes':n,'successes_replayed':int(success.sum()),'successes_reported':int(np.array(report['successes'])[ids].sum()),'max_travel_per_episode_mm':(travel.max(0)*1000).tolist(),'late_mean_normal_tracking_error_mm':float(delta[late,:,0].mean()*1000),'late_mean_tool_rmse_mm':float(np.sqrt(np.square(delta[late]).sum(-1).mean())*1000),'late_predicate_pass_fraction':{key:float(value[late].mean()) for key,value in predicates.items()},'maximum_realized_motor_force_N':float(np.abs(trace['motor_force'][:,ids]).max()),'initial_tool_positions_m':np.array(report['initial_tool_position_m'])[ids].tolist(),'reset_offsets_m':np.array(report['initial_offsets_m'])[ids].tolist()}
  if 'contact_force' in trace:record['late_mean_contact_force_N']=trace['contact_force'][late][:,ids].mean((0,1)).tolist()
  else:record['contact_force_limitation']='Blue archive did not log contact wrenches; spring travel proves interaction but does not identify contacting links or total wrench.'
  assert record['successes_replayed']==record['successes_reported']
  result['robots'][robot]=record
 blue,rex=(result['robots'][r] for r in ['blue','rex']);result['paired_reset_offsets_exact']=blue['reset_offsets_m']==rex['reset_offsets_m'];result['maximum_initial_tcp_difference_m']=float(np.abs(np.array(blue['initial_tool_positions_m'])-rex['initial_tool_positions_m']).max())
 result['finding']='All Blue final-window predicates except 4 mm button travel pass; insufficient realized pressing depth is the observed bottleneck. The stored data alone do not isolate controller compliance, gripper contact geometry and hydrodynamics.'
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=audit(ROOT/'artifacts/held_arm_embodiments_20260924');args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps({r:{k:v for k,v in record.items() if k not in ['max_travel_per_episode_mm','initial_tool_positions_m','reset_offsets_m']} for r,record in result['robots'].items()},indent=2))
