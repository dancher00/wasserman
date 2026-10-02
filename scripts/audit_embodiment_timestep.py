"""Matched 240/480 Hz (or adjacent finer rates) trajectory audit, no success retuning."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def audit(a,b):
 x,y=(dict(np.load(p/'trace.npz',allow_pickle=False)) for p in [a,b]);ra,rb=(json.loads((p/'report.json').read_text()) for p in [a,b]);n=len(ra['seeds'])
 assert ra['seeds']==rb['seeds'] and ra['control_mode']==rb['control_mode'] and ra['stroke_m']==rb['stroke_m']
 assert np.allclose(x['time'],y['time'],atol=1e-8)
 assert ra.get('ik_mode','feedback')==rb.get('ik_mode','feedback')
 assert ra.get('offset_frame','world')==rb.get('offset_frame','world')
 assert ra.get('solver_type',1)==rb.get('solver_type',1)
 assert ra.get('acceleration_cap','legacy')==rb.get('acceleration_cap','legacy')
 assert ra.get('hydro_variant')==rb.get('hydro_variant') and ra.get('hold_travel_m',0)==rb.get('hold_travel_m',0)
 blue='base_quaternion' in x;key='base_quaternion' if blue else 'quaternion';mask=x['time']>=5
 diff=x['tool_position'][mask,:n]-y['tool_position'][mask,:n]
 tcp=float(np.sqrt(np.square(diff).sum(-1).mean()));qdiff=float(np.abs(x['joint_position'][mask,:n]-y['joint_position'][mask,:n]).max())
 dot=np.abs((x[key][mask,:n]*y[key][mask,:n]).sum(-1)).clip(0,1);angle=float(np.rad2deg(2*np.arccos(dot)).max())
 finite=all(np.isfinite(t[k]).all() for t in [x,y] for k in t)
 checks={'finite':finite,'tcp_rms_below_10mm':tcp<=.01,'attitude_max_difference_below_half_degree':angle<=.5,'joint_difference_below_002rad':qdiff<=.02,'success_invariance':ra['successes']==rb['successes']}
 if blue:
  # Older 240 Hz probe encoded its nominal two-step delay in the task config.
  checks['same_physical_motor_delay']=abs(ra.get('thruster_command_delay_s',2*ra['dt'])-rb.get('thruster_command_delay_s',2*rb['dt']))<1e-9
  if ra.get('hydro_variant')=='geometry-scaled':checks['no_powered_hydrodynamic_clipping']=all(np.count_nonzero(t['hydro_clip_counts'][-1,:n])==0 for t in [x,y])
  checks['same_acceleration_filter_time_constant']=abs(ra.get('acceleration_filter_tau_s',-ra['dt']/np.log(.8))-rb.get('acceleration_filter_tau_s',-rb['dt']/np.log(.8)))<1e-9
 else:
  limits=np.array([.17,.17,.15,.25,.30,.15,.15,.15]);speed=max(float((np.abs(t['joint_velocity'][:,:n])-limits).max()) for t in [x,y]);checks['sampled_native_joint_rates']=speed<=.003
 checks={key:bool(value) for key,value in checks.items()}
 return {'configuration':{key:ra.get(key,1 if key=='solver_type' else None) for key in ['offset_frame','solver_type','ik_mode','stroke_m','hold_travel_m','hydro_variant','acceleration_cap']},'passed':all(checks.values()),'checks':checks,'robot':'blue' if blue else 'rex','dt':[ra['dt'],rb['dt']],'seeds':ra['seeds'],'window':'5 seconds onward; powered episodes','tcp_rms_difference_mm':tcp*1000,'maximum_attitude_difference_deg':angle,'maximum_joint_difference_rad':qdiff,'successes':[ra['successes'],rb['successes']],'trace_sha256':[hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest() for p in [a,b]],'scope':'Selected motion/contact numerical check; not general dynamics or hardware validation'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.first,a.second)
 encoded=json.dumps(r,indent=2)+'\n'
 with a.output.open('x') as f:f.write(encoded)
 print(json.dumps(r,indent=2))
