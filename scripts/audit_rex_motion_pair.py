"""Compare matched development trajectories at two physics rates."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def audit(a,b):
 x,y=(np.load(p/'trace.npz') for p in [a,b]);ra,rb=(json.loads((p/'report.json').read_text()) for p in [a,b]);n=len(ra['seeds']);assert ra['seeds']==rb['seeds'];assert ra['control_mode']==rb['control_mode'];assert np.allclose(x['time'],y['time'],atol=1e-8)
 mask=x['time']>=5
 diff=x['tool_position'][mask,:n]-y['tool_position'][mask,:n]
 tcp=float(np.sqrt(np.square(diff).sum(-1).mean()))
 qdiff=float(np.abs(x['joint_position'][mask,:n]-y['joint_position'][mask,:n]).max())
 dot=np.abs((x['quaternion'][mask,:n]*y['quaternion'][mask,:n]).sum(-1)).clip(0,1)
 angle=float(np.rad2deg(2*np.arccos(dot)).max())
 finite=all(np.isfinite(t[key]).all() for t in [x,y] for key in t.files)
 motor=max(float(np.abs(t['motor_force']).max()) for t in [x,y])
 limits=np.array([.17,.17,.15,.25,.30,.15,.15,.15]);speed=max(float((np.abs(t['joint_velocity'][:,:n])-limits).max()) for t in [x,y])
 checks={'finite':finite,'motor_limits':motor<=1540.001,'tcp_rms_below_10mm':tcp<=.01,'attitude_max_difference_below_half_degree':angle<=.5,'joint_difference_below_002rad':qdiff<=.02,'sampled_joint_speed_excess_below_1percent':speed<=float(limits.max()*.01)}
 result={'passed':all(checks.values()),'checks':checks,'dt':[ra['dt'],rb['dt']],'seeds':ra['seeds'],'window':'5 seconds onward; powered development episodes only','tcp_rms_difference_mm':tcp*1000,'maximum_attitude_difference_deg':angle,'maximum_joint_difference_rad':qdiff,'maximum_motor_force_N':motor,'maximum_sampled_joint_speed_excess_rad_s':speed,'successes':[ra['successes'],rb['successes']],'trace_sha256':[hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest() for p in [a,b]],'scope':'Selected development motion/solver comparison, not validation of general hydrodynamics or hardware'}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.first,a.second)
 with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(r,indent=2))
