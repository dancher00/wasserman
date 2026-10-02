"""Compare URDF reference kinematics with independently recorded simulator TCPs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('blue',type=Path);p.add_argument('rex',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out={}
for robot,folder,urdf,tool,offset in [('blue',a.blue,'bluerov2_alpha_registered_v1/bluerov2_alpha_registered_v1.urdf','alpha_tool_link',[0,0,0]),('rex',a.rex,'rexrov2_oberon7_centered/rexrov2_oberon7_centered.urdf','oberon_end_effector',[.18,0,0])]:
 r=json.loads((folder/'report.json').read_text());t=np.load(folder/'trace.npz');source=ROOT/'src/wasman/assets/data/robots'/urdf
 model=pin.buildModelFromUrdf(str(source),pin.JointModelFreeFlyer());data=model.createData();indices=[model.joints[model.getJointId(n)].idx_q for n in r['joint_names']];frame=model.getFrameId(tool);errors=[]
 for i in range(0,len(t['time']),6):
  for n in range(len(r['seeds'])):
   q=pin.neutral(model);q[:3]=t['base_position' if robot=='blue' else 'position'][i,n];q[3:7]=t['base_quaternion' if robot=='blue' else 'quaternion'][i,n];q[indices]=t['joint_position'][i,n]
   pin.framesForwardKinematics(model,data,q);placement=data.oMf[frame];pred=placement.translation+placement.rotation@np.array(offset);errors.append(float(np.linalg.norm(pred-t['tool_position'][i,n])))
 out[robot]={'samples':len(errors),'frequency_hz':5,'max_fk_tcp_difference_m':max(errors),'passed':max(errors)<1e-4,'source_urdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'trace_sha256':hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest()}
encoded=json.dumps(out,indent=2)+'\n'
with a.output.open('x') as f:f.write(encoded)
print(encoded)
