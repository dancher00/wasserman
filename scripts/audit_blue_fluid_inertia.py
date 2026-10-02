"""Linearized delayed-added-inertia stability audit; not nonlinear simulator validation."""
import argparse,ast,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.linalg import eigh
from wasman.physics.hydrodynamics import LinkHydrodynamics
from wasman.assets.grasp_hydrodynamics import geometry_scaled_arm_hydrodynamics
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=ROOT/'src/wasman/assets/bluerov2_alpha.py';tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_neutral_volume','_arm_hydrodynamics'] or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BLUEROV2_ALPHA_HYDRODYNAMICS' for t in n.targets)]
scope={'LinkHydrodynamics':LinkHydrodynamics};exec(compile(ast.Module(nodes,type_ignores=[]),str(source),'exec'),scope);legacy=scope['BLUEROV2_ALPHA_HYDRODYNAMICS'];scaled=geometry_scaled_arm_hydrodynamics(legacy)
urdf=ROOT/'src/wasman/assets/data/robots/bluerov2_alpha_registered_v1/bluerov2_alpha_registered_v1.urdf';model=pin.buildModelFromUrdf(str(urdf),pin.JointModelFreeFlyer());data=model.createData();links={l.get('name'):l for l in ET.parse(urdf).findall('link')}
r=json.loads((a.folder/'report.json').read_text());t=np.load(a.folder/'trace.npz');qi=[model.joints[model.getJointId(n)].idx_q for n in r['joint_names']];result={}
for name,coeffs in [('legacy',legacy),('geometry_scaled',scaled)]:
 vals=[]
 for i in range(0,len(t['time']),15):
  q=pin.neutral(model);q[qi]=t['joint_position'][i,0];M=pin.crba(model,data,q);A=np.zeros_like(M)
  for c in coeffs:
   origin=links[c.name].find('inertial/origin');com=np.array([float(v) for v in origin.get('xyz','0 0 0').split()]) if origin is not None else np.zeros(3)
   J=pin.computeFrameJacobian(model,data,q,model.getFrameId(c.name),pin.ReferenceFrame.LOCAL).copy();J[:3]+=np.cross(J[3:].T,com).T;A+=J.T@np.diag(c.added_mass)@J
  vals.append(float(eigh(A,M,eigvals_only=True)[-1]))
 maximum=max(vals);bound=2/(1+maximum)
 result[name]={'max_generalized_added_to_rigid_inertia_ratio':maximum,'filtered_delayed_acceleration_linear_stability_alpha_upper_bound':bound,'alpha_240hz':.2,'linearized_alpha_stable':.2<bound,'samples':len(vals),'scope':'Linear delayed-inertial feedback only; omits nonlinear contacts/drives. Not proof of full closed-loop stability.'}
encoded=json.dumps(result,indent=2)+'\n'
with a.output.open('x') as f:f.write(encoded)
print(encoded)
