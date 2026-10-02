"""Check semantic tool-frame calibration and native action bounds."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from wasman.controllers.embodiment_ik import EmbodimentIK
pytestmark=pytest.mark.unit

def test_intrinsic_gripper_roll_does_not_command_unreachable_wrist_rotation():
 roll=1.942;orientation=[np.sin(roll/2),0,0,np.cos(roll/2)]
 def proxy(value):return NS(torch=torch.tensor(value,dtype=torch.float32))
 jac=torch.zeros(1,2,6,7);jac[0,1,3,6]=1
 robot=NS(device='cpu',data=NS(body_link_quat_w=proxy([[[0,0,0,1],orientation]]),body_link_pos_w=proxy([[[0,0,0],[1,0,0]]]),body_link_jacobian_w=NS(torch=jac),joint_pos=proxy([[0]]),soft_joint_pos_limits=proxy([[[-3,3]]])))
 ik=EmbodimentIK(robot,0,1,[0],1/30,[0,0,0],[.35],torch.zeros(1,3),torch.zeros(1,1),orientation)
 base,arm=ik.targets(torch.tensor([[1.,0,0]]),torch.tensor([[0.,0,0,1]]))
 assert torch.allclose(base,torch.zeros_like(base),atol=1e-6)
 assert torch.allclose(arm,torch.zeros_like(arm),atol=1e-6)

def test_adapter_bounds_prevent_hidden_action_clipping():
 def proxy(value):return NS(torch=torch.tensor(value,dtype=torch.float32))
 jac=torch.zeros(1,2,6,7);jac[0,1,3,6]=1
 robot=NS(device='cpu',data=NS(body_link_quat_w=proxy([[[0,0,0,1],[0,0,0,1]]]),body_link_pos_w=proxy([[[0,0,0],[1,0,0]]]),body_link_jacobian_w=NS(torch=jac),joint_pos=proxy([[0]]),soft_joint_pos_limits=proxy([[[-3,3]]])))
 ik=EmbodimentIK(robot,0,1,[0],1/30,[0,0,0],[.35],torch.zeros(1,3),torch.zeros(1,1),arm_reference_limits=torch.tensor([[[-.05,.05]]]))
 for _ in range(30):_,arm=ik.targets(torch.tensor([[1.,0,0]]),torch.tensor([[.70710678,0,0,.70710678]]))
 assert float(arm.abs().max())<=.050001

def test_persistent_tracking_error_accumulates_reference_with_native_rate_limit():
 def proxy(value):return NS(torch=torch.tensor(value,dtype=torch.float32))
 jac=torch.zeros(1,2,6,7)
 robot=NS(device='cpu',data=NS(body_link_quat_w=proxy([[[0,0,0,1],[0,0,0,1]]]),body_link_pos_w=proxy([[[0,0,0],[1,0,0]]]),body_link_jacobian_w=NS(torch=jac),joint_pos=proxy([[0]]),soft_joint_pos_limits=proxy([[[-3,3]]])))
 ik=EmbodimentIK(robot,0,1,[0],1/30,[0,0,0],[.35],torch.zeros(1,3),torch.zeros(1,1))
 refs=[]
 for _ in range(60):
  base,_=ik.targets(torch.tensor([[1.,0,.01]]),torch.tensor([[0.,0,0,1]]));refs.append(base.clone())
 # A compliant inner loop must receive a growing correction, not the same biased target.
 assert refs[-1][0,2]>.015
 increments=torch.diff(torch.stack(refs),dim=0)
 assert increments.abs().max()<=.1/30+1e-7

 # Continued impossible motion must stop accumulating after half a second of lead.
 for _ in range(600):
  base,_=ik.targets(torch.tensor([[1.,0,.01]]),torch.tensor([[0.,0,0,1]]))
 assert .049 <= base[0,2] <= .054
 # Reversing the requested direction must release the integrator immediately.
 before=base.clone()
 base,_=ik.targets(torch.tensor([[1.,0,-.01]]),torch.tensor([[0.,0,0,1]]))
 assert base[0,2]<before[0,2]
 # First contact freezes the acquired load bias without a command jump.
 before=base.clone()
 base,_=ik.targets(torch.tensor([[1.,0,-.01]]),torch.tensor([[0.,0,0,1]]),torch.tensor([True]))
 assert torch.allclose(base,before,atol=1e-7)
 for _ in range(60):
  base,_=ik.targets(torch.tensor([[1.,0,-.01]]),torch.tensor([[0.,0,0,1]]),torch.tensor([False]))
 assert torch.allclose(base,before,atol=1e-6)
 assert ik.contact_latched.item()

def test_reference_ik_keeps_joint_plan_when_arm_tracking_deflects(tmp_path):
 from wasman.controllers.embodiment_ik import ReferenceEmbodimentIK
 urdf=tmp_path/'arm.urdf'
 urdf.write_text('''<robot name="test"><link name="base"/><link name="arm"/><link name="tool"/>
 <joint name="joint" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="1"/></joint>
 <joint name="tip" type="fixed"><parent link="arm"/><child link="tool"/><origin xyz="1 0 0"/></joint></robot>''')
 def proxy(value):return NS(torch=torch.tensor(value,dtype=torch.float32))
 def robot():return NS(device='cpu',body_names=['base','arm','tool'],joint_names=['joint'],data=NS(body_link_quat_w=proxy([[[0,0,0,1]]*3]),body_link_pos_w=proxy([[[0,0,0],[0,0,0],[1,0,0]]]),joint_pos=proxy([[0]]),soft_joint_pos_limits=proxy([[[-3,3]]])))
 robots=[robot(),robot()]
 controllers=[ReferenceEmbodimentIK(r,0,2,[0],1/30,[0,0,0],[.35],torch.zeros(1,3),torch.zeros(1,1),urdf=urdf) for r in robots]
 robots[1].data.joint_pos.torch[:]=.2
 robots[1].data.body_link_pos_w.torch[:,2]=torch.tensor([np.cos(.2),np.sin(.2),0])
 robots[1].data.body_link_quat_w.torch[:,2]=torch.tensor([0,0,np.sin(.1),np.cos(.1)])
 for _ in range(90):
  commands=[c.targets(torch.tensor([[1.,.05,0]]),torch.tensor([[0.,0,0,1.]]),torch.tensor([False])) for c in controllers]
  assert torch.allclose(commands[0][1],commands[1][1],atol=1e-7)
 assert torch.allclose(commands[1][0]-commands[0][0],torch.tensor([[1-np.cos(.2),-np.sin(.2),0]],dtype=torch.float32),atol=1e-5)
 # A completed press holds native commands even if the next scripted goal advances.
 before=commands[1]
 after=controllers[1].targets(torch.tensor([[1.1,.05,0]]),torch.tensor([[0.,0,0,1.]]),torch.tensor([True]))
 assert all(torch.equal(x,y) for x,y in zip(before,after))

 # Global base tilt must not be mistaken for arm deflection by the level-frame adapter.
 r=robot()
 c=ReferenceEmbodimentIK(r,0,2,[0],1/30,[0,0,0],[.35],torch.zeros(1,3),torch.zeros(1,1),urdf=urdf,offset_frame='level')
 r.data.body_link_quat_w.torch[:]=torch.tensor([0,0,np.sin(.1),np.cos(.1)])
 r.data.body_link_pos_w.torch[:,2]=torch.tensor([np.cos(.2),np.sin(.2),0])
 for _ in range(30):
  base,arm=c.targets(torch.tensor([[1.,0,0]]),torch.tensor([[0.,0,0,1.]]))
 assert torch.allclose(base,torch.zeros_like(base),atol=1e-6)
 assert torch.allclose(arm,torch.zeros_like(arm),atol=1e-6)
