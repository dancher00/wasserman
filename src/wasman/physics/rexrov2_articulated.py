"""Base added-mass closure including moving-arm rigid-body reaction.

Native body masses, base added mass, buoyancy, drag and motor parameters are
unchanged. Pinocchio RNEA supplies the current base inertia and articulated bias.
Measured joint accelerations and contact wrench are one-step estimates; this is
an explicit approximation that requires timestep convergence for each study.
No arm added mass or hardware calibration is implied.
"""
from pathlib import Path
import numpy as np
import pinocchio as pin
import torch
from wasman.physics.rexrov2 import HeldArmRexHydrodynamics, cross_bias
from wasman.controllers.rexrov2_workspace import URDF


class ArticulatedRexHydrodynamics(HeldArmRexHydrodynamics):
    def __init__(self, parameters, device, joint_names, urdf=URDF):
        super().__init__(parameters, device)
        self.model=pin.buildModelFromUrdf(str(urdf),pin.JointModelFreeFlyer())
        self.model.gravity.linear[:]=0
        self.data=self.model.createData()
        self.q_indices=[self.model.joints[self.model.getJointId(n)].idx_q for n in joint_names]
        self.v_indices=[self.model.joints[self.model.getJointId(n)].idx_v for n in joint_names]
        self.previous_velocity=None
        self.current_inertia=None
        self.rigid_bias=None
        self.joint_acceleration=None

    def update_articulation(self, position, velocity, body_twist, dt):
        qj=position.detach().cpu().numpy();vj=velocity.detach().cpu().numpy();base=body_twist.detach().cpu().numpy()
        acceleration=np.zeros_like(vj) if self.previous_velocity is None else (vj-self.previous_velocity)/dt
        self.previous_velocity=vj.copy()
        matrices=[];biases=[]
        for joints,rates,acc,nu in zip(qj,vj,acceleration,base,strict=True):
            q=pin.neutral(self.model);q[self.q_indices]=joints
            v=np.zeros(self.model.nv);v[:6]=nu;v[self.v_indices]=rates
            a=np.zeros(self.model.nv);a[self.v_indices]=acc
            matrices.append(pin.crba(self.model,self.data,q)[:6,:6].copy())
            biases.append(pin.rnea(self.model,self.data,q,v,a)[:6].copy())
        self.current_inertia=torch.as_tensor(np.stack(matrices),dtype=body_twist.dtype,device=body_twist.device)
        self.rigid_bias=torch.as_tensor(np.stack(biases),dtype=body_twist.dtype,device=body_twist.device)
        self.joint_acceleration=torch.as_tensor(acceleration,dtype=body_twist.dtype,device=body_twist.device)

    def close_added_mass(self, body_twist, current_b, non_added_external):
        if self.current_inertia is None:raise RuntimeError('Call update_articulation before closing added mass')
        relative=body_twist.clone();relative[:,:3]-=current_b
        c=torch.zeros_like(body_twist);c[:,:3]=torch.cross(body_twist[:,3:],current_b,dim=-1)
        added_bias=cross_bias(self.added,relative)
        rhs=non_added_external-self.rigid_bias-added_bias-c@self.added.T
        acceleration=torch.linalg.solve(self.current_inertia+self.added,rhs.unsqueeze(-1)).squeeze(-1)
        return -(acceleration+c)@self.added.T-added_bias,acceleration
