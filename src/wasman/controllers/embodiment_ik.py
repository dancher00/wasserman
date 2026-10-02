"""Shared EE-target IK for floating robots with different arm dimensions.

Base orientation stays level. Damped least-squares solves base translation and
native arm joints. Targets go through the existing finite-force actuators/PID;
this controller never writes robot or mechanism poses during a rollout.
"""
import torch
from wasman.controllers.tool_pose import damped_step


class EmbodimentIK:
    def __init__(self,robot,base_id,tool_id,arm_ids,dt,tool_offset,arm_speeds,position,arm_position,tool_orientation_offset=None,arm_reference_limits=None):
        self.robot=robot;self.base_id=base_id;self.tool_id=tool_id;self.arm_ids=arm_ids;self.dt=dt
        self.offset=torch.as_tensor(tool_offset,device=robot.device,dtype=torch.float32)
        self.weights=torch.tensor([.35]*3+[1.]*len(arm_ids),device=robot.device)
        self.max_speed=torch.tensor([.10]*3+list(arm_speeds),device=robot.device)
        self.reference=torch.cat((position,arm_position),-1).clone()
        self.posture=arm_position.clone()
        self.orientation_offset=torch.tensor([0,0,0,1] if tool_orientation_offset is None else tool_orientation_offset,device=robot.device,dtype=torch.float32)
        self.reference_limits=arm_reference_limits
        self.contact_latched=torch.zeros(len(position),dtype=torch.bool,device=robot.device)
        self.contact_bias=torch.zeros_like(self.reference)

    def targets(self,target_position,target_quaternion,contact=None):
        from isaaclab.utils.math import quat_apply,quat_mul,compute_pose_error
        data=self.robot.data;quaternion=data.body_link_quat_w.torch[:,self.tool_id]
        offset=quat_apply(quaternion,self.offset.expand(len(quaternion),3))
        position=data.body_link_pos_w.torch[:,self.tool_id]+offset
        raw_target_quaternion=quat_mul(target_quaternion,self.orientation_offset.expand(len(quaternion),4))
        dp,dr=compute_pose_error(position,quaternion,target_position,raw_target_quaternion)
        error=torch.cat((dp,dr),-1)
        jac=data.body_link_jacobian_w.torch[:,self.tool_id,:,[6+j for j in self.arm_ids]].clone()
        jac[:,:3]+=torch.cross(jac[:,3:].transpose(1,2),offset[:,None].expand(-1,len(self.arm_ids),-1),dim=-1).transpose(1,2)
        base=torch.zeros(len(position),6,3,device=position.device);base[:,:3]=torch.eye(3,device=position.device)
        jac=torch.cat((base,jac),-1);delta=damped_step(jac,error,self.weights)
        actual=torch.cat((data.body_link_pos_w.torch[:,self.base_id],data.joint_pos.torch[:,self.arm_ids]),-1)
        # Same mild posture preference as the released EE interface.
        rest=torch.cat((torch.zeros_like(position),.1*(self.posture-actual[:,3:])),-1)
        delta+=rest-damped_step(jac,(jac@rest.unsqueeze(-1)).squeeze(-1),self.weights)
        # Resolved-rate integration removes static inner-loop tracking bias.
        # Unit task-space gain (1/s); identical for both native platforms.
        increment=delta.clamp(-self.max_speed,self.max_speed)*self.dt
        # Conditional integration: stop winding into an unreachable contact goal.
        # Maximum reference lead is half a second at the declared native speed.
        lead=self.reference-actual
        outward=(lead*increment)>0
        self.integration_blocked=(lead.abs()>=.5*self.max_speed)&outward
        if contact is not None:
            first=contact & ~self.contact_latched
            # Bumpless transition: retain learned load compensation at first touch.
            self.contact_bias[first]=(self.reference-actual-delta)[first]
            self.contact_latched|=contact
        contact_increment=(actual+delta+self.contact_bias-self.reference).clamp(-self.max_speed*self.dt,self.max_speed*self.dt)
        self.reference+=torch.where(self.contact_latched[:,None],contact_increment,torch.where(self.integration_blocked,0.,increment))
        limits=data.soft_joint_pos_limits.torch[:,self.arm_ids]
        if self.reference_limits is not None:
            limits=torch.stack((torch.maximum(limits[...,0],self.reference_limits[...,0]),torch.minimum(limits[...,1],self.reference_limits[...,1])),-1)
        self.reference[:,3:]=self.reference[:,3:].clamp(limits[...,0],limits[...,1])
        return self.reference[:,:3].clone(),self.reference[:,3:].clone()


class ReferenceEmbodimentIK(EmbodimentIK):
    """Reference-state IK with measured FK compensation through the floating base.

    Native joint references follow virtual kinematics, never measured tracking
    errors in redundant joints. The base compensates actual arm deflection.
    A task-progress hold freezes the native base/joint setpoints. No pose writes.
    """
    def __init__(self,*args,urdf,offset_frame="world",**kwargs):
        super().__init__(*args,**kwargs)
        import pinocchio as pin
        import numpy as np
        assert offset_frame in ("world","level")
        self.offset_frame=offset_frame
        self.pin=pin;self.np=np
        self.model=pin.buildModelFromUrdf(str(urdf),pin.JointModelFreeFlyer())
        self.pin_data=self.model.createData()
        self.frame=self.model.getFrameId(self.robot.body_names[self.tool_id])
        assert self.frame<self.model.nframes
        self.q_indices=[self.model.joints[self.model.getJointId(name)].idx_q for name in self.robot.joint_names]
        self.arm_v_indices=[self.model.joints[self.model.getJointId(self.robot.joint_names[j])].idx_v for j in self.arm_ids]
        q=self.robot.data.joint_pos.torch.detach().cpu().numpy()
        self.pin_q=np.tile(pin.neutral(self.model),(len(q),1))
        self.pin_q[:,self.q_indices]=q
        self.base_command=self.reference[:,:3].clone()
        self.arm_command=self.reference[:,3:].clone()
        self.last_tool_reference=None

    def _kinematics(self,with_jacobian):
        pin=self.pin;np=self.np;ref=self.reference.detach().cpu().numpy()
        self.pin_q[:,:3]=ref[:,:3]
        self.pin_q[:,[self.q_indices[j] for j in self.arm_ids]]=ref[:,3:]
        poses=[];quats=[];jacobians=[]
        offset=self.offset.cpu().numpy()
        for q in self.pin_q:
            pin.framesForwardKinematics(self.model,self.pin_data,q)
            placement=self.pin_data.oMf[self.frame];lever=placement.rotation@offset
            poses.append(placement.translation+lever);quats.append(pin.Quaternion(placement.rotation).coeffs())
            if with_jacobian:
                full=pin.computeFrameJacobian(self.model,self.pin_data,q,self.frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                j=full[:,self.arm_v_indices].copy();j[:3]+=np.cross(j[3:].T,lever).T
                base=np.zeros((6,3));base[:3]=np.eye(3)
                jacobians.append(np.concatenate((base,j),axis=1))
        tensor=lambda value:torch.as_tensor(np.array(value),dtype=self.reference.dtype,device=self.reference.device)
        return tensor(poses),tensor(quats),tensor(jacobians) if with_jacobian else None

    def targets(self,target_position,target_quaternion,contact=None):
        from isaaclab.utils.math import quat_apply,quat_mul,compute_pose_error
        position,quaternion,jac=self._kinematics(True)
        raw_target=quat_mul(target_quaternion,self.orientation_offset.expand(len(position),4))
        dp,dr=compute_pose_error(position,quaternion,target_position,raw_target)
        delta=damped_step(jac,torch.cat((dp,dr),-1),self.weights)
        rest=torch.cat((torch.zeros_like(position),.1*(self.posture-self.reference[:,3:])),-1)
        delta+=rest-damped_step(jac,(jac@rest.unsqueeze(-1)).squeeze(-1),self.weights)
        if contact is not None:self.contact_latched|=contact
        active=~self.contact_latched
        self.reference[active]+=delta.clamp(-self.max_speed,self.max_speed)[active]*self.dt
        limits=self.robot.data.soft_joint_pos_limits.torch[:,self.arm_ids]
        if self.reference_limits is not None:
            limits=torch.stack((torch.maximum(limits[...,0],self.reference_limits[...,0]),torch.minimum(limits[...,1],self.reference_limits[...,1])),-1)
        self.reference[:,3:]=self.reference[:,3:].clamp(limits[...,0],limits[...,1])
        planned,_,_=self._kinematics(False)
        data=self.robot.data
        actual_offset=data.body_link_pos_w.torch[:,self.tool_id]-data.body_link_pos_w.torch[:,self.base_id]
        actual_offset+=quat_apply(data.body_link_quat_w.torch[:,self.tool_id],self.offset.expand(len(position),3))
        if self.offset_frame=="level":
            # Compensate articulated arm deflection in the commanded level-base frame.
            # Feeding base tilt back into translation cancels contact's restoring geometry.
            from isaaclab.utils.math import quat_apply_inverse
            actual_offset=quat_apply_inverse(data.body_link_quat_w.torch[:,self.base_id],actual_offset)
        desired_base=planned-actual_offset
        increment=(desired_base-self.base_command).clamp(-.10*self.dt,.10*self.dt)
        self.base_command[active]+=increment[active]
        self.arm_command[active]=self.reference[active,3:]
        self.last_tool_reference=planned.clone()
        return self.base_command.clone(),self.arm_command.clone()
