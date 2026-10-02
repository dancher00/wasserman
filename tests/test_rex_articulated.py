"""Physical identities for the new opt-in moving-arm closure, not score tests."""
import numpy as np
import pytest
import torch
from wasman.controllers.rexrov2_workspace import RexWorkspace
from wasman.physics.rexrov2 import HeldArmRexHydrodynamics,cross_bias
from wasman.physics.rexrov2_centered import load_centered_parameters,FOLDED_Q
from wasman.physics.rexrov2_articulated import ArticulatedRexHydrodynamics
pytestmark=pytest.mark.unit

def model():
 w=RexWorkspace();names=list(w.model.names)[1:];p=load_centered_parameters(FOLDED_Q,workspace=w)
 return p,ArticulatedRexHydrodynamics(p,'cpu',names)

def test_held_arm_reduces_to_existing_closure():
 p,new=model();old=HeldArmRexHydrodynamics(p,'cpu');nu=torch.tensor([[.1,-.04,.02,.015,-.025,.04]]);q=torch.tensor(FOLDED_Q[None],dtype=torch.float32)
 new.update_articulation(q,torch.zeros_like(q),nu,1/240)
 assert torch.allclose(new.current_inertia[0],old.rigid,atol=2e-3,rtol=2e-6)
 assert torch.allclose(new.rigid_bias,cross_bias(old.rigid,nu),atol=1e-5,rtol=2e-5)
 load=torch.tensor([[20.,30.,-40.,15.,-10.,5.]]);current=torch.tensor([[.04,0.,0.]])
 for a,b in zip(new.close_added_mass(nu,current,load),old.close_added_mass(nu,current,load),strict=True):assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)

def test_joint_acceleration_produces_reaction_with_no_external_load():
 _,new=model();nu=torch.zeros(1,6);q=torch.tensor(FOLDED_Q[None],dtype=torch.float32)
 new.update_articulation(q,torch.zeros_like(q),nu,1/240)
 rates=torch.zeros_like(q);rates[:,1]=.001
 new.update_articulation(q,rates,nu,1/240)
 added,acc=new.close_added_mass(nu,torch.zeros(1,3),torch.zeros(1,6))
 assert torch.linalg.vector_norm(new.rigid_bias)>1
 assert torch.linalg.vector_norm(acc)>1e-4
 residual=((new.current_inertia+new.added)@acc.unsqueeze(-1)).squeeze(-1)+new.rigid_bias
 assert residual.abs().max()<1e-5
 assert torch.isfinite(added).all()
