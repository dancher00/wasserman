"""Isolate native articulation rotation integration from policy, fluid and contact."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dt', type=float, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--position-iterations',type=int)
p.add_argument('--solver-type', type=int, choices=[0, 1], default=1)
AppLauncher.add_app_launcher_args(p)
p.set_defaults(visualizer=['none'])
a = p.parse_args()
launcher = AppLauncher(a)
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_tasks.utils import resolve_task_config
import wasman.tasks
from wasman.assets.registered_bluerov import configure_registered_bluerov
from scipy.spatial.transform import Rotation

try:
    cfg, _ = resolve_task_config('Wasman-Underwater-PressButton-Direct', '', overrides=('physics=isaacsim_physx',))
    configure_registered_bluerov(cfg)
    cfg.sim.dt = a.dt
    cfg.sim.physics.solver_type = a.solver_type
    cfg.sim.gravity = (0., 0., 0.)
    sim = sim_utils.SimulationContext(cfg.sim)
    speeds = [.001, .003, .01, .03, .1]
    for i in range(len(speeds)):
        sim_utils.create_prim(f'/World/envs/env_{i}', 'Xform', translation=(i*10., 0., 0.))
    rcfg = cfg.scene.robot.copy()
    rcfg.prim_path = '/World/envs/env_[0-9]+/Robot'
    if a.position_iterations is not None:
        rcfg.spawn.articulation_props.solver_position_iteration_count=a.position_iterations
    robot = Articulation(rcfg)
    sim.reset()
    q = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=q)
    robot.set_joint_position_target(q)
    velocity = torch.zeros(len(speeds), 6, device=sim.device)
    velocity[:, 5] = torch.tensor(speeds, device=sim.device)
    robot.write_root_velocity_to_sim_index(root_velocity=velocity)
    robot.write_data_to_sim()
    sim.forward()
    robot.update(a.dt)
    initial = robot.data.root_quat_w.torch.cpu().numpy().copy()
    rows = []
    for step in range(round(2/a.dt)):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(a.dt)
        rows.append(dict(quaternion=robot.data.root_quat_w.torch.cpu().numpy().copy(), angular=robot.data.root_ang_vel_w.torch.cpu().numpy().copy(),raw_pose=robot.data._root_view.get_root_transforms().numpy().copy(),body_quaternion=robot.data.body_link_quat_w.torch[:,0].cpu().numpy().copy()))
    np.savez_compressed(a.output.with_suffix('.npz'),initial=initial,**{k:np.array([r[k] for r in rows]) for k in rows[0]})
    a.output.with_suffix('.py').write_bytes(Path(__file__).read_bytes())
    quat = np.array([r['quaternion'] for r in rows]); ang = np.array([r['angular'] for r in rows])
    measured = (Rotation.from_quat(initial).inv()*Rotation.from_quat(quat[-1])).as_rotvec()
    expected = ang.sum(0)*a.dt
    result = dict(dt=a.dt, solver_type=a.solver_type,position_iterations=rcfg.spawn.articulation_props.solver_position_iteration_count,raw_root_difference=float(np.abs(quat-np.array([r["raw_pose"] for r in rows])[:,:,3:]).max()),body_difference=float(np.abs(quat-np.array([r["body_quaternion"] for r in rows])).max()),seconds=2, speeds=speeds, measured_rotvec=measured.tolist(), integrated_angular_velocity=expected.tolist(), error_rad=np.linalg.norm(measured-expected,axis=1).tolist(), scope='Dry native robot, zero gravity, no contact or fluid, no EE/base controller. No task result.')
    with a.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
finally:
    launcher.app.close()
