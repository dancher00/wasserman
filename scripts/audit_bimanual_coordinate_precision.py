"""Audit recorded TCPs using native FK at original and near-zero world origins.

This supplements, never overwrites, the frozen CPU CAD/FK audit. PhysX uses
float32 world transforms; a long chain hundreds of metres from the origin can
exceed the original 100 micrometre CPU-vs-world comparison solely by rounding.
We measure that coordinate effect with identical recorded poses and native FK,
without integrating physics. The 100 micrometre tolerance remains unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
os.environ.setdefault('ACCEPT_EULA', 'Y')
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=['none'], enable_cameras=False)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
launcher = AppLauncher(args)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_tasks.utils import resolve_task_config
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import wasman.tasks
from wasman.assets.rexrov2_bimanual import BIMANUAL_REX_CFG
from wasman.controllers.bimanual_kinematics import BimanualKinematics
from wasman.physics.hydrodynamics import quat_apply_xyzw
from wasman.physics.rexrov2_bimanual import URDF


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    report = json.loads((args.run / 'report.json').read_text())
    geometry = json.loads((args.run / 'geometry-audit.json').read_text())
    for name, expected in report['source_files'].items():
        assert sha(Path(name)) == expected, name
    with np.load(args.run / 'trace.npz') as trace:
        d = {key: trace[key] for key in ['time', 'position', 'quaternion', 'joint_position', 'left_tool_position', 'tool_position']}
    n = len(report['seeds'])
    assert len(geometry['episodes']) == n
    assert d['time'][-1] >= report['seconds'] - 1 / 30 - 1e-6
    w = BimanualKinematics()
    ids = [w.model.joints[w.model.getJointId(name)].idx_q for name in report['joint_names']]
    cfg, _ = resolve_task_config('Wasman-Underwater-OpenHatch-Direct', '', overrides=('physics=isaacsim_physx',))
    cfg.sim.dt, cfg.sim.visualizer_cfgs = 1 / 240, []
    cfg.sim.physics.solver_type = 0
    sim = sim_utils.SimulationContext(cfg.sim)
    for env in range(n):
        sim_utils.create_prim(f'/World/envs/env_{env}', 'Xform', translation=(0, env * 12., 0))
    rcfg = BIMANUAL_REX_CFG.copy()
    rcfg.prim_path = '/World/envs/env_[0-9]+/Robot'
    robot = Articulation(rcfg)
    sim.reset()
    def tensor(value):
        return torch.tensor(value, dtype=torch.float32, device=sim.device)
    origins = tensor([[0, env * 12, 0] for env in range(n)])
    zeros = torch.zeros_like(origins)
    ordered = [report['joint_names'].index(name) for name in robot.joint_names]
    tools = [robot.find_bodies(side + '_oberon_end_effector')[0][0] for side in ['left', 'right']]
    tcp_offset = tensor([.16, 0, 0]).repeat(n, 1)
    root_velocity = torch.zeros(n, 6, device=sim.device)
    joint_velocity = torch.zeros(n, len(ordered), device=sim.device)
    # Every powered sample is checked; no phase filter or failed-sample exclusion.
    shape = (len(d['time']), n, 2, 3)
    correction = np.zeros(shape, dtype=np.float64)
    world_residual = np.zeros(shape, dtype=np.float64)
    local_residual = np.zeros(shape, dtype=np.float64)
    for i, time in enumerate(d['time']):
        state = tensor(d['joint_position'][i, :n][:, ordered])
        local_position = tensor(d['position'][i, :n])
        quaternion = tensor(d['quaternion'][i, :n])
        native = []
        for origin in [origins, zeros]:
            pose = torch.cat((local_position + origin, quaternion), -1)
            robot.write_root_pose_to_sim_index(root_pose=pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
            robot.write_joint_position_to_sim_index(position=state)
            robot.write_joint_velocity_to_sim_index(velocity=joint_velocity)
            robot.reset()
            sim.forward()  # No time integration, controller, forces or task contact.
            robot.update(1 / 240)
            observed_q = robot.data.joint_pos.torch
            assert torch.equal(observed_q, state), 'Native FK changed the assigned joint coordinates'
            tcp = torch.stack([
                robot.data.body_link_pos_w.torch[:, bid] + quat_apply_xyzw(
                    robot.data.body_link_quat_w.torch[:, bid], tcp_offset) - origin
                for bid in tools], dim=1)
            native.append(tcp.cpu().numpy().astype(np.float64))
        observed = np.stack((d['left_tool_position'][i, :n], d['tool_position'][i, :n]), axis=1).astype(np.float64)
        expected = np.zeros((n, 2, 3))
        for env in range(n):
            q = np.zeros(w.model.nq)
            q[ids] = d['joint_position'][i, env]
            rot = Rotation.from_quat(d['quaternion'][i, env])
            poses = w.poses(q)
            for side_index, side in enumerate(['left', 'right']):
                expected[env, side_index] = d['position'][i, env].astype(np.float64) + rot.apply(poses[side][0])
        correction[i] = native[0] - native[1]
        world_residual[i] = observed - native[0]
        local_residual[i] = native[1] - expected
        if i % 300 == 0:
            print(f'Coordinate audit {i}/{len(d["time"])} frames; {n} recorded poses per frame; no physics integration', flush=True)
    corrected = world_residual + local_residual
    rows = []
    for env, seed in enumerate(report['seeds']):
        row = {'seed': seed, 'samples': len(d['time']),
               'original_tcp_fk_max_error_m': geometry['episodes'][env]['tcp_fk_max_error_m']}
        for key, values in [('recorded_vs_native_world', world_residual), ('native_local_vs_cpu', local_residual),
                            ('coordinate_corrected_fk', corrected), ('world_coordinate_rounding', correction)]:
            row[key + '_max_error_m'] = {side: float(np.linalg.norm(values[:, env, si], axis=-1).max())
                                         for si, side in enumerate(['left', 'right'])}
        row['passed'] = all(max(row[key + '_max_error_m'].values()) < 1e-4 for key in
                            ['recorded_vs_native_world', 'native_local_vs_cpu', 'coordinate_corrected_fk'])
        rows.append(row)
    assert all(np.isfinite(x).all() for x in [correction, world_residual, local_residual])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'coordinate-residuals.npz', time=d['time'],
                        coordinate_correction_m=correction, recorded_vs_native_world_m=world_residual,
                        native_local_vs_cpu_m=local_residual)
    result = {'passed': all(row['passed'] for row in rows), 'method': __doc__,
              'threshold_m': 1e-4, 'physics_steps_after_pose_assignment': 0,
              'trace_sha256': sha(args.run / 'trace.npz'), 'geometry_audit_sha256': sha(args.run / 'geometry-audit.json'),
              'source_sha256': sha(Path(__file__)), 'native_urdf_sha256': sha(URDF),
              'residuals_sha256': sha(args.output / 'coordinate-residuals.npz'), 'episodes': rows,
              'scope': 'All powered 30 Hz samples; original CAD, joint and motor audits remain required. The original absolute-world FK failures remain recorded.',
              'source_files_verified': report['source_files']}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'passed': result['passed'], 'episodes': len(rows), 'samples_per_episode': len(d['time']),
                      'max_corrected_error_m': max(max(row['coordinate_corrected_fk_max_error_m'].values()) for row in rows)}, indent=2), flush=True)
    assert result['passed'], 'Coordinate-aware FK audit failed; preserve the record and investigate'


exit_code = 0
try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    launcher.app.close(exit_code=exit_code)
