"""Instrument frozen evaluators/experts for a separately registered current study.

The existing policy, action decoding, scene and success code execute unchanged.
Only the water vector is assigned, immediately before the camera-warmup step.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['OpenHatch', 'RotateValve'], required=True)
    p.add_argument('--stage', choices=['expert-pilot', 'benchmark', 'evaluation', 'video'], required=True)
    p.add_argument('--speed', type=float, choices=[0., .1, .2], required=True)
    p.add_argument('--protocol', type=Path, default=ROOT / 'research/current-contact-5090/protocol.json')
    p.add_argument('--video-index', type=int)
    a, rest = p.parse_known_args()
    out = Path(rest[rest.index('--output-dir') + 1])
    if os.environ.get('WASMAN_ASSET_PROFILE') != 'open-procedural-v1':
        raise ValueError('Only the open asset profile is authorized')
    protocol = json.loads(a.protocol.read_text())
    wrapper_hash = digest(__file__)
    if a.stage in ('evaluation', 'video') and protocol['status'] != 'frozen':
        raise ValueError('Main evaluation requires a frozen protocol')
    if '--record-video' in rest:
        raise ValueError('Use study video-index; never change the rollout batch')
    if (a.video_index is not None) != (a.stage == 'video'):
        raise ValueError('Video selection is explicit and limited to video stage')
    idx = rest.index('--seeds') + 1
    seeds = []
    while idx < len(rest) and not rest[idx].startswith('--'):
        seeds.append(int(rest[idx])); idx += 1
    cohort = 'expert_pilot' if a.stage == 'expert-pilot' else 'benchmark' if a.stage == 'benchmark' else 'evaluation'
    if seeds != protocol['resets'][a.task][cohort]:
        raise ValueError('Ordered seed cohort mismatch')
    if a.stage in ('evaluation', 'video'):
        if rest[rest.index('--purpose') + 1] != 'research':
            raise ValueError('Main research run must use research purpose')
    import gymnasium as gym
    from wasman.learning.revision_protocol import validate_runtime_snapshot
    validate_runtime_snapshot()
    import wasman.learning.revision_diagnostics as diagnostics

    # Replace only the old controller-study cohort gate, not physics or inference.
    def validate_current_cohort(config, seeds, purpose, integral_multiplier, hydro_condition):
        allowed = protocol['resets'][a.task]['benchmark' if a.stage == 'benchmark' else 'evaluation']
        if list(seeds) != allowed or integral_multiplier != 1. or hydro_condition != 'nominal':
            raise ValueError('Current study cohort or fixed condition mismatch')
        if config['pilot'] or config['model'] != 'DP' or config['seed'] not in [17, 43, 101]:
            raise ValueError('Only the six final DP checkpoints are authorized')
        model = protocol['models'][f"{a.task}-DP-{config['seed']}"]
        checkpoint = Path(rest[rest.index('--checkpoint') + 1])
        if digest(checkpoint) != model['checkpoint_sha256']:
            raise ValueError('Checkpoint bytes mismatch')
        if config != json.loads((ROOT / model['config_path']).read_text()):
            raise ValueError('Embedded and registered model configurations differ')
    diagnostics.validate_condition_cohort = validate_current_cohort
    original_make = gym.make

    def make(task, *, cfg, **kwargs):
        import torch
        if cfg.current_speed_range != (0., 0.) or cfg.current_vertical_range != (0., 0.) or cfg.turbulence_sigma != 0.:
            raise ValueError('Frozen evaluator must disable random current and turbulence')
        if cfg.scene.num_envs != (2 if a.stage == 'expert-pilot' else 30):
            raise ValueError('Full 30-environment policy batches are mandatory')
        def portable(v):
            if isinstance(v, float) and not math.isfinite(v):
                return {'nonfinite_config_float': str(v)}
            if v is None or isinstance(v, (str, bool, int, float)):
                return v
            if isinstance(v, dict):
                return {str(k): portable(x) for k, x in v.items()}
            if isinstance(v, (tuple, list)):
                return [portable(x) for x in v]
            if isinstance(v, Path):
                return str(v)
            if callable(v):
                return f'{v.__module__}.{v.__qualname__}'
            if dataclasses.is_dataclass(v):
                return portable(dataclasses.asdict(v))
            if hasattr(v, 'tolist'):
                return v.tolist()
            raise TypeError(f'Unrecorded configuration type: {type(v)}')
        resolved = portable(cfg.to_dict())
        (out / 'resolved-config.json').write_text(json.dumps(resolved, indent=2, allow_nan=False)+'\n')
        env = original_make(task, cfg=cfg, **kwargs)
        raw = env.unwrapped
        records = []
        initial = {}
        calls = 0
        writer = None
        video_finished = False
        video_count = 0
        started = time.monotonic()
        step_times = []
        vector = torch.tensor([0., a.speed, 0.], device=raw.device)
        def snap():
            values = dict(
                robot_root=raw.robot.data.root_state_w.torch,
                base_position=raw.robot.data.body_link_pos_w.torch[:, raw._base_body_id] - raw.scene.env_origins,
                base_quaternion=raw.robot.data.body_link_quat_w.torch[:, raw._base_body_id],
                base_linear_velocity=raw.robot.data.body_com_lin_vel_w.torch[:, raw._base_body_id],
                base_angular_velocity=raw.robot.data.body_com_ang_vel_w.torch[:, raw._base_body_id],
                joints=raw.robot.data.joint_pos.torch,
                joint_velocities=raw.robot.data.joint_vel.torch,
                mechanism_root=raw.button.data.root_state_w.torch,
                mechanism_joints=raw.button.data.joint_pos.torch,
                mechanism_velocities=raw.button.data.joint_vel.torch,
                env_origins=raw.scene.env_origins,
                volume_scale=raw._hydrodynamics.volume_scale,
                damping_scale=raw._hydrodynamics.damping_scale,
                added_mass_scale=raw._hydrodynamics.added_mass_scale,
                current=raw.current_w,
                motor_force=raw._thrusters.force,
                motor_requested_force=raw._thrusters.requested_force,
                motor_saturation_scale=raw._thrusters.saturation_scale,
                station_position_error=raw._station_position_error_w,
                station_orientation_error=raw._station_orientation_error_b,
                attitude=raw._base_attitude_error,
                contact_force_vectors=raw.finger_force_vectors(),
                near_handle=raw._distance < cfg.tool_contact_distance,
            )
            return {k: v.detach().cpu().clone() for k, v in values.items()}
        original_step = env.step
        def step(action):
            nonlocal calls, writer, video_finished, video_count
            t = time.monotonic()
            if calls == 0:
                initial['before_current_and_warmup'] = snap()
                if torch.count_nonzero(raw.current_w):
                    raise ValueError('Current was nonzero before intervention')
            if calls == 1:
                initial['before_first_command'] = snap()
            raw._mean_current_w[:] = vector
            raw._turbulent_current_w.zero_()
            result = original_step(action)
            record = snap()
            record['action'] = action.detach().cpu().clone()
            record['terminated'] = (result[2] | result[3]).detach().cpu().clone()
            record['success_signal'] = result[4]['wasman_success'].detach().cpu().clone()
            if calls == 0:
                initial['after_warmup'] = record
            else:
                records.append(record)
                if a.video_index is not None and not video_finished:
                    if writer is None:
                        import imageio_ffmpeg
                        camera = raw.scene['gripper_camera']
                        h, w = camera.data.output['rgb'].torch.shape[1:3]
                        writer = imageio_ffmpeg.write_frames(str(out / 'wrist.mp4'), (w,h), fps=30,
                            codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=8,
                            output_params=['-movflags','+faststart'])
                        writer.send(None)
                    writer.send(raw.scene['gripper_camera'].data.output['rgb'].torch[a.video_index,...,:3].contiguous().cpu().numpy())
                    video_count += 1
                    video_finished = bool(record['terminated'][a.video_index] or record['success_signal'][a.video_index])
            calls += 1
            step_times.append(time.monotonic()-t)
            return result
        env.step = step
        original_close = env.close
        def close():
            if writer is not None:
                writer.close()
            torch.save(dict(initial=initial, trace=records), out / 'current-physical.pt')
            record = dict(study=protocol['study'], stage=a.stage, task=a.task, speed_m_s=a.speed,
                current_world_m_s=[0.,a.speed,0.], control_dt=raw.step_dt,
                activation='Immediately before first camera-warmup env.step; held through all subsequent steps',
                warmup_control_steps=1, scored_steps=len(records), total_env_step_seconds=sum(step_times),
                wrapped_runtime_seconds=time.monotonic()-started, video_index=a.video_index,
                video_frames=video_count,
                motor_force_limits_N=raw._thrusters.force_knots[[0,-1]].detach().cpu().tolist(),
                config_sha256=digest(out/'resolved-config.json'), protocol_sha256=digest(a.protocol),
                wrapper_sha256=wrapper_hash, underlying_evaluator_sha256=digest(script),
                max_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_torch_reserved_bytes=torch.cuda.max_memory_reserved())
            (out/'current-condition.json').write_text(json.dumps(record,indent=2)+'\n')
            original_close()
        env.close = close
        return env
    gym.make = make
    name = 'hatch' if a.task == 'OpenHatch' else 'valve'
    script = ROOT / f"scripts/{'collect' if a.stage == 'expert-pilot' else 'evaluate'}_{name}_visual.py"
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
