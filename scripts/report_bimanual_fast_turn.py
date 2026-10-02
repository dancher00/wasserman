"""Reduce the three rapid-command demonstrations without scoring a new benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = []
initial = None
sources = None
for mode in ['free', 'support', 'two-hands']:
    root = a.root / mode
    report = json.loads((root / 'report.json').read_text())
    geometry = json.loads((root / 'geometry-audit.json').read_text())
    replay = json.loads((root / 'independent-replay.json').read_text())
    assert report['seeds'] == [93003] and report['seconds'] == 50 and report['dt'] == 1/240
    assert report['physics_change'] is None and report['requested_turn_speed_rad_s'] == 1.4
    assert initial is None or initial == report['initial_offsets_m']
    assert sources is None or sources == report['source_files']
    initial, sources = report['initial_offsets_m'], report['source_files']
    assert replay['complete'] and replay['finite'] and replay['recorded_success_agrees']
    g = geometry['episodes'][0]
    assert not g['cad_collisions'] and g['max_motor_force_N'] <= 1540.001
    # Keep a joint-limit violation as a visible failure diagnostic, never a clean audit.
    joint_limit_passed = g['max_joint_limit_excess_rad'] <= .002
    assert max(g['tcp_fk_max_error_m'].values()) < 1e-4
    with np.load(root / 'trace.npz') as d:
        rot = Rotation.from_quat(d['quaternion'][:, 0].astype(float))
        orientation = np.rad2deg(rot.magnitude())
        tilt = np.rad2deg(np.arccos(np.clip(rot.apply([0, 0, 1])[:, 2], -1, 1)))
        angles = np.rad2deg(d['valve_angle'][:, 0])
        load = np.abs(d['motor_force'][:, 0]).max(axis=1)/1540*100
        index = int(orientation.argmax())
        row = {'mode': mode, 'seed': 93003, 'horizon_s': 50,
               'peak_orientation_error_deg': float(orientation[index]),
               'peak_orientation_time_s': float(d['time'][index]),
               'peak_tilt_deg': float(tilt.max()),
               'overturned': bool(np.any(tilt >= 90)),
               'maximum_valve_rotation_deg': float(angles.max()),
               'final_valve_rotation_deg': float(angles[-1]),
               'peak_thruster_load_percent': float(load.max()),
               'joint_limit_audit_passed': joint_limit_passed,
               'joint_limit_tolerance_rad': .002,
               'max_joint_limit_excess_rad': g['max_joint_limit_excess_rad'],
               'self_collision_count': len(g['cad_collisions']),
               'trace_sha256': hashlib.sha256((root/'trace.npz').read_bytes()).hexdigest(),
               'geometry_audit_sha256': hashlib.sha256((root/'geometry-audit.json').read_bytes()).hexdigest(),
               'series': [{'time_s': float(d['time'][i]), 'orientation_error_deg': float(orientation[i]), 'tilt_deg': float(tilt[i]), 'valve_rotation_deg': float(angles[i]), 'thruster_load_percent': float(load[i])} for i in range(0,len(angles),3)]}
    rows.append(row)
result = {'study': 'bimanual-fast-turn-v1', 'kind': 'three matched demonstrations, not a success-rate benchmark',
          'requested_turn_speed_rad_s': 1.4, 'original_turn_speed_rad_s': .07,
          'requested_turn_angle_deg': 90, 'requested_turn_duration_s': float(np.pi/2/1.4),
          'horizon_s': 50, 'seed': 93003, 'initial_offsets_m': initial,
          'unchanged': ['robot mass and inertia', 'arm and jaw gains and effort/velocity limits', 'valve friction and damping', 'station-keeping controller', '1540 N thruster limits', '240 Hz physics and 30 Hz commands'],
          'physical_source_sha256': sources,
          'scope': 'One new initial state shared by the three modes. Grasp acquisition is unchanged; a 20x faster command requests one 90-degree turn, then hold/release/withdrawal. No new 170-degree task success rate is claimed. Peak metrics cover the full recorded 50 seconds; curves are displayed at 10 Hz.',
          'modes': rows}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='series'} for r in rows],indent=2))
