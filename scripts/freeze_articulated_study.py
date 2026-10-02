"""Freeze sources and protocol only after declared development gates pass."""
import argparse,hashlib,json,datetime,importlib.metadata,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--blue-gate',type=Path,required=True);p.add_argument('--rex-gate',type=Path,required=True);p.add_argument('--motion-gate',type=Path,required=True);p.add_argument('--cad-gate',type=Path,required=True);p.add_argument('--fk-gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
gates=[]
for file in [a.blue_gate,a.rex_gate,a.motion_gate]:
 r=json.loads(file.read_text());assert r['passed'],f'Failed development gate: {file}'
 gates.append({'path':str(file),'sha256':digest(file),'result':r})
for gate in gates[:2]:
 c=gate['result']['configuration']
 assert (c['ik_mode'],c['stroke_m'],c['hold_travel_m'])==('reference',.14,.006),'Gate belongs to a different controller'
 assert c['offset_frame']=='level','Final controller must use the validated frame correction'
assert gates[0]['result']['configuration']['solver_type']==gates[1]['result']['configuration']['solver_type'],'Both contact gates must validate the same solver'
assert gates[0]['result']['configuration']['hydro_variant']=='geometry-scaled' and gates[0]['result']['configuration']['acceleration_cap']=='none'
common_rates=set(gates[0]['result']['dt']) & set(gates[1]['result']['dt'])
assert common_rates,'A common final rate must occur in both passed numerical checks'
cad=json.loads(a.cad_gate.read_text());fk=json.loads(a.fk_gate.read_text())
assert cad['passed'] and all(v['passed'] for v in fk.values()),'Geometry/kinematics audit failed'
files=list((ROOT/'src/wasman').rglob('*.py'))
for part in ['src/wasman/physics/data','src/wasman/assets/data/robots','src/wasman/assets/data/objects']:
 files.extend(f for f in (ROOT/part).rglob('*') if f.is_file() and '__pycache__' not in str(f) and not f.name.startswith('.'))
files.extend(ROOT/f for f in ['scripts/freeze_articulated_study.py','scripts/audit_articulated_cad.py','scripts/audit_blue_rotation_integration.py','scripts/plot_articulated_embodiments.py','scripts/run_articulated_embodiments.py','scripts/audit_blue_fluid_inertia.py','scripts/audit_embodiment_fk.py','scripts/probe_blue_articulated.py','scripts/probe_rex_articulated.py','scripts/audit_embodiment_timestep.py','scripts/report_articulated_embodiments.py','scripts/record_articulated_observer.py','scripts/rex_rotor_presentation.py','configs/studies/rex-centered-button-pose.json','docs/studies/embodiment-v2-protocol.md','docs/studies/embodiment-v3-protocol.md','uv.lock'])
result={'study':'embodiment_v3_20260927','offset_frame':'level','geometry_gates':{'cad':cad,'fk':fk},'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'final_seeds':list(range(86000,86030)),'demo_seed':85000,'solver_type':gates[0]['result']['configuration']['solver_type'],'final_dt':max(common_rates),'stroke_m':.14,'ik_mode':'reference','hold_travel_m':.006,'blue_hydro_variant':'geometry-scaled','blue_acceleration_cap':'none','seconds':20,'control_hz':30,'gates':gates,'isaaclab_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'.deps/IsaacLab',text=True).strip(),'isaaclab_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT/'.deps/IsaacLab')).hexdigest(),'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','pin','imageio-ffmpeg','warp-lang']},'source_sha256':{str(f.relative_to(ROOT)):digest(f) for f in sorted(set(files))},'note':'Final states must be unopened before this manifest; original v1 evidence remains unchanged.'}
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(f'Frozen {len(files)} source/asset files; dt={result["final_dt"]}')
