"""Run one frozen scripted PressButton embodiment evaluation (no policy training)."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--robot',choices=['blue','rex'],required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--split',choices=['test','development','demo'],default='test')
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--check-only',action='store_true')
a=p.parse_args();manifest=json.loads(a.manifest.read_text())
assert all(g['result']['passed'] for g in manifest['gates']), 'A numerical gate failed'
errors=[]
for name,expected in manifest['source_sha256'].items():
 file=ROOT/name
 if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:errors.append(name)
if errors:raise SystemExit('Frozen runtime differs: '+', '.join(errors[:20]))
seeds=manifest['final_seeds'] if a.split=='test' else [85000,85001,85002] if a.split=='development' else [manifest['demo_seed']]
cmd=[sys.executable,str(ROOT/f'scripts/probe_{a.robot}_articulated.py'),'--solver-type',str(manifest['solver_type']),'--offset-frame',manifest['offset_frame'],'--control-mode','ik','--ik-mode',manifest['ik_mode'],'--stroke',str(manifest['stroke_m']),'--hold-travel',str(manifest['hold_travel_m']),'--dt',str(manifest['final_dt']),'--seconds',str(manifest['seconds']),'--seeds',*map(str,seeds),'--output-dir',str(a.output_dir.resolve())]
cmd+=['--hydro-variant',manifest['blue_hydro_variant'],'--acceleration-cap',manifest['blue_acceleration_cap']] if a.robot=='blue' else ['--closure','articulated']
if a.split=='demo':cmd=[sys.executable,str(ROOT/'scripts/record_articulated_observer.py'),'--robot',a.robot,*cmd[2:]]
print(json.dumps({'split':a.split,'robot':a.robot,'seeds':seeds,'command':cmd},indent=2),flush=True)
if not a.check_only:
 if a.output_dir.exists():raise SystemExit('Choose a new output directory to preserve evidence')
 raise SystemExit(subprocess.call(cmd,cwd=ROOT))
