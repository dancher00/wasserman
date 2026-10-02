"""Reuse the frozen physics runner with a separately recorded fast command plan."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
probe = ROOT / 'scripts/probe_bimanual_valve.py'
source = probe.read_text()
old = 'from wasman.controllers.bimanual_turn_plan import BASE_POSITION, OPEN_JAW, RELEASE_JAW, ContactPlanClock, mode_plan'
new = old.replace('bimanual_turn_plan', 'bimanual_fast_turn')
assert source.count(old) == 1
source = source.replace(old, new)
key = '        "src/wasman/controllers/bimanual_turn_plan.py",'
assert source.count(key) == 1
source = source.replace(key, key + '\n        "src/wasman/controllers/bimanual_fast_turn.py",\n        "scripts/probe_bimanual_fast_turn.py",')
source = source.replace('kind="Twin Oberon RotateValve development",', 'kind="Rapid 90-degree command demonstration; not a benchmark cohort",\n        requested_turn_speed_rad_s=1.4,\n        physics_change=None,')
output = Path(sys.argv[sys.argv.index('--output-dir') + 1]).resolve()
if output.exists():
    raise SystemExit('Use a fresh output directory')
output.mkdir(parents=True)
(output / 'executed_probe.py').write_text(source)
(output / 'execution.json').write_text(json.dumps({
    'runner_sha256': hashlib.sha256(probe.read_bytes()).hexdigest(),
    'executed_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'change': 'Import the separate fast command plan and record its provenance; all physical implementation and limits unchanged',
    'requested_turn_speed_rad_s': 1.4,
    'seed_role': 'Separate demonstration; not part of the published test cohort',
}, indent=2) + '\n')
try:
    exec(compile(source, str(probe), 'exec'), {'__name__': '__main__', '__file__': str(probe)})
finally:
    # The independent phase replay must use the actual executed command plan.
    (output / 'bimanual_turn_plan.py').write_bytes((ROOT / 'src/wasman/controllers/bimanual_fast_turn.py').read_bytes())
