"""Render recorded states with the industrial support; optional fixed wide view."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
depth = 2.5
if '--pool-depth' in sys.argv:
    i = sys.argv.index('--pool-depth')
    depth = float(sys.argv[i+1])
    del sys.argv[i:i+2]
assert 2.5 <= depth <= 5
wide = '--wide' in sys.argv
if wide:
    sys.argv.remove('--wide')
target = Path(sys.argv[sys.argv.index('--output-dir') + 1]).resolve()
base = ROOT / 'scripts/render_bimanual_trace.py'
source = base.read_text()
old = "assert report['mode'] in ('two-hands', 'support', 'free') and report['seconds'] == 180"
assert source.count(old) == 1
source = source.replace(old, "assert report['mode'] in ('two-hands', 'support', 'free') and report['seconds'] >= 15")
start = source.index("if report['mode'] != 'two-hands':\n    rail =")
end = source.index("light = UsdLux.DomeLight.Define", start)
source = source[:start] + "support_presentation = None\nif report['mode'] != 'two-hands':\n    from bimanual_support_presentation import build_support\n    support_presentation = build_support(stage, ROOT)\n" + source[end:]
old = "selected = [0, 300, 900, 1350, 2400, 3300, 4200, 4800, 5399] if args.snapshots_only else range(len(data['time']))"
assert source.count(old) == 1
source = source.replace(old, "selected = np.linspace(0, len(data['time'])-1, 9, dtype=int).tolist() if args.snapshots_only else range(len(data['time']))")
if depth != 2.5:
    anchor = "UsdGeom.Xformable(pool).AddTranslateOp(opSuffix='presentation').Set(Gf.Vec3d(-8, 7.5, 0))"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + f"\nUsdGeom.Xformable(pool).AddScaleOp(opSuffix='water_depth').Set(Gf.Vec3f(1, 1, {depth/2.5}))")
    source = source.replace('float(2.5 - high[2])', f'float({depth} - high[2])')
    source = source.replace('water_depth_m=2.5', f'water_depth_m={depth}')
if wide:
    old = '    u = np.clip((data[\'time\'][index] - 3) / 9, 0, 1)'
    assert source.count(old) == 1
    source = source.replace(old, '    u = 0.0  # Fixed wide camera preserves visible base rotation.')
source = source.replace("    presentation='standard-pool-ship-green-v1', presentation_only=True,", "    presentation='standard-pool-industrial-support-v2', presentation_only=True,\n    support_presentation=support_presentation,\n    fixed_wide_camera=" + str(wide) + ",")
anchor = "launcher = AppLauncher(args)"
assert source.count(anchor) == 1
source = source.replace(anchor, "_record_renderer_variant()\n" + anchor)

def record_variant():
    (target / 'executed_renderer.py').write_text(source)
    (target / 'bimanual_support_presentation.py').write_bytes((ROOT / 'scripts/bimanual_support_presentation.py').read_bytes())
    (target / 'renderer-variant.json').write_text(json.dumps({
        'original_renderer_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
        'executed_renderer_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'wide': wide, 'pool_depth_m': depth, 'physical_state_replay_only': True,
        'provenance_written_before_application_start': True,
    }, indent=2) + '\n')

exec(compile(source, str(base), 'exec'), {
    '__name__': '__main__', '__file__': str(base),
    '_record_renderer_variant': record_variant,
})
