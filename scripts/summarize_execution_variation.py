"""Summarize existing paired executions; never replace the primary scores."""
import hashlib
import itertools
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def summarize():
    source = ROOT / 'research/revision-v2-portability-results.json'
    data = json.loads(source.read_text())
    rows = data['pairs']
    models = ['ACT', 'DP', 'BC']
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    assert len(rows) == 54 and len({(r['task'], r['model'], r['training_seed']) for r in rows}) == 54
    grouped = []
    for task, model in itertools.product(tasks, models):
        run = [r for r in rows if (r['task'], r['model']) == (task, model)]
        assert len(run) == 3
        a = sum(r['primary_successes'] for r in run)
        b = sum(r['repeated_successes'] for r in run)
        grouped.append(dict(task=task, model=model, primary_successes=a, repeated_successes=b,
                            episodes=90, drift_pp=(b-a)/90*100,
                            agreeing_outcomes=sum(r['paired_agreement'] for r in run)))
    macro = {}
    for model in models:
        run = [r for r in grouped if r['model'] == model]
        a = sum(r['primary_successes'] for r in run)/540*100
        b = sum(r['repeated_successes'] for r in run)/540*100
        macro[model] = dict(primary_percent=a, repeated_percent=b, drift_pp=b-a)
    contrasts = []
    sign = lambda n: (n > 0) - (n < 0)
    for task, (left, right) in itertools.product(tasks, itertools.combinations(models, 2)):
        a = next(r for r in grouped if (r['task'], r['model']) == (task, left))
        b = next(r for r in grouped if (r['task'], r['model']) == (task, right))
        p = a['primary_successes']-b['primary_successes']
        r = a['repeated_successes']-b['repeated_successes']
        contrasts.append(dict(task=task, contrast=f'{left}-{right}', primary_difference_pp=p/90*100,
                              repeated_difference_pp=r/90*100, strict_reversal=sign(p)*sign(r)<0,
                              identical_direction=sign(p)==sign(r)))
    result = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='Descriptive execution sensitivity, not new training replicates, intervals or a causal mechanism test.',
                  cohorts=54, episodes=1620, agreement=data['agreeing_outcomes'], rows=grouped,
                  macro=macro, contrasts=contrasts,
                  strict_direction_reversals=sum(r['strict_reversal'] for r in contrasts))
    (ROOT/'research/revision-v2-execution-impact.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Execution sensitivity of the reported conclusions', '',
             'Same final checkpoints, three training identities and 30 task resets on two workstations. These paired repetitions do not replace primary scores.', '',
             '| Model | Primary mean (%) | Repeat mean (%) | Drift (pp) |', '|---|---:|---:|---:|']
    for model, r in macro.items(): lines.append(f"| {model} | {r['primary_percent']:.1f} | {r['repeated_percent']:.1f} | {r['drift_pp']:+.1f} |")
    lines += ['', 'The equal-task ordering DP > ACT > BC is preserved. Across the 18 task-level pairwise model contrasts, no nonzero difference reverses direction. The ACT/DP tie on RotateValve becomes a one-episode DP advantage across 90 episodes; this is not evidence of a resolved difference.', '',
              '| Task | Model | Primary /90 | Repeat /90 | Drift (pp) | Matching /90 |', '|---|---|---:|---:|---:|---:|']
    for r in grouped: lines.append(f"| {r['task']} | {r['model']} | {r['primary_successes']} | {r['repeated_successes']} | {r['drift_pp']:+.1f} | {r['agreeing_outcomes']} |")
    lines += ['', 'Outcome agreement is 1,438/1,620 (88.8%). Small aggregate drift can conceal opposing episode flips. The largest absolute task/model mean drift is 7.8 pp (PullLever ACT).', '',
              'The separate same-GPU seed-17 DP repeat agrees on 169/180 outcomes (93.9%); task counts change by at most one success out of 30, despite seven Shell and three Slider outcome flips. This does not identify the numerical source of execution variability. Controller, action-interface and current contrasts were not rerun across workstations in this comparison.', '',
              'Regenerate: `python scripts/summarize_execution_variation.py`. Source hashes and all 18 comparisons: `research/revision-v2-execution-impact.json`.']
    (ROOT/'docs/revision-v2-execution-impact.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__ == '__main__':
    r=summarize(); print(json.dumps({'macro':r['macro'],'strict_direction_reversals':r['strict_direction_reversals']},indent=2))
