"""Run the frozen finite campaign, then analysis/figures and selected video reruns."""
import json,subprocess,time
from current_contact_study import ROOT
P=ROOT/'artifacts/current-contact-5090'
python=str(ROOT/'.venv/bin/python')
commands=[['scripts/run_current_contact.py','--stage','evaluation'],['scripts/analyze_current_contact.py'],['scripts/plot_current_contact.py'],['scripts/run_current_contact.py','--stage','video']]
with (P/'pipeline.jsonl').open('a',buffering=1) as f:
    for command in commands:
        f.write(json.dumps(dict(event='start',command=command,unix=time.time()))+'\n')
        p=subprocess.run([python,*command],cwd=ROOT)
        f.write(json.dumps(dict(event='finish',command=command,returncode=p.returncode,unix=time.time()))+'\n')
        if p.returncode:raise SystemExit(p.returncode)
(P/'pipeline-completed.json').write_text(json.dumps(dict(completed_unix=time.time(),scope='Evaluation, physical analysis, plots and selected video reruns; final review/release verification remains separate.'))+'\n')
