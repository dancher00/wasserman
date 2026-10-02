"""Guard against turning the bounded handoff into a different experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import second_machine_check as check

class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.runtime=self.root/'runtime'; self.meta=self.root/'metadata'
        for p in [self.root/'research',self.runtime/'scripts',self.runtime/'logs',self.meta]:p.mkdir(parents=True)
        (self.runtime/'scripts/collect_valve_visual.py').write_bytes(b'expert')
        (self.runtime/'logs/model.pt').write_bytes(b'weights');(self.meta/'eval.py').write_bytes(b'evaluator')
        sha=lambda b:hashlib.sha256(b).hexdigest()
        (self.root/'research/paper-release.json').write_text(json.dumps({'runtimes':{'core':{'commit':'frozen'}}}))
        (self.root/'research/paper-results-index.json').write_text(json.dumps({'records':[{'id':'baseline-RotateValve-DP','backend':'eval.py','backend_sha256':sha(b'evaluator'),'checkpoint':'logs/model.pt','checkpoint_sha256':sha(b'weights')}]}))
        self.patch=patch.object(check,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.runtime_check=patch.object(check,'check_runtime');self.runtime_mock=self.runtime_check.start();self.addCleanup(self.runtime_check.stop)
    def test_fixed_full_episode_and_pinned_runtime(self):
        for mode in ['expert','dp']:
            p=check.plan(self.runtime,self.meta,mode,self.root/'fresh')
            self.runtime_mock.assert_called_with(self.runtime,'frozen')
            self.assertEqual(p['command'][p['command'].index('--seeds')+1:][:4],['2500','--steps','2240','--output-dir'])
            self.assertEqual(p['command'].count('--seeds'),1)
    def test_modified_weight_rejected(self):
        (self.runtime/'logs/model.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Checkpoint'):check.plan(self.runtime,self.meta,'dp',self.root/'fresh')
    def test_modified_evaluator_rejected(self):
        (self.meta/'eval.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'evaluator'):check.plan(self.runtime,self.meta,'dp',self.root/'fresh')
    def test_previous_output_preserved(self):
        with self.assertRaisesRegex(ValueError,'fresh'):check.plan(self.runtime,self.meta,'expert',self.meta)

if __name__=='__main__':unittest.main()
