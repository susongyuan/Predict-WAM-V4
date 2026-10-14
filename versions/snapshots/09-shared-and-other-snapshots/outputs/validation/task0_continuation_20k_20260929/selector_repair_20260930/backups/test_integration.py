#!/usr/bin/env python3
"""CPU-only launcher/controller/source integration, no systemd or GPU calls."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / 'continuation_plan.json'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_wam_v3_stages as S

class Integration(unittest.TestCase):
    def args(self):
        cp = json.loads(PLAN.read_text())
        return SimpleNamespace(continuation_plan=str(PLAN), budget=50000,
            config=cp['parent']['config'], cache_ledger='data/dino_cache_task000/sha256_ledger_micro16_v1.json',
            approval=str(HERE / 'no_gpu_approval.json'), resume_source=cp['parent']['checkpoint'],
            instance=10, prefix=cp['run_prefix'] + 'cpucheck')

    def test_controller_30k_four_stages_and_recovery(self):
        a = self.args(); p = S.new_plan(a); S.validate_plan(p, a)
        self.assertEqual(p['start_step'], 30000)
        self.assertEqual(p['checkpoints'], [35000,40000,45000,50000])
        S.validate_plan(json.loads(json.dumps(p)), a)
        for field, value in [('checkpoints',[5000,35000,40000,45000,50000]),('total_optimizer_steps',60000),('start_step',0)]:
            bad = copy.deepcopy(p); bad[field] = value
            with self.assertRaises(ValueError): S.validate_plan(bad, a)
        with patch.object(S.subprocess, 'run', side_effect=AssertionError('no subprocess')), patch.object(S, 'wait_release', side_effect=AssertionError('no GPU lock')):
            with self.assertRaisesRegex(ValueError, 'resource-window'): S.execute(p, a, HERE)

    def test_launcher_dry_run_and_execution_block(self):
        a = self.args()
        env = dict(os.environ, NAME=a.prefix, MODE='long', MAX_STEPS='35000',
            CONFIG=a.config, CACHE_LEDGER=a.cache_ledger, APPROVAL=a.approval,
            RESUME=a.resume_source, CONTINUATION_PLAN=str(PLAN), DRY_RUN='1')
        for key in ('INIT','RUN_NAME','RESUME_SHA256'): env.pop(key,None)
        cmd = ['bash', str(ROOT/'scripts/run_wam_v3_unit.sh')]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('--continuation-plan '+str(PLAN),run.stdout)
        self.assertIn('--max-steps 35000',run.stdout)
        env['DRY_RUN']='0'
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode,2)
        self.assertIn('resource-window authorization unresolved',run.stderr)
        self.assertFalse((ROOT/'outputs/training'/a.prefix).exists())

    def test_runtime_training_dependency_and_live_inventory(self):
        cp = json.loads(PLAN.read_text()); migrations={r['path']:r for r in cp['source_migrations']}
        row=migrations['harness/orchestrator.py']
        old=ast.parse((ROOT/row['backup']).read_text()); new=ast.parse((ROOT/row['path']).read_text())
        self.assertEqual(len(old.body),len(new.body))
        for a,b in zip(old.body,new.body):
            if getattr(a,'name',None) not in ('Contract','Orchestrator','_self_test','self_test'):
                self.assertEqual(ast.dump(a),ast.dump(b))
        manifest=json.loads((ROOT/cp['parent']['manifest']).read_text())
        for filename,digest in manifest['source_sha256'].items():
            rel=str(Path(filename).relative_to(ROOT))
            if rel.startswith(('harness/','scripts/')):
                self.assertEqual(S.digest(filename),migrations.get(rel,{}).get('to_sha256',digest),rel)
        with patch.dict(os.environ, {'CONTINUATION_PLAN':'untrusted'}):
            self.assertNotIn('CONTINUATION_PLAN',S.clean_env())

if __name__=='__main__': unittest.main(verbosity=2)
