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

    def test_selector_scope_and_legacy_budgets(self):
        import select_wam_v3_dev_best as selector
        a=self.args(); plan=S.new_plan(a); selector.validate_plan(plan,ROOT)
        for change in ({'start_step':0}, {'start_step':30000.0},
                       {'checkpoints':list(range(5000,50001,5000))},
                       {'checkpoints':[35000,45000,50000]}, {'total_optimizer_steps':60000}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                selector.validate_plan({**plan,**change},ROOT)
        bad=copy.deepcopy(plan); bad['training']['continuation_plan_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA256'): selector.validate_plan(bad,ROOT)
        for budget in (30000,50000):
            a.continuation_plan=None; a.budget=budget
            legacy=S.new_plan(a); legacy.pop('start_step')
            S.validate_plan(legacy,a); selector.validate_plan(legacy,ROOT)
            self.assertEqual(len(legacy['checkpoints']),budget//5000)

    def cpu_chain(self, interrupt=None):
        # Real execute/validators/selector; mocked authorization ONLY inside a
        # disposable fixture AFTER all process/video/lock boundaries are fake.
        # No production CLI/environment guard bypass is introduced.
        from contextlib import ExitStack
        from test_select_wam_v3_dev_best import SelectionTest
        f=SelectionTest(); f.setUp(); self.addCleanup(f.doCleanups)
        root,out=f.root,f.out
        def receipt(path,step,model_only=False):
            f.put(path.with_name(path.name+'.sha256.json'),dict(step=step,
                size=path.stat().st_size,sha256=S.digest(path),model_only=model_only))
        parent=root/'parent_last.pt'; parent.write_bytes(b'CPU not a pickle'); receipt(parent,30000)
        cp=json.loads(PLAN.read_text()); cp['run_prefix']='audit'
        cp['parent'].update(checkpoint=str(parent),sha256=S.digest(parent),
                            config_sha256=S.digest(root/'config.yaml'))
        f.put(root/'continuation.json',cp); f.put(root/'approval.json',{'CPU_only':True})
        a=SimpleNamespace(prefix='audit',continuation_plan='continuation.json',budget=50000,
            config='config.yaml',cache_ledger='unused',approval='approval.json',
            resume_source=str(parent),instance=10)
        trained=[]; evaluated=[]; published=[]; active=set(); events=[]; armed=[interrupt]
        def train(cmd,**kw):
            self.assertEqual(cmd,[str(root/'scripts/run_wam_v3_unit.sh')])
            env=kw['env']; step=int(env['MAX_STEPS']); resumed=Path(env['RESUME'])
            expected=parent if step==35000 else root/'outputs/training'/f'audit_s{step-5000:06d}'/'checkpoint_last.pt'
            self.assertEqual(resumed,expected)
            self.assertEqual(S.validate_last(resumed)['step'],step-5000)
            self.assertEqual(env['RESUME_SHA256'],S.digest(resumed))
            self.assertEqual(env['CONTINUATION_PLAN'],str(root/'continuation.json'))
            if step>35000: self.assertIn(step-5000,published)
            run=root/'outputs/training'/env['NAME']; run.mkdir(parents=True)
            ck=run/f'checkpoint_step_{step:06d}.pt'
            ck.write_bytes(f'CPU fixture at {step}; deliberately not a torch pickle'.encode()); receipt(ck,step,True)
            last=run/'checkpoint_last.pt'; last.write_bytes(b'untouched last/optimizer'); receipt(last,step)
            f.put(run/'summary.json',{'completed_updates':step})
            f.put(run/'status.json',{'status':'budget_complete'})
            active.add(f'wam-train-{env["NAME"]}.service'); trained.append(step); events.append(('train',step))
            return SimpleNamespace(returncode=0)
        def unit_live(unit):
            if unit in active: active.remove(unit); return True
            return False
        def evaluate(cmd,**kw):
            self.assertEqual(cmd,[str(root/'scripts/run_task0_closed_loop.sh')])
            step=int(Path(kw['env']['CKPT']).stem.split('_')[-1]); self.assertEqual(trained[-1],step)
            paths=f.fixture(step,False,0.9 if step==35000 else 0.1)
            paths['stage'].unlink()  # Completion receipt must come from real controller.
            evaluated.append(step); events.append(('eval',step))
            return SimpleNamespace(pid=987654321,wait=lambda:0)
        real_publish=S.publish_best
        def publish(path):
            index=real_publish(path); self.assertEqual(index['rejected_count'],0)
            latest=max(c['step'] for c in index['candidates'])
            published.append(latest); events.append(('select',latest))
            if armed[0]=='after_selector':
                armed[0]=None; raise RuntimeError('CPU injected interruption after first selector')
            return index
        def video(phys,target,step):
            if armed[0]=='after_eval':
                armed[0]=None; raise RuntimeError('CPU injected interruption after first evaluation')
            path=target/f'video_{step:06d}.json'
            f.put(path,{'CPU_fixture_only':True,'not_physical_validation':True}); return str(path)
        with ExitStack() as stack:
            for obj,key,value in [(S,'ROOT',root),(S.subprocess,'run',train),
                (S.subprocess,'Popen',evaluate),(S,'unit_live',unit_live),
                (S,'wait_release',lambda:None),(S.time,'sleep',lambda _:None),
                (S,'process_token',lambda _:None),(S,'verify_video',video),(S,'publish_best',publish)]:
                stack.enter_context(patch.object(obj,key,value))
            plan=S.new_plan(a); S.validate_plan(plan,a); f.put(out/'plan.json',plan)
            with self.assertRaisesRegex(ValueError,'resource-window'): S.execute(plan,a,out)
            self.assertEqual(events,[])
            stack.enter_context(patch.object(S,'require_execution_authorization',lambda _:None))
            rc=S.execute(plan,a,out)
            if interrupt:
                self.assertEqual(rc,3); self.assertEqual(trained,[35000]); self.assertEqual(evaluated,[35000])
                self.assertEqual(S.read(out/'status.json')['phase'],'stopped_infrastructure_failure')
                S.validate_plan(S.read(out/'plan.json'),a)
                self.assertEqual(S.execute(S.read(out/'plan.json'),a,out),0)
            else: self.assertEqual(rc,0)
            expected=[35000,40000,45000,50000]
            self.assertEqual(trained,expected); self.assertEqual(evaluated,expected)
            index=S.read(out/'best_task.json')
            self.assertEqual([c['step'] for c in index['candidates']],expected)
            self.assertEqual(index['pending'],[]); self.assertEqual(index['rejected'],[])
            self.assertEqual(index['best']['step'],35000); self.assertFalse(index['has_successful_checkpoint'])
            self.assertEqual(S.read(out/'status.json')['phase'],'budget_complete')
            self.assertEqual(len(list(out.glob('stage_*.json'))),4)
            self.assertEqual(S.execute(plan,a,out),0)
            self.assertEqual(trained,expected); self.assertEqual(evaluated,expected)
            print('CPU_CHAIN',interrupt,events,flush=True)

    def test_cpu_four_stage_train_eval_select_resume(self): self.cpu_chain()
    def test_cpu_restart_after_eval_before_receipt(self): self.cpu_chain('after_eval')
    def test_cpu_restart_after_selector(self): self.cpu_chain('after_selector')

if __name__=='__main__': unittest.main(verbosity=2)

