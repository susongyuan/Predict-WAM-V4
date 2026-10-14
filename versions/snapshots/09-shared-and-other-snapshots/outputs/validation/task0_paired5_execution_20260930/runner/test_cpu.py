#!/usr/bin/env python3
"""Synthetic control-plane tests only; no GPU launch or physical scores."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock
import paired5 as r
import admission as a

class Safety(unittest.TestCase):
    def test_01_source_and_receipt_binding(self):
        self.assertEqual(a.authorize()['gpu_utilization_percent_allowed'],100)
        self.assertEqual(len(r.verify_code()),64)

    def test_02_no_implicit_execute(self):
        with patch.object(r,'dry_run',return_value={'cpu':True}) as dry, patch.object(r,'acquire_gpu',side_effect=AssertionError('GPU reached')):
            with contextlib.redirect_stdout(io.StringIO()): r.main(['--variant','original','--seeds','1'])
            dry.assert_called_once()

    def test_03_unauthorized_env_does_not_bypass(self):
        with patch.object(a,'AUTH',Path('/definitely/missing/authorization.json')),patch.dict(os.environ,{'P2_ALLOW_GPU':'1','CUDA_VISIBLE_DEVICES':'0'}),patch.object(r,'dispatch_lock',side_effect=AssertionError('lock reached')):
            with self.assertRaisesRegex(RuntimeError,'Authorization'): r.main(['--execute','--seeds','1'])

    def test_04_direct_gpu_api_requires_execute_permit(self):
        with patch.object(r,'workers',side_effect=AssertionError('process scan reached')):
            with self.assertRaisesRegex(RuntimeError,'explicit --execute'): r.acquire_gpu('synthetic')
        with patch.object(r,'receipt_state',side_effect=AssertionError('state reached')):
            with self.assertRaisesRegex(RuntimeError,'explicit --execute'): r.run_one('original',1,'synthetic')

    def test_05_duplicate_bad_seed_refused(self):
        for seeds in ([1,1],[],[2,1],[-1],[5]):
            with self.assertRaises(RuntimeError): r.selected('original',seeds)

    def test_06_seed0_reused_only(self):
        status,row=r.receipt_state('original',0)
        self.assertEqual(status,'complete'); self.assertTrue(row['reused'])
        self.assertFalse(row['benchmark_toggle_success']); self.assertEqual(row['actual_steps'],3225)
        with patch.object(r,'require_permit'):
            with self.assertRaisesRegex(RuntimeError,'cannot be launched'): r.run_one('original',0,'synthetic',r._PERMIT)

    def test_07_all_complete_skip_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'STATE',Path(tmp)),patch.object(r,'acquire_gpu',side_effect=AssertionError('GPU reached')),patch.object(r,'dry_run',side_effect=AssertionError('probe reached')):
            with contextlib.redirect_stdout(io.StringIO()): r.main(['--execute','--variant','original','--seeds','0'])

    def test_08_partial_and_error_stop_no_rerun(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'STATE',Path(tmp)):
            d=Path(tmp)/'original/seed1';d.mkdir(parents=True)
            for file in ('pending.json','infrastructure_error.json'):
                (d/file).write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'no rerun'): r.receipt_state('original',1)
            with patch.object(r,'acquire_gpu',side_effect=AssertionError('GPU reached')):
                with self.assertRaisesRegex(RuntimeError,'no rerun'): r.main(['--execute','--seeds','1'])

    def test_09_single_dispatch_no_chain(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'STATE',Path(tmp)),patch.object(r,'dry_run',side_effect=AssertionError('probe reached')):
            with self.assertRaisesRegex(RuntimeError,'Single-dispatch'): r.main(['--execute','--seeds','1','2'])

    def test_10_empty_complete_receipt_does_not_skip(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'STATE',Path(tmp)):
            d=Path(tmp)/'improved/seed2'; d.mkdir(parents=True)
            receipt={'status':'complete','variant':'improved','seed':2,'run_dir':str(r.run_path('improved',2)),
                     'authorization_sha256':a.AUTH_SHA,'code_integrity_sha256':r.sha(r.BASE/'runner/code_integrity.json'),'evidence_sha256':{}}
            (d/'complete.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError,'directory missing'): r.receipt_state('improved',2)
            with patch.object(r,'acquire_gpu',side_effect=AssertionError('GPU reached')):
                with self.assertRaisesRegex(RuntimeError,'directory missing'):
                    r.main(['--execute','--variant','improved','--seeds','2'])
            receipt['seed']=3;(d/'complete.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError,'Invalid complete'): r.receipt_state('improved',2)

    @contextlib.contextmanager
    def completed_fixture(self):
        # Synthetic, temporary control-plane evidence, never a physical result.
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'STATE',Path(tmp)/'state'):
            run=Path(tmp)/'run';(run/'eval/videos').mkdir(parents=True)
            (run/'manifest.json').write_text('{}')
            (run/'eval/videos/fixture.mp4').write_bytes(b'synthetic-not-a-video')
            d=r.STATE/'improved/seed2';d.mkdir(parents=True)
            receipt={'status':'complete','variant':'improved','seed':2,'run_dir':str(run),
                     'authorization_sha256':a.AUTH_SHA,'code_integrity_sha256':r.sha(r.BASE/'runner/code_integrity.json'),
                     'success':False,'q_score':0.0,'steps':5,'physical_outcome':'failure',
                     'video':{'all_verified':True},'exit_confirmation':[{},{}],
                     'evidence_sha256':{p:r.sha(run/p) for p in ('manifest.json','eval/videos/fixture.mp4')}}
            current={k:v for k,v in receipt.items() if k!='exit_confirmation'};current['video']=None
            (d/'complete.json').write_text(json.dumps(receipt))
            with patch.object(r,'run_path',return_value=run):
                yield d,receipt,current

    def test_19_verified_failure_reused_without_gpu(self):
        with self.completed_fixture() as (d,p,current),patch.object(r,'validate_result',return_value=current) as verify:
            self.assertEqual(r.receipt_state('improved',2)[0],'complete')
            verify.assert_called_with('improved',2,decode=False)
            with patch.object(r,'acquire_gpu',side_effect=AssertionError('GPU reached')):
                with contextlib.redirect_stdout(io.StringIO()):r.main(['--execute','--variant','improved','--seeds','2'])

    def test_20_missing_fields_or_empty_evidence_rejected(self):
        with self.completed_fixture() as (d,p,current),patch.object(r,'validate_result',return_value=current):
            for key in ('success','steps','q_score','video','exit_confirmation','evidence_sha256'):
                bad=dict(p);bad.pop(key)
                (d/'complete.json').write_text(json.dumps(bad))
                with self.subTest(key=key),self.assertRaises(RuntimeError):r.receipt_state('improved',2)
            (d/'complete.json').write_text(json.dumps({**p,'evidence_sha256':{}}))
            with self.assertRaisesRegex(RuntimeError,'inventory empty'):r.receipt_state('improved',2)

    def test_21_omitted_evidence_or_changed_result_rejected(self):
        with self.completed_fixture() as (d,p,current),patch.object(r,'validate_result',return_value=current):
            bad={**p,'evidence_sha256':{'eval/videos/fixture.mp4':p['evidence_sha256']['eval/videos/fixture.mp4']}}
            (d/'complete.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError,'result/evidence mismatch'):r.receipt_state('improved',2)
            (d/'complete.json').write_text(json.dumps({**p,'success':True}))
            with self.assertRaisesRegex(RuntimeError,'result/evidence mismatch'):r.receipt_state('improved',2)

    def test_22_actual_manifest_validation_is_required(self):
        with self.completed_fixture():
            with self.assertRaisesRegex(RuntimeError,'Manifest mismatch'):r.receipt_state('improved',2)

    def test_23_error_and_complete_conflict_rejected(self):
        with self.completed_fixture() as (d,p,current):
            (d/'infrastructure_error.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Complete/error'):r.receipt_state('improved',2)

    def test_11_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'STATE',Path(tmp)/'state'),patch.object(r,'run_path',return_value=Path(tmp)):
            with self.assertRaisesRegex(RuntimeError,'overwrite'): r.receipt_state('original',1)

    def test_12_explicit_execute_visibility_not_inherited_cpu_mask(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            self.assertEqual(r.execute_environment()['CUDA_VISIBLE_DEVICES'],'0')
            self.assertEqual(r.configure('original').cpu_env()['CUDA_VISIBLE_DEVICES'],'')

    def test_13_resource_boundaries(self):
        x={'cpu_used_percent':84.9,'memory_used_percent':84.9,'gpu':[{'vram_used_mib':15299,'vram_total_mib':16311,'utilization_percent':100}]}
        self.assertIsNone(a.violation(x))
        x['gpu'][0]['vram_used_mib']=15300;self.assertIn('VRAM',a.violation(x))
        x['gpu'][0]['vram_used_mib']=14580;x['cpu_used_percent']=85;self.assertIn('CPU',a.violation(x))
        x['cpu_used_percent']=0;x['memory_used_percent']=85;self.assertIn('RAM',a.violation(x))
        x['memory_used_percent']=0;x['gpu']=[];self.assertIn('unavailable',a.violation(x))

    def test_14_lock_no_steal(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock=Path(tmp)/'gpu.lock';lock.write_text('another-job\n')
            with patch.object(r,'GPU_LOCK',lock),patch.object(r,'require_permit'),patch.object(r,'workers',return_value=[]):
                with self.assertRaisesRegex(RuntimeError,'no steal'): r.acquire_gpu('synthetic',r._PERMIT)
            self.assertEqual(lock.read_text(),'another-job\n')

    def test_15_live_worker_blocks_before_gpu_query(self):
        with patch.object(r,'require_permit'),patch.object(r,'workers',return_value=[{'pid':1}]),patch.object(r,'module',side_effect=AssertionError('GPU query reached')):
            with self.assertRaisesRegex(RuntimeError,'already running'):r.acquire_gpu('synthetic',r._PERMIT)

    def test_16_direct_wrapper_without_ticket_blocks_before_launch(self):
        cmd=r.configure('original').command(1)
        result=subprocess.run(cmd,cwd=r.PROJECT,env=r.configure('original').cpu_env(),capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('pending.json',result.stderr)
        self.assertFalse(r.run_path('original',1).exists())

    def test_17_error_retained_and_second_call_refused(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'STATE',Path(tmp)/'state'),patch.object(r,'GPU_LOCK',Path(tmp)/'gpu.lock'),patch.object(r,'require_permit'),patch.object(r,'verify_image'),patch.object(r,'assert_exited',return_value=[{'synthetic':'stopped'}]):
            r.GPU_LOCK.write_text('synthetic\n')
            child=Mock(pid=987654321);child.wait.return_value=3;child.poll.return_value=3
            with patch.object(r.subprocess,'Popen',return_value=child) as launch,patch.object(r.subprocess,'run'):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'exit 3'):r.run_one('original',1,'synthetic',r._PERMIT)
                self.assertTrue((r.STATE/'original/seed1/infrastructure_error.json').exists())
                with self.assertRaisesRegex(RuntimeError,'no rerun'):r.run_one('original',1,'synthetic',r._PERMIT)
                self.assertEqual(launch.call_count,1)
            self.assertEqual(r.GPU_LOCK.read_text(),'synthetic\n')

    def test_18_complete_receipts_never_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';r.publish_once(p,{'synthetic':1})
            with self.assertRaises(FileExistsError):r.publish_once(p,{'synthetic':2})
            self.assertEqual(r.read(p),{'synthetic':1})

if __name__=='__main__':unittest.main(verbosity=2)
