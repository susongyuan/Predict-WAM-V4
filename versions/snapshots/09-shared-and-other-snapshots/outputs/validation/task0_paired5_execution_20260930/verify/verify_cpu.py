#!/usr/bin/env python3
"""Independent CPU verification; writes only this verify directory, no GPU query/launch."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

BASE=Path('/home/user/Robotic/project2/outputs/validation/task0_paired5_execution_20260930')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'runner'))
import paired5 as r


def save(name, data):
    with (HERE/name).open('x') as out: json.dump(data,out,indent=2,ensure_ascii=False)


def invoke(name,args,env=None):
    proc=subprocess.run(args,cwd=BASE,env=env or {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},capture_output=True,text=True,timeout=150)
    with (HERE/(name+'.stdout')).open('x') as out: out.write(proc.stdout)
    with (HERE/(name+'.stderr')).open('x') as out: out.write(proc.stderr)
    return {'returncode':proc.returncode,'stdout':str(HERE/(name+'.stdout')),'stderr':str(HERE/(name+'.stderr'))}

results={'source_integrity_sha256':r.verify_code(),'gpu_started':False,'nvidia_smi_called':False,'large_payload_content_hashed':False}
results['existing_independent_cpu_commands']=json.loads((HERE/'cpu_commands.json').read_text())
results['own_safety_test_log']=str(HERE/'safety_tests.stderr')
results['initial_attempt_stopped_without_overwrite']='Existing original_dry.stdout caused exclusive-write refusal; no evidence replaced. Reuse independently produced dry-run results instead.'
for variant in ('original','improved'):
    # Exercise actual package/controller/assisted-grasp/asset checks that parser-only dry probe omits.
    # hardware_snapshot is replaced before use; inspect_environment runs only a CPU -c wrapper.
    code="""import json,behavior_preflight as p
p.hardware_snapshot=lambda:{'cpu_used_percent':1,'memory_used_percent':30,'gpu':[]}
x=p.collect()
print(json.dumps({'blockers':x['blockers'],'interfaces':x['interfaces'],'python':x['python'],'packages':x['packages'],'authorization':x['license_acceptance']},ensure_ascii=False))
"""
    results[variant+'_preflight_cpu']=invoke(variant+'_preflight_cpu',[str(r.image(variant)/'.conda-envs/lilawam/bin/python'),'-B','-c',code])

with tempfile.TemporaryDirectory(prefix='receipt-counterexample-',dir=HERE) as tmp:
    state=Path(tmp)/'state'
    run=Path(tmp)/'nonexistent-output'
    d=state/'original/seed1';d.mkdir(parents=True)
    forged={'status':'complete','variant':'original','seed':1,'run_dir':str(run),'authorization_sha256':r.AUTH_SHA,'code_integrity_sha256':r.sha(BASE/'runner/code_integrity.json'),'evidence_sha256':{}}
    (d/'complete.json').write_text(json.dumps(forged))
    with patch.object(r,'STATE',state),patch.object(r,'run_path',return_value=run),patch.object(r,'acquire_gpu',side_effect=AssertionError('must not launch')):
        state_name,receipt=r.receipt_state('original',1)
        out=io.StringIO()
        with contextlib.redirect_stdout(out): r.main(['--execute','--variant','original','--seeds','1'])
        results['empty_complete_receipt_counterexample']={'output_exists':run.exists(),'accepted_state':state_name,'execute_output':out.getvalue(),'receipt_missing_physical_outcome':all(x not in forged for x in ('success','q_score','steps','video'))}

results['post_integrity_sha256']=r.verify_code()
results['lock_exists'] = r.GPU_LOCK.exists() or r.GPU_LOCK.is_symlink()
results['workers_snapshot']=r.workers()
save('cpu_verification.json',results)
print(json.dumps(results,indent=2,ensure_ascii=False))
