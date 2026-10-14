#!/usr/bin/env python3
"""Budget50k wrapper around tested generic serial pipeline; original source stays frozen."""
import argparse
import copy
import json
from pathlib import Path
import signal
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_robotwin_multitask_pipeline as core


def ready(parent,out):
    p=out/'handoff.json'
    if not p.exists():
        s=out/'handoff_status.json'
        if s.exists() and core.read(s).get('status')=='failed':raise RuntimeError('safe handoff failed: '+str(core.read(s)))
        return False
    h=core.read(p)
    if h.get('status')!='complete' or h.get('parent_step')!=5000 or h.get('new_total')!=50000 or h.get('source_changes')!=[]:
        raise RuntimeError('invalid50k handoff receipt')
    if h.get('no_native_evaluation_started') is not True or 'MainPID=0' not in h.get('old_unit_properties',''):
        raise RuntimeError('parent still running or already evaluated')
    if core.sha(h['parent_checkpoint'])!=h['parent_sha256']:raise RuntimeError('handoff checkpoint no longer immutable')
    return True


def run(path):
    path=Path(path).resolve();manifest=core.read(path);out=Path(manifest['out']);parent=Path(manifest['parent_run'])
    with (out/'extension_wrapper_claim.json').open('x') as f:json.dump({'started_at':core.now(),'manifest_sha256':core.sha(path)},f)
    wait={'status':'waiting_handoff','phase':'original_training_to5000','total_budget':50000,'updated_at':core.now(),'verified_complete':False,'record_video':False}
    def cancelled(signum,frame):raise InterruptedError('extension wrapper signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    started=time.monotonic()
    try:
        while not ready(parent,out):
            if core.changes(manifest):raise RuntimeError('extension source changed while awaiting handoff')
            if time.monotonic()-started>7200:raise TimeoutError('extension handoff wait deadline')
            wait.update(updated_at=core.now(),training=core.read(parent/'train/status.json'))
            core.atomic(out/'wait_status.json',wait);core.atomic(parent/'experiment_status.json',wait)
            time.sleep(10)
        core.append_progress('50k续训流水线已取得安全5000步父检查点；先CPU合同检查和独立2步CUDA续训检查，再保留完整优化器继续至总50000。评测只用final50k，5任务各10次、无视频。')
        # Existing engine is budget-agnostic except a human-readable startup label.
        original_doc=core.update_doc
        def doc(text):return original_doc(text.replace('联合训练20k','联合训练50k'))
        core.update_doc=doc
        stop=threading.Event();report_errors=[]
        def sync():
            while not stop.wait(15):
                try:
                    p=out/'pipeline_status.json'
                    if p.exists():
                        value=core.read(p);value.update(total_budget=50000,continued_from_step=5000,active_run=str(out),record_video=False)
                        if (out/'completion.json').exists():value['verified_complete']=core.read(out/'completion.json')['verified_complete']
                        core.atomic(parent/'experiment_status.json',value)
                except Exception as exc:report_errors.append(str(exc))
        thread=threading.Thread(target=sync,daemon=True);thread.start()
        try:code=core.run(path)
        finally:
            stop.set();thread.join(timeout=20);core.update_doc=original_doc
        final=core.read(out/'pipeline_status.json');final.update(total_budget=50000,continued_from_step=5000,active_run=str(out),record_video=False,
                    verified_complete=core.read(out/'completion.json')['verified_complete'],sync_reporting_errors=report_errors)
        core.atomic(parent/'experiment_status.json',final)
        return code
    except BaseException as exc:
        failure={'status':'failed','phase':'50k_extension','updated_at':core.now(),'total_budget':50000,'error':repr(exc),'verified_complete':False}
        core.atomic(out/'wrapper_failure.json',failure);core.atomic(parent/'experiment_status.json',failure)
        core.append_progress('50k扩展流程停止：'+str(exc)+'。原训练/测试证据保留，不回退到20k自动测试。')
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:raise SystemExit(run(a.manifest))
    m=core.read(a.manifest);print(json.dumps({'source_changes':core.changes(m),'stages':[s['name'] for s in m['stages']],'total_budget':50000}))
