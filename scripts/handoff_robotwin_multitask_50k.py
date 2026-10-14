#!/usr/bin/env python3
"""One user-authorized handoff after original step5000 checkpoint+VAL. Never edit weights."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_robotwin_multitask_pipeline import atomic,read,sha,now,changes,append_progress


def inspect_ready(root):
    state=read(root/'train/status.json')
    if state.get('status') not in ('running','completed_fixed_budget'):
        raise RuntimeError('original training stopped unexpectedly: '+str(state))
    receipt=read(root/'train/checkpoint_last.pt.sha256.json')
    if receipt.get('step',0)>5000:
        raise RuntimeError('missed registered step5000 handoff; do not choose a different checkpoint silently')
    if state.get('step',0)<5000 or receipt.get('step')!=5000 or not (root/'train/validation_step_005000.json').is_file():
        return None
    if receipt.get('model_only') is not False:
        raise RuntimeError('complete optimizer checkpoint is required')
    if changes(read(root/'pipeline_manifest.json')):
        raise RuntimeError('original frozen source changed before handoff')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-run',required=True);p.add_argument('--unit',required=True);a=p.parse_args()
    root=Path(a.parent_run).resolve();out=root/'extension_50k';out.mkdir(exist_ok=True)
    if (out/'handoff.json').exists():raise FileExistsError('handoff already recorded')
    started=time.monotonic()
    atomic(out/'handoff_status.json',dict(status='waiting_step5000_and_val',at=now(),old_total=20000,new_total=50000))
    try:
        while True:
            receipt=inspect_ready(root)
            if receipt is not None:break
            if time.monotonic()-started>7200:raise TimeoutError('step5000 handoff wait deadline')
            time.sleep(2)
        before=read(root/'train/status.json')
        atomic(out/'handoff_intent.json',dict(at=now(),reason='User explicitly changed total budget to50k',unit=a.unit,parent_receipt=receipt,observed_before_stop=before))
        subprocess.run(['systemctl','--user','stop',a.unit],check=True,timeout=100)
        props=subprocess.check_output(['systemctl','--user','show',a.unit,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' not in props:raise RuntimeError('original unit has not fully stopped')
        path=root/'train/checkpoint_last.pt';current=read(path.with_name(path.name+'.sha256.json'))
        if current!=receipt or current['sha256']!=sha(path):raise RuntimeError('parent safe checkpoint changed/corrupted during handoff')
        after=read(root/'train/status.json');pipeline=read(root/'completion.json')
        if pipeline.get('source_changes')!=[]:raise RuntimeError('original source integrity did not hold')
        step=after.get('step_observed',after.get('step',5000))
        # metrics carries the last fully performed optimizer update if stop landed after a status write.
        lines=(root/'train/metrics.jsonl').read_text().splitlines()
        performed=max((json.loads(x).get('step',0) for x in lines if x.strip() and json.loads(x).get('event')=='optimizer_step'),default=step)
        result=dict(status='complete',at=now(),reason='authorized_budget_extension_not_training_failure',parent_step=5000,
            parent_checkpoint=str(path),parent_sha256=current['sha256'],parent_receipt=current,source_changes=[],old_unit=a.unit,old_unit_properties=props,
            original_training_last_observed=after,original_last_optimizer_step=performed,
            discarded_uncheckpointed_updates=max(0,performed-5000),replay_from_safe_step=5000,new_total=50000,
            original_completion_sha256=sha(root/'completion.json'),original_results_preserved=True,no_native_evaluation_started=not (root/'eval').exists())
        if not result['no_native_evaluation_started']:raise RuntimeError('unexpected original eval directory; refuse duplicate testing')
        atomic(out/'handoff.json',result);atomic(out/'handoff_status.json',result)
        append_progress(f'已按50k新预算在原5000步+VAL后交接：保存完整Adam/RNG/采样状态，旧unit已停止并释放GPU；原回执标记为用户预算变更导致的停止，不算模型失败。将从安全5000步继续至50000；未落盘更新{result["discarded_uncheckpointed_updates"]}步如实记入handoff.json，原生测试尚未开始。')
    except BaseException as exc:
        atomic(out/'handoff_status.json',dict(status='failed',at=now(),error=repr(exc)))
        raise

if __name__=='__main__':main()
