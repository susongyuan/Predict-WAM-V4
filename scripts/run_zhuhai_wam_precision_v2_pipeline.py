#!/usr/bin/env python3
"""Frozen Precision-WAM V2 30k queue: paired five-episode checks at 20/25/30k."""
from pathlib import Path
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_zhuhai_wam_hjepa_pipeline as base
from harness import zhuhai_precision_v2_training as precision
from scripts.progress1_log import log
BASE_VERIFY_RESULT=base.verify_result


def configure():
    base.h=precision;base.log=log
    base.TRAIN=ROOT/'scripts/train_zhuhai_wam_precision_v2.py'
    base.EVAL=ROOT/'scripts/eval_zhuhai_wam_precision_v2.py'
    base.verify_result=verify_result


def verify_result(stage):
    value=BASE_VERIFY_RESULT(stage)
    if stage['kind']=='train':
        if stage['mode']=='tiny' and value.get('precision_gradient_admission_passed') is not True:
            raise ValueError('Precision clock/dense-refiner GPU gradient admission not passed')
        ck=Path(stage['out'])/f"checkpoint_step_{stage['step']:06d}.pt"
        receipt=precision.read_json(ck.with_name(ck.name+'.sha256.json'))
        if receipt.get('sha256')!=precision.sha256_file(ck):raise ValueError('Precision milestone content hash mismatch')
    return value


def build_plan(args):
    configure();plan=base.build_plan(args)
    plan['schema']='zhuhai-precision-wam-task2-30k-pipeline-v2'
    plan['precision']=precision.PRECISION_SPEC
    # Latest user request supersedes the extra ten-episode final stage BEFORE
    # any V2 rollout. Keep all three fixed, paired five-seed milestone checks.
    plan['stages']=[s for s in plan['stages'] if s['name']!='final_30000']
    plan['final_seeds']=[]
    plan['checkpoint_selection']='fixed final30k; report paired 20k/25k/30k, no best-of selection'
    plan['evaluation_protocol_revision']='2026-10-10: latest user asks five episodes at each 5k in final10k; no additional final10'
    plan['planned_physical_episodes']=15
    plan['protocol_change']='V2 independent run; paired fixed-seed 5x3; original full physical scoring/video-off; task1 DR200 requires separate admission'
    plan['runtime_source_note']='V1 stopped externally at zero updates; raw DINO cache reused, never V1 model outputs'
    paths=[ROOT/p for p in precision.EXTENSION_SOURCES]
    plan['source_hashes'].update({str(p):precision.sha256_file(p) for p in paths})
    return plan


def resource_snapshot():
    memory={k.rstrip(':'):int(v.split()[0])*1024 for k,v in
            (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    value=dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        mem_available_bytes=memory['MemAvailable'],swap_used_bytes=memory['SwapTotal']-memory['SwapFree'])
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,timeout=10,check=True)
        used,total,util=map(int,result.stdout.strip().splitlines()[0].split(','))
        value.update(gpu_used_mib=used,gpu_total_mib=total,gpu_utilization_percent=util)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:value['gpu_telemetry_error']=str(exc)
    return value


def run_stage(stage):
    """Preserve logs/telemetry and terminate only this stage's process group."""
    root=Path(stage['out']).parent;path=root/(stage['name']+'.resources.jsonl')
    with (root/(stage['name']+'.log')).open('xb') as stream, path.open('x') as telemetry:
        process=subprocess.Popen(stage['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                row=resource_snapshot();row.update(stage=stage['name'],child_pid=process.pid)
                telemetry.write(json.dumps(row,allow_nan=False)+'\n');telemetry.flush()
                try:return process.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=180)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise


def summarize(plan):
    """Report observed outcomes only; failure/unfinished stages are not successes."""
    rows=[]
    for stage in plan['stages']:
        if stage['kind']!='eval':continue
        path=Path(stage['out'])/'summary.json'
        if not path.is_file():
            rows.append(dict(step=stage['step'],status='not_completed',successes=None,episodes=5,success_rate=None));continue
        raw=precision.read_json(path)
        if raw.get('status')!='completed':
            rows.append(dict(step=stage['step'],status=raw.get('status','incomplete'),successes=None,episodes=5,success_rate=None));continue
        BASE_VERIFY_RESULT(stage)
        results=raw['episodes'];n=len(results);k=sum(r['full_task_success'] for r in results)
        z=1.959963984540054;p=k/n;den=1+z*z/n
        center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        rows.append(dict(step=stage['step'],status='completed',successes=k,episodes=n,success_rate=k/n,
            wilson95=[max(0.,center-half),min(1.,center+half)],seeds=stage['seeds'],evidence=str(path)))
    value=dict(schema='precision-v2-paired-milestone-results-v1',time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        results=rows,fixed_final_step=30000,paired_small_sample_not_independent_generalization=True,
        record_video=False,third_view=False,checkpoint_selection=plan['checkpoint_selection'])
    precision.write_json(Path(plan['out'])/'milestone_results.json',value)
    return value


def execute(plan,runner=run_stage):
    try:return base.execute(plan,runner)
    finally:
        if (Path(plan['out'])/'plan.json').is_file():summarize(plan)


def main(argv=None):
    configure();args=base.parser().parse_args(argv);plan=build_plan(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    if not args.gpu_lock_owner:raise ValueError('External GPU owner required')
    precision.verify_owner(args.gpu_lock_owner);return execute(plan)

if __name__=='__main__':raise SystemExit(main())
