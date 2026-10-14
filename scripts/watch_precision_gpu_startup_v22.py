"""Watch our evaluation startup/heartbeat and pinned GPU health; preserve failures."""
from pathlib import Path
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import signal
import subprocess
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts.supervise_precision_dagger50k_v12 import unit_state,read
from scripts.progress1_log import log


def execute(args):
    p=e.read_json(args.primary);c=e.read_json(args.conditional)
    out=Path(args.out);out.mkdir(exist_ok=False)
    units=[(args.primary_unit,'supervise_precision_dagger_milestones_v13.py',p),
        (args.conditional_unit,'run_precision_dagger_diagnostic_branch_v22.py',c)]
    env=dict(os.environ,**c['runtime_environment']);bad_nvml=0
    while True:
        participants=[];problems=[]
        for unit,source,plan in units:
            state=unit_state(unit);pid=int(state['MainPID'])
            if not pid:continue
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            e.require(source in command and state['ActiveState']=='active','Unexpected guarded participant')
            participants.append((pid,unit))
            for directory in plan['evaluation_outs'].values():
                for file in Path(directory).glob('zhuhai_task2/seed_*/result.json'):
                    raw=read(file);age=time.time()-file.stat().st_mtime
                    if raw.get('status')=='starting' and age>300:
                        problems.append(dict(pid=pid,unit=unit,reason='evaluation startup exceeds300s',path=str(file),age_seconds=age))
                    elif raw.get('status')=='running' and age>600:
                        problems.append(dict(pid=pid,unit=unit,reason='evaluation action heartbeat stale600s',path=str(file),age_seconds=age))
        if not participants:
            e.write_json(out/'status.json',dict(status='completed',reason='all monitored task supervisors exited'))
            return 0
        try:
            check=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],env=env,
                capture_output=True,text=True,timeout=10)
            ok=check.returncode==0 and check.stdout.strip()=='595.91.07'
            detail=dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr)
        except subprocess.TimeoutExpired:ok=False;detail={'error':'NVML query timeout'}
        bad_nvml=0 if ok else bad_nvml+1
        if bad_nvml>=3:
            problems.extend(dict(pid=pid,unit=unit,reason='pinned GPU NVML failed3 consecutive checks',nvml=detail) for pid,unit in participants)
        now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
        if problems:
            e.write_json(out/'failure_guard.json',dict(time=now,problems=problems,policy_results_not_reclassified=True))
            log('GPU启动/心跳守护暂停：'+str(problems)+'；所有既有成绩和启动异常保留，不自动重试。','gpu_startup_guard')
            for pid in set(row['pid'] for row in problems):os.kill(pid,signal.SIGINT)
            return 2
        e.write_json(out/'status.json',dict(status='watching',time=now,participants=participants,
            matched_driver=True,startup_limit_seconds=300,action_heartbeat_limit_seconds=600))
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('primary','conditional','primary-unit','conditional-unit','out'):
        parser.add_argument('--'+flag,required=True)
    raise SystemExit(execute(parser.parse_args()))
