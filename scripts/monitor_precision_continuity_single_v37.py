#!/usr/bin/env python3
"""Read-only half-hour state receipt for the preregistered continuity cohort."""
from pathlib import Path
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import gpu_lock
from scripts.progress1_log import log


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(args.directory).resolve();out=Path(args.out)
    if out.exists():raise ValueError('preserve previous monitor receipts')
    def read(name):
        p=root/name;return json.loads(p.read_text()) if p.exists() else None
    plan=read('plan.json');summary=read('summary.json');active=read('active_participant.json')
    env=dict(os.environ,**plan['child_env'],XDG_RUNTIME_DIR='/run/user/1000',
        DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    result=subprocess.run(['systemctl','--user','show',plan['systemd_unit'],
        '--property=ActiveState,SubState,MainPID,Result'],env=env,text=True,capture_output=True,check=True)
    state=dict(line.split('=',1) for line in result.stdout.splitlines())
    live={};writers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=(p/'cmdline').read_bytes().decode().split(chr(0));argv=[x for x in argv if x]
        except OSError:continue
        if plan['gpu_lock_owner'] in argv:live[p.name]=dict(command=argv,worker='--worker' in argv)
        if any(Path(x).name in ('eval_precision_continuity10_v36.py','eval_precision_continuity10_v36_r2.py','eval_precision_continuity1_v37.py') for x in argv):
            writers.append(int(p.name))
    current=read('zhuhai_task2/seed_'+str(active['seed'])+'/result.json') if active else None
    participant=live.get(str(active['pid'])) if active else None
    health=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],env=env,
        text=True,capture_output=True,timeout=10)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    value=dict(time=stamp,unit=state,summary_status=summary['status'],completed=len(summary['episodes']),
        full_successes=summary['full_successes'],scores=[r['observed_base_score'] for r in summary['episodes']],
        watchdog=read('watchdog_status.json'),active_participant=active,
        active_command_matches=participant is not None and participant['command']==active['command'],
        actual_children=live,writer_pids=writers,unique_writer=len(writers)==1,
        gpu_owner=gpu_lock.owner(),pinned_driver_healthy=health.returncode==0 and health.stdout.strip()=='595.91.07',
        current=current and {k:current.get(k) for k in ('seed','status','control_steps','partial_base_score','error','record_video','third_view')},
        failure_guard=read('failure_guard.json'),owner_retained_guard=read('owner_retained_guard.json'),
        default_video_off_except_preregistered_first=True,previous_zero_attempt_guards_preserved=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)
    log(f"RTC单例状态核对：{value['completed']}/1、整任务{value['full_successes']}；unit{state['ActiveState']}，子命令一致{value['active_command_matches']}、writer{writers}、driver健康{value['pinned_driver_healthy']}、guard{bool(value['failure_guard'])}。",'root')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
