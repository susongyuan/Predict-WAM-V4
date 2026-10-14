#!/usr/bin/env python3
"""Independent CPU re-admission of v24 archives; never starts an evaluator."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_precision_continuity1_v37 as runner
from scripts import precision_continuity_evidence_v37 as e
from scripts import gpu_lock
from scripts.analyze_precision_dagger_milestones_v19 import episode


def binding(path):
    p=Path(path)
    return dict(path=str(p.resolve()),sha256=e.sha256_file(p),size=p.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();directory=Path(args.directory).resolve();out=Path(args.out)
    e.require(not out.exists(),'Preserve previous independent audit')
    data=(directory/'summary.json').read_bytes();cohort=json.loads(data)
    plan=e.read_json(directory/'plan.json');runner.verify(plan)
    metadata,contract=runner.old.metadata_milestone(plan['checkpoint']['path'],80000)
    e.require(metadata==plan['checkpoint']==cohort['checkpoint'],'Checkpoint binding differs')
    e.require(cohort['seeds']==list(runner.SEEDS) and cohort['planned_episodes']==1
              and cohort['checkpoint_step']==80000,'Requested cohort changed')
    e.require(cohort['record_video'] is True and cohort['third_view'] is True and cohort['recorded_seeds']==[932106]
              and cohort['randomized'] is False,'Evaluation protocol changed')
    complete=cohort['status']=='completed'
    e.require(complete or args.allow_partial,'Not completed')
    e.require([r['seed'] for r in cohort['episodes']]==list(runner.SEEDS)[:len(cohort['episodes'])],
              'Unexpected seed order, duplicate or missing admitted attempt')
    e.require(not (directory/'failure_guard.json').exists(),'Guard exists; preserve and investigate')
    allowed_video=directory/e.TASK/'seed_932106'/'third_person_full.mp4'
    e.require(all(p==allowed_video for p in directory.rglob('*.mp4')),
              'Observer video occurred outside the preregistered first case')
    attempts=[e.read_json(p) for p in directory.glob('attempt_seed_*.json')]
    e.require(len(attempts)==len({a['seed'] for a in attempts}), 'Duplicate attempt receipt')
    commands={x['seed']:x for x in plan['commands']}
    for a in attempts:
        e.require(a==commands[a['seed']],'Actual attempt does not match frozen command')
    rows=[]
    source_hash=plan['extension_sources'][str(Path(runner.runtime.__file__).resolve())]
    for admitted in cohort['episodes']:
        fresh=runner.classify(admitted['raw_result_path'],metadata,contract,source_hash)
        e.require(fresh==admitted,'Independent evidence, score, safety or braking replay differs')
        raw=e.read_json(admitted['raw_result_path'])
        e.require(raw['system_mode']=='pure_wam','Baseline received full-stack intervention')
        code=e.read_json(directory/f'exit_seed_{admitted["seed"]}.json')['returncode']
        valid=(0,2) if admitted['status'] in ('control_safety_abort','physics_safety_abort') else (0,)
        e.require(code in valid and admitted['attempt_count']==1,'Exit/attempt mismatch')
        physical=episode(admitted)
        rows.append(dict(seed=admitted['seed'],status=admitted['status'],control_steps=admitted['control_steps'],
                         score=admitted['observed_base_score'],full_task_success=admitted['full_task_success'],
                         failure_stage=physical['failure_stage'],objects=physical['objects'],
                         evidence_files_rehashed=len(admitted['evidence_files']),
                         deterministic_command_replay=True,raw_result_sha256=admitted['raw_result_sha256']))
        print(f'{admitted["seed"]}: full hash/observation/action/score/braking re-admission PASS',flush=True)
    e.require(cohort['full_successes']==sum(r['full_task_success'] for r in rows),'Aggregate differs')
    receipts={}
    process_check=None
    if complete:
        e.require(len(rows)==1 and len(attempts)==1
                  and len(list((directory/e.TASK).glob('seed_*')))==1,'Exact10 attempts missing')
        e.require(cohort['success_rate']==cohort['full_successes']/1,'Success denominator differs')
        stored=e.read_json(directory/'failure_analysis.json');fresh=runner.failure_analysis(directory,cohort)
        for key in fresh:
            if key!='time':
                e.require(fresh[key]==stored[key],'Independent failure analysis differs: '+key)
        final=e.read_json(directory/'final_receipt.json')
        e.require(final['status']=='completed' and final['tests']==1
                  and final['full_successes']==cohort['full_successes']
                  and final['scores']==[r['score'] for r in rows]
                  and final['attempts_not_retried'] is True,'Terminal receipt differs')
        env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
        result=subprocess.run(['systemctl','--user','show',plan['systemd_unit'],
            '--property=ActiveState,SubState,MainPID,Result'],env=env,text=True,capture_output=True,check=True)
        process_check=dict(unit=result.stdout,gpu_lock_owner=gpu_lock.owner())
        e.require('ActiveState=inactive' in result.stdout and 'MainPID=0' in result.stdout
                  and 'Result=success' in result.stdout,'Writer has not finished cleanly')
        e.require(gpu_lock.owner()!=plan['gpu_lock_owner'],'Own GPU owner not released')
        remaining=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                argv=(proc/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            if plan['gpu_lock_owner'].encode() in argv:
                remaining.append(int(proc.name))
        e.require(not remaining,'Own evaluator/worker still running')
        process_check['own_child_pids']=remaining
        receipts={n:binding(directory/n) for n in ('failure_analysis.json','final_receipt.json')}
    value=dict(schema='independent-rtc-single-archive-audit-v37',
               time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
               status='passed_completed' if complete else 'passed_completed_subset_only',
               completed_count=len(rows),full_successes=cohort['full_successes'],scores=[r['score'] for r in rows],
               stage_counts=dict(Counter(r['failure_stage'] for r in rows)),episodes=rows,
               summary_snapshot=dict(path=str(directory/'summary.json'),sha256=hashlib.sha256(data).hexdigest()),
               plan=binding(directory/'plan.json'),original_five=plan['original_five'],paired_original20=plan['paired_original20'],checkpoint=metadata,
               original_five_preserved=True,all_admitted_archives_independently_rehashed=True,
               process_release_check=process_check,terminal_receipts=receipts,auditor=binding(__file__))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(out=str(out.resolve()),status=value['status'],completed=len(rows),successes=cohort['full_successes'])),flush=True)


if __name__=='__main__':main()
