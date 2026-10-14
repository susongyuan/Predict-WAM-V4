#!/usr/bin/env python3
"""Record the requested half-hour checks without launching model/GPU work."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.supervise_precision_dagger50k_v12 import read, unit_state
from scripts.eval_zhuhai_precision_fixed_attempts import write_json
from scripts.progress1_log import log


def check(plan, units, out):
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    training = Path(plan['training'])
    row = dict(time=now.isoformat(timespec='seconds'),
        training=read(training/'train/status.json'),
        pipeline=read(Path(plan['out'])/'status.json').get('stage'),
        units={name: unit_state(unit) for name, unit in units.items()},
        evaluations={}, latest_firstgrasp=None, interval_seconds=1800)
    probes = sorted((training/'train/firstgrasp').glob('diagnostic_*.json'))
    if probes:
        probe = read(probes[-1])
        row['latest_firstgrasp'] = dict(step=probe['step'],
            val=probe['metrics']['val/all/all']['raw_first8'],
            teacher_state_only=True, not_physical_success_rate=True)
    for step in ('70000', '80000'):
        cohort = read(Path(plan['evaluation_outs'][step])/'summary.json')
        row['evaluations'][step] = dict(status=cohort.get('status'),
            tested=len(cohort.get('episodes', [])), successes=cohort.get('full_successes'),
            planned=5, error=cohort.get('error'))
    row['next_check'] = (now+timedelta(seconds=1800)).isoformat(timespec='seconds')
    write_json(out/'latest.json', row)
    with (out/'checks.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    train = row['training']
    message = (f"半小时核查：正式{train.get('step', plan['resume_step'])}/80000，"
        f"训练{train.get('status', 'starting')}，流程{row['pipeline']}；"
        + '; '.join(f"{s}已测{r['tested']}/5、成功{r['successes']}" for s, r in row['evaluations'].items())
        + f"；异常{train.get('error') or '无'}。")
    log(message, 'root')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'units', 'out'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    os.environ.update(XDG_RUNTIME_DIR='/run/user/1000',
        DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    plan, units = read(args.plan), read(args.units)
    out = Path(args.out); out.mkdir(exist_ok=False)
    next_check = 0
    while True:
        terminal = read(Path(plan['out'])/'status.json').get('stage') in ('completed', 'halted')
        if time.monotonic() >= next_check or terminal:
            check(plan, units, out)
            next_check = time.monotonic()+1800
        if terminal: return 0
        time.sleep(30)


if __name__ == '__main__': raise SystemExit(main())
