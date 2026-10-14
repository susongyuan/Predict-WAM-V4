#!/usr/bin/env python3
"""Serial, frozen-interface Task90 5k/10k/15k manipulation checks; no videos."""
import json
from pathlib import Path
import subprocess
import sys
import time

from gpu_lock import owner, release
from summarize_task090_checkpoints import ROOT, summarize_run

OUT = ROOT / 'outputs/validation/task090_takeover_20261001'
FIRST = 'task090_wam_s015000_dev10_seed0_privbase_preposition_v4_firstgrasp'
FIRST_UNIT = 'task090-wam-s015000-dev10-seed0-preposition-v4'
FIRST_PID = json.loads((OUT / 'prepositionv4_15k_seed0_launch.json').read_text())['pid']
STATE = OUT / 'preposition_v4_matrix_state.json'
MARKER = '\n## 8. 当前自动测试矩阵\n'


def unit_state(unit):
    return subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'],
                          capture_output=True, text=True).stdout.strip()


def record(state):
    state['updated_local'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    progress = ROOT / 'progress.md'
    text = progress.read_text().split(MARKER)[0]
    detail = ('- 预定位辅助操作，固定v4接口，无录像；目标15k/5k/10k各seed0/1/2。\n'
              f'- 状态：{state["status"]}；当前run：{state.get("current_run")}；unit：{state.get("unit")}；PID：{state.get("pid")}。\n'
              f'- 已完成有效操作回合：{len(state["completed"])} / 9；不是成功次数。完整记录：{STATE.relative_to(ROOT)}。\n'
              '- 停止：先终止本矩阵脚本，随后停止当前unit，确认同PID的wam-policy/behavior-run子unit退出后释放自己的GPU锁。\n')
    progress.write_text(text + MARKER + detail)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def wait_clean(unit, pid):
    deadline = time.monotonic() + 2400
    while unit_state(unit) in ('active', 'activating', 'deactivating'):
        if time.monotonic() >= deadline:
            raise RuntimeError('parent has not ended; lock retained')
        time.sleep(3)
    for child in (f'wam-policy-{pid}', f'behavior-run-{pid}'):
        if unit_state(child) in ('active', 'activating', 'deactivating'):
            subprocess.run(['systemctl', '--user', 'stop', child], check=True, timeout=130)
        if unit_state(child) in ('active', 'activating', 'deactivating'):
            raise RuntimeError('child still active; lock retained: ' + child)
    if owner() not in (None, unit):
        raise RuntimeError('GPU lock owned by another job; not releasing')
    release(unit)


def valid_operation(name):
    rows = summarize_run(ROOT / 'outputs/behavior/closed_loop' / name)
    valid = [row for row in rows if row.get('wam_chunks', 0) > 0 and row.get('policy_step_alignment') is True
             and row.get('first_grasp_status') in ('verified_assisted_lift', 'no_qualified_lift')]
    if len(valid) != 1:
        raise RuntimeError('not a valid observed manipulation attempt; stop repetitions: ' + name)
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_task090_checkpoints.py'),
                    '--out', str(ROOT / 'outputs/validation/task090_checkpoint_results')], check=True)
    return {key: valid[0].get(key) for key in ('run', 'checkpoint_step', 'seed', 'wam_chunks', 'first_grasp_status',
                                             'first_grasp_success', 'task_success', 'q_score', 'evaluation_complete')}


def main():
    if STATE.exists():
        raise RuntimeError('matrix state already exists; inspect before resuming, never duplicate runs')
    template = json.loads((OUT / 'prepositionv4_15k_seed0_launch.json').read_text())['argv']
    state = dict(status='waiting_first', current_run=FIRST, unit=FIRST_UNIT, pid=FIRST_PID, completed=[])
    record(state)
    try:
        wait_clean(FIRST_UNIT, FIRST_PID)
        state['completed'].append(valid_operation(FIRST))
        for step, seed in [(15000, 1), (15000, 2)] + [(s, k) for s in (5000, 10000) for k in (0, 1, 2)]:
            tag = f'{step:06d}'
            name = f'task090_wam_s{tag}_dev10_seed{seed}_privbase_preposition_v4_firstgrasp'
            unit = f'task090-wam-s{tag}-dev10-seed{seed}-preposition-v4'
            if (ROOT / 'outputs/behavior/closed_loop' / name).exists():
                raise RuntimeError('refusing existing run: ' + name)
            if owner() is not None:
                raise RuntimeError('GPU lock is not free')
            cmd = [item.replace(FIRST_UNIT, unit) for item in template]
            cmd[cmd.index('--run-name') + 1] = name
            cmd[cmd.index('--seed') + 1] = str(seed)
            checkpoint = ROOT / f'outputs/training/task090_v4_50k_20260930_s{tag}/checkpoint_step_{tag}.pt'
            if not checkpoint.is_file():
                raise RuntimeError('missing fixed checkpoint')
            cmd[cmd.index('--checkpoint') + 1] = str(checkpoint)
            cmd = [('StandardOutput=append:' + str(OUT / (name + '_unit.log')))
                   if item.startswith('StandardOutput=append:') else item for item in cmd]
            state.update(status='starting', current_run=name, unit=unit, pid=None)
            record(state)  # Persist the action before starting the GPU job.
            (OUT / (name + '_launch.json')).write_text(json.dumps({'unit': unit, 'argv': cmd}, indent=2) + '\n')
            subprocess.run(cmd, check=True)
            pid = 0
            for _ in range(100):
                value = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True).strip()
                pid = int(value or 0)
                if pid:
                    break
                time.sleep(.2)
            if not pid:
                raise RuntimeError('no main PID; inspect launch, lock retained')
            state.update(status='running', pid=pid)
            record(state)
            wait_clean(unit, pid)
            state['completed'].append(valid_operation(name))
            state.update(status='run_complete', pid=None)
            record(state)
        state.update(status='matrix_complete', current_run=None, unit=None, pid=None)
        record(state)
    except Exception as error:
        state.update(status='stopped_on_error', error=str(error))
        record(state)
        raise


if __name__ == '__main__':
    main()
