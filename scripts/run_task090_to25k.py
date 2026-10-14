#!/usr/bin/env python3
"""User-authorized 15k -> 20k -> 25k continuation; physical checks deferred to 25k."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from gpu_lock import owner
from run_task090_preposition_matrix import wait_clean, unit_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/validation/task090_takeover_20261001'
PYTHON = ROOT / '.conda-envs/lilawam/bin/python'
STATE = OUT / 'to25k_state.json'
MARKER = '\n## 9. 当前25k续训队列\n'


def save(state):
    state['updated_local'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    progress = ROOT / 'progress.md'
    text = progress.read_text().split(MARKER)[0]
    progress.write_text(text + MARKER +
        f'- 状态：{state["status"]}；unit：{state.get("unit")}；PID：{state.get("pid")}；已确认步数：{state.get("observed_step", 15000)}。\n'
        '- 本轮用户允许先至25k再测试，20k保留checkpoint但不阻塞于物理评估。\n'
        '- 停止先停task090-to25k-queue.service，再对当前trainer主PID发SIGTERM，等待完整last落盘及自己的锁释放。\n'
        f'- 队列状态与错误：{STATE.relative_to(ROOT)}。\n')
    print(json.dumps(state, ensure_ascii=False), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if STATE.exists():
        raise RuntimeError('queue already has state; inspect it instead of duplicating training')
    first = json.loads((OUT / 'prepositionv4_15k_seed0_launch.json').read_text())
    state = dict(status='waiting_final_diagnostic', observed_step=15000, unit=first['unit'], pid=first['pid'], completed=[])
    save(state)
    try:
        wait_clean(first['unit'], first['pid'])
        for target in (20000, 25000):
            start, tag = target - 5000, f'{target:06d}'
            name = f'task090_v3_50k_20260930_s{tag}'
            unit = 'wam-train-' + name
            run = ROOT / 'outputs/training' / name
            source = ROOT / f'outputs/training/task090_v3_50k_20260930_s{start:06d}/checkpoint_last.pt'
            if run.exists() or owner() is not None:
                raise RuntimeError('existing output or occupied GPU; no duplicate launch')
            source_sha = sha(source)
            if target == 20000 and source_sha != '97e158bd98ec48242a0421bf0796c386590ea3be2f6356636cce95cc5e8df549':
                raise RuntimeError('15k complete checkpoint differs from verified source')
            cmd = ['systemd-run', '--user', '--collect', '--unit=' + unit, '--working-directory=' + str(ROOT),
                   '-p', 'CPUQuota=800%', '-p', 'MemoryMax=16G', '-p', 'MemorySwapMax=0', '-p', 'LimitNOFILE=8192',
                   '-p', 'KillMode=mixed', '-p', 'TimeoutStopSec=900',
                   '-p', f'ExecStartPre={PYTHON} {ROOT}/scripts/gpu_lock.py acquire --owner {unit} --max-wait 0',
                   '-p', f'ExecStopPost={PYTHON} {ROOT}/scripts/gpu_lock.py release --owner {unit}',
                   '-p', 'StandardOutput=append:' + str(OUT / f'to25k_s{tag}_unit.log'), '-p', 'StandardError=inherit',
                   '--setenv=P2_TASK_ID=90', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OMP_NUM_THREADS=8',
                   '--setenv=OPENBLAS_NUM_THREADS=8', '--setenv=MKL_NUM_THREADS=8',
                   str(PYTHON), str(ROOT / 'scripts/train_wam_task_v3.py'),
                   '--config', str(ROOT / 'configs/lilawam_task090_multiview_manip_v3_50k_train.yaml'),
                   '--run-name', name, '--max-steps', str(target), '--resume-source', str(source),
                   '--resume-source-sha256', source_sha, '--device', 'cuda', '--num-workers', '0']
            (OUT / f'to25k_s{tag}_launch.json').write_text(json.dumps({'argv': cmd, 'source_sha256': source_sha}, indent=2) + '\n')
            state.update(status='starting_training', unit=unit, pid=None, target_step=target, observed_step=start)
            save(state)
            subprocess.run(cmd, check=True)
            while unit_state(unit) in ('active', 'activating', 'deactivating'):
                value = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True).strip()
                state.update(status='training', pid=int(value or 0))
                log = run / 'log.jsonl'
                if log.exists():
                    for line in reversed(log.read_text().splitlines()):
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(row.get('step'), int):
                            state['observed_step'] = row['step']
                            break
                save(state)
                time.sleep(60)
            status = json.loads((run / 'status.json').read_text())
            if status.get('status') != 'segment_complete' or status.get('global_step', status.get('step')) != target:
                raise RuntimeError('training unit ended without completed requested segment: ' + json.dumps(status))
            if not (run / f'checkpoint_step_{tag}.pt').exists() or not (run / 'checkpoint_last.pt').exists():
                raise RuntimeError('completed segment missing fixed/complete checkpoints')
            state['completed'].append({'step': target, 'run': str(run.relative_to(ROOT))})
            state.update(status='segment_complete', observed_step=target, pid=None)
            save(state)
        state.update(status='awaiting_25k_three_physical_tests', pid=None, unit=None)
        save(state)
    except Exception as error:
        state.update(status='stopped_on_error', error=str(error))
        save(state)
        raise


if __name__ == '__main__':
    main()
