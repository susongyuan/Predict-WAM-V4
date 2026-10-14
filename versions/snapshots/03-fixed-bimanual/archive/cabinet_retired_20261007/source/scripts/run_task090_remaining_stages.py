#!/usr/bin/env python3
"""Continue the authorized Task90 budget to 50k, checking each 5k checkpoint."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from gpu_lock import owner
from run_task090_preposition_matrix import unit_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/validation/task090_remaining_stages_20261002'
TEMPLATE = ROOT / 'outputs/validation/task090_takeover_20261001/to25k_s025000_launch.json'
MARKER = '\n## 12. 当前50k预算内连续队列\n'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    original = json.loads(TEMPLATE.read_text())['argv']
    state = dict(status='prepared', completed=[], budget_end=50000)

    def save(**changes):
        state.update(changes, updated_local=time.strftime('%Y-%m-%d %H:%M:%S %z'))
        (OUT / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        p = ROOT / 'progress.md'
        prefix = p.read_text().split(MARKER)[0]
        p.write_text(prefix + MARKER +
            f'- 状态：{state["status"]}；目标步数：{state.get("target_step")}；最新确认：{state.get("observed_step")}；unit：{state.get("unit")}；PID：{state.get("pid")}。\n'
            f'- 全部状态：{OUT.relative_to(ROOT)}/state.json；测试阶段看其test_state_path内的当前run/PID。\n'
            '- 原50k预算内每5k三测；队列结束不等于任务成功。停止先停task090-remaining-stages.service，再对当前训练主PID发SIGTERM保存，或停止测试父子unit后释放自己的GPU锁。\n')
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def wait_unlock():
        for _ in range(30):
            if owner() is None:
                return
            time.sleep(2)
        raise RuntimeError('GPU occupied; no concurrent or duplicate work')

    save()
    try:
        for target in (35000, 40000, 45000, 50000):
            source_step, tag = target - 5000, f'{target:06d}'
            source_run = ROOT / f'outputs/training/task090_v3_50k_20260930_s{source_step:06d}'
            source_status = json.loads((source_run / 'status.json').read_text())
            if source_status.get('status') != 'segment_complete' or source_status.get('step') != source_step:
                raise RuntimeError('previous training segment is incomplete')
            source = source_run / 'checkpoint_last.pt'
            digest = hashlib.sha256()
            with source.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024**2), b''):
                    digest.update(block)
            run_name = f'task090_v3_50k_20260930_s{tag}'
            run = ROOT / 'outputs/training' / run_name
            if run.exists():
                raise RuntimeError('refusing to overwrite an existing training segment')
            wait_unlock()
            unit = 'wam-train-' + run_name
            cmd = [item.replace('wam-train-task090_v3_50k_20260930_s025000', unit) for item in original]
            for flag, value in (('--run-name', run_name), ('--max-steps', str(target)),
                                ('--resume-source', str(source)), ('--resume-source-sha256', digest.hexdigest())):
                cmd[cmd.index(flag) + 1] = value
            cmd = [('StandardOutput=append:' + str(OUT / f'train_{tag}.log'))
                   if item.startswith('StandardOutput=append:') else item for item in cmd]
            (OUT / f'train_{tag}_launch.json').write_text(json.dumps({'argv': cmd, 'source_sha256': digest.hexdigest()}, indent=2) + '\n')
            save(status='starting_training', target_step=target, observed_step=source_step, unit=unit, pid=None)
            subprocess.run(cmd, check=True)
            while unit_state(unit) in ('active', 'activating', 'deactivating'):
                pid = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True).strip()
                step = state['observed_step']
                log = run / 'log.jsonl'
                if log.exists():
                    for line in reversed(log.read_text().splitlines()):
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(row.get('step'), int):
                            step = row['step']
                            break
                save(status='training', pid=int(pid or 0), observed_step=step)
                time.sleep(60)
            status = json.loads((run / 'status.json').read_text())
            if status.get('status') != 'segment_complete' or status.get('step') != target:
                raise RuntimeError('training ended before requested checkpoint')
            if not (run / f'checkpoint_step_{tag}.pt').is_file():
                raise RuntimeError('fixed checkpoint missing')
            wait_unlock()
            tests = OUT / f'triplet_{tag}'
            save(status='three_physical_checks', observed_step=target, unit=None, pid=None,
                 test_state_path=str(tests / 'state.json'))
            with (OUT / f'triplet_{tag}.log').open('x') as log:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_task090_checkpoint_triplet.py'),
                    '--step', str(target), '--out', str(tests)], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('three-check runner stopped; preserved its state, no next training')
            checks = json.loads((tests / 'state.json').read_text())
            if checks.get('status') != 'triplet_complete' or len(checks.get('completed', [])) != 3:
                raise RuntimeError('three scored records not complete')
            rows = checks['completed']
            state['completed'].append(dict(step=target, trials=3,
                first_grasp_successes=sum(r.get('first_grasp_success') is True for r in rows),
                task_successes=sum(r.get('task_success') is True for r in rows),
                clean_exits=sum(r.get('evaluation_complete') is True for r in rows)))
            save(status='stage_checked')
        save(status='authorized_training_budget_complete', unit=None, pid=None)
    except Exception as error:
        save(status='stopped_on_error', error=str(error))
        raise


if __name__ == '__main__':
    main()
