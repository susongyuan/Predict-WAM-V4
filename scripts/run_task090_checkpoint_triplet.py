#!/usr/bin/env python3
"""Run three fixed-seed Task90 manipulation checks, serially and without video."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from gpu_lock import owner
from run_task090_preposition_matrix import wait_clean, unit_state
from summarize_task090_checkpoints import ROOT, summarize_run

TEMPLATE_DIR = ROOT / 'outputs/validation/task090_takeover_20261001'
TEMPLATE_UNIT = 'task090-wam-s015000-dev10-seed0-preposition-v4'
MARKER = '\n## 10. 当前检查点三次测试\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not 0 < args.step <= 50000 or args.step % 5000:
        parser.error('step must identify an existing fixed 5k checkpoint')
    tag = f'{args.step:06d}'
    checkpoint = ROOT / f'outputs/training/task090_v3_50k_20260930_s{tag}/checkpoint_step_{tag}.pt'
    if not checkpoint.is_file():
        raise RuntimeError('fixed checkpoint missing')
    args.out.mkdir(parents=True, exist_ok=False)
    template = json.loads((TEMPLATE_DIR / 'prepositionv4_15k_seed0_launch.json').read_text())['argv']
    state = dict(status='prepared', checkpoint_step=args.step, seeds=[0, 1, 2], completed=[])

    def save():
        state['updated_local'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
        (args.out / 'state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
        progress = ROOT / 'progress.md'
        text = progress.read_text().split(MARKER)[0]
        progress.write_text(text + MARKER +
            f'- 检查点：{args.step}；状态：{state["status"]}；已结束{len(state["completed"])}/3回合（不是成功次数）。\n'
            f'- 当前run：{state.get("run")}；unit：{state.get("unit")}；父PID：{state.get("pid")}。\n'
            f'- 完整结果：{args.out}/state.json；无录像、预定位辅助、WAM操作；不冒称完整任务通过。\n'
            '- 停止先停triplet队列，再停当前run父unit，核同PID的wam-policy/behavior-run子unit退出后释放自己的GPU锁。\n')
        print(json.dumps(state, ensure_ascii=False), flush=True)

    save()
    try:
        for seed in (0, 1, 2):
            name = f'task090_wam_s{tag}_dev10_seed{seed}_privbase_preposition_v4_rigidlog_v2_firstgrasp'
            unit = f'task090-wam-s{tag}-dev10-seed{seed}-preposition-v4'
            run = ROOT / 'outputs/behavior/closed_loop' / name
            if run.exists() or owner() is not None:
                raise RuntimeError('existing run or occupied GPU; refusing duplicate launch')
            cmd = [item.replace(TEMPLATE_UNIT, unit) for item in template]
            cmd[cmd.index('--run-name') + 1] = name
            cmd[cmd.index('--seed') + 1] = str(seed)
            cmd[cmd.index('--checkpoint') + 1] = str(checkpoint)
            index = cmd.index('--orchestrate') + 1
            orchestrate = json.loads(cmd[index])
            orchestrate['onboard_measurement_log'] = {
                'producer': 'rigid_v2',
                'every': 1,
                'calibration_path': str(ROOT / 'configs/task090_perception_geometry_v2.json'),
                'dino_artifact': str(ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz')}
            orchestrate['manip_snapshot_every'] = 150
            orchestrate['manip_snapshot_sensors'] = True
            cmd[index] = json.dumps(orchestrate)
            cmd = [('StandardOutput=append:' + str(args.out.resolve() / f'seed{seed}_unit.log'))
                   if item.startswith('StandardOutput=append:') else item for item in cmd]
            if '--write-video' in cmd:
                raise RuntimeError('triplet protocol is no-video')
            (args.out / f'seed{seed}_launch.json').write_text(json.dumps({'unit': unit, 'argv': cmd}, indent=2) + '\n')
            state.update(status='starting', run=name, unit=unit, pid=None)
            save()
            subprocess.run(cmd, check=True)
            pid = 0
            for _ in range(100):
                value = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True).strip()
                pid = int(value or 0)
                if pid:
                    break
                time.sleep(.2)
            if not pid:
                raise RuntimeError('no parent PID; inspect launcher before releasing any lock')
            state.update(status='running', pid=pid)
            save()
            wait_clean(unit, pid)
            rows = summarize_run(run)
            subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_task090_checkpoints.py'),
                            '--out', str(ROOT / 'outputs/validation/task090_checkpoint_results')], check=True)
            state['completed'].extend(rows)  # Keep failures and not-reached cases, not only successful operations.
            state.update(status='run_finished', pid=None)
            save()
            if len(rows) != 1 or rows[0].get('status') != 'scored' or rows[0].get('policy_step_alignment') is not True:
                raise RuntimeError('missing/misaligned scored episode; inspect before repeating')
            if rows[0].get('interface_sha256') != state['completed'][0].get('interface_sha256'):
                raise RuntimeError('runtime interface changed inside triplet')
            if rows[0].get('seed') != seed or rows[0].get('checkpoint_step') != args.step:
                raise RuntimeError('checkpoint or effective seed mismatch')
        state.update(status='triplet_complete', run=None, unit=None, pid=None)
        save()
    except Exception as error:
        state.update(status='stopped_on_error', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
