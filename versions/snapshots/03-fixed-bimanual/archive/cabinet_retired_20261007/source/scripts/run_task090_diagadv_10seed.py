#!/usr/bin/env python3
"""DIAGNOSTIC batch: 10 seeds of the frozen 50k checkpoint with the rule layer's advancement unblocked.

orchestrate.diagnostic_advance="on_contract_budget" makes every contract end at its budget and lets the
program advance even when no verified evidence exists. It NEVER produces success=True, never satisfies a
postcondition and never updates reliability/memory. This measures whether the POLICY can drive the task to
the evaluator's own goal check; it is NOT a system-capability or verified-success result.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_lock import owner
from run_task090_preposition_matrix import wait_clean, unit_state
from summarize_task090_checkpoints import ROOT, summarize_run

TEMPLATE_DIR = ROOT / 'outputs/validation/task090_takeover_20261001'
TEMPLATE_UNIT = 'task090-wam-s015000-dev10-seed0-preposition-v4'
SEEDS = tuple(range(10))
MAX_STEPS = 6000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    ap.add_argument('--max-steps', type=int, default=MAX_STEPS)
    args = ap.parse_args()
    tag = '050000'
    checkpoint = ROOT / f'outputs/training/task090_v3_50k_20260930_s{tag}/checkpoint_step_{tag}.pt'
    if not checkpoint.is_file():
        raise RuntimeError('fixed checkpoint missing')
    args.out.mkdir(parents=True, exist_ok=True)
    template = json.loads((TEMPLATE_DIR / 'prepositionv4_15k_seed0_launch.json').read_text())['argv']
    state_path = args.out / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state['seeds'] = list(args.seeds)
    else:
        state = dict(status='prepared', checkpoint_step=50000, seeds=list(args.seeds), completed=[],
                     diagnostic_advance='on_contract_budget',
                     label='POLICY DIAGNOSTIC: rule layer cannot block advancement; no verified success claimed')
    state['max_steps'] = args.max_steps

    def save():
        state['updated_local'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
        (args.out / 'state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({k: state.get(k) for k in ('status', 'run', 'unit', 'pid')}, ensure_ascii=False), flush=True)

    save()
    for seed in args.seeds:
        name = f'task090_wam_s{tag}_dev10_seed{seed}_privbase_preposition_v4_diagadv'
        unit = f'task090-wam-s{tag}-dev10-seed{seed}-diagadv'
        run = ROOT / 'outputs/behavior/closed_loop' / name
        if run.exists() or owner() is not None:
            raise RuntimeError('existing run or occupied GPU; refusing duplicate launch')
        cmd = [item.replace(TEMPLATE_UNIT, unit) for item in template]
        cmd[cmd.index('--run-name') + 1] = name
        cmd[cmd.index('--seed') + 1] = str(seed)
        cmd[cmd.index('--checkpoint') + 1] = str(checkpoint)
        cmd[cmd.index('--max-steps') + 1] = str(args.max_steps)
        cmd[cmd.index('--timeout') + 1] = '3600'
        cmd += ['--run-purpose', 'diagnostic_only',
                '--write-video', '--video-view', 'viewer', '--video-width', '1920', '--video-height', '1080']
        index = cmd.index('--orchestrate') + 1
        orch = json.loads(cmd[index])
        orch['diagnostic_advance'] = 'on_contract_budget'
        orch['_diagnostic_advance_note'] = ('Rule-layer advancement unblocked for a POLICY diagnostic only. '
                                            'success stays None; nothing is verified; not system capability.')
        orch['onboard_measurement_log'] = {
            'producer': 'rigid_v2', 'every': 1,
            'calibration_path': str(ROOT / 'configs/task090_perception_geometry_v2.json'),
            'dino_artifact': str(ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz')}
        orch['manip_snapshot_every'] = 150
        orch['manip_snapshot_sensors'] = True
        cmd[index] = json.dumps(orch)
        cmd = [('StandardOutput=append:' + str(args.out.resolve() / f'seed{seed}_unit.log'))
               if item.startswith('StandardOutput=append:') else item for item in cmd]
        (args.out / f'seed{seed}_launch.json').write_text(json.dumps({'unit': unit, 'argv': cmd}, indent=2) + '\n')
        state.update(status='starting', run=name, unit=unit, pid=None)
        save()
        subprocess.run(cmd, check=True)
        pid = 0
        for _ in range(100):
            value = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'],
                                            text=True).strip()
            pid = int(value or 0)
            if pid:
                break
            time.sleep(.2)
        if not pid:
            raise RuntimeError('no parent PID; inspect launcher before releasing any lock')
        state.update(status='running', pid=pid)
        save()
        try:
            wait_clean(unit, pid)
            rows = summarize_run(run)
            state['completed'].append({'seed': seed, 'run': name, 'rows': rows})
            state.update(status='running' if seed != args.seeds[-1] else 'batch_complete')
        except Exception as error:
            state['completed'].append({'seed': seed, 'run': name, 'error': type(error).__name__ + ':' + str(error)})
            state.update(status='aborted_on_error')
            save()
            raise
        save()
    state.update(status='batch_complete', run=None, unit=None, pid=None)
    save()


if __name__ == '__main__':
    main()
