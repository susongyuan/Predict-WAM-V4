#!/usr/bin/env python3
"""Durable serial GPU bootstrap; never labels missing physical providers complete."""
import os
import argparse
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import atomic_json
from scripts.run_dustbin_pure20 import bus_environment, now
from scripts.run_robotwin_checkpoint_queue import unit_state, read_result
from scripts.gpu_lock import owner

OUT = ROOT / 'outputs/construction/dustbin_full_middle_20261007'
BANK = OUT / 'vtt_bank'
ADAPTER = OUT / 'vtt_adapter_1000'
CACHE_UNIT = 'p2-dustbin-full-cache-20261007'
TRAIN_UNIT = 'p2-dustbin-task-vtt-20261007'
TRAIN = ROOT / 'outputs/runs/put_bottles_dustbin/wam_30k_20261007/train'


def clean(unit):
    state = unit_state(unit)
    done = state.get('MainPID') == '0' and state.get('SubState') in ('dead', 'exited', 'failed')
    if done and not (state.get('Result') == 'success' and state.get('ExecMainStatus') == '0'
                     and state.get('ExecMainCode') == '1' and state.get('LoadState') == 'loaded'):
        raise RuntimeError('unclean unit: ' + str(state))
    return done


def main():
    global ADAPTER, TRAIN_UNIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-after-source-audit', action='store_true')
    args = parser.parse_args()
    bus_environment()
    if args.resume_after_source_audit:
        if not (OUT / 'source_audit_abort.json').is_file() or not clean(CACHE_UNIT):
            raise ValueError('explicit preserved source-audit abort and completed cache required')
        old = unit_state(TRAIN_UNIT)
        if old.get('MainPID') != '0':
            raise ValueError('original adapter still running')
        ADAPTER = OUT / 'vtt_adapter_1000_v2'
        TRAIN_UNIT = 'p2-dustbin-task-vtt-v2-20261007'
        stages = [dict(stage='source_audit_abort', receipt=read_result(OUT / 'source_audit_abort.json'))]
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        stages = []
    def status(stage, **kwargs):
        atomic_json(OUT / 'status.json', dict(stage=stage, updated_at=now(), stages=stages,
                    full_system_physical_ready=False, **kwargs))
    status('waiting_for_train_val_cache')
    deadline = time.monotonic() + 7200
    while not clean(CACHE_UNIT):
        if time.monotonic() > deadline:
            raise TimeoutError('feature cache deadline')
        time.sleep(10)
    if owner() is not None:
        raise RuntimeError('cache exited without releasing GPU lock')
    stages.append(dict(stage='train_val_cache', status='completed', unit=unit_state(CACHE_UNIT)))
    status('building_train_only_vtt_bank')
    if not args.resume_after_source_audit:
        subprocess.run([str(ROOT / '.venvs/wam-fixed/bin/python'), '-B',
            'scripts/build_robotwin_task_vtt_bank.py', '--split-manifest',
            'data/benchmarks/robotwin2/put_bottles_dustbin_50_v1/split_manifest.json',
            '--cache', 'data/dustbin_dino_features', '--task', 'put_bottles_dustbin',
            '--out', str(BANK)], check=True, timeout=600, cwd=ROOT)
    stages.append(dict(stage='TRAIN_vtt_bank', status='completed', result=read_result(BANK / 'result.json')))
    status('training_frozen_base_residual_vtt_adapter')
    subprocess.run(['systemd-run', '--user', '--quiet', '--unit=' + TRAIN_UNIT,
        '--property=WorkingDirectory=' + str(ROOT), '--property=RuntimeMaxSec=3600',
        '--property=TimeoutStopSec=30', '--property=KillMode=control-group',
        '--property=RemainAfterExit=yes',
        '--property=ExecStopPost=/usr/bin/python3 ' + str(ROOT / 'scripts/gpu_lock.py') + ' release --owner ' + TRAIN_UNIT,
        '--setenv=PYTHONUNBUFFERED=1', '/usr/bin/python3', 'scripts/gpu_lock.py', 'run',
        '--owner', TRAIN_UNIT, '--max-wait', '10', '--',
        str(ROOT / '.venvs/wam-fixed/bin/python'), '-B', 'scripts/train_robotwin_task_vtt_adapter.py',
        '--out', str(ADAPTER), '--training-run-dir', str(TRAIN),
        '--checkpoint', str(TRAIN / 'checkpoint_step_025000.pt'), '--bank', str(BANK),
        '--gpu-lock-owner', TRAIN_UNIT, '--steps', '1000'], check=True, timeout=30, cwd=ROOT)
    while not clean(TRAIN_UNIT):
        time.sleep(10)
    result = read_result(ADAPTER / 'result.json')
    stages.append(dict(stage='vtt_adapter_offline_compatibility', result=result,
                       physical_success_verified=False))
    if owner() is not None:
        raise RuntimeError('adapter exited without releasing GPU lock')
    # A rejected offline adapter does not contaminate or prevent pure-WAM testing.
    status('running_original_fixed20_pure_offscreen_repair', vtt_adapter_admitted=result.get('offline_compatibility_admitted'))
    code = subprocess.run([sys.executable, '-B', 'scripts/resume_dustbin_pure20_offscreen.py', '--execute'], cwd=ROOT).returncode
    stages.append(dict(stage='pure_fixed20', exit_code=code))
    status('bootstrap_complete_full_providers_pending',
           pure_fixed20_exit_code=code,
           pending=['sensor-only three-bottle identity/hand binding', 'independent containment/release/stability predicates',
                    'action-risk calibration', 'failure-state recovery policy', 'full physical paired evaluation'])
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        if OUT.exists():
            atomic_json(OUT / 'error.json', dict(error=f'{type(exc).__name__}: {exc}',
                        updated_at=now(), full_system_physical_ready=False))
        raise
