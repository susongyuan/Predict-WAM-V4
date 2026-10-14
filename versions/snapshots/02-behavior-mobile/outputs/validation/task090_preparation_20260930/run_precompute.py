#!/usr/bin/env python3
"""Main-session Task90 GPU preprocessing with the existing guard and GPU lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/user/Robotic/project2')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from wam_v3_resource_authorization import validate_authorization, resource_environment
from gpu_lock import acquire, release
from behavior_resources import run_guarded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['vtt', 'dino'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Unsafe run id')
    cfg = ROOT/'configs/lilawam_task090_multiview_manip_v3_50k_train.yaml'
    py = ROOT/'.conda-envs/lilawam/bin/python'
    if args.kind == 'vtt':
        command = [str(py), '-u', str(ROOT/'scripts/precompute_skill_vtt.py'), '--batch', '32', '--stride', '2']
    else:
        command = [str(py), '-u', str(ROOT/'scripts/cache_dino_features.py'), '--config', str(cfg),
                   '--out', str(ROOT/'data/dino_cache_task090'), '--batch', '96', '--workers', '8',
                   '--duty', '0.8', '--vram-abort-mib', '15300', '--ram-abort-percent', '85']
    out = ROOT/'outputs/validation/task090_preparation_20260930'/f'{args.kind}_{args.run_id}'
    if not args.execute:
        print(json.dumps({'command': command, 'output': str(out), 'gpu_started': False}, indent=2)); return
    contract = validate_authorization(os.environ.get('GPU_AUTHORIZATION'))
    if os.environ.get('P2_TASK_ID') != '90':
        raise ValueError('Expected explicit P2_TASK_ID=90')
    out.mkdir(parents=True, exist_ok=False)
    (out/'request.json').write_text(json.dumps({'kind': args.kind, 'command': command,
        'config_sha256': hashlib.sha256(cfg.read_bytes()).hexdigest(), 'resource_authorization': contract,
        'started_unix': time.time(), 'training_started': False}, indent=2)+'\n')
    owner = f'task090-{args.kind}-{args.run_id}-{os.getpid()}'
    acquire(owner, max_wait=0)
    result = None
    try:
        env = {**os.environ, **resource_environment(contract), 'P2_TASK_ID': '90', 'CUDA_VISIBLE_DEVICES': '0',
               'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        result = run_guarded(command, env, out, timeout=None, memory_max_bytes=16*1024**3)
    finally:
        # A telemetry/stop error is not permission to release a live child unit's GPU.
        unit = f'behavior-run-{os.getpid()}.service'
        p = subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState','-p','MainPID'],
                           capture_output=True, text=True, timeout=20)
        props = dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
        exited = props.get('LoadState') == 'not-found' or (p.returncode == 0 and props.get('ActiveState') in ('inactive','failed') and props.get('MainPID') == '0')
        if exited:
            released = release(owner)
        else:
            released = False
        (out/'completion.json').write_text(json.dumps({'returncode': result, 'child_unit': unit,
            'child_properties': props, 'child_exit_confirmed': exited, 'gpu_lock_released': released,
            'training_started': False, 'finished_unix': time.time()}, indent=2)+'\n')
    if not exited or not released:
        raise SystemExit('Child exit/lock release unconfirmed; stop and inspect before any further GPU job')
    raise SystemExit(result if result is not None else 2)


if __name__ == '__main__':
    main()
