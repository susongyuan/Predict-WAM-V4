#!/usr/bin/env python3
"""Persistently wait for a completed Task90 checkpoint, then run its three checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from gpu_lock import owner

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    tag = f'{args.step:06d}'
    unit = f'wam-train-task090_v3_50k_20260930_s{tag}'
    run = ROOT / f'outputs/training/task090_v3_50k_20260930_s{tag}'
    receipt = args.out.parent / (args.out.name + '_handoff.json')
    if receipt.exists() or args.out.exists():
        raise RuntimeError('handoff or triplet already exists; inspect before resuming')

    def save(status, **extra):
        receipt.write_text(json.dumps(dict(status=status, step=args.step, training_unit=unit,
            updated_local=time.strftime('%Y-%m-%d %H:%M:%S %z'), **extra), indent=2, ensure_ascii=False) + '\n')
        print(status, extra, flush=True)

    save('waiting_for_training')
    try:
        while True:
            state = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'],
                                   capture_output=True, text=True).stdout.strip()
            if state not in ('active', 'activating', 'deactivating'):
                break
            time.sleep(30)
        status = json.loads((run / 'status.json').read_text())
        if status.get('status') != 'segment_complete' or status.get('step') != args.step:
            raise RuntimeError('training did not finish the requested checkpoint: ' + json.dumps(status))
        if not (run / f'checkpoint_step_{tag}.pt').is_file():
            raise RuntimeError('fixed checkpoint missing')
        for _ in range(30):
            if owner() is None:
                break
            time.sleep(2)
        if owner() is not None:
            raise RuntimeError('GPU still owned; no concurrent testing')
        save('starting_three_checks')
        command = [sys.executable, str(ROOT / 'scripts/run_task090_checkpoint_triplet.py'),
                   '--step', str(args.step), '--out', str(args.out)]
        result = subprocess.run(command)
        if result.returncode:
            raise RuntimeError('triplet stopped; inspect its state and preserved failures')
        state = json.loads((args.out / 'state.json').read_text())
        save('checks_finished', triplet_status=state.get('status'),
             trials=len(state.get('completed', [])),
             clean_exits=sum(x.get('evaluation_complete') is True for x in state.get('completed', [])),
             first_grasp_successes=sum(x.get('first_grasp_success') is True for x in state.get('completed', [])),
             task_successes=sum(x.get('task_success') is True for x in state.get('completed', [])))
    except Exception as error:
        save('stopped_on_error', error=str(error))
        raise


if __name__ == '__main__':
    main()
