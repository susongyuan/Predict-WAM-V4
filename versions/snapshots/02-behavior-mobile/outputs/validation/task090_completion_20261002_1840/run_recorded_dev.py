#!/usr/bin/env python3
"""One selected development recording, never a queue or a success retry loop."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT / 'outputs/validation/task090_completion_20261002_1840/recorded_dev02'
NAME = 'task090_s035k_dev10_seed1_privbase_preposition_onboard_hd_20261002'
UNIT = 'task090-onboard-hd-dev02-20261002'
RUN = ROOT / 'outputs/behavior/closed_loop' / NAME
SOURCE = ROOT / 'outputs/validation/task090_takeover_20261002_1319/recorded_revalidation_35k_seed1/launch.json'
PYTHON = ROOT / '.conda-envs/lilawam/bin/python'


def state(unit):
    output = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID'],
                            text=True, capture_output=True, check=True).stdout
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def main():
    assert not RUN.exists(), 'Never overwrite a prior run'
    assert not (OUT / 'launch.json').exists(), 'This one-shot launch was already attempted'
    for unit in ('task090-remaining-stages.service', UNIT + '.service'):
        assert state(unit).get('MainPID', '0') == '0', 'A queue or this run is already active'
    lock = Path('/tmp/project2_gpu.lock')
    assert not lock.exists() and not lock.is_symlink(), 'GPU lock is already owned'
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    active = [line for line in processes.splitlines()
              if any(marker in line for marker in ('scripts/train_wam_task_v3.py', 'harness.policy_server',
                                                   'scripts/behavior_closed_loop.py', 'omnigibson/eval/eval.py'))]
    assert not active, 'Existing model or simulator process: ' + repr(active)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    used, total = map(int, gpu.split(','))
    assert used < 4000 and used + 8192 + 2100 < 15300, 'Insufficient measured GPU headroom'
    old = json.loads(SOURCE.read_text())
    argv = [part.replace(old['unit'], UNIT) for part in old['argv']]
    argv.insert(1, '--wait')
    cfg = json.loads(argv[argv.index('--orchestrate') + 1])
    assert cfg['postconditions'] == 'task090_observed' and cfg['candidate_outcome_policy'] == 'hold'
    cfg['onboard_recording'] = {'enabled': True, 'compressed': True}
    # Avoid redundant compressed snapshots: the received-observation archive contains all views.
    cfg['manip_snapshot_sensors'] = False
    argv[argv.index('--orchestrate') + 1] = json.dumps(cfg)
    argv[argv.index('--run-name') + 1] = NAME
    argv[argv.index('--max-steps') + 1] = '600'
    argv = [f'StandardOutput=append:{OUT}/unit.log' if x.startswith('StandardOutput=append:') else x for x in argv]
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {'registered_at': datetime.datetime.now().astimezone().isoformat(),
               'status': 'dispatching', 'unit': UNIT, 'run_dir': str(RUN), 'source_launch': str(SOURCE),
               'argv': argv, 'orchestrate': cfg, 'gpu_mib_before': {'used': used, 'total': total},
               'scope': 'selected 35k/seed1 development evidence recording, 600-step first-grasp window; not full-task evaluation',
               'new_tracker_deployed': False, 'completion_gate_admitted': False,
               'protocol_changes': ['1080p viewer video', 'all-phase policy-received sensor archive',
                                    '600 rather than 2323 episode steps; manipulation budget unchanged'],
               'stop': f'stop {UNIT}.service, then its wam-policy-PID and behavior-run-PID children; release only this owner after all exited'}
    (OUT / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'unit': UNIT, 'run_dir': str(RUN), 'status': 'dispatching'}), flush=True)
    result = subprocess.run(argv)
    # Parent writes its PID-named child units in the runtime log. Do not release a
    # lock solely because the parent exited: inspect remaining policy/sim units.
    units = subprocess.check_output(['systemctl', '--user', 'list-units', '--state=active,activating,deactivating',
                                     '--no-legend', '--no-pager', 'wam-policy-*', 'behavior-run-*'], text=True)
    receipt.update(status='parent_command_ended', returncode=result.returncode,
                   ended_at=datetime.datetime.now().astimezone().isoformat(), remaining_children=units)
    if not units.strip():
        release = subprocess.run([str(PYTHON), str(ROOT / 'scripts/gpu_lock.py'), 'release', '--owner', UNIT],
                                 text=True, capture_output=True)
        receipt['lock_release'] = {'returncode': release.returncode, 'stdout': release.stdout, 'stderr': release.stderr}
    else:
        receipt['lock_release'] = {'status': 'not_attempted_children_still_active'}
    (OUT / 'termination.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'returncode': result.returncode,
                      'remaining_children': units}), flush=True)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
