#!/usr/bin/env python3
"""Two CPU-only restart counterexamples from the review's earlier inline run.

The review archived the original stdout separately; this saved script was not
rerun when archived. All controller subprocesses and time are mocked by Harness;
only disposable temporary roots receive fixture approvals/checkpoints/locks.
Run with: /usr/bin/python3 -B authorized_restart_counterexamples.py
"""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/home/user/Robotic/project2')
sys.path.insert(0, str(ROOT / 'scripts'))
from test_wam_v3_stages import Harness
import run_wam_v3_stages as S
import wam_v3_resource_authorization as R

contract = R.validate_authorization(str(R.RECEIPT))
hashes = S.entrypoint_source_hashes()


def authorized(h):
    h.setup_continue()
    h.put(h.root / 'approval.json', {'approved': True})
    plan = h.read('plan.json')
    plan['resource'] = contract
    plan['training'].update(approval_sha256=h.m.digest(h.root / 'approval.json'), entrypoint_source_sha256=hashes)
    h.m.entrypoint_source_hashes = lambda: hashes
    h.put(h.out / 'plan.json', plan)


with TemporaryDirectory() as td:
    h = Harness(Path(td))
    authorized(h)
    h.states = ['active', 'deactivating', 'inactive']
    h.m.GPU_LOCK.write_text('wam-train-audit_s005000')
    print('own-stage attach:', h.invoke('--continue-existing', bare=True))
    print('status:', h.read())
    print('subprocess calls:', h.m.subprocess.run.call_count)

with TemporaryDirectory() as td:
    h = Harness(Path(td))
    authorized(h)
    h.make_physical()
    h.put(h.out / 'status.json', {'step_target': 5000, 'returncode': 1, 'phase': 'stopped_infrastructure_failure'})
    h.m.GPU_LOCK.write_text('other-experiment')
    print('first retry blocked:', h.invoke('--continue-existing', bare=True))
    print('retained failure:', h.read())
    h.m.GPU_LOCK.unlink()
    print('second retry:', h.invoke('--continue-existing', bare=True))
    print('final status:', h.read())
    print('additional training stages:', len(h.train_envs))
