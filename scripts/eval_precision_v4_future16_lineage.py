#!/usr/bin/env python3
"""Run the frozen Future16 queue with strict lineage of its copied initial state.

The performance fork copied the original B4 step-80000 checkpoint unchanged,
then recorded the authorized B23 performance contract in its own manifest.
Only that initialization witness belongs to the original directory manifest.
The two scored checkpoints continue through the unchanged strict loader.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v4_future16 as future
from scripts import eval_precision_v4_future16_final5 as original

queue = original.queue
_read_contract = future.read_contract
_make_plan = original.make_plan
_training = None
_lineage = None
PERFORMANCE_SOURCES = {
    str(ROOT / 'scripts/run_precision_future16_memory.py'),
    str(ROOT / 'scripts/run_precision_future16_prefetch.py'),
    str(ROOT / 'scripts/run_precision_future16_fastnorm.py'),
}


def validate_performance_fork(initial, current):
    """Admit only the documented batch/performance changes, never model changes."""
    transition = current['runtime_performance_profile']['batch_transition']
    queue.e.require(transition['old_batch_size'] == initial['v4_training']['schedule']['batch_size']
                    and transition['new_batch_size'] == current['v4_training']['schedule']['batch_size']
                    and transition['at_updates'] == 5000,
                    'Initial-state lineage does not match the recorded batch transition')
    normalized = copy.deepcopy(current)
    normalized.pop('runtime_performance_profile')
    normalized['v4_training']['schedule']['batch_size'] = transition['old_batch_size']
    for collection in (('source_files',), ('hjepa', 'code_hashes')):
        old, new = initial, normalized
        for key in collection:
            old, new = old[key], new[key]
        added = set(new) - set(old)
        queue.e.require(added <= PERFORMANCE_SOURCES and set(old) <= set(new)
                        and all(new[key] == digest for key, digest in old.items()),
                        'Initial-state lineage changed non-performance frozen sources')
        for key in added:
            del new[key]
    queue.e.require(normalized == initial, 'Initial-state lineage changed a non-performance training contract')


def read_contract(path):
    global _lineage
    path = Path(path).resolve(strict=True)
    if _training is None or path != _training / 'checkpoint_step_080000.pt':
        return _read_contract(path)
    current = queue.e.read_json(_training / 'manifest.json')['contract']
    parent = Path(current['runtime_performance_profile']['batch_transition']['parent_training']).resolve(strict=True)
    queue.e.require(queue.BASE in parent.parents and parent != _training, 'Invalid original initialization directory')
    original_path = parent / path.name
    # Both receipts are recomputed from bytes before borrowing the old manifest.
    copied_receipt, original_receipt = queue.receipt(path), queue.receipt(original_path)
    queue.e.require(all(copied_receipt[key] == original_receipt[key] for key in ('sha256', 'size', 'step')),
                    'Copied initialization is not byte-identical to the original checkpoint')
    initial, receipt = _read_contract(original_path)
    validate_performance_fork(initial, current)
    _lineage = dict(status='passed', copied_checkpoint=copied_receipt, original_checkpoint=original_receipt,
                    original_manifest=str(parent / 'manifest.json'),
                    original_contract_sha256=queue.e.digest(initial),
                    fork_contract_sha256=queue.e.digest(current),
                    reason='Unchanged step-80000 initialization is bound to its original B4 manifest; scored B23 milestones retain their own strict manifest checks.')
    return initial, receipt


def make_plan(args):
    global _training
    _training = args.training.resolve(strict=True)
    plan = _make_plan(args)
    queue.e.require(_lineage is not None, 'Initialization lineage audit did not run')
    plan['training_audit']['initial_checkpoint_lineage'] = _lineage
    plan['frozen_sources'][str(Path(__file__).resolve())] = queue.e.sha256_file(__file__)
    test = ROOT / 'scripts/test_eval_precision_v4_lineage.py'
    plan['frozen_sources'][str(test)] = queue.e.sha256_file(test)
    return plan


def main():
    future.read_contract = read_contract
    original.make_plan = make_plan
    return original.main()


if __name__ == '__main__':
    raise SystemExit(main())
