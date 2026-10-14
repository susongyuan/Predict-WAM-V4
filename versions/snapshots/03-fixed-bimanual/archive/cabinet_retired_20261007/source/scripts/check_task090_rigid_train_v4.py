#!/usr/bin/env python3
"""One frozen TRAIN check, reusing v3's exact decoder-free 323-frame evaluator.

This is a process-local adapter, NOT a modification/deployment of the v3 runtime.
It keeps v3's source/data assertions and independent 1cm residual checks, adding
immutable-bank audits and first-loss comparisons. No threshold search, new DINO,
dev/test/GT access or physical grasp labels. A later recovery never replaces the
primary endpoint (first loss from the initial binding).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import psutil
from harness.task090_rigid_observation_v4 import Task090RigidObservationV4
from scripts import check_task090_rigid_train_v3 as reused

OUT_ROOT = ROOT/'outputs/validation/task090_completion_20261003_0156/rigid_v4'
V3_SUMMARY = ROOT/'outputs/validation/task090_completion_20261002_1840/rigid_v3/train_check_03_consistent/summary.json'
V4_SOURCES = [Path(__file__), ROOT/'harness/task090_rigid_observation_v4.py',
              ROOT/'scripts/test_task090_rigid_observation_v4.py',
              ROOT/'scripts/check_task090_rigid_train_v3.py',
              ROOT/'scripts/test_task090_rigid_observation_v2.py', V3_SUMMARY]


def bank_hashes(bank):
    return {name: hashlib.sha256(getattr(bank, name).tobytes()).hexdigest()
            for name in ('gray', 'uv', 'reference', 'errors')}


def run(out):
    old_summary = json.loads(V3_SUMMARY.read_text())
    old = {row['episode']: row for row in old_summary['results']}
    assert {ep: row['first_loss']['frame'] for ep, row in old.items()} == {18000: 616, 18001: 948}
    source_hashes = {str(path): reused.sha(path) for path in V4_SOURCES}
    original_audit, original_write = reused.audit, reused.write_json
    audits = {}

    class AuditedV4(Task090RigidObservationV4):
        def update(self, *args, **kwargs):
            before_bank = self.original_bank
            before = None if before_bank is None else bank_hashes(before_bank)
            report = super().update(*args, **kwargs)
            bank = self.original_bank
            if before_bank is not None:
                assert bank is before_bank and bank_hashes(bank) == before
            ep = kwargs['episode']
            diag = report['rigid_tracking']
            assert diag['tracker_version'] == 'original_bank_rgbd_v4'
            assert self.total_added == 0 and diag['dependent_reference_points_used'] == 0
            assert not report['admitted'] and not report['verified']
            if bank is not None:
                assert all(not getattr(bank, key).flags.writeable for key in ('gray', 'uv', 'reference', 'errors'))
                if before_bank is None:
                    with (out/f'episode_{ep}_original_bank.npz').open('xb') as file:
                        np.savez(file, gray=bank.gray, uv=bank.uv, reference=bank.reference,
                                 errors=bank.errors, origin_floor=self.origin_floor,
                                 origin_margin=self.origin_margin)
                    audits[ep] = dict(bank_sha256=bank_hashes(bank), bank_size=len(bank.uv),
                                      checked_frames=0, maximum_accepted=0, reobserved_count=0)
                audits[ep]['checked_frames'] += 1
                audits[ep]['maximum_accepted'] = max(audits[ep]['maximum_accepted'], int(self.last_accepted_mask.sum()))
                audits[ep]['reobserved_count'] += diag.get('reobserved_original_anchors', 0)
                if diag['status'] in ('tracking_initialized', 'rigid_motion_measured', 'anchor_recovery_pending'):
                    np.testing.assert_array_equal(self.anchor_reference, bank.reference[self.last_accepted_mask])
                    np.testing.assert_array_equal(self.reference, self.anchor_reference)
                    np.testing.assert_array_equal(self.errors, bank.errors[self.last_accepted_mask])
            return report

    def audit(ep, frame, diag):
        event = original_audit(ep, frame, diag)
        for key in ('original_bank_size', 'original_projection_seeds', 'original_anchors_after_lk',
                    'original_anchors_after_hand_separation', 'last_accepted_original_count',
                    'reobserved_original_anchors', 'accepted_original_ids',
                    'assumed_measurement_allowance_m', 'dependent_reference_points_used'):
            event[key] = diag.get(key)
        return event

    def write_json(path, value):
        if path.name == 'provenance.json':
            value['source_sha256'].update(source_hashes)
            value.update(tracker_version='original_bank_rgbd_v4', evaluator_reused=str(Path(reused.__file__)),
                         original_banks=audits, original_frame_manifest_copy='source/frames.jsonl')
        elif path.name == 'summary.json':
            for row in value['results']:
                baseline = old[row['episode']]
                first = row['first_loss']
                row.update(baseline_v3_first_loss=baseline['first_loss'],
                           first_loss_delta_vs_v3_control_steps=None if first is None else first['frame']-baseline['first_loss']['frame'],
                           initial_binding_matches_v3=row['initial_binding'] == baseline['initial_binding'],
                           original_bank_audit=audits[row['episode']])
            value.update(tracker_version='original_bank_rgbd_v4', verified=False,
                         original_only_no_replenishment=True, thresholds_searched=False,
                         target_identity_admitted=False, grasp_admitted=False)
            value['limitations'].append('Projection is only an LK search seed; repeated-texture identity is not certified.')
        original_write(path, value)

    # Reuse without copying its dataset loop or changing a source file. These
    # assignments live only in this standalone checker process and are restored.
    original_class = reused.Task090RigidObservationV3
    reused.Task090RigidObservationV3, reused.audit, reused.write_json = AuditedV4, audit, write_json
    try:
        reused.run(out)
    finally:
        reused.Task090RigidObservationV3, reused.audit, reused.write_json = original_class, original_audit, original_write
    for path, digest in source_hashes.items():
        assert reused.sha(path) == digest, f'v4 check source changed: {path}'
    provenance = json.loads((out/'provenance.json').read_text())
    # Freeze small source inputs and the original frame manifest, never payload
    # caches; exact RGB/depth and DINO row hashes were checked by the reused loop.
    source = out/'source'
    source.mkdir()
    for path in (ROOT/'harness/task090_rigid_observation_v4.py', Path(__file__),
                 ROOT/'scripts/test_task090_rigid_observation_v4.py', reused.SOURCE):
        with (source/path.name).open('xb') as file:
            file.write(path.read_bytes())
        assert reused.sha(source/path.name) == provenance['source_sha256'][str(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT_ROOT/'train_check_01')
    args = parser.parse_args()
    out = args.output.resolve()
    if OUT_ROOT.resolve() not in out.parents:
        raise ValueError('output must be a new directory under rigid_v4')
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    reused.PROCESS.cpu_affinity([min(reused.PROCESS.cpu_affinity())])
    psutil.cpu_percent(interval=None)
    reused.START = time.monotonic()
    reused.write_json(out/'plan.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), status='registered_before_cpu_execution',
        tracker_version='original_bank_rgbd_v4', windows_inclusive=reused.WINDOWS,
        sample_count=323, source=str(reused.SOURCE), cpu_only=True, cpu_limit=1,
        diagnostic_only=True, admitted=False, progress_preregistered='2026-10-03 02:36 CST',
        rigid_tolerance_m=.01, lk_fb_tolerance_px=1.5, max_frame_gap=6,
        original_anchor_minimum=6, no_replenishment=True, no_new_threshold_fitting=True,
        no_stage_labels_as_physical_truth=True, pretrained_weights_unchanged=True,
        source_sha256={str(path): reused.sha(path) for path in V4_SOURCES}))
    try:
        run(out)
    except BaseException as error:
        reused.write_json(out/'error.json', dict(status='failed_not_admitted', error=repr(error),
                          traceback=traceback.format_exc(), resources=reused.resources()))
        raise


if __name__ == '__main__':
    main()
