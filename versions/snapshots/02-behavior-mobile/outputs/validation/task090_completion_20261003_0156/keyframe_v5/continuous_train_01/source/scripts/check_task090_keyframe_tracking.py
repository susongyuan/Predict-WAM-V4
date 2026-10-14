#!/usr/bin/env python3
"""One isolated v5 replay using the unchanged 832-frame checker implementation.

An explicit function-global adapter injects the v5 factory, output root, source
snapshot extension and summary writer. It does not edit/patch the old checker,
tracker modules, default runtime or sensor reader. No automatic replay/retries.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import psutil
from harness.task090_rigid_observation_v5 import Task090RigidObservationV5, VERSION
from scripts import check_task090_continuous_tracking as continuous

CURRENT = ROOT/'outputs/validation/task090_completion_20261003_0156/keyframe_v5'
IDENTITY = CURRENT.parent/'remaining_observability/identity'
OLD_CONTINUOUS = continuous.OUT_ROOT/'continuous_train_01'
sha = continuous.sha256
write_json = continuous.reused.write_json


def v5_factory(*args, **kwargs):
    """The old loop's constructor slot is explicitly v5, including USD injection."""
    tracker = Task090RigidObservationV5(*args, **kwargs)
    assert type(tracker) is Task090RigidObservationV5
    return tracker


def longest(rows, statuses):
    best, current, start, best_span = 0, 0, None, None
    for row in rows:
        if row['rigid_tracking']['status'] in statuses:
            if current == 0:
                start = row['source_frame']['frame']
            current += 1
            if current > best:
                best, best_span = current, [start, row['source_frame']['frame']]
        else:
            current = 0
    return dict(samples=best, frames_inclusive=best_span)


def complete_summary(out, value):
    """Augment, never replace the original first-binding/first-loss endpoint."""
    value['tracker_version'] = VERSION
    value['explicit_factory'] = __name__ + '.v5_factory -> Task090RigidObservationV5'
    value['reader_and_independent_checker'] = str(Path(continuous.__file__))
    value['local_counterexample_is_not_full_window_evidence'] = True
    failures = []
    for summary in value['results']:
        ep = summary['episode']
        rows = [json.loads(line) for line in (out/f'episode_{ep}_measurements.jsonl').open()]
        assert len(rows) == summary['samples']
        for row in rows:
            diag = row['rigid_tracking']
            assert diag['tracker_version'] == VERSION
            assert not diag['admitted'] and not diag['verified']
            assert diag['appearance_keyframe_committed'] == (diag['status'] in continuous.ACCEPTED)
            if diag['appearance_keyframe_committed']:
                assert diag['appearance_keyframe_step'] == row['source_frame']['frame']
            if diag['status'] == 'unknown':
                failures.append(dict(episode=ep, frame=row['source_frame']['frame'],
                                     reason=diag['reason'], rigid_tracking=diag))
        summary['longest_rigid_measured_run'] = longest(rows, {'rigid_motion_measured'})
        summary['longest_accepted_observation_run_including_initialization_and_recovery'] = longest(rows, set(continuous.ACCEPTED))
        summary['unknown_frames'] = sum(r['rigid_tracking']['status'] == 'unknown' for r in rows)
        summary['appearance_commits'] = sum(r['rigid_tracking']['appearance_keyframe_committed'] for r in rows)
        summary['all_unknown_frame_indices'] = [r['source_frame']['frame'] for r in rows if r['rigid_tracking']['status'] == 'unknown']
        summary['depth_consensus_rejected_points'] = sum(len(r['rigid_tracking'].get('depth_samples', {}).get('rejected_original_ids', [])) for r in rows)
        old = json.loads((OLD_CONTINUOUS/'summary.json').read_text())
        baseline = next(s for s in old['results'] if s['episode'] == ep)
        baseline_failure = baseline['first_loss']['frame']
        at_old_loss = next(r for r in rows if r['source_frame']['frame'] == baseline_failure)
        summary['old_v4_first_loss_comparison'] = dict(
            old_first_loss=baseline_failure, v5_status_at_that_frame=at_old_loss['rigid_tracking']['status'],
            v5_reason_at_that_frame=at_old_loss['rigid_tracking'].get('reason'),
            v5_inliers_at_that_frame=at_old_loss['rigid_tracking'].get('original_anchor_inliers'),
            not_a_replacement_for_v5_initial_track=True)
    with (out/'all_failures.jsonl').open('x') as file:
        for row in failures:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    value['unknown_frames'] = len(failures)
    value['all_failures_path'] = str(out/'all_failures.jsonl')
    value['failure_reasons'] = dict(Counter(row['reason'] for row in failures))
    first = next((s for s in value['results'] if s['first_loss'] is not None), None)
    if first is not None:
        loss = first['first_loss']
        value['next_precise_blocker'] = dict(episode=first['episode'], frame=loss['frame'],
            reason=loss['reason'] or loss['status'],
            action='Inspect this first failed original-ID RGB-D consensus, including per-source counts and depth samples; no threshold search or automatic rerun.')
    else:
        value['next_precise_blocker'] = dict(reason='tracking_does_not_supply_independent_support_or_grasp_labels',
            action='Obtain independent object support/holding evidence; do not wire this CPU result to policy.')


def run(out):
    verified = json.loads((IDENTITY/'verified_identity_evidence.json').read_text())
    assert verified['direct_selected_results_verified'] == verified['short_selected_results'] == 12
    assert not verified['admitted'] and not verified['verified']
    for path, digest in {**verified['source_code_hashes'], **verified['evidence_files']}.items():
        assert sha(path) == digest, path
    assert sha(verified['verifier_script_path']) == verified['verifier_script_sha256']
    protected = [ROOT/'progress.md', *ROOT.glob('configs/task090*.json'),
                 ROOT/'harness/task090_rigid_observation_v4.py', ROOT/'harness/robot_self_geometry.py',
                 Path(continuous.__file__), OLD_CONTINUOUS/'summary.json', OLD_CONTINUOUS/'provenance.json']
    protected_hashes = {str(p): sha(p) for p in protected}
    extra = [ROOT/'harness/task090_rigid_observation_v5.py',
             ROOT/'scripts/test_task090_rigid_observation_v5.py', Path(__file__),
             IDENTITY/'verified_identity_evidence.json', IDENTITY/'short_keyframe_counterexample.json',
             IDENTITY/'diagnose_selected.py', IDENTITY/'verify_local_evidence.py',
             OLD_CONTINUOUS/'summary.json', OLD_CONTINUOUS/'provenance.json']

    def snapshot_v5(destination, sources):
        return continuous.snapshot(destination, list(sources) + extra)

    def write_v5(path, value):
        if Path(path).name == 'summary.json':
            complete_summary(out, value)
        elif Path(path).name == 'provenance.json':
            value.update(explicit_tracker_factory='Task090RigidObservationV5',
                         function_adapter='FunctionType: unchanged continuous.run code, isolated globals',
                         reused_checker_source_sha256=sha(continuous.__file__),
                         v5_source_sha256={str(p): sha(p) for p in extra},
                         protected_source_sha256=protected_hashes)
        return write_json(path, value)

    # Only the isolated new function sees these replacements. Old module globals
    # and its code stay byte-identical. Reader, checks and 832-frame loop are reused.
    namespace = dict(vars(continuous))
    namespace.update(Task090RigidObservationV4=v5_factory, OUT_ROOT=CURRENT, snapshot=snapshot_v5,
                     reused=SimpleNamespace(**{**vars(continuous.reused), 'write_json': write_v5}))
    replay = FunctionType(continuous.run.__code__, namespace, 'run_continuous_v5')
    write_json(out/'adapter.json', dict(
        original_function='scripts.check_task090_continuous_tracking.run',
        original_source_sha256=sha(continuous.__file__), new_factory='Task090RigidObservationV5',
        global_bindings_replaced=['Task090RigidObservationV4 (constructor slot only)', 'OUT_ROOT', 'snapshot', 'reused.write_json'],
        original_reader_residual_sensor_sha_and_bank_checks_unchanged=True,
        protected_source_sha256=protected_hashes, tracker_version=VERSION))
    replay(out)
    assert all(sha(path) == digest for path, digest in protected_hashes.items())
    write_json(out/'protected_unchanged.json', dict(unchanged=True, sha256=protected_hashes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=CURRENT/'continuous_train_01')
    args = parser.parse_args()
    out = args.output.resolve()
    if CURRENT.resolve() not in out.parents:
        raise ValueError('output must be new and inside keyframe_v5')
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    continuous.reused.PROCESS.cpu_affinity([min(continuous.reused.PROCESS.cpu_affinity())])
    psutil.cpu_percent(interval=None)
    continuous.reused.START = time.monotonic()
    write_json(out/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        tracker_version=VERSION, explicit_factory='Task090RigidObservationV5',
        sample_count=832, windows_inclusive=continuous.WINDOWS, source=str(continuous.SOURCE),
        frame_manifest_sha256=continuous.FRAME_SHA256, max_frame_gap=6, source_gap=1,
        rigid_tolerance_m=.01, sensor_margin_m=.01, lk_fb_tolerance_px=1.5, minimum_original_points=6,
        appearance_keyframes_max=1, immutable_original_fallback=True,
        no_automatic_retry=True, thresholds_searched=False, cpu_ram_limit_percent=85,
        initial_cached_dino_only=True, new_dino_extractions=0, videos_decoded=0,
        gpu_used=False, simulator_executed=False, policy_executed=False,
        physical_labels_available=0, admitted=False, verified=False))
    try:
        run(out)
    except BaseException as error:
        write_json(out/'error.json', dict(status='failed_not_admitted', error=repr(error),
            traceback=traceback.format_exc(), admitted=False, verified=False,
            resources=continuous.reused.resources()))
        raise


if __name__ == '__main__':
    main()
