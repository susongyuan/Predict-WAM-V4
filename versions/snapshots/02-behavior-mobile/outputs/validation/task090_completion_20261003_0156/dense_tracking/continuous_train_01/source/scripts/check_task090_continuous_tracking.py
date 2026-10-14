#!/usr/bin/env python3
"""One 832-frame gap1 TRAIN diagnostic with explicitly injected USD self geometry.

Decoder/GPU/GT/policy free. Reuse the v3 resource guard, audit and residual checker
primitives and v4 immutable-bank checks; never rewrite the old 323-frame result.
An existing destination is refused, including partial/failed attempts. No retries.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from collections import Counter
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
from PIL import Image

from harness.robot_self_geometry import RobotSelfGeometry, EXPORT, GEOMETRY_SHA256, sha256
from harness.task090_rigid_observation_v4 import Task090RigidObservationV4
from scripts import check_task090_rigid_train_v3 as reused
from scripts.check_task090_rigid_train_v4 import bank_hashes
from scripts.build_task090_dino_localizer import cached_frame, CACHE

OUT_ROOT = EXPORT/'implementation'
SOURCE = EXPORT.parent/'continuous_train/extracted/frames.jsonl'
MANIFEST = SOURCE.with_name('manifest.json')
FRAME_SHA256 = '072067be165485ba820cba33d59f7c9187c2fb72022582b9ffa449573c82fd87'
WINDOWS = {18000: (480, 891), 18001: (870, 1289)}
ACCEPTED = ('tracking_initialized', 'rigid_motion_measured', 'anchor_recovery_pending')


def initial_features(episode, frame, identity):
    """Exact cache request only; unavailable/mismatched cache becomes unknown."""
    try:
        if identity['episode'] != episode or identity['frame_index'] != frame:
            raise ValueError('initial_cache_identity_mismatch')
        features = cached_frame(episode, frame)
        digest = hashlib.sha256(features.tobytes()).hexdigest()
        if not np.isfinite(features).all() or digest != identity['feature_float32_sha256']:
            raise ValueError('initial_cache_feature_sha256_mismatch')
        return features, dict(status='exact_cached_frame', episode=episode, frame=frame,
                              feature_float32_sha256=digest)
    except (ValueError, KeyError, OSError) as error:
        return None, dict(status='unknown_no_matching_initial_cache', episode=episode,
                          frame=frame, reason=str(error))


def checked_row(row, ep, frame, first, revision):
    if (row['episode'] != ep or row['frame_index'] != frame or row['split'] != 'train' or
            row['dataset_revision'] != revision or row['sample_fps'] != 30 or
            row['frame_gap'] != (None if first else 1) or row['depth_unit'] != 'm' or
            row['applied_action_frame'] != frame-1 or not row['applied_action_verified'] or
            row['action_source_row_index'] != row['dataset_index']-1 or
            any(row[k] is not None for k in ('physical_success_label', 'grasp_relation_label', 'floor_contact_label'))):
        raise ValueError('continuous_frame_contract_mismatch')
    for key, shape in (('state61', (61,)), ('camera_pose7', (7,)), ('applied_action', (23,))):
        value = np.asarray(row[key], float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('continuous_invalid_lowdim:' + key)
    if abs(np.linalg.norm(np.asarray(row['camera_pose7'])[3:])-1) > .001:
        raise ValueError('continuous_invalid_camera_quaternion')


def snapshot(out, paths):
    hashes = {}
    for path in sorted(set(map(Path, paths))):
        target = out/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        with target.open('xb') as file:
            file.write(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return hashes


def run(out):
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['frame_manifest_sha256'] == sha256(SOURCE) == FRAME_SHA256
    assert manifest['sample_count'] == 832 and manifest['sample_fps'] == 30
    assert manifest['continuous'] and manifest['frame_gap'] == 1 and not manifest['missing_frames']
    assert manifest['action_contract']['previous_action_row_offset'] == -1
    assert manifest['physical_success_label'] is None and not manifest['admitted']
    rows = [json.loads(line) for line in SOURCE.open()]
    assert len(rows) == 832
    expected_keys = [(ep, frame) for ep, (lo, hi) in WINDOWS.items() for frame in range(lo, hi+1)]
    assert [(r['episode'], r['frame_index']) for r in rows] == expected_keys
    # Reuse the frozen v2/observation/localizer provenance, not their dataset loop.
    frozen = json.loads(reused.FROZEN.read_text())['sha256']
    for relative in ('harness/task090_rigid_observation_v2.py', 'harness/task090_observation.py',
                     'harness/task090_dino_localizer.py'):
        assert sha256(ROOT/relative) == frozen[relative], relative
    tests = json.loads((OUT_ROOT/'cpu_tests.json').read_text())
    assert tests['status'] == 'PASS' and tests['failures'] == tests['errors'] == 0
    for path, digest in tests['source_sha256'].items():
        assert sha256(path) == digest, 'source changed after CPU tests: ' + path
    geometry = RobotSelfGeometry()
    # Original exact DINO row identity is in the frozen 323-frame sensor manifest.
    prior_provenance = json.loads((reused.BASE/'provenance.json').read_text())
    assert sha256(reused.SOURCE) == prior_provenance['frame_manifest_sha256']
    starts = {(ep, lo) for ep, (lo, hi) in WINDOWS.items()}
    cache_rows = {(row['episode'], row['frame_index']): row
                  for row in map(json.loads, reused.SOURCE.open())
                  if (row['episode'], row['frame_index']) in starts}
    old_root = EXPORT.parent/'rigid_v4'
    # Old counterexample log and 323 summary are evidence only, never replayed.
    old_paths = [old_root/'cpu_tests.log', old_root/'train_check_01/summary.json',
                 old_root/'train_check_01/provenance.json']
    old_hashes = {str(p): sha256(p) for p in old_paths}
    sources = [Path(__file__), ROOT/'scripts/test_robot_self_geometry.py',
               ROOT/'harness/robot_self_geometry.py', ROOT/'harness/task090_rigid_observation_v4.py',
               ROOT/'harness/task090_rigid_observation_v3.py',
               ROOT/'scripts/check_task090_rigid_train_v3.py', ROOT/'scripts/check_task090_rigid_train_v4.py',
               ROOT/'scripts/test_task090_rigid_observation_v4.py',
               ROOT/'scripts/test_task090_rigid_observation_v3.py', ROOT/'scripts/test_task090_rigid_observation_v2.py',
               ROOT/'scripts/build_task090_dino_localizer.py', ROOT/'harness/object_perception.py',
               ROOT/'harness/task090_perception.py', ROOT/'harness/task090_predicates.py',
               ROOT/'harness/task1_perception_adapter.py', reused.CALIBRATION, reused.PROTOTYPES,
               reused.PROTOTYPES.with_suffix('.json'), reused.FROZEN, SOURCE, MANIFEST,
               reused.SOURCE, reused.BASE/'provenance.json', OUT_ROOT/'cpu_tests.json', OUT_ROOT/'cpu_tests.log',
               EXPORT/'geometry.json', EXPORT/'validation.json', EXPORT/'collision_vertices.npz'] + old_paths
    sources += [ROOT/p for p in ('harness/task090_rigid_observation_v2.py',
                                 'harness/task090_observation.py', 'harness/task090_dino_localizer.py')]
    # Snapshot small FK/geometry sources; reference the 131MB USD by verified SHA,
    # not a new copy of the asset or of any sensor/feature cache.
    sources += [Path(p) for p in geometry.source_sha256 if not p.endswith('.usda')]
    cache_stats = {}
    for ep in WINDOWS:
        for suffix in ('meta.json', 'frames.npy'):
            sources.append(CACHE/f'episode_{ep}.{suffix}')
        sources.append(ROOT/f'data/frame_cache/observation.rgb.zed_link_camera_0/episode_{ep}.meta.json')
        path = CACHE/f'episode_{ep}.feat.npy'
        stat = path.stat()
        cache_stats[str(path)] = (stat.st_size, stat.st_mtime_ns)
    hashes = snapshot(out, sources)
    reused.write_json(out/'source_sha256.json', hashes)
    summaries, init_cache, total_checked, sensor_hashes = [], [], 0, 0
    for ep, (lo, hi) in WINDOWS.items():
        producer = Task090RigidObservationV4(reused.CALIBRATION, reused.PROTOTYPES,
                                             max_frame_gap=6, self_filter=geometry)
        history, binding = [], None
        bank_digest = None
        ep_rows = [r for r in rows if r['episode'] == ep]
        with (out/f'episode_{ep}_measurements.jsonl').open('x') as result:
            for index, row in enumerate(ep_rows):
                reused.guard()
                frame = lo + index
                checked_row(row, ep, frame, index == 0, manifest['dataset_revision'])
                for key in ('rgb', 'depth_m'):
                    assert sha256(row[key]) == row[key+'_sha256'], (ep, frame, key)
                    sensor_hashes += 1
                with Image.open(row['rgb']) as image:
                    rgb = np.asarray(image).copy()
                depth = np.load(row['depth_m'], allow_pickle=False)
                assert rgb.shape == (720, 720, 3) and rgb.dtype == np.uint8
                assert depth.shape == (720, 720) and depth.dtype == np.float32 and np.isfinite(depth).all()
                features, feature_audit = None, None
                if index == 0:
                    original = cache_rows.get((ep, frame))
                    identity = {} if original is None else dict(original['cache_identity'], episode=ep)
                    if original is not None:
                        assert all(original[k+'_sha256'] == row[k+'_sha256'] for k in ('rgb', 'depth_m'))
                    features, feature_audit = initial_features(ep, frame, identity)
                    init_cache.append(feature_audit)
                observation = {'robot_r1::robot_r1:zed_link:Camera:0::rgb': rgb,
                               'robot_r1::robot_r1:zed_link:Camera:0::depth_linear': depth,
                               'robot_r1::cam_rel_poses': np.asarray(row['camera_pose7'])}
                before_bank = producer.original_bank
                report = producer.update(observation, np.asarray(row['state61']), episode=ep, step=frame,
                                         head_features=features, action=np.asarray(row['applied_action']))
                diag = report['rigid_tracking']
                assert not report['admitted'] and not report['verified']
                assert not diag['admitted'] and not diag['verified']
                assert producer.total_added == diag['dependent_reference_points_used'] == 0
                bank = producer.original_bank
                if bank is not None:
                    if before_bank is None:
                        bank_digest = bank_hashes(bank)
                        with (out/f'episode_{ep}_original_bank.npz').open('xb') as file:
                            np.savez(file, gray=bank.gray, uv=bank.uv, reference=bank.reference, errors=bank.errors,
                                     origin_floor=producer.origin_floor, origin_margin=producer.origin_margin)
                        binding = dict(frame=frame, points=len(bank.uv), bank_sha256=bank_digest)
                    else:
                        assert bank is before_bank and bank_hashes(bank) == bank_digest
                    assert all(not getattr(bank, key).flags.writeable for key in ('gray', 'uv', 'reference', 'errors'))
                measured = None
                if diag['status'] in ACCEPTED:
                    ids = np.flatnonzero(producer.last_accepted_mask)
                    np.testing.assert_array_equal(producer.anchor_reference, bank.reference[ids])
                    np.testing.assert_array_equal(producer.reference, producer.anchor_reference)
                    np.testing.assert_array_equal(producer.errors, bank.errors[ids])
                    r, t = producer.previous_pose
                    points = reused.sample_points(producer.previous_camera, depth, producer.anchor_uv)
                    residual = np.linalg.norm(producer.anchor_reference@r.T+t-points, axis=1)
                    assert len(residual) >= 6 and residual.max() <= reused.RIGID_TOLERANCE_M
                    assert geometry(points, np.asarray(row['state61'])).all()
                    measured = dict(original_ids=ids.tolist(), uv=producer.anchor_uv.tolist(),
                                    current_points_base_m=points.tolist(), rotation=r.tolist(), translation_m=t.tolist(),
                                    independently_recomputed_max_residual_m=float(residual.max()))
                    total_checked += 1
                    if diag['status'] != 'tracking_initialized':
                        assert diag['pose_source'] == 'original_anchors_only'
                        assert diag['sample_gap_steps'] == 1 and not diag['sparse_sampling']
                        assert diag['original_anchor_pose_error_max_m'] <= .01
                if features is None and index == 0:
                    assert diag['status'] == 'unknown'
                event = reused.audit(ep, frame, diag)
                for key, value in diag.items():
                    if ('anchor' in key or 'self_filter' in key or 'self_rejected' in key or
                            key in ('candidate_grasp', 'candidate_observations', 'gripper_contact_candidate',
                                    'nearest_tracked_point_to_eef_m', 'hand_motion_residual_m')):
                        event[key] = value
                event.update(right_finger_sum_m=float(np.asarray(row['state61'])[49:51].sum()),
                             left_finger_sum_m=float(np.asarray(row['state61'])[24:26].sum()),
                             source_gap_steps=None if index == 0 else 1)
                history.append(event)
                report['source_frame'] = dict(episode=ep, frame=frame, source_fps=30, sample_gap_steps=event['source_gap_steps'],
                    rgb=row['rgb'], depth_m=row['depth_m'], rgb_sha256=row['rgb_sha256'], depth_m_sha256=row['depth_m_sha256'],
                    applied_action_frame=frame-1, action_semantics='recorded previous-frame command; not executed',
                    dino_used=index == 0 and features is not None, dino_initialization=feature_audit,
                    state_action_source='source snapshot frames.jsonl', physical_success_label=None)
                report['independent_measurement_check'] = measured
                report['audit'] = event
                result.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+'\n')
                if (index+1) % 100 == 0:
                    result.flush()
                    print(json.dumps(dict(episode=ep, completed=index+1, frame=frame, status=diag['status'],
                                          resources=reused.resources())), flush=True)
        first_loss = next((r for r in history if
                           (binding is None or r['frame'] > binding['frame']) and
                           r['status'] != 'rigid_motion_measured'), None)
        first_close = next((r['frame'] for r in history if r['right_finger_sum_m'] < .095), None)
        after_close = [r for r in history if first_close is not None and r['frame'] >= first_close]
        bounds = [r['conditional_min_height_gain_m'] for r in history if r['conditional_min_height_gain_m'] is not None]
        measured_rows = [r for r in history if r['status'] == 'rigid_motion_measured']
        summary = dict(episode=ep, frames_inclusive=[lo, hi], samples=len(history), initial_binding=binding,
            first_loss=first_loss, first_gripper_below_095m=first_close,
            measured_frames=len(measured_rows), measured_frames_after_first_closure=sum(r['status']=='rigid_motion_measured' for r in after_close),
            measured_frames_while_finger_sum_below_095m=sum(r['right_finger_sum_m'] < .095 for r in measured_rows),
            initial_track_samples=0 if binding is None else sum(binding['frame'] <= r['frame'] <
                (first_loss['frame'] if first_loss else hi+1) for r in history),
            status_counts=dict(Counter(r['reason'] or r['status'] for r in history)),
            height_bound_range_m=None if not bounds else [min(bounds), max(bounds)],
            height_bound_frames_after_first_closure=sum(r['conditional_min_height_gain_m'] is not None for r in after_close),
            candidate_grasp_frames=sum(bool(r.get('candidate_grasp')) for r in history),
            candidate_observation_frames=sum((r.get('candidate_observations') or 0) > 0 for r in history),
            original_anchors_self_rejected_total=sum(r.get('original_anchors_self_rejected', 0) for r in history),
            initial_anchors_self_rejected_total=sum(r.get('initial_anchors_self_rejected', 0) for r in history),
            physical_success_label=None, admitted=False, verified=False)
        summaries.append(summary)
        print(json.dumps(dict(episode_complete=ep, summary=summary, resources=reused.resources())), flush=True)
    for path, digest in hashes.items():
        assert sha256(path) == digest, 'source changed during run: ' + path
    for path, digest in old_hashes.items():
        assert sha256(path) == digest, 'old v4 evidence changed'
    for path, stat in cache_stats.items():
        now = Path(path).stat()
        assert (now.st_size, now.st_mtime_ns) == stat
    reused.write_json(out/'provenance.json', dict(source_sha256=hashes, geometry_source_sha256=geometry.source_sha256,
        old_v4_evidence_sha256=old_hashes, old_v4_evidence_unchanged=True, exact_initial_cache=init_cache,
        sensor_hashes_verified=sensor_hashes, independent_accepted_frame_checks=total_checked,
        initial_features_only=True, post_initial_head_features=None, dataset_revision=manifest['dataset_revision']))
    summary = dict(status='cpu_continuous_train_diagnostic_complete_not_admission', samples=832, results=summaries,
        tracker_version='original_bank_rgbd_v4', self_filter_version=geometry.version,
        geometry_sha256=GEOMETRY_SHA256, admitted=False, verified=False, physical_labels_available=0,
        primary_endpoint='first loss from the initial binding, never best recovered track',
        source_frame_gap=1, source_fps=30, sensor_hashes_verified=sensor_hashes,
        independent_accepted_frame_checks=total_checked, initial_cache=init_cache,
        thresholds_searched=False, automatic_retries=0, uses_dev=False, uses_test=False, uses_hidden_truth=False,
        simulator_executed=False, gpu_used=False, policy_executed=False, new_dino_extractions=0, videos_decoded=0,
        resources=reused.resources(), limitations=[
            'AABB plus 1cm is conservative and can still over-mask nearby object surfaces.',
            'Only 12 distal robot links; no held object, proximal arms, torso, base or environment geometry.',
            'No PhysX cooked-hull/contact-skin/runtime-override or hardware finger-qpos validation.',
            'Initial DINO binding and repeated-texture identity remain unadmitted.',
            'Projection and EEF are not object measurements; accepted poses use current RGB-D original anchors only.',
            'Two TRAIN windows and schema/FK checks are not independent physical grasp labels or hardware freshness proof.',
            'Candidate and height diagnostics do not admit grasp, clearance, success or policy deployment.'])
    reused.write_json(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT_ROOT/'continuous_train_01')
    args = parser.parse_args()
    out = args.output.resolve()
    if OUT_ROOT.resolve() not in out.parents:
        raise ValueError('output must be a new directory under self_geometry_usd/implementation')
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    reused.PROCESS.cpu_affinity([min(reused.PROCESS.cpu_affinity())])
    psutil.cpu_percent(interval=None)
    reused.START = time.monotonic()
    reused.write_json(out/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        sample_count=832, windows_inclusive=WINDOWS, source=str(SOURCE), source_frame_sha256=FRAME_SHA256,
        max_frame_gap=6, actual_input_frame_gap=1, sensor_margin_m=.01, rigid_tolerance_m=.01,
        lk_fb_tolerance_px=1.5, original_anchor_minimum=6, new_threshold_fitting=False,
        cpu_affinity=reused.PROCESS.cpu_affinity(), cpu_ram_percent_limit=85, cpu_only=True,
        no_automatic_retry=True, initialization_cache_only=True, physical_success_label=None,
        admitted=False, verified=False, geometry_sha256=GEOMETRY_SHA256))
    try:
        run(out)
    except BaseException as error:
        reused.write_json(out/'error.json', dict(status='failed_not_admitted', error=repr(error),
            traceback=traceback.format_exc(), admitted=False, verified=False, resources=reused.resources()))
        raise


if __name__ == '__main__':
    main()
