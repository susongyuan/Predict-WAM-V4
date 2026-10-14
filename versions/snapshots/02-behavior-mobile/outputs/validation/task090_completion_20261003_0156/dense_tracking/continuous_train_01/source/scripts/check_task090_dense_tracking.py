#!/usr/bin/env python3
"""One sensor-only 832-frame TRAIN dense diagnostic. No v5 poses, GT or deployment."""
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import psutil
from PIL import Image
from harness.dense_rgbd_tracking import DenseSurfaceTracker
from harness.dense_rgbd_registration import CAPTURE, TOL, RESPONSE, MIN_FIT, MIN_HOLD, partition, transform
from harness.object_perception import Camera, camera_from_pose
from harness.robot_self_geometry import RobotSelfGeometry, sha256
from harness.task090_dino_localizer import Task090DinoLocalizer
from harness.task090_rigid_observation_v3 import Task090RigidObservationV3
from scripts import check_task090_continuous_tracking as reader

BASE = ROOT/'outputs/validation/task090_completion_20261003_0156'
OUT = BASE/'dense_tracking'
audit = reader.reused
WALL_LIMIT_S = 1800


def require_cpu_reports(registration, tracking):
    required = {'test_continuous_known_motion_exceeds_absolute_capture',
                'test_current_hold_only_does_not_change_fits', 'test_wrong_origin_cannot_invent_surface',
                'test_gap_six_can_measure_seven_expires', 'test_immutable_initial_arrays_and_ids',
                'test_relabelled_rgbd_payload_never_counts_twice', 'test_task_adapter_closed_binding_is_rejected',
                'test_default_preserves_before_hypotheses_and_result'}
    cases = {row['case'] for row in tracking.get('cases', []) if row.get('passed') is True}
    old_cases = {row['case'] for row in registration.get('cases', []) if row.get('passed') is True}
    if (registration.get('failed') != 0 or registration.get('passed') != 24 or
            registration.get('leakage_checks_passed') is not True or
            not {'qualified_known_rigid', 'qualified_camera_motion'} <= old_cases or
            tracking.get('failed') != 0 or tracking.get('passed', 0) < 25 or
            tracking.get('leakage_checks_passed') is not True or not required <= cases):
        raise RuntimeError('cpu_preflight_failed_no_sensor_payload_processed')


def guard():
    ram, cpu = psutil.virtual_memory().percent, psutil.cpu_percent(interval=None)
    audit.PEAK_RAM, audit.PEAK_CPU = max(audit.PEAK_RAM, ram), max(audit.PEAK_CPU, cpu)
    if ram >= 85 or cpu >= 85:
        raise RuntimeError(f'CPU/RAM85 guard: cpu={cpu}, ram={ram}')
    if audit.PROCESS.memory_info().rss > 2*1024**3 or time.monotonic()-audit.START > WALL_LIMIT_S:
        raise RuntimeError('dense diagnostic exceeded preregistered 2GiB / 1800 seconds')


def appearance(rgb, depth, calibration):
    return Task090RigidObservationV3._appearance(SimpleNamespace(calibration=calibration, target='trash_can'), rgb, depth) > 0


def initial_mask(rgb, depth, camera, features, state, localizer, calibration):
    if float(state[49:51].sum()) < .095:
        raise ValueError('initial_closed_gripper_no_semantic_binding')
    if features is None:
        raise ValueError('missing_exact_initial_cache')
    proposal = localizer.surface(rgb, depth, camera, features, audit.CALIBRATION)
    if proposal is None:
        raise ValueError('no_initial_dino_surface')
    # Reuse v3's existing DINO bounding box + HSV candidate, not its pose tracker.
    x0, y0, x1, y1 = proposal['detection']['bbox_xyxy']
    mask = np.zeros(depth.shape, bool)
    mask[y0:y1, x0:x1] = appearance(rgb, depth, calibration)[y0:y1, x0:x1]
    return mask, proposal['detection']


def install_input_guard():
    """Fail closed on accidental old trajectory reads, including in reused helpers."""
    allowed_jsonl = {reader.SOURCE.resolve(), audit.SOURCE.resolve()}
    denied = []
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        reading = (mode is not None and ('r' in mode or '+' in mode)) or (mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if reading and (('keyframe_v5' in str(path)) or (path.suffix == '.jsonl' and path not in allowed_jsonl)):
            denied.append(str(path))
            raise RuntimeError('forbidden_trajectory_input:' + str(path))
    sys.addaudithook(hook)
    return denied


def run(out, registration_report, tracking_report):
    old, tests = [json.loads(path.read_text()) for path in (registration_report, tracking_report)]
    require_cpu_reports(old, tests)
    for report in (old, tests):
        for path, digest in report['source_sha256'].items():
            if sha256(path) != digest:
                raise RuntimeError('CPU-tested source changed: '+path)
    assert sha256(reader.SOURCE) == reader.FRAME_SHA256
    manifest = json.loads(reader.MANIFEST.read_text())
    assert manifest['sample_count'] == 832 and manifest['frame_manifest_sha256'] == reader.FRAME_SHA256
    assert manifest['sample_fps'] == 30 and manifest['continuous'] and not manifest['missing_frames']
    assert manifest['physical_success_label'] is None and manifest['admitted'] is False
    denied = install_input_guard()
    rows = [json.loads(line) for line in reader.SOURCE.open()]
    expected = [(ep, frame) for ep,(lo,hi) in reader.WINDOWS.items() for frame in range(lo,hi+1)]
    assert [(r['episode'],r['frame_index']) for r in rows] == expected
    # This older SENSOR manifest is used only for the two exact cache identities;
    # no saved measurement, sparse bank, v5 trace or action is passed to the tracker.
    provenance = json.loads((audit.BASE/'provenance.json').read_text())
    assert sha256(audit.SOURCE) == provenance['frame_manifest_sha256']
    starts = {(ep, lo) for ep,(lo,hi) in reader.WINDOWS.items()}
    cache_rows = {(row['episode'],row['frame_index']):row for row in map(json.loads, audit.SOURCE.open())
                  if (row['episode'],row['frame_index']) in starts}
    assert set(cache_rows) == starts
    calibration = json.loads(audit.CALIBRATION.read_text())
    geometry, localizer = RobotSelfGeometry(), Task090DinoLocalizer(audit.PROTOTYPES)
    contract = json.loads((BASE/'camera_contract/nominal_contract.json').read_text())
    assert contract['evaluation_override']['computed_nominal_fx'] == 306
    sources = [Path(__file__), ROOT/'harness/dense_rgbd_tracking.py', ROOT/'harness/dense_rgbd_registration.py',
        ROOT/'scripts/test_dense_rgbd_tracking.py', ROOT/'scripts/test_dense_rgbd_registration.py',
        Path(reader.__file__), Path(audit.__file__), ROOT/'scripts/check_task090_rigid_train_v4.py',
        ROOT/'scripts/build_task090_dino_localizer.py', ROOT/'harness/object_perception.py',
        ROOT/'harness/robot_self_geometry.py', ROOT/'harness/task090_rigid_observation_v3.py',
        ROOT/'harness/task090_dino_localizer.py', ROOT/'harness/task090_perception.py',
        ROOT/'harness/task090_observation.py', ROOT/'harness/task090_rigid_observation_v2.py',
        ROOT/'harness/task090_rigid_observation_v4.py', ROOT/'harness/task1_perception_adapter.py',
        ROOT/'harness/task090_predicates.py', audit.CALIBRATION, audit.PROTOTYPES,
        audit.PROTOTYPES.with_suffix('.json'), reader.SOURCE, reader.MANIFEST, audit.SOURCE,
        audit.BASE/'provenance.json', registration_report, tracking_report, OUT/'plan.json',
        BASE/'camera_contract/nominal_contract.json']
    sources += [Path(p) for p in geometry.source_sha256 if not p.endswith('.usda')]
    cache_stats = {}
    for ep in reader.WINDOWS:
        sources += [reader.CACHE/f'episode_{ep}.{suffix}' for suffix in ('meta.json','frames.npy')]
        sources.append(ROOT/f'data/frame_cache/observation.rgb.zed_link_camera_0/episode_{ep}.meta.json')
        path = reader.CACHE/f'episode_{ep}.feat.npy'; stat = path.stat()
        cache_stats[str(path)] = [stat.st_size, stat.st_mtime_ns]
    hashes = reader.snapshot(out, sources)
    audit.write_json(out/'source_sha256.json', hashes)
    sensor_hashes, summaries, init_cache, independent_checks = {}, [], [], 0
    for ep,(lo,hi) in reader.WINDOWS.items():
        tracker = DenseSurfaceTracker(depth_max=calibration['depth_max_m'], self_filter=geometry,
                                      max_frame_gap=6, max_age_s=6/30.)
        tracker.reset(ep, 'trash_can')
        history, reference = [], None
        trace_dir = out/f'episode_{ep}_fits'; trace_dir.mkdir()
        with (out/f'episode_{ep}_frames.jsonl').open('x') as stream:
            for index, row in enumerate(r for r in rows if r['episode'] == ep):
                guard(); began = time.monotonic(); frame = row['frame_index']
                reader.checked_row(row, ep, frame, index == 0, manifest['dataset_revision'])
                for key in ('rgb', 'depth_m'):
                    digest = sha256(row[key]); assert digest == row[key+'_sha256']
                    sensor_hashes[row[key]] = digest
                with Image.open(row['rgb']) as image:
                    rgb = np.asarray(image).copy()
                depth = np.load(row['depth_m'], allow_pickle=False)
                assert rgb.shape == (720,720,3) and rgb.dtype == np.uint8
                assert depth.shape == (720,720) and depth.dtype == np.float32
                pose = camera_from_pose(np.asarray(row['camera_pose7']))
                if pose is None:
                    raise ValueError('invalid_current_camera_pose')
                camera, state = Camera(*pose,720,720), np.asarray(row['state61'])
                meta = dict(episode=ep, target='trash_can', step=frame, timestamp_s=frame/30.,
                            now_s=frame/30., observation_id=f'{ep}:{frame}')
                previous_pose, previous_step = tracker.accepted_pose, tracker.last_accepted_step
                binding, feature_audit = None, None
                if index == 0:
                    original = cache_rows[ep,frame]
                    assert all(original[k+'_sha256'] == row[k+'_sha256'] for k in ('rgb','depth_m'))
                    features, feature_audit = reader.initial_features(ep, frame, dict(original['cache_identity'], episode=ep))
                    init_cache.append(feature_audit)
                    try:
                        mask, binding = initial_mask(rgb, depth, camera, features, state, localizer, calibration)
                    except ValueError as error:
                        mask = np.zeros(depth.shape, bool); binding = dict(status='unknown', reason=str(error))
                    report, trace = tracker.initialize(rgb, depth, camera, state, mask,
                        binding_id=f'exact-initial-dino:{ep}:{frame}', **meta)
                    reference = tracker.reference
                    if reference is not None:
                        with (out/f'episode_{ep}_reference.npz').open('xb') as file:
                            np.savez_compressed(file, rgb=reference.rgb, depth=reference.depth, state=reference.state,
                                **{f'camera_{k}':v for k,v in reference.camera.items()}, **reference.cloud)
                        audit.write_json(out/f'episode_{ep}_binding.json', dict(reference_id=reference.reference_id,
                            episode=ep, frame=frame, binding=binding, initial_cache=feature_audit, status=report['status'],
                            point_coordinates='immutable actual initial Camera base', raw_pixel_ids=True))
                    del features
                else:
                    report, trace = tracker.update(rgb, depth, camera, state, appearance(rgb, depth, calibration), **meta)
                    assert tracker.reference is reference
                    if report['search_initial_pose'] is not None:
                        np.testing.assert_array_equal(report['search_initial_pose'], previous_pose)
                        assert report['search_origin_step'] == previous_step
                    if not report['pose_updated']:
                        assert tracker.accepted_pose is previous_pose and tracker.last_accepted_step == previous_step
                assert not any(report[k] for k in ('admitted','verified','grasp'))
                assert report['cumulative_error_bound_m'] is None and report['world_pose'] is None
                check = None
                if report['pose_updated']:
                    measurement = report['measurement']; ids = trace['fit_support_indices']
                    original_points = reference.cloud['points'][trace['source_fit_indices']]
                    np.testing.assert_array_equal(original_points, trace['fit_source_points'])
                    error = np.linalg.norm(transform(original_points, tracker.accepted_pose)-trace['fit_target_points'], axis=1)
                    np.testing.assert_allclose(error, trace['fit_distance_m'], atol=1e-12)
                    assert len(ids) >= MIN_FIT and error[ids].max() <= TOL
                    assert measurement['holdout']['inliers_1cm'] >= MIN_HOLD and measurement['holdout']['p95_m'] <= TOL
                    assert min(measurement['observability']['pose_response']) >= RESPONSE
                    assert min(measurement['observability']['normal_response']) >= RESPONSE
                    assert measurement['local_surface_displacement_m'] <= CAPTURE
                    check = dict(original_source_preserved=True, fit_support_count=len(ids), fit_support_max_m=float(error[ids].max()),
                                 actual_true_pose_error_m=None, inherited_error_not_reset=True)
                    independent_checks += 1
                if 'hypotheses' in trace:
                    fit, hold = partition(dict(points=trace['current_points'], holdout=trace['current_holdout']))
                    stencils = trace['current_raw_stencil_indices']
                    assert not np.intersect1d(stencils[fit],stencils[hold]).size
                    keys = ('hypotheses','search_seeds','ambiguity_probe_seeds','fit_scores','fit_source_points','fit_target_points',
                            'fit_distance_m','fit_inliers','fit_support_indices','hold_distance_m','hold_point_distance_m',
                            'source_fit_indices','source_hold_indices','current_uv','current_holdout',
                            'current_raw_stencil_indices','current_candidate_mask','current_robot_mask')
                    with (trace_dir/f'{frame}.npz').open('xb') as file:
                        np.savez_compressed(file, **{key:trace[key] for key in keys})
                report.update(frame=frame, sensor_source={k:row[k] for k in ('rgb','depth_m','rgb_sha256','depth_m_sha256')},
                    camera_pose7=row['camera_pose7'], reference_frame=lo, initial_dino_cache=feature_audit,
                    current_pose_is_measurement=report['pose_updated'], independent_measurement_check=check,
                    right_finger_sum_m=float(state[49:51].sum()), left_finger_sum_m=float(state[24:26].sum()),
                    source_gap_steps=None if index==0 else 1, source_fps=30,
                    action_input_to_tracker=False, frame_wall_seconds=time.monotonic()-began,
                    rss_mib=audit.PROCESS.memory_info().rss/1024**2)
                history.append(report)
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+'\n')
                if (index+1)%100 == 0 or index <= 1:
                    stream.flush()
                    print(json.dumps(dict(episode=ep, frame=frame, observations=index+1, status=report['status'],
                                          reasons=report['reasons'], resources=audit.resources())), flush=True)
        first_loss = next((r for r in history if (r['frame']==lo and r['status']!='reference_initialized') or
                           (r['frame']>lo and r['status']!='surface_motion_measured')), None)
        close = next((r['frame'] for r in history if r['right_finger_sum_m'] < .095), None)
        measured = [r for r in history if r['pose_updated']]
        after = [r for r in measured if close is not None and r['frame'] >= close]
        before = [r for r in measured if close is not None and r['frame'] < close]
        measured_frames = {r['frame'] for r in measured}
        summary = dict(episode=ep, frames_inclusive=[lo,hi], samples=len(history), initial_status=history[0]['status'],
            reference_id=None if reference is None else reference.reference_id, reference_immutable=True,
            first_loss=first_loss, measured_frames=len(measured), feasible_motion_candidates=len(measured),
            first_gripper_below_095m=close, measured_frames_after_first_closure=len(after),
            measured_frames_while_finger_sum_below_095m=sum(r['right_finger_sum_m']<.095 for r in measured),
            measured_across_first_closure=close in measured_frames and close-1 in measured_frames if close is not None else False,
            last_measurement_before_closure=before[-1]['frame'] if before else None,
            first_measurement_after_closure=after[0]['frame'] if after else None,
            initial_continuous_samples=len(history) if first_loss is None else first_loss['frame']-lo,
            status_counts=dict(Counter(r['status'] for r in history)),
            reason_counts=dict(Counter(reason for r in history for reason in r['reasons'])),
            cumulative_error_bound_m=None, part_identity='unknown', instance_scale='unknown',
            physical_success_label=None, candidate_grasp_frames=0, verified=False, admitted=False)
        summaries.append(summary)
        print(json.dumps(dict(episode_complete=ep, measured_frames=len(measured), first_loss=None if first_loss is None else first_loss['frame'],
                             measured_after_first_closure=len(after), resources=audit.resources())), flush=True)
    assert len(sensor_hashes)==1664 and not denied
    assert all(sha256(path)==value for path,value in hashes.items())
    assert all([Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==value for p,value in cache_stats.items())
    audit.write_json(out/'provenance.json', dict(source_sha256=hashes, sensor_sha256=sensor_hashes,
        geometry_source_sha256=geometry.source_sha256, initial_cache=init_cache, initial_cache_calls=2,
        forbidden_trajectory_reads=denied, post_initial_features_used=False, old_v5_poses_read=False,
        action_input_to_tracker=False, output_frame='reference base to current Camera base; not world',
        sensor_sha_checks=1664, independent_accepted_checks=independent_checks))
    summary = dict(status='completed_sensor_only_train_diagnostic_not_admission', samples=832, results=summaries,
        old_registration_cpu_passed=old['passed'], tracking_cpu_passed=tests['passed'],
        feasible_motion_candidates=independent_checks, source_frame_gap=1, max_frame_gap=6,
        sensor_sha_checks=1664, initial_cache_calls=2, gpu_used=False, simulator_executed=False,
        new_dino_extractions=0, videos_decoded=0, policy_deployed=False, thresholds_searched=False,
        old_v5_poses_read=False, automatic_retries=0, cumulative_error_bound_m=None,
        physical_success_label=None, verified=False, admitted=False, resources=audit.resources(),
        limitations=['Predicted ROI is local measured-surface vicinity, not a complete object envelope.',
            'Initial DINO/HSV binding, component/body membership and instance scale are unverified.',
            'Fixed partial initial surface cannot certify identity through occlusion, articulation or loss.',
            'Fit/held consistency and RMS are not cumulative true error bounds or physical grasp labels.',
            'Current Camera base conventions are preserved; no world pose without base odometry.',
            'Only existing distal self-geometry; calibration, masks and surface models retain their limitations.',
            'Replay clock is not a hardware freshness validation; measured closure crossings are not grasps.'])
    audit.write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-report', type=Path, required=True)
    parser.add_argument('--tracking-report', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1); audit.PROCESS.cpu_affinity([min(audit.PROCESS.cpu_affinity())]); psutil.cpu_percent(interval=None)
    audit.START = time.monotonic()
    # A durable one-shot marker prevents rerunning even under a different folder.
    audit.write_json(OUT/'full_window_started.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, cpu_wall_limit_s=WALL_LIMIT_S, cpu_affinity=audit.PROCESS.cpu_affinity(),
        system_cpu_ram_limit_percent=85, rss_limit_gib=2, sample_count=832, no_automatic_retry=True))
    out = OUT/'continuous_train_01'; out.mkdir(exist_ok=False)
    try:
        summary = run(out, args.registration_report.resolve(), args.tracking_report.resolve())
        print(json.dumps(dict(status=summary['status'], samples=summary['samples'],
                             feasible_motion_candidates=summary['feasible_motion_candidates'], resources=summary['resources'])), flush=True)
    except BaseException as error:
        audit.write_json(out/'failure.json', dict(status='failed_no_automatic_retry', error=repr(error),
            traceback=traceback.format_exc(), resources=audit.resources()))
        raise


if __name__ == '__main__':
    main()
