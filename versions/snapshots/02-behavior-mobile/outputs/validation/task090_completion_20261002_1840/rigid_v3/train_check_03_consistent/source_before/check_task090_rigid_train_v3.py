#!/usr/bin/env python3
"""TRAIN-only v3 continuity check; reuse 323 decoded frames and exact DINO rows.

No video decoding, new feature/sensor cache, simulator, GPU, training or test data.
The primary endpoint is the FIRST unknown after the initial binding, never the
best recovered/later track. Stage annotations are not read as physical labels.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import cv2
import numpy as np
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.task090_rigid_observation_v3 import Task090RigidObservationV3
from scripts.build_task090_dino_localizer import cached_frame, CACHE

BASE = ROOT / 'outputs/validation/task090_takeover_20261002_1319/rigid_sampling_15fps'
SOURCE = BASE / 'frames.jsonl'
OUT_ROOT = ROOT / 'outputs/validation/task090_completion_20261002_1840/rigid_v3'
CALIBRATION = ROOT / 'configs/task090_perception_geometry_v2.json'
PROTOTYPES = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz'
FROZEN = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/rigid_v2_runtime_frozen.json'
WINDOWS = {18000: (480, 804), 18001: (870, 1188)}
PROCESS = psutil.Process()
START = time.monotonic()
PEAK_RAM = PEAK_CPU = 0.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def guard():
    global PEAK_RAM, PEAK_CPU
    ram, cpu = psutil.virtual_memory().percent, psutil.cpu_percent(interval=None)
    PEAK_RAM, PEAK_CPU = max(PEAK_RAM, ram), max(PEAK_CPU, cpu)
    if ram >= 85 or cpu >= 85:
        raise RuntimeError(f'CPU/RAM85 guard: cpu={cpu}, ram={ram}')
    if PROCESS.memory_info().rss > 2 * 1024**3 or time.monotonic()-START > 900:
        raise RuntimeError('bounded CPU diagnostic exceeded 2GiB or 900 seconds')


def resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(pid=os.getpid(), wall_seconds=time.monotonic()-START,
                cpu_seconds=usage.ru_utime+usage.ru_stime, max_rss_mib=usage.ru_maxrss/1024,
                peak_system_ram_percent=PEAK_RAM, peak_system_cpu_percent=PEAK_CPU,
                cpu_affinity=PROCESS.cpu_affinity(), opencv_threads=cv2.getNumThreads(),
                python=sys.executable, process_count=1, gpu_used=False,
                niceness=PROCESS.nice(), process_io=PROCESS.io_counters()._asdict())


def audit(ep, frame, diag):
    return dict(episode=ep, frame=frame, status=diag['status'], reason=diag.get('reason'),
                points=diag.get('points'), overlap_inliers=diag.get('overlap_inliers'),
                original_anchor_inliers=diag.get('original_anchor_inliers'),
                original_anchor_fit_rms_m=diag.get('original_anchor_fit_rms_m'),
                original_anchor_pose_error_max_m=diag.get('original_anchor_pose_error_max_m'),
                original_anchor_pose_inliers_at_1cm=diag.get('original_anchor_pose_inliers_at_1cm'),
                original_anchors_after_lk_depth_appearance=diag.get('original_anchors_after_lk_depth_appearance'),
                original_anchors_retained=diag.get('original_anchors_retained'),
                total_replenished_points=diag.get('total_replenished_points', 0),
                replenished_points=diag.get('replenished_points', 0),
                fit_rms_m=diag.get('fit_rms_m'),
                cumulative_reference_error_m=diag.get('cumulative_reference_error_m'),
                conditional_min_height_gain_m=diag.get('conditional_min_height_gain_m'))


def run(out):
    source_sha = sha(SOURCE)
    old_provenance = json.loads((BASE/'provenance.json').read_text())
    assert old_provenance['frame_manifest_sha256'] == source_sha
    old_summary = json.loads((BASE/'summary.json').read_text())
    frozen = json.loads(FROZEN.read_text())['sha256']
    for relative in ('harness/task090_rigid_observation_v2.py', 'harness/task090_observation.py',
                     'harness/task090_dino_localizer.py'):
        assert sha(ROOT/relative) == frozen[relative], f'frozen source changed: {relative}'
    rows = [json.loads(line) for line in SOURCE.open()]
    assert len(rows) == 323
    by_ep = {ep: [r for r in rows if r['episode'] == ep] for ep in WINDOWS}
    for ep, (lo, hi) in WINDOWS.items():
        assert [r['frame_index'] for r in by_ep[ep]] == list(range(lo, hi+1, 2))
    source_paths = [Path(__file__), ROOT/'harness/task090_rigid_observation_v3.py',
                    ROOT/'scripts/test_task090_rigid_observation_v3.py', CALIBRATION,
                    PROTOTYPES, PROTOTYPES.with_suffix('.json'), SOURCE, BASE/'summary.json',
                    FROZEN, ROOT/'scripts/build_task090_dino_localizer.py']
    source_hashes = {str(path): sha(path) for path in source_paths}
    for relative in frozen:
        path = ROOT/relative
        if path.is_file():
            source_hashes[str(path)] = sha(path)
    summaries = []
    for ep, ep_rows in by_ep.items():
        producer = Task090RigidObservationV3(CALIBRATION, PROTOTYPES, max_frame_gap=6)
        feature_path = CACHE/f'episode_{ep}.feat.npy'
        feature_stat = (feature_path.stat().st_size, feature_path.stat().st_mtime_ns)
        old_rows = [json.loads(line) for line in (BASE/f'episode_{ep}/measurements_15fps.jsonl').open()]
        old_by_frame = {r['sampling_audit']['frame']: r for r in old_rows}
        history, binding = [], None
        bounds = []
        with (out/f'episode_{ep}_measurements.jsonl').open('x') as result:
            for row in ep_rows:
                guard()
                frame = row['frame_index']
                assert row['split'] == 'train' and row['physical_success_label'] is None
                assert row['applied_action_frame'] == frame-1
                assert row['cache_identity']['frame_index'] == frame
                assert row['dataset_revision'] == old_summary['dataset_revision']
                for key in ('rgb', 'depth_m'):
                    assert sha(row[key]) == row[key+'_sha256']
                with Image.open(row['rgb']) as image:
                    rgb = np.asarray(image).copy()
                depth = np.load(row['depth_m'], allow_pickle=False)
                features = cached_frame(ep, frame)
                assert hashlib.sha256(features.tobytes()).hexdigest() == row['cache_identity']['feature_float32_sha256']
                obs = {'robot_r1::robot_r1:zed_link:Camera:0::rgb': rgb,
                       'robot_r1::robot_r1:zed_link:Camera:0::depth_linear': depth,
                       'robot_r1::cam_rel_poses': np.asarray(row['camera_pose7'])}
                report = producer.update(obs, np.asarray(row['state61']), episode=ep, step=frame,
                                         head_features=features, action=np.asarray(row['applied_action']))
                diag = report['rigid_tracking']
                assert report['admitted'] is False and report['verified'] is False
                assert diag['candidate_grasp'] is False and diag['verified'] is False
                if diag['status'] == 'rigid_motion_measured':
                    assert diag['sample_gap_steps'] == 2 and diag['sparse_sampling']
                    assert diag['original_anchor_inliers'] >= 6 and diag['overlap_inliers'] >= 6
                event = audit(ep, frame, diag)
                event.update(right_finger_sum_m=row['right_finger_sum_m'],
                             baseline_v2_status=old_by_frame[frame]['rigid_tracking']['status'])
                history.append(event)
                if len(history) == 1:
                    assert diag['status'] == 'tracking_initialized', report
                    binding = dict(frame=frame, points=diag['points'],
                                   uv_sha256=hashlib.sha256(producer.uv.tobytes()).hexdigest(),
                                   reference_sha256=hashlib.sha256(producer.reference.tobytes()).hexdigest())
                if 'conditional_min_height_gain_m' in diag:
                    bounds.append(diag['conditional_min_height_gain_m'])
                report['source_frame'] = dict(episode=ep, frame=frame, physical_success_label=None,
                                             source_fps=30, sample_fps=15, sample_gap_steps=2,
                                             rgb=row['rgb'], depth=row['depth_m'],
                                             applied_action_frame=frame-1,
                                             action_semantics='recorded previous-frame action; not executed',
                                             feature_float32_sha256=row['cache_identity']['feature_float32_sha256'])
                report['audit'] = event
                result.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+'\n')
        baseline = next(r for r in old_summary['results'] if r['episode'] == ep and r['fps'] == 15)
        failed = [r for r in history[1:] if r['status'] != 'rigid_motion_measured']
        first_loss = failed[0] if failed else None
        initial = history if first_loss is None else [r for r in history if r['frame'] < first_loss['frame']]
        old_loss = baseline['first_loss']
        summary = dict(episode=ep, role='TRAIN fit-source, no new fitting' if ep == 18000 else 'held-out TRAIN episode',
                       samples=len(history), initial_binding=binding,
                       initial_binding_matches_v2=(binding == baseline['initial_binding']),
                       first_loss=first_loss, baseline_v2_first_loss=old_loss,
                       first_loss_delta_control_steps=None if first_loss is None else first_loss['frame']-old_loss['frame'],
                       last_initial_track_observation=initial[-1], initial_track_samples=len(initial),
                       replenished_before_first_loss=sum(r['replenished_points'] for r in initial),
                       total_replenished_points=producer.total_added,
                       cumulative_reference_error_high_water_m=producer.error_high_water,
                       maximum_original_anchor_inliers=max((r['original_anchor_inliers'] or 0) for r in history),
                       min_original_anchor_inliers_measured=min(r['original_anchor_inliers'] for r in history
                           if r['status'] == 'rigid_motion_measured') if len(initial)>1 else None,
                       status_counts=dict(Counter(r['reason'] or r['status'] for r in history)),
                       first_gripper_below_095m=next((r['frame'] for r in history if r['right_finger_sum_m'] < .095), None),
                       bound_range=None if not bounds else [min(bounds), max(bounds)],
                       candidate_grasp_frames=0, physical_success_label=None, admitted=False)
        summaries.append(summary)
        assert feature_stat == (feature_path.stat().st_size, feature_path.stat().st_mtime_ns)
        print(json.dumps(dict(episode_complete=ep, summary=summary, resources=resources()), ensure_ascii=False), flush=True)
    for name, value in source_hashes.items():
        assert sha(name) == value, f'source changed during run: {name}'
    write_json(out/'provenance.json', dict(source_sha256=source_hashes, original_index_reused=True,
                                         exact_dino_rows_verified=323, sensor_hashes_verified=646,
                                         source_frozen_unchanged=True, source_revision=old_summary['dataset_revision']))
    summary = dict(status='cpu_train_diagnostic_complete_not_admission', results=summaries,
                   samples=323, candidate_grasp_frames=0, admitted=False, physical_labels_available=0,
                   uses_dev=False, uses_test=False, uses_hidden_truth=False, simulator_executed=False,
                   policy_executed=False, training_executed=False, videos_decoded=0, new_cache_files=0,
                   primary_endpoint='first loss from original binding, not best recovery or later track',
                   sampling='15fps = 2 control steps, never 30Hz evidence', resources=resources(),
                   limitations=['Initial DINO/appearance binding remains unadmitted and can be wrong.',
                                'Original anchor checks are geometric, not independent semantic ground truth.',
                                'Error margins are conservative engineering assumptions, not sensor certification.',
                                'Only two TRAIN episodes; no physical grasp labels, task success or generalization claims.',
                                'No independent test-time admission or success-gate integration.'])
    write_json(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT_ROOT/'train_check_01')
    args = parser.parse_args()
    out = args.output.resolve()
    if OUT_ROOT.resolve() not in out.parents:
        raise ValueError('output must be a new directory under rigid_v3')
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    PROCESS.cpu_affinity([min(PROCESS.cpu_affinity())])
    psutil.cpu_percent(interval=None)
    write_json(out/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                   status='registered_before_cpu_execution', windows_inclusive=WINDOWS,
                                   sample_count=323, source=str(SOURCE), cpu_only=True,
                                   diagnostic_only=True, admitted=False, progress_preregistered='2026-10-02 19:00',
                                   rigid_tolerance_m=.01, lk_fb_tolerance_px=1.5, max_frame_gap=6,
                                   original_anchor_minimum=6, overlap_minimum=6,
                                   no_new_threshold_fitting=True, no_stage_labels_as_physical_truth=True))
    try:
        run(out)
    except BaseException as error:
        write_json(out/'error.json', dict(status='failed_not_admitted', error=repr(error),
                                         traceback=traceback.format_exc(), resources=resources()))
        raise


if __name__ == '__main__':
    main()
