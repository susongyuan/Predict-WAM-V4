#!/usr/bin/env python3
"""Bounded TRAIN-only 15fps vs 5fps check; frozen v2, no physical admission.

Uses original native RGB/depth decoders and existing exact DINO cache. The common
open-hand start is kept even when it precedes the closure window substantially.
No downloads, services, GPU, controller, training, or frozen-source edits.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['OPENCV_FOR_THREADS_NUM'] = '1'

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

import cv2
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_task090_grasp_evidence import (rgb_frame, video_reference, decode,
    dequantize_depth, codec, DATA, LOWDIM, RGB, DEPTH, POSE, REVISION, sha256)
# The reused CLI module sets thread environment defaults; restore this runner's budget.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from harness.task090_rigid_observation_v2 import Task090RigidObservation
from scripts.build_task090_dino_localizer import cached_frame, CACHE

OUT = ROOT / 'outputs/validation/task090_takeover_20261002_1319/rigid_sampling_15fps'
DENSE = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dense_train_v1'
BASE = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/rigid_train_check_v2b'
FROZEN = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/rigid_v2_runtime_frozen.json'
CALIBRATION = ROOT / 'configs/task090_perception_geometry_v2.json'
PROTOTYPES = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz'
# Inclusive, predeclared windows: original first binding through closure/old loss.
WINDOWS = {18000: (480, 804), 18001: (870, 1188)}
PROCESS = psutil.Process()
START = time.monotonic()
PEAK_RAM_PERCENT = 0.
PEAK_CPU_PERCENT = 0.
COUNTS = Counter()


def write_json(path, value):
    with path.open('x') as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


def stat_identity(path, digest=True):
    stat = path.stat()
    result = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  inode=stat.st_ino, device=stat.st_dev)
    if digest:
        result['sha256'] = sha256(path)
    return result


def guard():
    global PEAK_RAM_PERCENT, PEAK_CPU_PERCENT
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent(interval=None)
    PEAK_RAM_PERCENT = max(PEAK_RAM_PERCENT, ram)
    PEAK_CPU_PERCENT = max(PEAK_CPU_PERCENT, cpu)
    if ram >= 85 or cpu >= 85:
        raise RuntimeError(f'CPU/RAM85 guard: cpu={cpu}, ram={ram}')
    if time.monotonic() - START > 900:
        raise RuntimeError('bounded CPU diagnostic exceeded 900 seconds')
    if PROCESS.memory_info().rss > 2 * 1024**3:
        raise RuntimeError('diagnostic exceeded its 2GiB RSS ceiling')
    assert not torch.cuda.is_initialized()


def resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(pid=os.getpid(), wall_seconds=time.monotonic()-START,
                cpu_seconds=usage.ru_utime+usage.ru_stime, max_rss_mib=usage.ru_maxrss/1024,
                peak_system_ram_percent=PEAK_RAM_PERCENT, peak_system_cpu_percent=PEAK_CPU_PERCENT,
                cpu_affinity=PROCESS.cpu_affinity(), torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(), opencv_threads=cv2.getNumThreads(),
                arrow_cpu_threads=pa.cpu_count(), decoder_threads=1, cuda_initialized=torch.cuda.is_initialized(),
                process_io=PROCESS.io_counters()._asdict(), niceness=PROCESS.nice(),
                process_count=1, python=sys.executable, counts=dict(COUNTS))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    PROCESS.cpu_affinity([min(PROCESS.cpu_affinity())])
    psutil.cpu_percent(interval=None)
    write_json(OUT/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        windows_inclusive=WINDOWS, frames_15fps={ep:list(range(lo,hi+1,2)) for ep,(lo,hi) in WINDOWS.items()},
        frames_5fps={ep:list(range(lo,hi+1,6)) for ep,(lo,hi) in WINDOWS.items()},
        sample_gap_15fps=2, sample_gap_5fps=6, unchanged_max_frame_gap=6,
        fixed_v2_parameters=True, fit_episode=18000, held_out_train_episode=18001,
        fit_performed=False, no_new_threshold_selection=True,
        init_rule='same original early open-gripper frame and exact same material-point initialization',
        prior_first_loss_correction='18001 first loss is 948, not 1128; 1128 follows later rebinding',
        physical_success_label=None, admitted=False, cpu_only=True, maximum_unique_frames=323,
        progress_preregistration='progress.md 2026-10-02 13:43; no progress edits by this worker'))
    try:
        run()
    except BaseException as error:
        write_json(OUT/'error.json', dict(status='stopped_no_admission', error=repr(error),
            traceback=traceback.format_exc(), physical_success_label=None, admitted=False,
            resources=resources()))
        raise


def run():
    guard()
    frozen = json.loads(FROZEN.read_text())['sha256']
    frozen_paths = [ROOT/name for name in ('harness/task090_rigid_observation_v2.py',
                    'harness/task090_observation.py', 'harness/task090_dino_localizer.py')]
    for path in frozen_paths:
        assert sha256(path) == frozen[str(path.relative_to(ROOT))], f'frozen mismatch: {path}'
    info_path, split_path, index_path = DATA/'meta/info.json', DATA/'split_task_090.json', LOWDIM/'index.json'
    info, split, index = [json.loads(p.read_text()) for p in (info_path, split_path, index_path)]
    segments_path = DATA/'skill_segments_task090.json'
    segments = json.loads(segments_path.read_text())
    assert index['revision'] == segments['dataset_revision'] == REVISION
    assert index['split_sha256'] == segments['split_sha256'] == sha256(split_path)
    assert set(WINDOWS) <= set(split['train_episode_indices'])
    assert not set(WINDOWS) & set(split['validation_episode_indices'])
    assert info['fps'] == info['features'][RGB]['info']['video.fps'] == 30
    enc = info['features'][DEPTH]['info']
    assert enc['video.fps'] == 30 and enc['video.pix_fmt'] == 'gray12le' and enc['video.use_log']
    params = {key:enc['video.'+key] for key in ('depth_min','depth_max','shift','use_log','pix_fmt')}
    dense_manifest = json.loads((DENSE/'manifest.json').read_text())
    assert dense_manifest['dataset_revision'] == REVISION
    assert sha256(DENSE/'frames.jsonl') == dense_manifest['frame_manifest_sha256']
    old_frames = {(r['episode'],r['frame_index']):r for r in
                  map(json.loads, (DENSE/'frames.jsonl').open())}
    old_results = {(r['source_frame']['episode'],r['source_frame']['frame']):r for r in
                   map(json.loads, (BASE/'measurements.jsonl').open())}
    sources = frozen_paths + [CALIBRATION, PROTOTYPES, PROTOTYPES.with_suffix('.json'),
        FROZEN, info_path, split_path, index_path, segments_path, Path(__file__), Path(codec.__file__),
        ROOT/'scripts/prepare_task090_grasp_evidence.py', ROOT/'scripts/prepare_task090_depth_samples.py',
        ROOT/'scripts/check_task090_rigid_train_v2.py', ROOT/'scripts/build_task090_dino_localizer.py',
        DENSE/'manifest.json', DENSE/'frames.jsonl', BASE/'measurements.jsonl']
    for name in ('DOWNLOAD_PROVENANCE_task0090_head_depth.json', 'DOWNLOAD_PROVENANCE_task0090_rgb.json'):
        path = DATA/name
        assert json.loads(path.read_text())['revision'] == REVISION
        sources.append(path)
    metadata = {}
    for path in sorted((DATA/'meta/episodes/chunk-090').glob('*.parquet')):
        rows = pq.read_table(path, filters=[('episode_index','in',list(WINDOWS))], use_threads=False).to_pylist()
        if rows:
            sources.append(path)
        for row in rows:
            assert row['episode_index'] not in metadata
            metadata[row['episode_index']] = row
    assert set(metadata) == set(WINDOWS)
    provenance = {str(path):stat_identity(path) for path in sources}
    summaries, comparison, sensor_manifest = [], [], OUT/'frames.jsonl'
    with sensor_manifest.open('x') as sensor_out:
        for ep, (lo, hi) in WINDOWS.items():
            guard()
            meta = metadata[ep]
            frames = list(range(lo,hi+1,2))
            assert meta['task_index'] == 90 and index['episodes'][str(ep)]['split'] == 'train'
            annotation_path = DATA/meta['annotation_path']
            raw = next(s for s in json.loads(annotation_path.read_text())['skill_annotation']
                       if s['skill_description'] == ['pick up from'] and s['object_id'] == [['trash_can','floors']])
            assert raw['frame_duration'][0] == lo and hi < raw['frame_duration'][1]
            provenance[str(annotation_path)] = stat_identity(annotation_path)
            rgb_path, rgb_offset = video_reference(meta, info, RGB)
            depth_path, depth_offset = video_reference(meta, info, DEPTH)
            for path in (rgb_path, depth_path):
                identity = stat_identity(path, digest=False)
                prior = dense_manifest['source_provenance'][str(path)]
                assert identity['bytes'] == prior['bytes'] and identity['mtime_ns'] == prior['mtime_ns']
                identity.update(prior_sha256=prior['sha256'], full_payload_rehashed=False,
                    identity_check='prior video SHA plus current size/mtime; decoded per-frame hashes recorded')
                provenance[str(path)] = identity
            state_path = LOWDIM/f'episode_{ep}_state.npy'
            states = np.load(state_path,mmap_mode='r',allow_pickle=False)
            assert states.shape == (meta['length'],61)
            assert hashlib.sha256(states.tobytes(order='C')).hexdigest() == index['episodes'][str(ep)]['state_sha256']
            provenance[str(state_path)] = stat_identity(state_path)
            cache_meta_path = CACHE/f'episode_{ep}.meta.json'
            source_meta_path = ROOT/f'data/frame_cache/{RGB}/episode_{ep}.meta.json'
            cache_meta = json.loads(cache_meta_path.read_text())
            assert cache_meta['complete'] and cache_meta['episode'] == ep and cache_meta['stride'] == 2
            assert cache_meta['source_frame_meta_sha256'] == sha256(source_meta_path)
            frame_meta = json.loads(source_meta_path.read_text())
            assert DATA/frame_meta['video_path'] == rgb_path and frame_meta['from_timestamp'] == rgb_offset
            indices_path = CACHE/f'episode_{ep}.frames.npy'
            indices = np.load(indices_path,allow_pickle=False)
            assert np.all(np.diff(indices)>0) and set(frames) <= set(indices.tolist())
            for path in (cache_meta_path, source_meta_path, indices_path):
                provenance[str(path)] = stat_identity(path)
            feat_path = CACHE/f'episode_{ep}.feat.npy'
            provenance[str(feat_path)] = stat_identity(feat_path, digest=False)
            provenance[str(feat_path)]['identity_check'] = 'mmap exact indexed rows; SHA of each used float32 feature; not whole-cache payload SHA'
            data_path = DATA/info['data_path'].format(chunk_index=meta['data/chunk_index'],file_index=meta['data/file_index'])
            wanted = sorted(set(frames) | {f-1 for f in frames})
            rows = pq.read_table(data_path,columns=['episode_index','frame_index','timestamp','index','observation.state',POSE,'action'],
                filters=[('episode_index','=',ep),('frame_index','in',wanted)],use_threads=False).to_pylist()
            row_map = {r['frame_index']:r for r in rows}
            assert len(row_map) == len(wanted) and sorted(row_map) == wanted
            prior = dense_manifest['source_provenance'][str(data_path)]
            identity = stat_identity(data_path,digest=False)
            assert identity['bytes'] == prior['bytes'] and identity['mtime_ns'] == prior['mtime_ns']
            identity.update(prior_sha256=prior['sha256'], full_payload_rehashed=False)
            provenance[str(data_path)] = identity
            ep_dir = OUT/f'episode_{ep}'
            ep_dir.mkdir()
            producers = {fps:Task090RigidObservation(CALIBRATION,PROTOTYPES,max_frame_gap=6) for fps in (5,15)}
            histories = {fps:[] for fps in producers}
            bindings = {}
            with (ep_dir/'measurements_5fps.jsonl').open('x') as low, (ep_dir/'measurements_15fps.jsonl').open('x') as high:
                result_files = {5:low,15:high}
                for frame in frames:
                    guard()
                    row = row_map[frame]
                    state = np.asarray(row['observation.state'],np.float32)
                    pose = np.asarray(row[POSE],np.float32)
                    action = np.asarray(row_map[frame-1]['action'],float)
                    assert row['index'] == meta['dataset_from_index']+frame
                    assert np.array_equal(state,states[frame]) and state.shape == (61,) and pose.shape == (7,)
                    assert np.isfinite(state).all() and np.isfinite(pose).all() and action.shape == (23,) and np.isfinite(action).all()
                    assert abs(np.linalg.norm(pose[3:])-1) < .001 and abs(np.linalg.norm(state[45:49])-1) < .001
                    assert abs(row['timestamp']-frame/30) < 1e-4
                    reused = (ep,frame) in old_frames
                    if reused:
                        evidence = dict(old_frames[(ep,frame)])
                        assert evidence['split'] == 'train' and evidence['physical_success_label'] is None
                        assert evidence['rgb_source'] == str(rgb_path) and evidence['depth_source'] == str(depth_path)
                        assert np.array_equal(np.asarray(evidence['state61'],np.float32),state)
                        assert np.array_equal(np.asarray(evidence['camera_pose7'],np.float32),pose)
                        for key in ('rgb','depth_m'):
                            assert sha256(evidence[key]) == evidence[key+'_sha256']
                        with Image.open(evidence['rgb']) as image:
                            rgb = np.asarray(image).copy()
                        depth = np.load(evidence['depth_m'],allow_pickle=False)
                        COUNTS['reused_rgbd_frames'] += 1
                    else:
                        rgb, rgb_pts = rgb_frame(rgb_path,rgb_offset+frame/30,.5/30)
                        q, depth_pts, pix = decode(depth_path,depth_offset+frame/30,.5/30)
                        depth = dequantize_depth(q,**params,output_unit='m').squeeze(-1)
                        reference = np.exp(q.astype(np.float64)/4095*(math.log(params['depth_max']+params['shift'])-
                            math.log(params['depth_min']+params['shift']))+math.log(params['depth_min']+params['shift']))-params['shift']
                        reference = np.clip(reference,params['depth_min'],params['depth_max'])
                        codec_error = float(np.max(np.abs(depth-reference)))
                        assert codec_error < 1e-5 and abs((rgb_pts-rgb_offset)-(depth_pts-depth_offset)) < 1e-6
                        png, npy = ep_dir/f'f{frame:05d}_head_rgb.png', ep_dir/f'f{frame:05d}_head_depth_m.npy'
                        Image.fromarray(rgb).save(png)
                        np.save(npy,depth,allow_pickle=False)
                        evidence = dict(episode=ep,frame_index=frame,split='train',raw_episode_id=meta['raw_episode_id'],
                            train_instance_id=meta['task_instance_id'],dataset_index=row['index'],timestamp_s=row['timestamp'],
                            rgb=str(png),depth_m=str(npy),rgb_sha256=sha256(png),depth_m_sha256=sha256(npy),
                            rgb_source=str(rgb_path),depth_source=str(depth_path),lowdim_source=str(data_path),
                            rgb_stream_episode_offset_s=rgb_offset,depth_stream_episode_offset_s=depth_offset,
                            rgb_requested_pts_s=rgb_offset+frame/30,rgb_decoded_pts_s=rgb_pts,
                            depth_requested_pts_s=depth_offset+frame/30,depth_decoded_pts_s=depth_pts,
                            depth_original_pixel_format=pix,depth_unit='m',codec_reference_max_error_m=codec_error,
                            state61=state.tolist(),camera_pose7=pose.tolist(),physical_success_label=None)
                        COUNTS['newly_decoded_rgbd_frames'] += 1
                    assert rgb.shape == (720,720,3) and depth.shape == (720,720) and depth.dtype == np.float32 and np.isfinite(depth).all()
                    for stream, offset in (('rgb',rgb_offset),('depth',depth_offset)):
                        assert abs(evidence[stream+'_decoded_pts_s']-offset-frame/30) < 1e-6
                    features = cached_frame(ep,frame)
                    evidence.update(dataset_revision=REVISION,reused_existing_rgbd=reused,
                        cache_identity=dict(path=str(feat_path),row=int(np.searchsorted(indices,frame)),frame_index=frame,
                            feature_float32_sha256=hashlib.sha256(features.tobytes()).hexdigest(),source_meta_sha256=sha256(source_meta_path)),
                        applied_action_frame=frame-1,applied_action=action.tolist(),
                        action_semantics='recorded action at frame-1, diagnostic only; never executed',
                        right_finger_sum_m=float(state[49:51].sum()),physical_success_label=None)
                    sensor_out.write(json.dumps(evidence,allow_nan=False)+'\n')
                    sensor_out.flush()
                    COUNTS['unique_frames'] += 1
                    obs = {'robot_r1::robot_r1:zed_link:Camera:0::rgb':rgb,
                           'robot_r1::robot_r1:zed_link:Camera:0::depth_linear':depth,
                           'robot_r1::cam_rel_poses':pose}
                    for fps,producer in producers.items():
                        if fps == 5 and (frame-lo)%6:
                            continue
                        before = 0 if producer.uv is None else len(producer.uv)
                        report = producer.update(obs,state,episode=ep,step=frame,head_features=features,action=action)
                        rigid = report['rigid_tracking']
                        assert not report['admitted'] and not rigid['candidate_grasp']
                        if rigid['status'] == 'rigid_motion_measured':
                            assert rigid['sample_gap_steps'] == 30//fps and rigid['sparse_sampling']
                        if frame == lo:
                            assert float(state[49:51].sum()) >= .095 and rigid['status'] == 'tracking_initialized'
                            bindings[fps] = dict(frame=frame,points=rigid['points'],
                                uv_sha256=hashlib.sha256(producer.uv.tobytes()).hexdigest(),
                                reference_sha256=hashlib.sha256(producer.reference.tobytes()).hexdigest())
                        audit = dict(frame=frame,status=rigid['status'],reason=rigid.get('reason'),
                            points=rigid.get('points'),previous_points=before,right_finger_sum_m=float(state[49:51].sum()),
                            applied_action_22=float(action[22]))
                        histories[fps].append(audit)
                        report['sampling_audit'] = dict(**audit,source_fps=30,sample_fps=fps,
                            physical_success_label=None,feature_sha256=evidence['cache_identity']['feature_float32_sha256'])
                        result_files[fps].write(json.dumps(report,allow_nan=False)+'\n')
                        result_files[fps].flush()
                        if fps == 5:
                            old = old_results[(ep,frame)]['rigid_tracking']
                            assert all(rigid.get(key) == old.get(key) for key in ('status','reason','points','candidate_grasp')), (ep,frame,'baseline mismatch')
                            COUNTS['5fps_baseline_status_point_matches'] += 1
                    if COUNTS['unique_frames']%30 == 0:
                        print(json.dumps(dict(episode=ep,frame=frame,counts=dict(COUNTS),resources=resources())),flush=True)
            assert bindings[5] == bindings[15]
            pair = {}
            for fps, history in histories.items():
                losses = [r for r in history if r['status']=='unknown' and r['previous_points']>0]
                closures = [r for r in history if r['right_finger_sum_m'] < .095]
                measured = [r for r in history if r['status']=='rigid_motion_measured']
                record = dict(episode=ep,role='fit_source_no_refit' if ep==18000 else 'held_out_train',fps=fps,
                    samples=len(history),window_inclusive=[lo,hi],initial_binding=bindings[fps],
                    first_loss=losses[0] if losses else None,all_losses=losses,
                    initializations=[r['frame'] for r in history if r['status']=='tracking_initialized'],
                    first_gripper_below_095m=closures[0] if closures else None,
                    rigid_motion_samples=len(measured),reason_counts=dict(Counter(r['reason'] or r['status'] for r in history)),
                    measured_after_first_closure=sum(r['frame']>=closures[0]['frame'] for r in measured) if closures else 0,
                    physical_success_label=None,candidate_grasp_frames=0,admitted=False)
                summaries.append(record)
                pair[fps] = record
            comparison.append(dict(episode=ep,common_initial_binding_exact=True,
                first_loss_5fps=pair[5]['first_loss'],first_loss_15fps=pair[15]['first_loss'],
                first_loss_delta_control_steps=None if not pair[5]['first_loss'] or not pair[15]['first_loss'] else
                    pair[15]['first_loss']['frame']-pair[5]['first_loss']['frame'],
                primary_endpoint='first loss from original shared open-hand binding, not best later rebind'))
            states._mmap.close()
            print(json.dumps(dict(episode_complete=ep,comparison=comparison[-1])),flush=True)
    guard()
    for path in sources:
        assert sha256(path) == provenance[str(path)]['sha256'], f'source changed while running: {path}'
    for ep in WINDOWS:
        path = CACHE/f'episode_{ep}.feat.npy'
        assert stat_identity(path,digest=False) == {k:v for k,v in provenance[str(path)].items() if k!='identity_check'}
    assert COUNTS['unique_frames'] == 323 and COUNTS['5fps_baseline_status_point_matches'] == 109
    write_json(OUT/'provenance.json',dict(dataset_revision=REVISION,source_provenance=provenance,
        codec_parameters=enc,full_video_cache_payload_rehash=False,
        frame_manifest=str(sensor_manifest),frame_manifest_sha256=sha256(sensor_manifest),
        source_frozen_unchanged=True,resources=resources()))
    summary = dict(status='cpu_sampling_diagnostic_complete_not_admission',dataset_revision=REVISION,
        samples=dict(COUNTS),results=summaries,comparison=comparison,resources=resources(),
        physical_success_label=None,admitted=False,uses_dev=False,uses_test=False,uses_hidden_truth=False,
        simulator_executed=False,policy_executed=False,training_executed=False,
        limitations=['No independent physical labels; no grasp or floor-clearance admission.',
            '15fps is sample_gap=2, not control-rate 30Hz; frozen candidate gate requires gap=1.',
            '2 TRAIN episodes only; not generalization or full-task evidence.',
            'Comparison only tests temporal sampling with unchanged v2 parameters and shared initialization.',
            'Raw video/whole DINO payload not rehashed to limit training I/O contention; identity scope is explicit.'])
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
