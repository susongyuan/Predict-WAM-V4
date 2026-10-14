#!/usr/bin/env python3
"""One fixed CPU TRAIN18001 724..1289 diagnostic; no retries or physical labels.

Only this script/its small driver and early_baseline_run02 outputs may be written.
Reuse the exact continuous reader/codec, cached-spatial FrameList/model factory,
and v5 tracker in separate existing Python environments.
The CPU bf16 extraction is separately attributed, not a claim of GPU bit identity.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['P2_TASK_ID'] = '90'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

from collections import Counter
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT/'outputs/validation/task090_completion_20261003_0156'
RUN_ROOT = CURRENT/'early_baseline_run02'
OUT = None  # Explicit, new per-stage attempt directory; never the old failure.
REVISION = '4f50b44796641a4d526a19d9aeadc8aa51e2f2c2'
RGB = 'observation.rgb.zed_link_camera_0'
HANDOFF_VERSION = 'early_baseline_split_v1'
EP, LO, BOUNDARY, HI = 18001, 724, 870, 1289
START = time.monotonic()
COUNTS = Counter()
STAGE = 'entry'
SOURCES = {}
# Display-only inspection regions fixed from the existing native RGB BEFORE depth
# or DINO are read. These are NOT annotations supplied to any detector or tracker.
INSPECTION_ROIS = {
    'visible_front_surface': [218, 480, 285, 534],
    'front_lower_edge_and_immediate_background': [220, 538, 300, 575],
    'adjacent_ground_front': [220, 580, 320, 620],
    'adjacent_ground_right': [315, 500, 341, 547],
}


def write_json(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def remember(path):
    path = Path(path).resolve()
    stat = path.stat()
    value = dict(sha256=sha(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    assert path.stat().st_mtime_ns == stat.st_mtime_ns, str(path)
    SOURCES[str(path)] = value
    return value['sha256']


def resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(wall_seconds=time.monotonic()-START, cpu_seconds=usage.ru_utime+usage.ru_stime,
                  peak_rss_mib=usage.ru_maxrss/1024, pid=os.getpid(), python=sys.executable,
                  counts=dict(COUNTS), gpu_used=False)
    if 'budget' in globals():
        result.update(budget.snapshot())
        result.update(wall_seconds=time.monotonic()-START, counts=dict(COUNTS))
    return result


def stage(name):
    global STAGE
    STAGE = name
    event = dict(stage=name, utc=datetime.now(timezone.utc).isoformat(), resources=resources())
    with (OUT/'stages.jsonl').open('a') as file:
        file.write(json.dumps(event, allow_nan=False)+'\n')
    print(json.dumps(event), flush=True)


def np_save(path, array):
    with Path(path).open('xb') as file:
        np.save(file, array, allow_pickle=False)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def check_sources_unchanged():
    for path, original in SOURCES.items():
        stat = Path(path).stat()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (
            original['bytes'], original['mtime_ns'], original['inode']), 'source changed: '+path
        # All small source/protocol files are rehashed; large immutable inputs were
        # streamed in full before use and must retain their stat identity.
        if stat.st_size < 16*1024**2:
            assert sha(path) == original['sha256'], path


def inspect_depth(row, rgb, depth):
    from harness.object_perception import Camera, camera_from_pose
    from harness.task090_perception import observed_floor_plane
    camera = Camera(*camera_from_pose(row['camera_pose7']), 720, 720)
    floor = observed_floor_plane(depth, camera)
    stats = {}
    for name, (x0, y0, x1, y1) in INSPECTION_ROIS.items():
        vv, uu = np.mgrid[y0:y1, x0:x1]
        dd = depth[vv, uu]
        good = np.isfinite(dd) & (dd > .05) & (dd < 6.)
        points = camera.backproject(uu[good], vv[good], dd[good])
        clearance = None if floor is None else points[:, 2]-np.c_[points[:, :2], np.ones(len(points))]@floor['coefficients']
        stats[name] = dict(bbox_xyxy=[x0, y0, x1, y1], pixels=dd.size, finite_pixels=int(np.isfinite(dd).sum()),
            valid_depth_pixels=int(good.sum()), depth_min_m=float(dd.min()), depth_max_m=float(dd.max()),
            depth_p02_p50_p98_m=np.percentile(dd, [2, 50, 98]).tolist(),
            observed_height_above_fitted_plane_p02_p50_p98_m=None if clearance is None else np.percentile(clearance, [2, 50, 98]).tolist())
    result = dict(episode=EP, frame=LO, rgb=row['rgb'], depth_m=row['depth_m'],
        rgb_sha256=row['rgb_sha256'], depth_m_sha256=row['depth_m_sha256'],
        rgb_pts_s=row['rgb_decoded_pts_s'], depth_pts_s=row['depth_decoded_pts_s'],
        source='assistant native RGB inspection plus exact synchronized depth statistics; not a human label',
        regions_are_reporting_only=True, regions_used_by_initializer=False,
        whole_projected_outline_visible_2d='Yes', front_lower_edge_visible_2d='Yes', adjacent_ground_visible_2d='Yes',
        raw_depth_regions=stats, unchanged_observed_floor_estimator=jsonable(floor),
        actual_support_contact='Unknown', whole_bottom_surface='Unknown', complete_3d_envelope='Unknown',
        physical_success_label=None, physical_support_label=None, grasp_relation_label=None,
        verified=False, admitted=False,
        caveat='A resolved visible front edge and floor depth do not reveal the underside/backside or certify contact.')
    write_json(OUT/'initial_depth_coverage.json', result)
    assert floor is not None, 'frame724_floor_not_observed_by_unchanged_estimator'
    assert all(value['valid_depth_pixels'] == value['pixels'] for value in stats.values()), 'frame724_inspection_depth_missing'


def prepare_frames(check_only=False):
    stage('source_and_lowdim_preflight')
    old_root = CURRENT/'continuous_train/extracted'
    old_manifest_path = old_root/'manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    old_frames = old_root/'frames.jsonl'
    assert remember(old_frames) == old_manifest['frame_manifest_sha256'] == continuous.FRAME_SHA256
    assert old_manifest['sample_count'] == 832 and not old_manifest['admitted']
    old_rows = {r['frame_index']: r for r in map(json.loads, old_frames.open()) if r['episode'] == EP}
    assert sorted(old_rows) == list(range(BOUNDARY, HI+1))
    old_source_path = Path(old_manifest['source_provenance'])
    old_sources = json.loads(old_source_path.read_text())
    data = prep.DATA
    info_path, split_path = data/'meta/info.json', data/'split_task_090.json'
    segments_path, index_path = data/'skill_segments_task090.json', prep.LOWDIM/'index.json'
    info, split, segments, index = [json.loads(p.read_text()) for p in (info_path, split_path, segments_path, index_path)]
    assert index['revision'] == segments['dataset_revision'] == prep.REVISION
    assert index['split_sha256'] == segments['split_sha256'] == sha(split_path)
    assert EP in split['train_episode_indices'] and EP not in split['validation_episode_indices']
    metadata = []
    for path in sorted((data/'meta/episodes/chunk-090').glob('*.parquet')):
        found = prep.pq.read_table(path, filters=[('episode_index', '=', EP)], use_threads=False).to_pylist()
        if found:
            remember(path)
            metadata.extend(found)
    assert len(metadata) == 1
    meta = metadata[0]
    assert meta['task_index'] == 90 and index['episodes'][str(EP)]['split'] == 'train'
    annotation_path = data/meta['annotation_path']
    annotation = json.loads(annotation_path.read_text())
    raw = next(s for s in annotation['skill_annotation'] if s['skill_description'] == ['pick up from'] and s['object_id'] == [['trash_can', 'floors']])
    segment = next(s for s in segments['episodes'][str(EP)]['segments'] if s['skill_idx'] == raw['skill_idx'])
    assert prep.checked_window(raw, segment, BOUNDARY, meta['length']) == list(range(BOUNDARY, HI+1))
    enc = info['features'][prep.DEPTH]['info']
    assert info['fps'] == enc['video.fps'] == info['features'][prep.RGB]['info']['video.fps'] == 30
    assert enc['video.pix_fmt'] == 'gray12le' and enc['video.use_log']
    params = {key: enc['video.'+key] for key in ('depth_min', 'depth_max', 'shift', 'use_log', 'pix_fmt')}
    rgb_source, rgb_offset = prep.video_reference(meta, info, prep.RGB)
    depth_source, depth_offset = prep.video_reference(meta, info, prep.DEPTH)
    data_path = data/info['data_path'].format(chunk_index=meta['data/chunk_index'], file_index=meta['data/file_index'])
    state_path, action_path = [prep.LOWDIM/f'episode_{EP}_{kind}.npy' for kind in ('state', 'action')]
    sources = [old_manifest_path, old_source_path, info_path, split_path, segments_path, index_path, annotation_path,
               rgb_source, depth_source, data_path, state_path, action_path, prep.WRAPPER, prep.DATA_WRAPPER,
               Path(prep.__file__), ROOT/'scripts/prepare_task090_grasp_evidence.py',
               ROOT/'scripts/prepare_task090_depth_samples.py', Path(prep.codec.__file__),
               CURRENT/'support_coverage/coverage.json']
    for path in sources:
        budget.guard()
        digest = remember(path)
        if str(path) in old_sources and path.is_relative_to(data):
            assert digest == old_sources[str(path)]['sha256'], str(path)
    states, actions = [np.load(path, mmap_mode='r', allow_pickle=False) for path in (state_path, action_path)]
    for kind, array, dim in (('state', states, 61), ('action', actions, 23)):
        assert array.shape == (meta['length'], dim) and array.dtype == np.float32
        assert hashlib.sha256(array.tobytes()).hexdigest() == index['episodes'][str(EP)][kind+'_sha256']
    wanted = list(range(LO-1, HI+1))
    lowdim = prep.pq.read_table(data_path, columns=['episode_index', 'frame_index', 'timestamp', 'index', 'observation.state', prep.POSE, 'action'],
        filters=[('episode_index', '=', EP), ('frame_index', 'in', wanted)], use_threads=False).to_pylist()
    lowdim.sort(key=lambda r: r['frame_index'])
    raw_rows = prep.exact_rows(lowdim, EP, wanted, meta['dataset_from_index'])
    contract = prep.action_pairing_contract(prep.WRAPPER.read_text())
    native = json.loads((CURRENT/'support_coverage/coverage.json').read_text())['best_navigation_candidate']
    assert (native['episode'], native['frame'], native['actual_video_pts_s']) == (EP, LO, 138.3)
    assert remember(native['image']) == native['sha256']
    with Image.open(native['image']) as im:
        native_rgb = np.asarray(im).copy()
    free = prep.shutil.disk_usage(OUT).free
    assert free > 2*1024**3, 'need extraction space and 1GiB reserve'
    write_json(OUT/'metadata.json', dict(episode_metadata=meta, revision=prep.REVISION, sample_count_planned=566,
        inclusive_frames=[LO, HI], native_resolution=[720, 720], source_fps=30, gap=1,
        missing_rgbd_frames=146, existing_rgbd_references=420, codec_parameters=enc,
        action_contract=contract, temporal_selection_only=raw, nav_to_manipulation_frame=BOUNDARY,
        annotation_is_not_physical_label=True, original_source_paths=[str(rgb_source), str(depth_source), str(data_path)],
        physical_success_label=None, verified=False, admitted=False))
    if check_only:
        return
    claim_execution('prepare_started')
    directory = OUT/f'episode_{EP}'
    directory.mkdir()
    rows = []
    max_codec = max_pts = 0.
    missing = list(range(LO, BOUNDARY))
    stage('single_seek_rgbd_decode_and_exact_cache_reference')
    with ExitStack() as stack, (OUT/'frames.jsonl').open('x') as output, (OUT/'frame_sha256.jsonl').open('x') as hashes:
        rgb_reader = prep.MissingFrameReader(rgb_source, rgb_offset, missing, 'rgb', COUNTS)
        stack.callback(rgb_reader.close)
        depth_reader = prep.MissingFrameReader(depth_source, depth_offset, missing, 'depth', COUNTS)
        stack.callback(depth_reader.close)
        for frame in range(LO, HI+1):
            budget.guard()
            raw_row, previous = raw_rows[frame], raw_rows[frame-1]
            state, pose, action = prep.checked_state_action(raw_row, previous, states, actions)
            if frame < BOUNDARY:
                rgb, rgb_pts, _ = rgb_reader.get(frame)
                q, depth_pts, pix = depth_reader.get(frame)
                depth, error = prep.metric_depth(q, params)
                png, npy = directory/f'f{frame:05d}_head_rgb.png', directory/f'f{frame:05d}_head_depth_m.npy'
                with png.open('xb') as file:
                    Image.fromarray(rgb).save(file, format='PNG', compress_level=3)
                np_save(npy, depth)
                row = dict(episode=EP, raw_episode_id=meta['raw_episode_id'], train_instance_id=meta['task_instance_id'],
                    split='train', dataset_revision=prep.REVISION, frame_index=frame, dataset_index=raw_row['index'],
                    timestamp_s=raw_row['timestamp'], frame_time_s=frame/30, sample_fps=30,
                    rgb=str(png), depth_m=str(npy), rgb_sha256=sha(png), depth_m_sha256=sha(npy),
                    rgb_source=str(rgb_source), depth_source=str(depth_source), lowdim_source=str(data_path),
                    rgb_stream_episode_offset_s=rgb_offset, depth_stream_episode_offset_s=depth_offset,
                    rgb_requested_pts_s=rgb_offset+frame/30, depth_requested_pts_s=depth_offset+frame/30,
                    rgb_decoded_pts_s=rgb_pts, depth_decoded_pts_s=depth_pts, depth_original_pixel_format=pix,
                    depth_unit='m', depth_shape=[720,720], depth_range_m=[float(depth.min()),float(depth.max())],
                    codec_reference_max_error_m=error, state61=state.tolist(), camera_pose7=pose.tolist(),
                    applied_action_frame=frame-1, applied_action=action.tolist(), applied_action_verified=True,
                    action_semantics=contract['semantics'], action_source_row_index=previous['index'],
                    action_source_sha256=SOURCES[str(data_path)]['sha256'], reused_existing_rgbd=False,
                    reused_from_frame_manifests=[], annotation_temporal_position_not_outcome='before',
                    physical_success_label=None, grasp_relation_label=None, floor_contact_label=None,
                    depth_saturated_min_pixels=int((q==0).sum()), depth_saturated_max_pixels=int((q==4095).sum()))
                COUNTS['new_rgbd_frames'] += 1
            else:
                row = dict(old_rows[frame])
                assert row['dataset_index'] == raw_row['index'] and row['action_source_sha256'] == SOURCES[str(data_path)]['sha256']
                for key, value in (('state61',state), ('camera_pose7',pose), ('applied_action',action)):
                    np.testing.assert_array_equal(np.asarray(row[key], np.float32), value)
                for kind, source, offset in (('rgb',rgb_source,rgb_offset), ('depth',depth_source,depth_offset)):
                    assert row[kind+'_source'] == str(source) and row[kind+'_stream_episode_offset_s'] == offset
                    prep.pts_error(row[kind+'_requested_pts_s'], offset+frame/30)
                for key in ('rgb', 'depth_m'):
                    prep.validate_cached_path(row[key], row[key+'_sha256'], roots=prep.PAYLOAD_ROOTS+(old_root,))
                with Image.open(row['rgb']) as im:
                    rgb = np.asarray(im).copy()
                depth = np.load(row['depth_m'], allow_pickle=False)
                rgb_pts, depth_pts, error = row['rgb_decoded_pts_s'], row['depth_decoded_pts_s'], row['codec_reference_max_error_m']
                row['reused_existing_rgbd'] = True
                row['early_baseline_reference_manifest'] = str(old_frames)
                COUNTS['existing_rgbd_frames_referenced'] += 1
            row.update(frame_gap=None if frame == LO else 1, verified=False, admitted=False)
            assert rgb.shape == (720,720,3) and rgb.dtype == np.uint8
            assert depth.shape == (720,720) and depth.dtype == np.float32 and np.isfinite(depth).all()
            assert params['depth_min']-1e-5 <= depth.min() <= depth.max() <= params['depth_max']+1e-5
            assert np.isfinite(error) and error < 1e-5
            max_codec = max(max_codec,error)
            for a,b in ((rgb_pts,rgb_offset+frame/30), (depth_pts,depth_offset+frame/30), (rgb_pts-rgb_offset,depth_pts-depth_offset)):
                max_pts = max(max_pts, prep.pts_error(a,b))
            continuous.checked_row(row, EP, frame, frame == LO, prep.REVISION)
            output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush()
            hashes.write(json.dumps(dict(frame=frame, rgb=row['rgb'], depth_m=row['depth_m'], rgb_sha256=row['rgb_sha256'],
                depth_m_sha256=row['depth_m_sha256'], state61_float32_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                previous_action_float32_sha256=hashlib.sha256(action.tobytes()).hexdigest(),
                camera_pose7_float32_sha256=hashlib.sha256(pose.tobytes()).hexdigest()))+'\n')
            COUNTS['sensor_payloads_checked'] += 2
            COUNTS['extracted_manifest_rows'] += 1
            rows.append(row)
            if frame == LO:
                np.testing.assert_array_equal(rgb, native_rgb)
                prep.pts_error(rgb_pts, 138.3)
                inspect_depth(row, rgb, depth)
            if (frame-LO+1) % 100 == 0:
                print(json.dumps(dict(stage='preparing',frame=frame,rows=len(rows))), flush=True)
    assert len(rows) == 566 and COUNTS['rgb_seeks'] == COUNTS['depth_seeks'] == 1
    write_json(OUT/'extraction.json', dict(status='566_exact_frames_prepared_not_yet_tracking', samples=len(rows),
        frame_manifest_sha256=sha(OUT/'frames.jsonl'), frame_sha_manifest_sha256=sha(OUT/'frame_sha256.jsonl'),
        max_codec_reference_error_m=max_codec, max_pts_error_s=max_pts, missing_frames=[],
        per_stream_seeks=1, original_870_1289_payloads_referenced_not_copied=True, resources=resources(),
        physical_success_label=None, verified=False, admitted=False))
    return rows


def exact_initial_features():
    stage('exact_frozen_dino_source_preflight')
    from scripts import cache_dino_features as extractor
    from harness.frame_cache import FramePack
    cache = continuous.CACHE
    cfg_path = cache.parent/'cache_config.json'
    cfg = json.loads(cfg_path.read_text())
    frame_meta_path = ROOT/f'data/frame_cache/{prep.RGB}/episode_{EP}.meta.json'
    frame_meta = json.loads(frame_meta_path.read_text())
    dino_meta_path = cache/f'episode_{EP}.meta.json'
    dino_meta = json.loads(dino_meta_path.read_text())
    indices_path = cache/f'episode_{EP}.frames.npy'
    indices = np.load(indices_path, allow_pickle=False)
    assert dino_meta['complete'] and dino_meta['feat_layers'] == cfg['feat_layers'] == [-12,-8,-4]
    assert dino_meta['source_frame_meta_sha256'] == sha(frame_meta_path)
    assert cfg['tokens_per_image'] == 201 and cfg['encoder'] == 'models/dinov3-vitl16-pretrain-lvd1689m'
    audit = dict(episode=EP, frame=LO, existing_cache_min_frame=int(indices.min()),
        exact_724_in_original_cache=bool(LO in indices), feature_shape=[3,201,1024],
        unrelated_870_substitution=False, candidate_search='existing Task090 head spatial cache and local exact-frame feature filenames; no other exact frozen feature found',
        physical_success_label=None, verified=False, admitted=False)
    training_path = ROOT/'outputs/training/task090_v3_50k_20260930_s050000/manifest.json'
    training = json.loads(training_path.read_text())
    # cache_dino_features imports this inside main; reuse the original helper.
    from harness.wam_behavior import load_upstream
    _, runner = load_upstream()
    for path in (Path(extractor.__file__), ROOT/'harness/wam_behavior.py', Path(runner.__file__), cfg_path, dino_meta_path, indices_path):
        assert remember(path) == training['source_sha256'][str(path)], 'frozen training/cache source changed: '+str(path)
    remember(training_path)
    for path in (frame_meta_path, frame_meta_path.with_name(f'episode_{EP}.idx.npy'),
                 ROOT/'harness/frame_cache.py'):
        remember(path)
    if LO in indices:
        features = continuous.cached_frame(EP,LO)
        audit.update(status='exact_existing_frozen_cache', forward_calls=0,
            feature_float32_sha256=hashlib.sha256(features.tobytes()).hexdigest())
        write_json(OUT/'dino_source.json',audit)
        return features
    assert int(indices.min()) == BOUNDARY
    encoder = ROOT/cfg['encoder']
    provenance_path = encoder/'DOWNLOAD_PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text())
    assert provenance['revision'] == cfg['encoder_revision'] == 'ea8dc2863c51be0a264bab82070e3e8836b02d51'
    model_config = json.loads((encoder/'config.json').read_text())
    assert (model_config['hidden_size'],model_config['num_register_tokens'],model_config['patch_size']) == (1024,4,16)
    # This assertion prevents the factory's optional dummy forward entirely.
    remember(provenance_path)
    for name in ('config.json','preprocessor_config.json','model.safetensors'):
        budget.guard()
        assert remember(encoder/name) == provenance['files'][name]['sha256']
    pack = FramePack(frame_meta_path.parent,EP)
    assert frame_meta['from_timestamp']+LO/30 == 138.3 and pack.has(LO)
    assert remember(pack.path) == frame_meta['sha256']
    jpeg = pack.jpeg(LO)
    with (OUT/'dino_input_224.jpg').open('xb') as file:
        file.write(jpeg)
    source_list = extractor.FrameList(frame_meta_path.parent,EP,np.array([LO],np.int64))
    pixels = source_list[0].unsqueeze(0)
    assert pixels.shape == (1,3,224,224) and pixels.dtype == torch.float32
    pixels_sha = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
    start,end = map(int,pack.offsets[LO//pack.stride:LO//pack.stride+2])
    audit.update(status='preflight_passed_cpu_forward_not_started', encoder=str(encoder), encoder_revision=cfg['encoder_revision'],
        weight_sha256=SOURCES[str(encoder/'model.safetensors')]['sha256'],
        extractor_source_sha256=SOURCES[str(Path(extractor.__file__))]['sha256'],
        model_factory_source_sha256=SOURCES[str(Path(runner.__file__))]['sha256'],
        input_jpeg_sha256=hashlib.sha256(jpeg).hexdigest(), input_jpeg_bytes=[start,end],
        input_pack=str(pack.path), input_pack_sha256=frame_meta['sha256'], input_frame=LO,
        normalization='original cache_dino_features.FrameList -> FramePack.rgb -> wam_behavior.normalize_image, float32 CPU then bf16',
        input_normalized_float32_sha256=pixels_sha, native_rgb_used_by_tracker=[720,720], encoder_input=[224,224],
        native_image_not_resized_for_new_features=True, feat_layers=cfg['feat_layers'], dtype='torch.bfloat16',
        prefix_tokens=5, spatial_grid=[14,14], device='cpu', same_original_local_ModelFactory=True,
        original_cache_library_versions_recorded=False,
        cpu_vs_original_gpu_bit_identity_claimed=False,
        runtime_versions={module.__name__:module.__version__ for module in (torch,prep.av,np,cv2,prep.pa)},
        one_forward_limit=1, no_dummy_forward=True, no_alternative_encoder_or_dtype=True)
    write_json(OUT/'dino_preflight.json',audit)
    stage('frozen_encoder_cpu_load')
    enc, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(str(encoder),torch.bfloat16,torch.device('cpu'))
    assert (hidden,registers,patch) == (1024,4,16)
    assert not enc.training and all(not p.requires_grad and p.device.type == 'cpu' and p.dtype == torch.bfloat16 for p in enc.parameters())
    import transformers
    implementation = Path(inspect.getfile(type(enc)))
    remember(implementation)
    audit.update(transformers_version=transformers.__version__, actual_encoder_class=type(enc).__name__,
        implementation_source=str(implementation), implementation_sha256=SOURCES[str(implementation)]['sha256'],
        attention_implementation=enc.config._attn_implementation)
    budget.guard()
    stage('one_frozen_dino_cpu_forward')
    before = time.monotonic()
    COUNTS['dino_forward_attempts'] += 1
    with torch.no_grad():
        px = pixels.to(torch.device('cpu'),torch.bfloat16)
        hs = enc(pixel_values=px,output_hidden_states=True,return_dict=True).hidden_states
        h = torch.stack([hs[layer] for layer in cfg['feat_layers']],dim=1)
        assert h.shape == (1,3,201,1024) and h.dtype == torch.bfloat16 and torch.isfinite(h).all()
        bits = h.contiguous().view(torch.int16).cpu().numpy().view(np.uint16).copy()
    COUNTS['dino_forward_completed'] += 1
    path = OUT/'ep18001_f00724.feat.npy'
    np_save(path,bits)
    features = (bits[0].astype(np.uint32)<<16).view(np.float32).copy()
    np.testing.assert_array_equal(features,h[0].float().numpy())
    audit.update(status='one_exact_frame_frozen_cpu_forward_complete',forward_calls=1,forward_seconds=time.monotonic()-before,
        feature_file=str(path),feature_file_sha256=sha(path),feature_float32_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
        original_cache_modified=False,resources=resources())
    write_json(OUT/'dino_source.json',audit)
    del enc,hs,h,px,pixels,bits,source_list,pack
    gc.collect()
    budget.guard()
    return features


def tracking(rows,features,geometry):
    from harness.task090_rigid_observation_v5 import Task090RigidObservationV5,VERSION
    stage('single_continuous_v5_replay_724_to_1289')
    tracker = Task090RigidObservationV5(continuous.reused.CALIBRATION,continuous.reused.PROTOTYPES,max_frame_gap=6,self_filter=geometry)
    original_measure = tracker.measure.update
    proposals = []
    def capture_proposal(*args,**kwargs):
        result = original_measure(*args,**kwargs)
        proposals.append(result)
        return result
    tracker.measure.update = capture_proposal  # Audit only; exact original return value.
    history, binding, bank_digest, first_loss = [],None,None,None
    boundaries = {}
    with (OUT/'episode_18001_measurements.jsonl').open('x') as output, (OUT/'all_failures.jsonl').open('x') as failures:
        for index,row in enumerate(rows):
            budget.guard()
            frame=row['frame_index']
            continuous.checked_row(row,EP,frame,index==0,prep.REVISION)
            for key in ('rgb','depth_m'):
                assert sha(row[key]) == row[key+'_sha256']
                COUNTS['tracking_sensor_sha_checks'] += 1
            with Image.open(row['rgb']) as im:
                rgb=np.asarray(im).copy()
            depth=np.load(row['depth_m'],allow_pickle=False)
            observation={'robot_r1::robot_r1:zed_link:Camera:0::rgb':rgb,
                         'robot_r1::robot_r1:zed_link:Camera:0::depth_linear':depth,
                         'robot_r1::cam_rel_poses':np.asarray(row['camera_pose7'])}
            before_bank=tracker.original_bank
            report=tracker.update(observation,np.asarray(row['state61']),episode=EP,step=frame,
                head_features=features if index==0 else None,action=np.asarray(row['applied_action']))
            diag=report['rigid_tracking']
            assert diag['tracker_version']==VERSION and not diag['verified'] and not diag['admitted']
            assert not report['verified'] and not report['admitted']
            assert tracker.total_added==diag['dependent_reference_points_used']==0
            bank=tracker.original_bank
            if bank is not None:
                if before_bank is None:
                    assert frame==LO, 'late initialization/rebinding forbidden'
                    bank_digest=continuous.bank_hashes(bank)
                    with (OUT/'episode_18001_original_bank.npz').open('xb') as file:
                        np.savez(file,gray=bank.gray,uv=bank.uv,reference=bank.reference,errors=bank.errors,
                                 origin_floor=tracker.origin_floor,origin_margin=tracker.origin_margin)
                    binding=dict(frame=frame,points=len(bank.uv),track_id=tracker.track_id,bank_sha256=bank_digest)
                else:
                    assert bank is before_bank and continuous.bank_hashes(bank)==bank_digest
                assert all(not getattr(bank,key).flags.writeable for key in ('gray','uv','reference','errors'))
                assert tracker.track_id==binding['track_id']
                COUNTS['immutable_bank_frame_checks']+=1
            measured=None
            if diag['status'] in continuous.ACCEPTED:
                ids=np.flatnonzero(tracker.last_accepted_mask)
                np.testing.assert_array_equal(tracker.anchor_reference,bank.reference[ids])
                np.testing.assert_array_equal(tracker.reference,tracker.anchor_reference)
                np.testing.assert_array_equal(tracker.errors,bank.errors[ids])
                r,t=tracker.previous_pose
                points=continuous.reused.sample_points(tracker.previous_camera,depth,tracker.anchor_uv)
                residual=np.linalg.norm(tracker.anchor_reference@r.T+t-points,axis=1)
                assert len(residual)>=6 and residual.max()<=continuous.reused.RIGID_TOLERANCE_M
                assert geometry(points,np.asarray(row['state61'])).all()
                measured=dict(original_ids=ids.tolist(),uv=tracker.anchor_uv.tolist(),current_points_base_m=points.tolist(),
                    rotation=r.tolist(),translation_m=t.tolist(),independently_recomputed_max_residual_m=float(residual.max()))
                COUNTS['independent_accepted_frame_checks']+=1
                if frame!=LO:
                    assert diag['pose_source']=='original_anchors_only' and diag['sample_gap_steps']==1
            assert diag['appearance_keyframe_committed']==(diag['status'] in continuous.ACCEPTED)
            event=continuous.reused.audit(EP,frame,diag)
            event.update(track_id_retained_in_producer=tracker.track_id,right_finger_sum_m=float(np.asarray(row['state61'])[49:51].sum()),
                candidate_grasp=diag.get('candidate_grasp'),candidate_observations=diag.get('candidate_observations',0),
                appearance_keyframe_step=diag['appearance_keyframe_step'])
            if first_loss is None and frame>LO and diag['status']!='rigid_motion_measured':
                first_loss=event
            report.update(source_frame=dict(episode=EP,frame=frame,source_fps=30,source_gap=None if index==0 else 1,
                rgb=row['rgb'],depth_m=row['depth_m'],rgb_sha256=row['rgb_sha256'],depth_m_sha256=row['depth_m_sha256'],
                applied_action_frame=frame-1,dino_used=index==0,physical_success_label=None),
                audit=event,independent_measurement_check=measured,bank_sha256=bank_digest,
                physical_success_label=None,grasp_relation_label=None,floor_contact_label=None)
            output.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n');output.flush()
            history.append(event)
            COUNTS['tracking_frames_processed']+=1
            if diag['status']=='unknown':
                failures.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n');failures.flush()
            if frame in (BOUNDARY-1,BOUNDARY):
                boundaries[str(frame)]=dict(event=event,bank_sha256=bank_digest,measured=measured is not None)
            if frame==LO:
                roi_counts={name:dict(count=int(((bank.uv[:,0]>=box[0])&(bank.uv[:,0]<box[2])&(bank.uv[:,1]>=box[1])&(bank.uv[:,1]<box[3])).sum()),bbox_xyxy=box)
                    for name,box in INSPECTION_ROIS.items()} if bank is not None else None
                write_json(OUT/'initialization.json',dict(status=diag['status'],original_initializer_proposals=proposals,
                    original_bank=binding,bank_reporting_roi_counts=roi_counts,report=report,
                    lower_edge_coverage_semantics='Counts in display-only strip include any background; not certified object ownership or support.',
                    actual_support_contact='Unknown',complete_3d_envelope='Unknown',physical_success_label=None,verified=False,admitted=False))
                if diag['status']!='tracking_initialized':
                    stage('stopped_initialization_failed_no_other_frame_or_retry')
                    break
            if (index+1)%100==0:
                print(json.dumps(dict(stage='tracking',frame=frame,processed=len(history),status=diag['status'])),flush=True)
    assert len(proposals)==1, 'initializer must be called exactly once'
    measured_rows=[r for r in history if r['status']=='rigid_motion_measured']
    first_close=next((r['frame'] for r in history if r['right_finger_sum_m']<.095),None)
    bounds=[r['conditional_min_height_gain_m'] for r in history if r['conditional_min_height_gain_m'] is not None]
    bridge=len(boundaries)==2 and all(r['measured'] for r in boundaries.values())
    complete=len(history)==566
    summary=dict(status='cpu_early_baseline_complete_not_admission' if complete else 'stopped_initialization_failed_not_admission',
        frames_prepared=len(rows),tracking_frames_processed=len(history),full_566_frame_diagnostic_completed=complete,
        episode=EP,frames_requested_inclusive=[LO,HI],initial_binding=binding,first_loss=first_loss,
        first_gripper_below_095m=first_close,measured_frames=len(measured_rows),
        measured_frames_after_first_closure=sum(first_close is not None and r['frame']>=first_close for r in measured_rows),
        measured_frames_while_finger_sum_below_095m=sum(r['right_finger_sum_m']<.095 for r in measured_rows),
        height_bound_range_m=None if not bounds else [min(bounds),max(bounds)],
        candidate_grasp_frames=sum(bool(r['candidate_grasp']) for r in history),
        candidate_observation_frames=sum(r['candidate_observations']>0 for r in history),
        status_counts=dict(Counter(r['status'] for r in history)),failure_reasons=dict(Counter(r['reason'] for r in history if r['status']=='unknown')),
        nav_manipulation_boundary=boundaries,same_retained_bank_and_track_across_870=len(boundaries)==2 and binding is not None,
        measured_same_track_across_869_870=bridge,
        uninterrupted_initial_track_to_manipulation=bridge and (first_loss is None or first_loss['frame']>BOUNDARY),
        primary_endpoint='First loss from frame724 binding, not a best recovered track or random success rate.',
        initialization_calls=len(proposals),rebinding_at_870=False,post_initial_dino_features=None,
        tracker_version=VERSION,self_filter_version=geometry.version,thresholds_searched=False,automatic_retries=0,
        actual_support_contact='Unknown',absolute_support_clearance='Unknown',physical_success_label=None,
        physical_support_label=None,grasp_relation_label=None,physical_labels_available=0,verified=False,admitted=False,
        simulator_executed=False,policy_executed=False,WAM_executed=False,uses_hidden_truth=False,
        resources=resources())
    check_sources_unchanged()
    write_json(OUT/'source_provenance.json',SOURCES)
    write_json(OUT/'protected_unchanged.json',dict(unchanged=True,source_files=len(SOURCES),
        method='full initial hashes; final stat identity plus rehash for sources under16MiB',verified=False,admitted=False))
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def run():
    global prep,continuous,np,cv2,torch,Image,budget
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    import prepare_task090_continuous_evidence as prep
    from scripts import check_task090_continuous_tracking as continuous
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    budget=prep.Budget()
    cv2.setNumThreads(1)
    budget.guard()
    stage('imports_and_cpu_budget_ready')
    test_path=CURRENT/'keyframe_v5/cpu_tests.json'
    tests=json.loads(test_path.read_text())
    assert tests['status']=='PASS' and tests['failures']==tests['errors']==0
    for path,digest in tests['source_sha256'].items():
        assert remember(path)==digest
    for path in [Path(__file__),test_path,continuous.reused.PROTOTYPES,continuous.reused.PROTOTYPES.with_suffix('.json'),
        continuous.reused.FROZEN,ROOT/'harness/task090_observation.py',ROOT/'harness/task090_dino_localizer.py',
        ROOT/'harness/object_perception.py',ROOT/'harness/task090_perception.py',ROOT/'harness/task090_predicates.py',
        ROOT/'harness/task1_perception_adapter.py',ROOT/'scripts/check_task090_rigid_train_v3.py',
        ROOT/'scripts/check_task090_rigid_train_v4.py',*ROOT.glob('configs/task090*.json'),*ROOT.glob('configs/lilawam_task090*.yaml')]:
        remember(path)
    frozen=json.loads(continuous.reused.FROZEN.read_text())['sha256']
    for rel in ('harness/task090_rigid_observation_v2.py','harness/task090_observation.py','harness/task090_dino_localizer.py'):
        assert sha(ROOT/rel)==frozen[rel]
    geometry=continuous.RobotSelfGeometry()
    for path,digest in geometry.source_sha256.items():
        assert remember(path)==digest
    rows=prepare_frames()
    features=exact_initial_features()
    tracking(rows,features,geometry)


def main():
    if len(sys.argv)!=1:
        raise ValueError('No alternate frame/window/model/threshold/output arguments allowed')
    OUT.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HOME']=str(OUT/'runtime_cache/huggingface')
    os.environ['MPLCONFIGDIR']=str(OUT/'runtime_cache/matplotlib')
    write_json(OUT/'plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),script=str(Path(__file__).resolve()),
        script_sha256=sha(__file__),episode=EP,frames_inclusive=[LO,HI],samples_planned=566,
        dino_forward_budget=1,max_cpu_threads=1,cpu_ram_limit_percent=85,per_process_rss_limit_gib=2,wall_limit_s=900,
        max_frame_gap=6,source_gap=1,rigid_tolerance_m=.01,minimum_original_points=6,lk_fb_tolerance_px=1.5,
        inspection_rois_reporting_only=INSPECTION_ROIS,unchanged_self_geometry=True,unchanged_v5=True,
        initialization_frame=LO,automatic_retries=0,alternate_initialization_frames=[],old_832_window_replayed=False,
        physical_success_label=None,verified=False,admitted=False,policy_executed=False,gpu_used=False))
    with (OUT/'execution.log').open('x',buffering=1) as log, redirect_stdout(log),redirect_stderr(log):
        try:
            run()
        except BaseException as error:
            failure=dict(status='failed_partial_evidence_retained_not_admission',failed_stage=STAGE,error=repr(error),
                traceback=traceback.format_exc(),actual_tracking_frames=COUNTS['tracking_frames_processed'],
                actual_extracted_manifest_rows=COUNTS['extracted_manifest_rows'],full_566_frame_diagnostic_completed=False,
                initialization='Unknown',physical_success_label=None,physical_support_label=None,
                verified=False,admitted=False,resources=resources())
            write_json(OUT/'failure.json',failure)
            if not (OUT/'source_provenance.json').exists():
                write_json(OUT/'source_provenance.json',SOURCES)
            print(json.dumps(failure,ensure_ascii=False),flush=True)
            return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
