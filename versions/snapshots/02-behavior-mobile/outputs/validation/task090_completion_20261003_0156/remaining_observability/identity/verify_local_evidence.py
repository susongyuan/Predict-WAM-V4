#!/usr/bin/env python3
"""Audit already-computed selected-window evidence; no tracker/video replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import psutil
ROOT = Path('/home/user/Robotic/project2')
BASE = ROOT/'outputs/validation/task090_completion_20261003_0156'
OUT = BASE/'remaining_observability/identity'
OLD = BASE/'self_geometry_usd/implementation/continuous_train_01'
start = time.monotonic()
proc = psutil.Process()
proc.cpu_affinity([min(proc.cpu_affinity())])
assert psutil.virtual_memory().percent < 85

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(name):
    return json.loads((OUT/name).read_text())

summary = read('machine_evidence.json')
direct = read('direct_failure_details.json')
short = read('short_keyframe_counterexample.json')
initial = read('initial_points.json')
for p, digest in summary['old_output_hashes'].items():
    assert sha(p) == digest, p
for p, digest in summary['source_hashes'].items():
    assert sha(p) == digest, p
assert sha(OUT/'diagnose_selected.py') == summary['diagnostic_script_sha256']
frozen = json.loads((OLD/'source_sha256.json').read_text())
code_hashes = {}
for rel in ('harness/task090_rigid_observation_v4.py', 'harness/task090_rigid_observation_v3.py',
            'harness/task090_rigid_observation_v2.py', 'harness/object_perception.py',
            'harness/robot_self_geometry.py', 'configs/task090_perception_geometry_v2.json'):
    p = ROOT/rel
    assert sha(p) == frozen[str(p)], rel
    code_hashes[str(p)] = sha(p)

frame_rows = {(r['episode'], r['frame_index']): r for r in map(json.loads, (BASE/'continuous_train/extracted/frames.jsonl').open())}
selected = {}
for ep, first, failure, crop0, crop1 in ((18000,480,797,(190,540,430,720),(180,490,500,720)),
                                        (18001,870,949,(470,545,635,720),(450,545,635,720))):
    records = {r['source_frame']['frame']: r for r in map(json.loads,(OLD/f'episode_{ep}_measurements.jsonl').open())}
    b = np.load(OLD/f'episode_{ep}_original_bank.npz')
    for r in [r for r in direct if r['episode'] == ep]:
        d = records[r['frame']]['rigid_tracking']
        assert r['lk_pass_count'] == d['original_anchors_after_lk']
        assert r['self_rejected'] == d['original_anchors_self_rejected']
        if r['fit']['status'] == 'unknown':
            assert d['reason'] == 'ValueError:' + r['fit']['reason']
        else:
            assert r['fit']['accepted_ids'] == records[r['frame']]['independent_measurement_check']['original_ids']
    for r in [r for r in short if r['episode'] == ep]:
        assert r['immutable_reference_sha256'] == hashlib.sha256(b['reference'].tobytes()).hexdigest()
        assert len(r['fit']['accepted_ids']) >= 6 and r['fit']['max_residual_m'] <= .01
    inspections = []
    for frame, box in ((first, crop0), (failure, crop1)):
        row = frame_rows[ep,frame]
        source = np.asarray(Image.open(row['rgb']))
        copy = OUT/f'ep{ep}_f{frame}_rgb.png'
        assert np.array_equal(source,np.asarray(Image.open(copy)))
        crop = OUT/f'ep{ep}_f{frame}_inspection_crop.png'
        x0,y0,x1,y1 = box
        assert np.array_equal(source[y0:y1,x0:x1], np.asarray(Image.open(crop)))
        inspections.append(dict(frame=frame, original_rgb_path=row['rgb'], original_rgb_sha256=row['rgb_sha256'],
                                inspected_copy=str(copy), inspected_copy_sha256=sha(copy),
                                native_unannotated_crop=str(crop), crop_xyxy=list(box),
                                pixel_equal_to_source=True))
    d = next(r for r in direct if (r['episode'],r['frame']) == (ep,failure))
    s = next(r for r in short if (r['episode'],r['frame']) == (ep,failure))
    differences = []
    for q in s['points']:
        if q['id'] not in s['fit']['accepted_ids']:
            continue
        p = next(p for p in d['points'] if p['id'] == q['id'])
        differences.append(dict(id=q['id'], original_uv=b['uv'][q['id']].tolist(),
            direct_uv=p['measured_uv'], short_uv=q['measured_uv'],
            uv_distance_px=float(np.linalg.norm(np.asarray(p['measured_uv'])-q['measured_uv'])),
            direct_lk_fb_px=p['fb_error_px'], short_lk_fb_px=q['fb_error_px'],
            direct_depth_m=p['measured']['depth_m'], short_depth_m=q['measured']['depth_m'],
            absolute_depth_difference_m=abs(p['measured']['depth_m']-q['measured']['depth_m']),
            direct_depth_span_5x5_m=p['measured']['depth_span_5x5_m']))
    selected[str(ep)] = dict(initial_frame=first, first_failure_frame=failure,
        immutable_bank_size=len(b['uv']), projected_count=len(d['projected_ids']),
        not_projected_original_ids=d['not_projected_ids'],
        direct_lk_count=d['lk_pass_count'], direct_self_rejected=d['self_rejected'],
        direct_fit=d['fit'], short_fit=s['fit'], point_comparisons=differences,
        image_inspection_provenance=inspections,
        first_failure_right_finger_sum_m=records[failure]['audit']['right_finger_sum_m'])

views = []
for camera in ('observation.rgb.left_realsense_link_camera_0','observation.rgb.right_realsense_link_camera_0'):
    for ep in (18000,18001):
        p = ROOT/'data/frame_cache'/camera/f'episode_{ep}.meta.json'
        m = json.loads(p.read_text())
        video = ROOT/'data/behavior-1k/2026-challenge-demos'/m['video_path']
        views.append(dict(episode=ep,camera=camera,metadata_path=str(p),metadata_sha256=sha(p),
                          native_video_exists=video.is_file(),native_video_path=str(video),
                          cached_size=m['size'],cached_stride=m['stride'],
                          decoded_in_this_diagnostic=False))
depth_views = sorted(p.name for p in (ROOT/'data/behavior-1k/2026-challenge-demos/videos').glob('observation.depth*'))

visual = {
    'source': 'assistant visual inspection of individually opened, unannotated native RGB images and pixel-index JSON; not GT or physical labels',
    'initial_target_surface_observable': {'18000': True, '18001': True},
    'initial_bank_semantically_validated_all_points': False,
    'initial_18000': 'The bin rim, front and inner regions are visible without robot-arm occlusion. Several landmarks lie on silhouettes or inner depth edges; a clean target-only surface cannot be asserted for every edge sample. The earlier intermediate impression that an arm was present at initialization was corrected by individual-file inspection.',
    'initial_18001': 'The bin is partially cut by the right/bottom image boundary, but its upper-left rim and inner boundary remain visible. It is not entirely out of frame, and the initialization is not visually a wholly different object.',
    'failure_18000': 'The right hand/forearm occludes a substantial right/front rim region. Left/back rim landmarks remain visible. All 40 projected seeds are in bounds, which is not a visibility certificate.',
    'failure_18001': 'The approaching open hand occludes lower-left bin regions, including approximately original IDs 2/3/13 under the last measured pose. Several upper-rim points remain visible. Only ID14 projects outside the image; offscreen loss alone cannot explain 22->5.',
    'texture': 'Large dark areas are nearly uniform; usable corner points are concentrated on rim/depth discontinuities. This is limited texture support, not zero observable texture. Gray patch standard deviations in initial_points.json are descriptive only.',
    'identity_limits': 'RGB-only inspection does not certify which inner brown structure is rigidly attached, nor subpixel silhouette depth ownership. No full-bank wrong-object binding is demonstrated; individual edge contamination remains possible.'
}
plan = {
    'preferred_only': 'Immutable-original-3D short-distance appearance-keyframe bank',
    'reuse': ['Task090RigidObservationV4.original_bank', 'Task090RigidObservationV3.previous_gray/anchor_uv',
              'seeded_flow_matches', 'sample_points', 'original_anchor_check', 'RobotSelfGeometry'],
    'implementation': 'Retain initial IDs/3D/errors permanently. Save gray+measured UV+IDs only after an accepted original-only RGB-D fit. Match against the recent accepted appearance view first; use the immutable initial view for reobservation, deduplicate by original ID and never count a projection as a measurement. Keep all existing 6-point/1cm/1.5px/expiry/recovery gates. Failed frames do not become keyframes. Never invert estimated poses to fabricate new reference 3D.',
    'depth_caveat': 'Record local depth discontinuity and reject samples whose pixel/depth ownership is ambiguous under the existing measurement uncertainty; do not average across foreground/background or widen the 1cm gate. The short counterexample does not fix edge sampling itself and may still become Unknown after conservative rejection.',
    'local_counterexample_scope': 'Accepted recorded seeds 793 and 945, then only 794-799 and 946-951. Existing output reused; no new full-window run. Original reference arrays remain byte-identical. This isolates a recoverable correspondence limitation, not long-horizon identity or physical grasp.',
    'if_original_surface_really_disappears': 'Do not keep tuning original-anchor LK. Existing head dense RGB-D can only extend an observed object surface during a verified overlap with the old surface, propagating registration uncertainty and rejecting symmetry/texture degeneracy. Completely disjoint new surfaces cannot certify identity. Wrist RGB videos exist but wrist-depth payloads are not local; reduced 224px stride2 caches and action-model multiview features are not synchronized metric object measurements. Cross-view registration, intrinsics/extrinsics/timing, target ownership and occlusion validation would be required before wrist RGB plus head depth could substitute. None is established here; no dense/multiview replacement is currently admitted.',
    'runtime_change_in_this_task': False
}
result = dict(status='verified_existing_local_diagnostic_not_admitted', original_outputs_unchanged=True,
    direct_selected_results_verified=12, short_selected_results=12, no_tracker_rerun_in_verifier=True,
    inherited_tracking_execution=dict(rgbd_pairs_loaded=summary['rgbd_pairs_loaded'],seconds=summary['seconds'],rss_mib=summary['rss_mib']),
    thresholds=summary['thresholds'], thresholds_searched=False, gpu_used=False, simulator_used=False,
    gt_used=False, hand_pose_generates_object_measurement=False, new_landmarks=False,
    physical_success_label=None, admitted=False, verified=False,
    cases=selected, visual_inspection=visual, preferred_implementation=plan,
    local_multiview_metadata=views, local_depth_view_directories=depth_views,
    source_code_hashes=code_hashes,
    evidence_files={str(OUT/n):sha(OUT/n) for n in ('diagnose_selected.py','machine_evidence.json','initial_points.json',
                    'direct_failure_details.json','short_keyframe_counterexample.json')},
    verifier_script_path=str(Path(__file__)),verifier_script_sha256=sha(__file__),
    audit_seconds=time.monotonic()-start,audit_rss_mib=proc.memory_info().rss/2**20)
with (OUT/'verified_identity_evidence.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2)
print(json.dumps(dict(status=result['status'],output=str(OUT/'verified_identity_evidence.json'),
                     direct_selected_results_verified=12,short_selected_results=12,
                     old_outputs_unchanged=True, audit_seconds=result['audit_seconds']),ensure_ascii=False))
