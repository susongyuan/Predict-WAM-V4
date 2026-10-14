#!/usr/bin/env python3
"""Exactly 21 cached TRAIN observations; no LK, model, decoder or simulator run."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import time
import traceback
from types import SimpleNamespace
import cv2
import numpy as np
import scipy
from PIL import Image
from harness.object_perception import Camera, camera_from_pose
from harness.robot_self_geometry import RobotSelfGeometry, sha256
from harness.task090_rigid_observation_v3 import Task090RigidObservationV3
from harness.dense_rgbd_registration import surface, register, transform, CAPTURE
from scripts import check_task090_continuous_tracking as reader
from scripts import test_dense_rgbd_registration as synthetic

audit = reader.reused
BASE = ROOT/'outputs/validation/task090_completion_20261003_0156'
OUT = BASE/'dense_body_measurement'
WINDOWS = [(18001, 762, BASE/'early_baseline_run02/attempt01/prepare/frames.jsonl', BASE/'early_baseline_run02/attempt01/track'),
           (18000, 803, reader.SOURCE, BASE/'keyframe_v5/continuous_train_01'),
           (18001, 1027, reader.SOURCE, BASE/'keyframe_v5/continuous_train_01')]


def rows(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file]


def candidate(depth, rgb, camera, bank, previous, geometry, state, calibration):
    measured = previous['independent_measurement_check']
    reference = bank['reference']
    points = reference @ np.asarray(measured['rotation']).T + measured['translation_m']
    u, v, z = camera.project(points)
    valid = np.isfinite(np.c_[u, v, z]).all(axis=1) & (z > .05)
    hull = np.zeros(depth.shape, np.uint8)
    if valid.sum() >= 3:
        uv = np.rint(np.c_[u[valid], v[valid]]).astype(np.int32)
        cv2.fillConvexPoly(hull, cv2.convexHull(uv), 1)
    appearance = Task090RigidObservationV3._appearance(SimpleNamespace(calibration=calibration, target='trash_can'), rgb, depth)
    prior_points = np.asarray(measured['current_points_base_m'])
    prior_z = (prior_points-camera.t) @ camera.R[:, 2]
    mask = (hull > 0) & (appearance > 0) & (depth >= prior_z.min()-CAPTURE) & (depth <= prior_z.max()+CAPTURE)
    yy, xx = np.nonzero(mask)
    robot = np.zeros(depth.shape, bool)
    robot[yy, xx] = ~geometry(camera.backproject(xx, yy, depth[yy, xx]), state)
    return mask, robot, hull


def run():
    run_dir = OUT/'local_check'; run_dir.mkdir(exist_ok=False)
    audit.START = time.monotonic(); audit.PROCESS.cpu_affinity([min(audit.PROCESS.cpu_affinity())]); cv2.setNumThreads(1)
    protected = [*ROOT.glob('configs/*.json'), *ROOT.glob('harness/*policy*.py'), ROOT/'harness/task090_rigid_observation_v5.py', ROOT/'harness/task090_predicates.py']
    sources = [ROOT/'harness/dense_rgbd_registration.py', Path(__file__), Path(synthetic.__file__), Path(reader.__file__), Path(audit.__file__),
               ROOT/'harness/object_perception.py', ROOT/'harness/robot_self_geometry.py', ROOT/'harness/task090_rigid_observation_v3.py',
               BASE/'camera_contract/nominal_contract.json', BASE/'early_baseline_run02/execution_audit.json', OUT/'plan.json', audit.CALIBRATION]
    sources += [p for ep, fail, manifest, track in WINDOWS for p in (manifest, track/f'episode_{ep}_measurements.jsonl', track/f'episode_{ep}_original_bank.npz', track/'summary.json')]
    hashes = {str(p): sha256(p) for p in set(sources+protected)}
    audit.write_json(run_dir/'preregistered_sources.json', dict(sha256=hashes, local_capture_surface_displacement_max_m=CAPTURE,
        normal_compatibility_abs_dot_min=.5, windows=[dict(episode=e, failure=f, frames=[f-3, f+3]) for e, f, _, _ in WINDOWS],
        synthetic_before_actual=True, independent_support_is_geometric_not_statistical=True))
    contract = json.loads((BASE/'camera_contract/nominal_contract.json').read_text())
    assert contract['evaluation_override']['computed_nominal_fx'] == 306
    assert sha256(reader.SOURCE) == reader.FRAME_SHA256
    early_audit = json.loads((BASE/'early_baseline_run02/execution_audit.json').read_text())
    assert sha256(WINDOWS[0][2]) == early_audit['source_checks']['frames_jsonl_sha256']
    tests = synthetic.run(OUT/'cpu_tests')
    print(json.dumps(dict(stage='cpu_counterexamples', passed=tests['passed'], failed=tests['failed'])), flush=True)
    geometry, calibration = RobotSelfGeometry(), json.loads(audit.CALIBRATION.read_text())
    results, sensor_hashes = [], {}
    for ep, failure, manifest, track in WINDOWS:
        selected = [r for r in rows(manifest) if r['episode'] == ep and failure-3 <= r['frame_index'] <= failure+3]
        assert [r['frame_index'] for r in selected] == list(range(failure-3, failure+4))
        history = rows(track/f'episode_{ep}_measurements.jsonl')
        accepted = [r for r in history if r.get('independent_measurement_check') is not None]
        bank = np.load(track/f'episode_{ep}_original_bank.npz', allow_pickle=False)
        folder = run_dir/f'{ep}_{failure}'; folder.mkdir()
        source = source_pose = None
        for index, row in enumerate(selected):
            audit.guard(); frame = row['frame_index']
            reader.checked_row(row, ep, frame, False, row['dataset_revision'])
            previous = max((r for r in accepted if r['source_frame']['frame'] <= frame-(index > 0)), key=lambda r:r['source_frame']['frame'])
            assert frame-previous['source_frame']['frame'] <= 6
            for key in ('rgb', 'depth_m'):
                sensor_hashes[row[key]] = sha256(row[key]); assert sensor_hashes[row[key]] == row[key+'_sha256']
            with Image.open(row['rgb']) as image:
                rgb = np.asarray(image).copy()
            depth = np.load(row['depth_m'], allow_pickle=False)
            assert rgb.shape == (720, 720, 3) and depth.shape == (720, 720)
            extrinsics = camera_from_pose(np.asarray(row['camera_pose7']))
            if extrinsics is None:
                raise ValueError('invalid_camera_pose')
            camera = Camera(*extrinsics, 720, 720)
            mask, robot, hull = candidate(depth, rgb, camera, bank, previous, geometry, np.asarray(row['state61']), calibration)
            cloud = surface(depth, camera, mask, robot, calibration['depth_max_m'])
            np.savez_compressed(folder/f'frame_{frame}.npz', depth=depth, camera_R=camera.R, camera_t=camera.t, candidate_hull=hull, **cloud)
            provenance = dict(episode=ep, frame=frame, failure_frame=failure, prediction_frame=previous['source_frame']['frame'],
                original_track=previous['rigid_tracking'].get('track_id'), source_frame=row, mask_pixels=int(mask.sum()),
                robot_excluded_pixels=int((mask & robot).sum()), retained_surface_pixels=len(cloud['points']))
            if index == 0:
                source = cloud; inherited = previous['rigid_tracking']['cumulative_reference_error_m']
                measured = previous['independent_measurement_check']; source_pose = np.eye(4)
                source_pose[:3, :3], source_pose[:3, 3] = measured['rotation'], measured['translation_m']
                audit.write_json(folder/'source_binding.json', dict(**provenance, prior_v5=previous, original_reference_pose=source_pose.tolist(),
                    inherited_error_m=inherited, original_error_bound_still_unknown=True, source_is_local_measurement_not_rebind=True))
                continue
            result, trace = register(source, cloud, inherited_error_m=inherited)
            pose = result['transform_source_to_current_base']
            result.update(provenance=provenance, source_frame_index=failure-3, original_reference_candidate_transform=None if pose is None else (np.asarray(pose)@source_pose).tolist(),
                original_reference_transform_depends_on_unadmitted_v5=True, cpu_suite_failures=tests['failed'])
            audit.write_json(folder/f'result_{frame}.json', result)
            np.savez_compressed(folder/f'correspondences_{frame}.npz', **trace)
            results.append(result)
        bank.close()
        print(json.dumps(dict(stage='fixed_window_complete', episode=ep, failure=failure, observations=7,
                             outcomes=[dict(frame=r['provenance']['frame'], status=r['status'], reasons=r['reasons']) for r in results[-6:]])), flush=True)
    assert len(results) == 18 and len(sensor_hashes) == 42
    assert all(sha256(path) == digest for path, digest in hashes.items()), 'source/protected input changed'
    audit.write_json(run_dir/'provenance.json', dict(source_sha256=hashes, sensor_sha256=sensor_hashes, geometry_sha256=geometry.source_sha256,
        protected_unchanged=True, numpy=np.__version__, scipy=scipy.__version__, cv2=cv2.__version__, resources=audit.resources()))
    summary = dict(status='fixed_local_diagnostic_only', observations=21, registrations=18, synthetic_passed=tests['passed'], synthetic_failed=tests['failed'],
        feasible_candidates=sum(r['status']=='feasible_measurement_candidate' for r in results), results=results,
        new_dino_extractions=0, videos_decoded=0, simulator_executed=False, gpu_used=False, policy_changed=False,
        part_identity='part_identity_unknown', physical_success=None, cumulative_error_bound_m=None, resources=audit.resources(),
        next_step='If geometry candidates survive CPU truth counterexamples: complete continuous tracking and independent relation validation. Otherwise obtain the specific missing independent surface directions/held-out overlap/body-link mapping and instance scale; no further LK patches or threshold sweep.')
    audit.write_json(OUT/'summary.json', summary)
    return summary


if __name__ == '__main__':
    try:
        value = run(); print(json.dumps({k:v for k,v in value.items() if k != 'results'}), flush=True)
    except BaseException as error:
        audit.write_json(OUT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), resources=audit.resources()))
        raise
