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
import argparse
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
from harness.dense_rgbd_registration import surface, register, transform, partition, CAPTURE, HOLDOUT_PROTOCOL
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


def require_cpu_pass(report):
    cases = {r['case']: r for r in report.get('cases', [])}
    required = ('qualified_known_rigid', 'qualified_camera_motion')
    if (report.get('failed') != 0 or report.get('leakage_checks_passed') is not True
            or any(k not in cases or not cases[k]['passed'] for k in required)):
        raise RuntimeError('cpu_preflight_failed_no_real_frames_processed')


def run(out=None, cpu_report=None):
    global OUT
    if out is not None:
        OUT = Path(out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT/'local_check'; run_dir.mkdir(exist_ok=False)
    audit.START = time.monotonic(); audit.PROCESS.cpu_affinity([min(audit.PROCESS.cpu_affinity())]); cv2.setNumThreads(1)
    # The independent lifecycle agent owns policy_server.py; it is not imported
    # or exercised here, so do not freeze unrelated concurrently edited source.
    protected = [*ROOT.glob('configs/*.json'), *[p for p in ROOT.glob('harness/*policy*.py') if p.name != 'policy_server.py'],
                 ROOT/'harness/task090_rigid_observation_v5.py', ROOT/'harness/task090_predicates.py']
    sources = [ROOT/'harness/dense_rgbd_registration.py', Path(__file__), Path(synthetic.__file__), Path(reader.__file__), Path(audit.__file__),
               ROOT/'harness/object_perception.py', ROOT/'harness/robot_self_geometry.py', ROOT/'harness/task090_rigid_observation_v3.py',
               BASE/'camera_contract/nominal_contract.json', BASE/'early_baseline_run02/execution_audit.json', OUT/'plan.json', audit.CALIBRATION]
    sources += [p for ep, fail, manifest, track in WINDOWS for p in (manifest, track/f'episode_{ep}_measurements.jsonl', track/f'episode_{ep}_original_bank.npz', track/'summary.json')]
    if cpu_report is not None:
        sources.append(Path(cpu_report).resolve())
    hashes = {str(p): sha256(p) for p in set(sources+protected)}
    audit.write_json(run_dir/'preregistered_sources.json', dict(sha256=hashes, local_capture_surface_displacement_max_m=CAPTURE,
        normal_compatibility_abs_dot_min=.5, windows=[dict(episode=e, failure=f, frames=[f-3, f+3]) for e, f, _, _ in WINDOWS],
        synthetic_before_actual=True, independent_support_is_geometric_not_statistical=True))
    contract = json.loads((BASE/'camera_contract/nominal_contract.json').read_text())
    assert contract['evaluation_override']['computed_nominal_fx'] == 306
    assert sha256(reader.SOURCE) == reader.FRAME_SHA256
    early_audit = json.loads((BASE/'early_baseline_run02/execution_audit.json').read_text())
    assert sha256(WINDOWS[0][2]) == early_audit['source_checks']['frames_jsonl_sha256']
    if cpu_report is None:
        tests = synthetic.run(OUT/'cpu_tests')
    else:
        tests = json.loads(Path(cpu_report).read_text())
        for path, digest in tests['source_sha256'].items():
            assert sha256(path) == digest, 'CPU-tested source changed: '+path
    print(json.dumps(dict(stage='cpu_counterexamples', passed=tests['passed'], failed=tests['failed'],
                         reused=cpu_report is not None)), flush=True)
    require_cpu_pass(tests)  # No real sensor payload, masks or registrations before this gate.
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
            fit_ids, held_ids = partition(cloud)
            assert not np.intersect1d(cloud['raw_stencil_indices'][fit_ids], cloud['raw_stencil_indices'][held_ids]).size
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
        holdout_protocol=HOLDOUT_PROTOCOL, cpu_report_reused=cpu_report is not None,
        raw_fit_hold_depth_stencils_disjoint=True, external_mask_and_calibration_independence_not_claimed=True,
        concurrently_owned_files_outside_check=['harness/policy_server.py'],
        part_identity='part_identity_unknown', physical_success=None, cumulative_error_bound_m=None, resources=audit.resources(),
        next_step='If geometry candidates survive CPU truth counterexamples: complete continuous tracking and independent relation validation. Otherwise obtain the specific missing independent surface directions/held-out overlap/body-link mapping and instance scale; no further LK patches or threshold sweep.')
    audit.write_json(OUT/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-report', type=Path)
    args = parser.parse_args()
    OUT = args.out.resolve()
    try:
        value = run(OUT, args.cpu_report)
        print(json.dumps({k:v for k,v in value.items() if k != 'results'}), flush=True)
    except BaseException as error:
        if OUT.exists() and not (OUT/'failure.json').exists():
            audit.write_json(OUT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), resources=audit.resources()))
        raise
