#!/usr/bin/env python3
"""Single CPU analytic counterexample suite; saves failures, never retunes gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from harness.object_perception import Camera
from harness.dense_rgbd_registration import measure, transform
from harness.robot_self_geometry import sha256
from scripts import check_task090_rigid_train_v3 as audit


def analytic_depth(camera, pose, kind='asymmetric'):
    """Exact ray/box or ray/sphere intersections, not a simulator/scene replay."""
    v, u = np.indices((camera.H, camera.W))
    rays = camera.rays(u, v)
    if kind == 'sphere':
        origin = camera.t-np.array([0., 0., 1.])
        a, b = np.sum(rays*rays, axis=-1), rays@origin
        disc = b*b-a*(origin@origin-.24**2)
        z = (-b-np.sqrt(np.maximum(disc, 0)))/a
        return np.where((disc >= 0) & (z > 0), z, 0)
    base = Rotation.from_rotvec([.3, -.5, .12]).as_matrix()
    rotation = pose[:3, :3] @ base
    centre = transform(np.array([[0., 0., 1.]]), pose)[0]
    origin, direction = (camera.t-centre)@rotation, rays@rotation
    z = np.full((camera.H, camera.W), np.inf)
    boxes = [(np.zeros(3), np.array([.4, .32, .28]))]
    if kind == 'asymmetric':
        boxes.append((np.array([.2, -.14, -.10]), np.array([.13, .09, .08])))
    for offset, size in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            a, b = (offset-size/2-origin)/direction, (offset+size/2-origin)/direction
        near, far = np.minimum(a, b).max(axis=-1), np.maximum(a, b).min(axis=-1)
        z = np.minimum(z, np.where((near > 0) & (far >= near), near, np.inf))
    return np.where(np.isfinite(z), z, 0)


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    camera = Camera(np.eye(3), np.zeros(3), 480, 480)
    truth = np.eye(4); truth[:3, :3] = Rotation.from_rotvec([.01, -.015, .005]).as_matrix()
    truth[:3, 3] = [.009, -.007, .006]
    source = analytic_depth(camera, np.eye(4))
    moved = analytic_depth(camera, truth)
    moving_camera = Camera(Rotation.from_rotvec([.025, -.01, .015]).as_matrix(), np.array([.008, -.009, .004]), 480, 480)
    plane = np.full_like(source, .95)
    occluded = moved.copy(); hand = np.zeros_like(source, bool); hand[210:240, 175:205] = True; occluded[hand] = .4
    far = moved.copy(); far[far > 0] += .6
    cases = [('known_rigid', source, moved, camera, truth, 'candidate'),
             ('camera_motion', source, analytic_depth(moving_camera, np.eye(4)), moving_camera, np.eye(4), 'candidate'),
             ('hand_occlusion', source, occluded, camera, truth, 'safe'),
             ('depth_edge', source, np.where(np.indices(source.shape)[1] < 200, .4, moved), camera, truth, 'safe'),
             ('single_plane', plane, plane, camera, np.eye(4), 'unobservable'),
             ('pure_background', plane, plane+.003, camera, np.eye(4), 'unobservable'),
             ('repeated_symmetric', analytic_depth(camera, np.eye(4), 'sphere'), analytic_depth(camera, np.eye(4), 'sphere'), camera, np.eye(4), 'unobservable'),
             ('wrong_target', source, plane, camera, np.eye(4), 'unobservable'),
             ('no_overlap', source, far, camera, np.eye(4), 'unobservable'),
             ('wrong_numeric_units', source*1000, moved*1000, camera, truth, 'unobservable'),
             ('wrong_declared_units', source, moved, camera, truth, 'exception')]
    rows = []
    for name, a, b, cam, expected, expectation in cases:
        audit.guard()
        trace, error = {}, None
        try:
            result, trace = measure(a, b, camera, cam, a > 0, b > 0, np.zeros_like(a, bool),
                                    hand if name == 'hand_occlusion' else np.zeros_like(b, bool), depth_max=6.,
                                    unit='mm' if expectation == 'exception' else 'm', inherited_error_m=.037)
            pose = result['transform_source_to_current_base']
            if pose is not None and len(trace.get('source_points', [])):
                points = trace['source_points']
                error = float(np.max(np.linalg.norm(transform(points, np.asarray(pose))-transform(points, expected), axis=1)))
            candidate = result['status'] == 'feasible_measurement_candidate'
            passed = ((expectation == 'candidate' and candidate and error <= .01) or
                      (expectation == 'unobservable' and not candidate) or
                      (expectation == 'safe' and (not candidate or error <= .01)))
            assert result['inherited_cumulative_error_m'] == .037 and result['cumulative_error_bound_m'] is None
            assert not any(k in result for k in ('verified', 'grasp', 'candidate_grasp'))
            for prefix, depth in [('source', a), ('current', b)]:
                u, v = trace[prefix+'_uv'].T
                cam0 = camera if prefix == 'source' else cam
                np.testing.assert_allclose(trace[prefix+'_points'], cam0.backproject(u, v, depth[v, u]))
        except ValueError as exc:
            result, passed = dict(status='exception', reason=str(exc)), expectation == 'exception'
        except AssertionError as exc:
            result, passed = dict(status='test_assertion_failed', reason=str(exc), measurement=result), False
        np.savez_compressed(out/(name+'.npz'), source_depth=a, current_depth=b, truth=expected,
                            source_camera_R=camera.R, source_camera_t=camera.t, current_camera_R=cam.R, current_camera_t=cam.t, **trace)
        rows.append(dict(case=name, expectation=expectation, passed=bool(passed), known_truth_surface_max_error_m=error, result=result))
    report = dict(passed=sum(r['passed'] for r in rows), failed=sum(not r['passed'] for r in rows), cases=rows,
                  resources=audit.resources(), source_sha256={str(p): sha256(p) for p in [Path(__file__), ROOT/'harness/dense_rgbd_registration.py']})
    audit.write_json(out/'results.json', report)
    return report


if __name__ == '__main__':
    audit.PROCESS.cpu_affinity([min(audit.PROCESS.cpu_affinity())])
    print(run(ROOT/'outputs/validation/task090_completion_20261003_0156/dense_body_measurement/cpu_tests'))
