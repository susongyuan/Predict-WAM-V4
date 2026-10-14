#!/usr/bin/env python3
"""CPU truth controls and held-depth dependency tests; never physical admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from harness.object_perception import Camera
from harness.dense_rgbd_registration import measure, surface, partition, support_report, transform, TOL, RESPONSE, HOLDOUT_PROTOCOL
from harness.robot_self_geometry import sha256
from scripts import check_task090_rigid_train_v3 as audit


def analytic_depth(camera, pose, kind='asymmetric'):
    """Exact ray intersections. Balanced view sees three independent box normals."""
    v, u = np.indices((camera.H, camera.W))
    rays = camera.rays(u, v)
    if kind == 'sphere':
        origin = camera.t-np.array([0., 0., 1.])
        a, b = np.sum(rays*rays, axis=-1), rays@origin
        disc = b*b-a*(origin@origin-.24**2)
        z = (-b-np.sqrt(np.maximum(disc, 0)))/a
        return np.where((disc >= 0) & (z > 0), z, 0)
    if kind == 'balanced':
        # Viewing direction in object axes has three equal-magnitude components.
        base = Rotation.from_euler('xyz', [np.pi/4, -np.arctan(1/np.sqrt(2)), 0.]).as_matrix()
        size = np.array([.4, .36, .32])
    else:
        base = Rotation.from_rotvec([.3, -.5, .12]).as_matrix()
        size = np.array([.4, .32, .28])
    rotation = pose[:3, :3] @ base
    centre = transform(np.array([[0., 0., 1.]]), pose)[0]
    origin, direction = (camera.t-centre)@rotation, rays@rotation
    z = np.full((camera.H, camera.W), np.inf)
    boxes = [(np.zeros(3), size)]
    if kind in ('asymmetric', 'balanced'):
        boxes.append((np.array([.2, -.14, -.10]), np.array([.13, .09, .08])))
    for offset, size in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            a, b = (offset-size/2-origin)/direction, (offset+size/2-origin)/direction
        near, far = np.minimum(a, b).max(axis=-1), np.maximum(a, b).min(axis=-1)
        z = np.minimum(z, np.where((near > 0) & (far >= near), near, np.inf))
    return np.where(np.isfinite(z), z, 0)


def assert_raw_fold_separation(trace):
    for prefix in ('source', 'current'):
        fit = trace[prefix+'_fit_indices']
        held = trace[prefix+'_hold_indices']
        stencils = trace[prefix+'_raw_stencil_indices']
        assert not np.intersect1d(stencils[fit], stencils[held]).size
        uv = trace[prefix+'_uv']
        assert np.all(uv[fit] % 2 == 0) and np.all(uv[held] % 2 == 1)


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    camera = Camera(np.eye(3), np.zeros(3), 480, 480)
    truth = np.eye(4); truth[:3, :3] = Rotation.from_rotvec([.01, -.015, .005]).as_matrix()
    truth[:3, 3] = [.009, -.007, .006]
    source = analytic_depth(camera, np.eye(4))
    moved = analytic_depth(camera, truth)
    qualified = analytic_depth(camera, np.eye(4), 'balanced')
    qualified_moved = analytic_depth(camera, truth, 'balanced')
    moving_camera = Camera(Rotation.from_rotvec([.025, -.01, .015]).as_matrix(), np.array([.008, -.009, .004]), 480, 480)
    plane = np.full_like(source, .95)
    occluded = moved.copy(); hand = np.zeros_like(source, bool); hand[210:240, 175:205] = True; occluded[hand] = .4
    far = moved.copy(); far[far > 0] += .6
    # Original weak-view arrays are retained, not relabelled as successful positives.
    # They may reject; if accepted, truth error and unchanged response gates must pass.
    cases = [('legacy_weak_known_rigid', source, moved, camera, truth, 'safe'),
             ('legacy_weak_camera_motion', source, analytic_depth(moving_camera, np.eye(4)), moving_camera, np.eye(4), 'safe'),
             ('qualified_known_rigid', qualified, qualified_moved, camera, truth, 'candidate'),
             ('qualified_camera_motion', qualified, analytic_depth(moving_camera, np.eye(4), 'balanced'), moving_camera, np.eye(4), 'candidate'),
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
            passed = ((expectation == 'candidate' and candidate and error <= TOL) or
                      (expectation == 'unobservable' and not candidate) or
                      (expectation == 'safe' and (not candidate or error <= TOL)))
            if candidate:
                assert min(result['observability']['normal_response']) >= RESPONSE
                assert min(result['observability']['pose_response']) >= RESPONSE
            assert result['inherited_cumulative_error_m'] == .037 and result['cumulative_error_bound_m'] is None
            assert not any(k in result for k in ('verified', 'grasp', 'candidate_grasp'))
            assert_raw_fold_separation(trace)
            for prefix, depth in [('source', a), ('current', b)]:
                u, v = trace[prefix+'_uv'].T
                cam0 = camera if prefix == 'source' else cam
                np.testing.assert_allclose(trace[prefix+'_points'], cam0.backproject(u, v, depth[v, u]))
        except ValueError as exc:
            result, passed = dict(status='exception', reason=str(exc)), expectation == 'exception'
        except (AssertionError, KeyError) as exc:
            result, passed = dict(status='test_assertion_failed', reason=str(exc), measurement=result), False
        np.savez_compressed(out/(name+'.npz'), source_depth=a, current_depth=b, truth=expected,
                            source_camera_R=camera.R, source_camera_t=camera.t, current_camera_R=cam.R, current_camera_t=cam.t, **trace)
        rows.append(dict(case=name, expectation=expectation, passed=bool(passed), known_truth_surface_max_error_m=error, result=result))

    protocol = []
    zeros = np.zeros_like(qualified, bool)
    def measured(a, b):
        # Freeze external masks/calibration: test dependency isolation in this fitter.
        return measure(a, b, camera, camera, qualified > 0, qualified_moved > 0, zeros, zeros, depth_max=6.)
    baseline, base_trace = measured(qualified, qualified_moved)
    for where in ('source', 'current', 'both_missing'):
        a, b = qualified.copy(), qualified_moved.copy()
        if where in ('source', 'both_missing'):
            a[1::2, 1::2] = 0 if where == 'both_missing' else a[1::2, 1::2]+.03
        if where in ('current', 'both_missing'):
            b[1::2, 1::2] = 0 if where == 'both_missing' else b[1::2, 1::2]+.03
        try:
            result, trace = measured(a, b)
            assert 'hypotheses' in base_trace and 'hypotheses' in trace
            for key in ('hypotheses', 'fit_scores', 'fit_centre', 'fit_source_points', 'fit_target_points', 'fit_distance_m', 'fit_inliers'):
                np.testing.assert_array_equal(base_trace[key], trace[key])
            assert result['selected_hypothesis'] == baseline['selected_hypothesis']
            np.testing.assert_array_equal(result['transform_source_to_current_base'], baseline['transform_source_to_current_base'])
            assert_raw_fold_separation(trace)
            if where == 'both_missing':
                assert result['status'] == 'unobservable' and result['holdout']['p95_m'] is None
            protocol.append(dict(case='hold_only_'+where, passed=True, result=result))
            np.savez_compressed(out/('hold_only_'+where+'.npz'), source_depth=a, current_depth=b, **trace)
        except (AssertionError, KeyError, ValueError) as exc:
            protocol.append(dict(case='hold_only_'+where, passed=False, error=str(exc)))

    # The protocol must expose, not conceal, unresolved pixel-sampling ambiguity.
    coarse = Camera(np.eye(3), np.zeros(3), 120, 120); coarse.f = 100.
    flat = np.full((120, 120), 2.)
    cloud = surface(flat, coarse, np.ones_like(flat, bool), np.zeros_like(flat, bool), 6.)
    _, held = partition(cloud)
    interior = cloud['points'][held][10:-10]
    distance = cKDTree(cloud['points'][held]).query(interior+[.02, 0, 0])[0]
    protocol.append(dict(case='coarse_sampling_true_pose_can_be_inconclusive', passed=bool(np.percentile(distance, 95) > TOL),
        held_p95_m=float(np.percentile(distance, 95)), expectation='limitation, not a physical misalignment label'))
    points = np.random.default_rng(11).uniform(-.2, .2, (144, 3))
    normals = np.tile([0., 0., 1.], (144, 1)); normals[:2] = [[1., 0., 0.], [0., 1., 0.]]
    weak = support_report(points, normals)
    protocol.append(dict(case='low_normal_response_not_admitted', passed=not weak['observable'], result=weak))
    strip = np.zeros_like(qualified); strip[:, 240] = 1.
    result, _ = measure(strip, strip, camera, camera, strip > 0, strip > 0, zeros, zeros, depth_max=6.)
    protocol.append(dict(case='narrow_surface_missing_stencils_unknown', passed=result['status'] == 'unobservable', result=result))
    from scripts.check_dense_rgbd_failure_windows import require_cpu_pass
    blocked = 0
    for report in ({'failed': 1, 'passed': 12, 'leakage_checks_passed': True},
                   {'failed': 0, 'passed': 20, 'leakage_checks_passed': False}):
        try:
            require_cpu_pass(report)
        except RuntimeError:
            blocked += 1
    protocol.append(dict(case='failed_cpu_or_leakage_blocks_real_frames', passed=blocked == 2))
    rows.extend(protocol)
    # A long stencil must test plane coherence, not confuse slope with an edge.
    yy, xx = np.indices(source.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        slant = 2. / (1. + (xx-240)/204.)
    slant_mask = (xx >= 220) & (xx < 260) & (yy >= 220) & (yy < 260)
    slant = np.where(slant_mask, slant, 0.)
    slant_cloud = surface(slant, camera, slant_mask, zeros, 6.)
    protocol.append(dict(case='perspective_slant_preserves_measured_points',
        passed=len(slant_cloud['points']) >= 600,
        retained=len(slant_cloud['points']), expected_minimum=600))
    stepped = np.where(xx < 240, .95, 2.)
    step_cloud = surface(stepped, camera, np.ones_like(zeros), zeros, 6.)
    boundary = (step_cloud['uv'][:, 0] >= 238) & (step_cloud['uv'][:, 0] <= 241)
    protocol.append(dict(case='cross_depth_surface_stencil_rejected', passed=not boundary.any(),
        retained_boundary_centres=int(boundary.sum())))
    source_paths = [Path(__file__), ROOT/'harness/dense_rgbd_registration.py', ROOT/'scripts/check_dense_rgbd_failure_windows.py']
    rows.extend(protocol[-2:])
    report = dict(passed=sum(r['passed'] for r in rows), failed=sum(not r['passed'] for r in rows), cases=rows,
        leakage_checks_passed=all(r['passed'] for r in protocol if r['case'].startswith('hold_only_')),
        raw_stencil_checks_in_all_measurement_cases=True, holdout_protocol=HOLDOUT_PROTOCOL,
        old_results_unchanged=True, physical_admission=False,
        positive_control_note='New balanced analytic three-face controls are mandatory positives; legacy weak-view arrays remain safety controls, not claimed successes.',
        limitations=['Shared masks/calibration can carry external dependencies; not end-to-end statistical independence.',
                     'Pixel interleaving cannot guarantee common-surface overlap or resolve undersampling.'],
        resources=audit.resources(), source_sha256={str(p): sha256(p) for p in source_paths})
    audit.write_json(out/'results.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit.PROCESS.cpu_affinity([min(audit.PROCESS.cpu_affinity())])
    result = run(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    sys.exit(0 if result['failed'] == 0 else 1)
