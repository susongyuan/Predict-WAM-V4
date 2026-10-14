"""Task-neutral, unadmitted RGB-D component registration; no tracking or grasp gate.

Coordinates are Camera's base convention. Returned transform maps source points
into current base coordinates, not a world-motion measurement without base odometry.
No interpolated depth, predicted point, robot pose or held-out surface enters ICP.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

TOL, SPACING, BLOCK, CAPTURE = .01, .02, .08, .10
MIN_FIT, MIN_HOLD, RESPONSE = 12, 6, .1


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def separated(points):
    """Actual samples >=2cm apart; neither voxel centroids nor independent pixels."""
    if not len(points):
        return np.empty(0, int)
    tree, used, indices = cKDTree(points), np.zeros(len(points), bool), []
    for i in range(len(points)):
        if not used[i]:
            indices.append(i)
            used[tree.query_ball_point(points[i], SPACING - 1e-12)] = True
    return np.asarray(indices)


def surface(depth, camera, candidate, robot_mask, depth_max, unit='m'):
    if unit != 'm':
        raise ValueError('depth_units_not_metres')
    depth, candidate, robot_mask = np.asarray(depth), np.asarray(candidate), np.asarray(robot_mask)
    if (depth.ndim != 2 or depth.shape != (camera.H, camera.W) or
            candidate.shape != depth.shape or robot_mask.shape != depth.shape or
            candidate.dtype != bool or robot_mask.dtype != bool):
        raise ValueError('unaligned_depth_or_mask')
    if (not np.isfinite(camera.R).all() or not np.isfinite(camera.t).all() or
            not np.allclose(camera.R.T @ camera.R, np.eye(3), atol=1e-6) or
            abs(np.linalg.det(camera.R)-1) > 1e-6 or camera.f <= 0):
        raise ValueError('invalid_camera')
    valid = candidate & ~robot_mask & np.isfinite(depth) & (depth > .05) & (depth < depth_max)
    keep = valid.copy()
    keep[[0, -1], :] = False
    keep[:, [0, -1]] = False
    for axis in (0, 1):
        for shift in (-1, 1):
            neighbour = np.roll(depth, shift, axis)
            keep &= np.roll(valid, shift, axis) & (np.abs(depth-neighbour) <= TOL)
    v, u = np.nonzero(keep)
    p = camera.backproject(u, v, depth[v, u])
    dx = camera.backproject(u+1, v, depth[v, u+1])-camera.backproject(u-1, v, depth[v, u-1])
    dy = camera.backproject(u, v+1, depth[v+1, u])-camera.backproject(u, v-1, depth[v-1, u])
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=1)
    good = np.isfinite(n).all(axis=1) & (length > 1e-12)
    return dict(points=p[good], normals=n[good]/length[good, None], uv=np.c_[u, v][good],
                valid_mask=valid, stencil_mask=keep, candidate_mask=candidate, robot_mask=robot_mask)


def partition(cloud, centre, basis):
    p = cloud['points']
    cells = np.floor(((p-centre) @ basis)/BLOCK).astype(int)
    held = (cells.sum(axis=1) % 4) == 0
    fit = ~held
    # Do not let fitted target points abut or include a held-out target region.
    if held.any() and fit.any():
        fit[fit] &= cKDTree(p[held]).query(p[fit])[0] >= SPACING
    return np.flatnonzero(fit), np.flatnonzero(held)


def support_report(points, normals):
    centre = points.mean(axis=0) if len(points) else np.zeros(3)
    radius = float(np.sqrt(np.mean(np.sum((points-centre)**2, axis=1)))) if len(points) else 0.
    if len(points) < 6 or radius < SPACING:
        return dict(count=len(points), radius_m=radius, normal_rank=0, pose_rank=0, observable=False)
    jacobian = np.c_[np.cross((points-centre)/radius, normals), normals]
    s = np.linalg.svd(jacobian, compute_uv=False)/np.sqrt(len(points))
    ns = np.linalg.svd(normals, compute_uv=False)/np.sqrt(len(points))
    return dict(count=len(points), radius_m=radius, span_m=np.ptp(points, axis=0).tolist(),
                normal_response=ns.tolist(), pose_response=s.tolist(), normal_rank=int((ns >= RESPONSE).sum()),
                pose_rank=int((s >= RESPONSE).sum()), weakest_twist=np.linalg.svd(jacobian, full_matrices=False)[2][-1].tolist(),
                observable=bool(ns[-1] >= RESPONSE and s[-1] >= RESPONSE))


def correspond(p, n, target, target_n, tree):
    d, j = tree.query(p)
    good = (d <= CAPTURE) & (np.abs(np.sum(n*target_n[j], axis=1)) >= .5)
    return d, j, good


def icp(p, n, q, qn, tree, seed):
    pose = seed.copy()
    for iteration in range(30):
        x, nx = transform(p, pose), n @ pose[:3, :3].T
        d, j, good = correspond(x, nx, q, qn, tree)
        if good.sum() < 6:
            break
        a, b, normal = x[good], q[j[good]], qn[j[good]]
        centre = a.mean(axis=0)
        residual = np.sum((a-b)*normal, axis=1)
        weight = np.sqrt(np.minimum(1., TOL/np.maximum(np.abs(residual), 1e-12)))
        jacobian = np.c_[np.cross(a-centre, normal), normal]
        delta = np.linalg.lstsq(jacobian*weight[:, None], -residual*weight, rcond=1e-3)[0]
        delta[:3] *= min(1., .2/max(np.linalg.norm(delta[:3]), 1e-12))
        delta[3:] *= min(1., .05/max(np.linalg.norm(delta[3:]), 1e-12))
        r = Rotation.from_rotvec(delta[:3]).as_matrix()
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = r, centre-r@centre+delta[3:]
        pose = step @ pose
        if np.linalg.norm(delta) < 1e-6:
            break
    x = transform(p, pose)
    d, j, good = correspond(x, n @ pose[:3, :3].T, q, qn, tree)
    return pose, d, j, good, iteration+1


def register(source, current, *, inherited_error_m=None, static_component_model=None):
    """Clouds come from surface(); no initial pose parameter or success output.

    Finite ambiguity probes do not prove global uniqueness. Unknown body/link
    mapping stays unknown even with a supplied static model; no instance fitting.
    """
    result = dict(status='unobservable', reasons=[], component='component_candidate', part_identity='part_identity_unknown',
                  static_model_supplied=static_component_model is not None, transform_source_to_current_base=None,
                  inherited_cumulative_error_m=inherited_error_m, cumulative_error_bound_m=None,
                  holdout_is_truth_bound=False, global_uniqueness_certified=False)
    trace = {}
    if min(len(source['points']), len(current['points'])) < MIN_FIT+MIN_HOLD:
        result['reasons'] = ['insufficient_surface_or_invalid_depth_scale']
        return result, trace
    centre = source['points'].mean(axis=0)
    basis = np.linalg.svd(source['points']-centre, full_matrices=False)[2].T
    sf, sh = partition(source, centre, basis)
    tf, th = partition(current, centre, basis)
    sf, sh = sf[separated(source['points'][sf])], sh[separated(source['points'][sh])]
    trace.update(source_fit_indices=sf, source_hold_indices=sh, current_fit_indices=tf, current_hold_indices=th,
                 partition_centre=centre, partition_basis=basis)
    result['support_counts'] = dict(source_pixels=len(source['points']), current_pixels=len(current['points']),
                                  source_fit=len(sf), source_hold=len(sh), current_fit_pixels=len(tf), current_hold_pixels=len(th))
    result['source_surface_support'] = support_report(source['points'][sf], source['normals'][sf])
    result['source_holdout_support'] = support_report(source['points'][sh], source['normals'][sh])
    result['independence_scope'] = '2cm geometric separation only; sensor correlations and target ownership unknown'
    if len(sf) < MIN_FIT or len(sh) < MIN_HOLD or not len(tf) or not len(th):
        result['reasons'] = ['insufficient_spatially_separated_fit_or_holdout']
        return result, trace
    p, n, q, qn = source['points'][sf], source['normals'][sf], current['points'][tf], current['normals'][tf]
    tree, holdtree = cKDTree(q), cKDTree(current['points'][th])
    radius = max(SPACING, np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2, axis=1))))
    seeds = [np.eye(4), np.eye(4)]
    seeds[1][:3, 3] = np.median(q, axis=0)-np.median(p, axis=0)
    for axis in np.eye(3):
        for sign in (-1, 1):
            seed = np.eye(4); seed[:3, 3] = sign*CAPTURE*axis; seeds.append(seed)
        for angle in (-CAPTURE/radius, CAPTURE/radius, np.pi):
            seed = np.eye(4); seed[:3, :3] = Rotation.from_rotvec(angle*axis).as_matrix()
            seed[:3, 3] = centre-seed[:3, :3]@centre; seeds.append(seed)
    fits = [icp(p, n, q, qn, tree, seed) for seed in seeds]
    scores = [float(np.mean(np.minimum(np.where(f[3], f[1], CAPTURE), CAPTURE)**2)) for f in fits]
    best = int(np.argmin(scores))  # Holdout never selects/refines the pose.
    pose, d, j, good, iterations = fits[best]
    inside = good & (d <= TOL)
    ids = np.flatnonzero(inside)
    ids = ids[separated(q[j[ids]])]
    x = transform(p, pose)
    observability = support_report(x[ids], qn[j[ids]])
    hd, hj = holdtree.query(transform(source['points'][sh], pose))
    trace.update(fit_source_points=p, fit_target_points=q[j], fit_distance_m=d, fit_inliers=inside,
                 fit_support_indices=ids, hold_distance_m=hd, hold_target_points=current['points'][th][hj],
                 hypotheses=np.asarray([f[0] for f in fits]))
    alternatives = []
    for i, f in enumerate(fits):
        coverage = float(np.mean(f[3] & (f[1] <= TOL)))
        hold95 = float(np.percentile(holdtree.query(transform(source['points'][sh], f[0]))[0], 95))
        displacement = float(np.sqrt(np.mean(np.sum((transform(p, f[0])-x)**2, axis=1))))
        alternatives.append(dict(index=i, fit_score=scores[i], coverage=coverage, hold_p95_m=hold95,
                                 surface_displacement_from_best_m=displacement, iterations=f[4]))
    if np.sqrt(np.mean(np.sum((x-p)**2, axis=1))) > CAPTURE:
        result['reasons'].append('outside_declared_local_capture_basin')
    if len(ids) < MIN_FIT or inside.mean() <= .5:
        result['reasons'].append('insufficient_majority_1cm_fit_support')
    if not observability['observable']:
        result['reasons'].append('normal_or_pose_rank_degenerate')
    if np.percentile(hd, 95) > TOL or (hd <= TOL).sum() < MIN_HOLD:
        result['reasons'].append('spatial_holdout_inconsistent_or_no_overlap')
    if any(a['coverage'] > .5 and a['hold_p95_m'] <= TOL and a['surface_displacement_from_best_m'] > 2*TOL for a in alternatives):
        result['reasons'].append('repeated_shape_or_ambiguous_transform')
    result.update(transform_source_to_current_base=pose.tolist(), selected_hypothesis=best, alternatives=alternatives,
                  observability=observability, fit_coverage=float(inside.mean()),
                  fit_inlier_rms_m=float(np.sqrt(np.mean(d[inside]**2))) if inside.any() else None,
                  fit_inlier_max_m=float(d[inside].max()) if inside.any() else None,
                  holdout=dict(count=len(hd), inliers_1cm=int((hd <= TOL).sum()), p95_m=float(np.percentile(hd, 95)),
                               max_m=float(hd.max()), rms_m=float(np.sqrt(np.mean(hd**2)))))
    if not result['reasons']:
        result['status'] = 'feasible_measurement_candidate'
    return result, trace


def measure(source_depth, current_depth, source_camera, current_camera, source_mask, current_mask,
            source_robot_mask, current_robot_mask, *, depth_max, unit='m', **kwargs):
    source = surface(source_depth, source_camera, source_mask, source_robot_mask, depth_max, unit)
    current = surface(current_depth, current_camera, current_mask, current_robot_mask, depth_max, unit)
    result, trace = register(source, current, **kwargs)
    trace.update({prefix+'_'+key: value for prefix, cloud in [('source', source), ('current', current)] for key, value in cloud.items()})
    return result, trace
