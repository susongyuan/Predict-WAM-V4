"""Task-neutral, unadmitted RGB-D component registration; no tracking or grasp gate.

Coordinates are Camera's base convention. Returned transform maps source points
into current base coordinates, not a world-motion measurement without base odometry.
No interpolated depth, predicted point, robot pose or held-out surface enters ICP.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

TOL, SPACING, CAPTURE = .01, .02, .10
MIN_FIT, MIN_HOLD, RESPONSE = 12, 6, .1
HOLDOUT_PROTOCOL = 'disjoint_raw_pixel_lattices_v2'


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
    vv, uu = np.indices(depth.shape)
    fit_phase = (vv % 2 == 0) & (uu % 2 == 0)
    hold_phase = (vv % 2 == 1) & (uu % 2 == 1)
    keep = valid & (fit_phase | hold_phase)
    keep[:2, :] = keep[-2:, :] = False
    keep[:, :2] = keep[:, -2:] = False
    # The entire normal/depth-validity stencil stays in one raw-pixel fold.
    # Do not use a one-pixel derivative or prefilter across the other fold.
    for axis in (0, 1):
        for shift in (-2, 2):
            neighbour = np.roll(depth, shift, axis)
            keep &= np.roll(valid, shift, axis) & (np.abs(depth-neighbour) <= TOL)
    v, u = np.nonzero(keep)
    p = camera.backproject(u, v, depth[v, u])
    dx = camera.backproject(u+2, v, depth[v, u+2])-camera.backproject(u-2, v, depth[v, u-2])
    dy = camera.backproject(u, v+2, depth[v+2, u])-camera.backproject(u, v-2, depth[v-2, u])
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=1)
    good = np.isfinite(n).all(axis=1) & (length > 1e-12)
    centre_ids = v*depth.shape[1]+u
    stencils = np.stack([centre_ids, centre_ids-2, centre_ids+2,
                         centre_ids-2*depth.shape[1], centre_ids+2*depth.shape[1]], axis=1)
    return dict(points=p[good], normals=n[good]/length[good, None], uv=np.c_[u, v][good],
                holdout=hold_phase[v, u][good], raw_stencil_indices=stencils[good],
                valid_mask=valid, stencil_mask=keep, candidate_mask=candidate, robot_mask=robot_mask)


def partition(cloud):
    """Both folds cover the image; membership does not use any depth value or pose."""
    held = np.asarray(cloud['holdout'])
    if held.dtype != bool or held.shape != (len(cloud['points']),):
        raise ValueError('missing_or_invalid_raw_pixel_holdout')
    return np.flatnonzero(~held), np.flatnonzero(held)


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
    sf, sh = partition(source)
    tf, th = partition(current)
    sf, sh = sf[separated(source['points'][sf])], sh[separated(source['points'][sh])]
    trace.update(source_fit_indices=sf, source_hold_indices=sh, current_fit_indices=tf, current_hold_indices=th)
    result['holdout_protocol'] = HOLDOUT_PROTOCOL
    result['support_counts'] = dict(source_pixels=len(source['points']), current_pixels=len(current['points']),
                                  source_fit=len(sf), source_hold=len(sh), current_fit_pixels=len(tf), current_hold_pixels=len(th))
    result['source_surface_support'] = support_report(source['points'][sf], source['normals'][sf])
    result['source_holdout_support'] = support_report(source['points'][sh], source['normals'][sh])
    result['independence_scope'] = ('Disjoint raw-depth lattices and normal stencils within this fitter; '
        'external masks/calibration may share inputs. Not statistical independence, '
        'material identity or guaranteed overlap under occlusion/undersampling.')
    if len(sf) < MIN_FIT or not len(tf):
        result['reasons'] = ['insufficient_fit_surface_or_invalid_depth_scale']
        return result, trace
    p, n, q, qn = source['points'][sf], source['normals'][sf], current['points'][tf], current['normals'][tf]
    centre = p.mean(axis=0)  # Only fitted points may affect seeds or pose selection.
    tree = cKDTree(q)
    holdtree = cKDTree(current['points'][th]) if len(th) else None
    radius = max(SPACING, np.sqrt(np.mean(np.sum((p-centre)**2, axis=1))))
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
    def held_distances(candidate_pose):
        if holdtree is None or not len(sh):
            return np.empty(0), np.empty(0, dtype=int), None
        distances, matches = holdtree.query(transform(source['points'][sh], candidate_pose))
        return distances, matches, float(np.percentile(distances, 95))

    hd, hj, hold95 = held_distances(pose)
    trace.update(fit_source_points=p, fit_target_points=q[j], fit_distance_m=d, fit_inliers=inside,
                 fit_support_indices=ids, hold_distance_m=hd,
                 hold_target_points=current['points'][th][hj],
                 hypotheses=np.asarray([f[0] for f in fits]), fit_scores=np.asarray(scores),
                 fit_centre=centre)
    alternatives = []
    for i, f in enumerate(fits):
        coverage = float(np.mean(f[3] & (f[1] <= TOL)))
        _, _, alternative_hold95 = held_distances(f[0])
        displacement = float(np.sqrt(np.mean(np.sum((transform(p, f[0])-x)**2, axis=1))))
        alternatives.append(dict(index=i, fit_score=scores[i], coverage=coverage, hold_p95_m=alternative_hold95,
                                 surface_displacement_from_best_m=displacement, iterations=f[4]))
    if np.sqrt(np.mean(np.sum((x-p)**2, axis=1))) > CAPTURE:
        result['reasons'].append('outside_declared_local_capture_basin')
    if len(ids) < MIN_FIT or inside.mean() <= .5:
        result['reasons'].append('insufficient_majority_1cm_fit_support')
    if not observability['observable']:
        result['reasons'].append('normal_or_pose_rank_degenerate')
    if hold95 is None or len(sh) < MIN_HOLD or (hd <= TOL).sum() < MIN_HOLD:
        result['reasons'].append('insufficient_held_surface_support')
    elif hold95 > TOL:
        result['reasons'].append('held_surface_inconsistent_or_insufficient_sampling_overlap')
    if any(a['coverage'] > .5 and a['hold_p95_m'] is not None and a['hold_p95_m'] <= TOL
           and a['surface_displacement_from_best_m'] > 2*TOL for a in alternatives):
        result['reasons'].append('repeated_shape_or_ambiguous_transform')
    result.update(transform_source_to_current_base=pose.tolist(), selected_hypothesis=best, alternatives=alternatives,
                  observability=observability, fit_coverage=float(inside.mean()),
                  fit_inlier_rms_m=float(np.sqrt(np.mean(d[inside]**2))) if inside.any() else None,
                  fit_inlier_max_m=float(d[inside].max()) if inside.any() else None,
                  holdout=dict(count=len(hd), inliers_1cm=int((hd <= TOL).sum()), p95_m=hold95,
                               max_m=float(hd.max()) if len(hd) else None,
                               rms_m=float(np.sqrt(np.mean(hd**2))) if len(hd) else None))
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
