#!/usr/bin/env python3
"""Validate the extracted static geometry; no pxr, simulation, FK replay, or frame I/O."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[n] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import numpy as np
import yaml

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/user/Robotic/project2')
START = time.monotonic()
e = json.loads((OUT/'extraction.json').read_text())
YAML = ROOT/'data/omnigibson/omnigibson-robot-assets/models/r1pro/r1pro.yaml'
robot = yaml.safe_load(YAML.read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, value, compact=False):
    with (OUT/name).open('x') as f:
        json.dump(value, f, indent=None if compact else 2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def angle(R):
    v = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]
    return float(np.arctan2(np.linalg.norm(v)/2, (np.trace(R)-1)/2))


def errors(A, B):
    return dict(position_m=float(np.linalg.norm(A[:3,3]-B[:3,3])),
                rotation_rad=angle(A[:3,:3]@B[:3,:3].T))


def motion(kind, axis, q):
    M = np.eye(4)
    if kind == 'prismatic':
        M[:3,3] = axis*q
    elif kind == 'revolute':
        x,y,z = axis
        K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        M[:3,:3] = np.eye(3) + np.sin(q)*K + (1-np.cos(q))*(K@K)
    return M


assert sha(OUT/'collision_vertices.npz') == e['arrays_sha256']
npz = np.load(OUT/'collision_vertices.npz', allow_pickle=False)
shapes, counts, serialization_checks = {}, {}, []
for name, link in e['links'].items():
    shapes[name] = []
    n = 0
    for s in link['shapes']:
        raw = npz[s['id']+'__points_mesh']
        points = npz[s['npz_key_points']]
        T = np.asarray(s['T_link_mesh_column'])
        derived = raw@T[:3,:3].T + T[:3,3]
        assert np.max(np.abs(derived-points)) < 1e-12
        b = dict(id=s['id'], source_prim=s['prim_path'], representation='link_local_aabb',
                 min_m=s['aabb_min_m'], max_m=s['aabb_max_m'], input_vertex_count=s['vertex_count'])
        # Check the actual downstream JSON number representation, not only in-memory bounds.
        b = json.loads(json.dumps(b, allow_nan=False))
        lo, hi = np.asarray(b['min_m']), np.asarray(b['max_m'])
        assert np.all((points >= lo) & (points <= hi))
        assert np.all(lo < hi)
        assert s['containment']['all_inside'] and s['containment']['outside_count'] == 0
        n += len(points)
        shapes[name].append(b)
        serialization_checks.append(dict(id=s['id'], checked=len(points), all_inside=True))
    counts[name] = dict(collision_meshes=len(shapes[name]), vertices=n,
        union_dimensions_m=(np.asarray(link['union_aabb_max_m'])-link['union_aabb_min_m']).tolist())
assert len(shapes) == 12 and sum(c['vertices'] for c in counts.values()) == 4852

state = e['prior_fk_evidence']['state_mapping']
qmap = {}
for group, names, unit in [('trunk_qpos', robot['articulated_trunk']['trunk_joint_names'], 'rad')]:
    start, end = state[group]
    assert end-start == len(names)
    qmap.update({n: dict(state61_index=i, unit=unit) for i,n in enumerate(names, start)})
for arm in ('left', 'right'):
    assert robot['manipulation']['eef_link_names'][arm] == arm+'_eef_link'
    for group, field, unit in [('arm_qpos', 'arm_joint_names', 'rad'), ('finger_qpos', 'finger_joint_names', 'm')]:
        start, end = state[arm+'_'+group]
        names = robot['manipulation'][field][arm]
        assert end-start == len(names)
        qmap.update({n: dict(state61_index=i, unit=unit) for i,n in enumerate(names, start)})

joint_checks, runtime_joints, differences, eef_checks = {}, {}, [], {}
for name, j in e['joints'].items():
    J0, J1, T = [np.asarray(j[k]) for k in (
        'T_parent_joint0_column', 'T_child_joint1_column', 'T_parent_child_at_q0_column')]
    assert np.max(np.abs(T-J0@np.linalg.inv(J1))) < 1e-12
    u = j['urdf']
    kind = {'PhysicsFixedJoint':'fixed', 'PhysicsPrismaticJoint':'prismatic', 'PhysicsRevoluteJoint':'revolute'}[j['usd_type']]
    token = j['usd_axis_token']
    axis_joint = np.eye(3)['XYZ'.index(token)] if token else np.zeros(3)
    axis_child = J1[:3,:3]@axis_joint
    axis_parent = J0[:3,:3]@axis_joint
    attrs = j['usd_authored_attributes']
    usd_limits = None if kind == 'fixed' else [attrs['physics:lowerLimit'], attrs['physics:upperLimit']]
    limits_si = None if usd_limits is None else (np.deg2rad(usd_limits).tolist() if kind == 'revolute' else usd_limits)
    runtime_joints[name] = dict(parent=j['parent'], child=j['child'], type=kind,
        T_parent_joint0=j['T_parent_joint0_column'], T_child_joint1=j['T_child_joint1_column'],
        axis_joint=axis_joint.tolist() if token else None,
        limits_si=limits_si, qpos=qmap.get(name), source_prim=j['prim_path'])
    if u is None:
        arm = name.split('_')[0]
        old = e['existing_eef_offsets'][arm]
        oldT = np.eye(4)
        oldT[:3,:3], oldT[:3,3] = old['eef_R'], old['eef_xyz']
        err = errors(T, oldT)
        assert err['position_m'] < 1e-6 and err['rotation_rad'] < 2e-6
        eef_checks[arm] = dict(parent=j['parent'], child=j['child'],
            T_gripper_eef=T.tolist(), nominal_translation_m=[0,0,-0.06], nominal_quaternion_wxyz=[0,0,1,0],
            existing_offset_difference=err, action='keep_existing_calibration_no_refit',
            warning='EEF is not the gripper frame: omitting fixed offset causes about 6cm and 180deg error.')
        continue
    assert kind == u['type'] and j['parent'] == u['parent'] and j['child'] == u['child']
    U = np.eye(4)
    U[:3,:3], U[:3,3] = rpy(*u['origin_rpy_rad']), u['origin_xyz_m']
    err = errors(T, U)
    assert err['position_m'] < 1e-7 and err['rotation_rad'] < 1e-6, (name, err)
    axis_err = None if kind == 'fixed' else float(np.linalg.norm(axis_child-u['axis']))
    assert axis_err is None or axis_err < 1e-6
    # Two endpoints and the midpoint check the local joint formula, NOT recorded-frame FK.
    trial_errors = []
    if kind != 'fixed':
        for q in [limits_si[0], sum(limits_si)/2, limits_si[1]]:
            trial_errors.append(errors(J0@motion(kind, axis_joint, q)@np.linalg.inv(J1),
                                       U@motion(kind, np.asarray(u['axis']), q)))
        assert max(d['position_m'] for d in trial_errors) < 1e-7
        assert max(d['rotation_rad'] for d in trial_errors) < 1e-6
    limit_delta = None if kind == 'fixed' else (np.asarray(limits_si)-[u['limit']['lower'],u['limit']['upper']]).tolist()
    check = dict(parent_child_match=True, type_match=True, urdf_origin_xyz_m=u['origin_xyz_m'],
        urdf_origin_rpy_rad=u['origin_rpy_rad'], usd_zero_frame_error=err,
        usd_axis_token=token, usd_axis_parent=axis_parent.tolist() if token else None,
        usd_axis_child=axis_child.tolist() if token else None, urdf_axis=u['axis'], axis_error=axis_err,
        usd_limits_authored=usd_limits, usd_limits_si=limits_si, urdf_limits=u['limit'],
        usd_minus_urdf_limits_si=limit_delta, qpos_binding=qmap.get(name), local_joint_probe_errors=trial_errors,
        authored_usd_qpos_state={k:v for k,v in attrs.items() if k.startswith('state:')})
    if 'finger_joint' in name:
        check['qpos_equivalence'] = 'q_USD_m = q_URDF_m; positive opens each finger along its own outward axis; no sign flip or offset.'
        check['representation_difference'] = ('USD Y with localRot0=localRot1=(0,1,0,0) wxyz maps to URDF -Y; both joint frames must be applied.'
            if name.endswith('2') else 'USD Y with identity joint frames matches URDF +Y.')
        check['physical_kinematic_difference_beyond_float_rounding'] = False
        check['proprio_sample_validation'] = 'NOT_REPEATED_OR_NEWLY_TESTED; slices reused from prior preflight, joint order from robot YAML.'
    if limit_delta is not None and np.max(np.abs(limit_delta)) > 1e-6:
        differences.append(dict(joint=name, field='limits', usd_minus_urdf_si=limit_delta,
            treatment='Retain both; runtime JSON gives USD asset limits. Do not silently clamp qpos.'))
    joint_checks[name] = check
assert len(joint_checks) == 28 and len(eef_checks) == 2

unknown = [
    'Authoritative source is this static USD, not byte-equivalence to missing URDF OBJ meshes; USD collision scales/components differ and no URDF scale is applied.',
    'Boxes contain every source vertex and hence each mathematical convex hull, but are not exact mesh surfaces: over-coverage can mask nearby object points.',
    'No PhysX cooking/contact/rest-offset envelope was simulated; cooked hull inflation, runtime asset overrides and contact skins remain unverified.',
    'Only the 12 named distal robot links are modeled; no held object, arm1-5, torso, base, environment or other object is included.',
    'Finger measured-qpos placement was not checked on recorded frames. Static USD and URDF finger joint kinematics agree; prior preflight checks EEF, not fingers.',
    'State61 slices are reused from preflight, names/order from r1pro.yaml. Runtime must bind named proprio correctly and reject unknown, stale, nonfinite or unsupported states rather than guess.',
    'Use metric rigid link poses. Runtime root scaling or deforming geometry is unsupported unless separately reconciled.',
    'Numeric box padding is 1e-9 m only, not a sensor/FK uncertainty allowance; downstream geometric/sensor uncertainty policy needs its own validation.',
    'No self-filter implementation/integration, 832-frame tracking, physical grasp verification or completion admission is performed.'
]
semantics = dict(
    matrices='4x4 nested row arrays acting on homogeneous COLUMN vectors; translation is matrix[:3,3].',
    extraction='Gf uses row vectors: p_link_row = p_mesh_row * ComputeRelativeTransform(mesh,link). Exported matrices are its transpose; all ancestor xform ops, local translation/orientation/scale and ordering are composed once.',
    geometry='82 separate link-local AABBs, one per explicit active PhysicsCollisionAPI+PhysicsMeshCollisionAPI convexHull mesh. Union of boxes is the conservative proxy; do not replace by intersection or shrink bounds.',
    units='All geometry and prismatic q are meters. Runtime revolute q is radians; USD authored angular limits are converted degrees->radians.',
    joint_formula='T_parent_child(q) = T_parent_joint0 @ Motion(axis_joint,q) @ inverse(T_child_joint1); Motion is rotation for revolute, translation for prismatic, identity for fixed.',
    pose_application='p_base = T_base_link @ [p_link,1]; membership: inverse(T_base_link) maps query into link frame, then any shape satisfies min_m <= xyz <= max_m on all axes.',
    pose_source='Current named proprio/FK only, not static USD placement. Existing URDF chain is locally equivalent within recorded rounding errors; use existing root/eef calibration without refitting.',
    eef_anchor='With calibrated measured EEF pose, T_base_gripper = T_base_eef @ inverse(T_gripper_eef_existing). Do not apply EEF offset again to geometry already in gripper frame.',
    root_offset='For existing URDF FK, add root_xyz from the unchanged calibration once in base coordinates. Do not add it again when anchoring directly from measured EEF.',
    hull_containment='A box is convex and contains every collision input vertex, therefore it contains every convex combination (the full convex hull); 1nm outward padding and nextafter prohibit shrinkage.',
    finger_axis='For joint2 on BOTH arms, USD token Y is in joint frame; localRot0 and localRot1 are 180deg about X, producing -Y in gripper/child. q remains positive opening, no sign flip.',
    state_scope='A standalone robot asset definition, not hidden environment GT. No world/object stage, live state, RGB-D, frame manifest or held-object geometry was read.')
sources = dict(e['source_files'])
for p in (YAML, OUT/'extraction.json', OUT/'collision_vertices.npz', Path(__file__)):
    sources[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
geometry = dict(schema='r1pro_usd_collision_aabb_v1', status='static_geometry_validated_runtime_integration_not_tested',
    source_usd_sha256=next(v['sha256'] for k,v in sources.items() if k.endswith('/r1pro.usda')),
    link_count=12, collision_mesh_count=82, input_vertex_count=4852, includes_held_object=False,
    units=e['units'], semantics=semantics, links=shapes, joints=runtime_joints,
    state61_mapping=state, existing_calibration=e['existing_eef_offsets'],
    eef_static_checks=eef_checks, limitations=unknown)
write('geometry.json', geometry, compact=True)
# Re-read the final deliverable, verifying every box after disk JSON serialization.
saved = json.loads((OUT/'geometry.json').read_text())
for boxes in saved['links'].values():
    for b in boxes:
        v = npz[b['id']+'__points_link_m']
        assert np.all((v >= np.asarray(b['min_m'])) & (v <= np.asarray(b['max_m'])))
prior = e['prior_fk_evidence']['independent_train_fk']
validation = dict(status='PASS_STATIC_USD_COLLISION_EXPORT', runtime_integration='NOT_TESTED',
    geometry_json=str(OUT/'geometry.json'), geometry_sha256=sha(OUT/'geometry.json'),
    sources=sources, counts=counts, all_vertices=dict(count=4852, collision_meshes=82, outside=0,
        json_roundtrip_all_inside=True, gf_transform_max_error_m=max(s['containment']['max_gf_transform_error_m'] for l in e['links'].values() for s in l['shapes']),
        mathematical_convex_hulls_contained=True, containment_checks=serialization_checks),
    tolerances=dict(local_joint_origin_position_m=1e-7, local_joint_rotation_rad=1e-6, local_joint_axis_norm_error=1e-6,
                    eef_existing_offset_position_m=1e-6, eef_existing_offset_rotation_rad=2e-6),
    joint_mapping_checks=joint_checks, significant_joint_limit_differences=differences,
    eef_offset_checks=eef_checks,
    reused_fk=dict(preflight=str(OUT.parent/'self_geometry/preflight.json'), original_frames=832,
        original_arm_fk_evaluations=1664, newly_evaluated_frames=0, rehashed_frame_files=0,
        prior_max_position_error_m=max(s['position_error_m']['max'] for s in prior),
        prior_max_orientation_error_deg=max(s['orientation_error_deg']['max'] for s in prior),
        scope='Existing EEF evidence reused verbatim; not a new finger-FK validation.'),
    conversion_semantics=semantics, limitations=unknown,
    execution=dict(usd_stage_opens=e['source_stage_opens'], extraction=e['resources'],
        validation_cpu_affinity=sorted(os.sched_getaffinity(0)), validation_elapsed_s=time.monotonic()-START,
        validation_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        pxr_imported_for_validation=False, gpu_used=False, simulation_started=False,
        world_or_object_state_read=False, tracking_frames=0),
    next_stage='Consume geometry.json without pxr; independently validate the self-filter and current proprio placement before any tracking/admission claims.')
write('validation.json', validation)
print(json.dumps(dict(status=validation['status'], geometry=str(OUT/'geometry.json'),
    validation=str(OUT/'validation.json'), geometry_bytes=(OUT/'geometry.json').stat().st_size,
    joints_checked=len(joint_checks), eef_checked=len(eef_checks), significant_limit_differences=differences,
    max_joint_origin_error_m=max(v['usd_zero_frame_error']['position_m'] for v in joint_checks.values()),
    max_joint_rotation_error_rad=max(v['usd_zero_frame_error']['rotation_rad'] for v in joint_checks.values()),
    eef_offset_checks=eef_checks, execution=validation['execution']), ensure_ascii=False))
