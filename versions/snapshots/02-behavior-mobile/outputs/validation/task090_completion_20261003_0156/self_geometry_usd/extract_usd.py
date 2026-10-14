#!/usr/bin/env python3
"""One CPU-only USD parse. Export only explicit robot collision meshes + static joints."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PXR_WORK_THREAD_LIMIT'):
    os.environ[name] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

ROOT = Path('/home/user/Robotic/project2')
OUT = Path(__file__).resolve().parent
ASSET = ROOT/'data/omnigibson/omnigibson-robot-assets/models/r1pro'
USD, URDF = ASSET/'usd/r1pro.usda', ASSET/'urdf/r1pro.urdf'
PREFLIGHT = OUT.parent/'self_geometry/preflight.json'
OFFSETS = ROOT/'configs/r1pro_eef_offsets.json'
LINKS = [f'{a}_{s}' for a in ('left', 'right') for s in (
    'arm_link6', 'arm_link7', 'gripper_link', 'realsense_link',
    'gripper_finger_link1', 'gripper_finger_link2')]
START = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def plain(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, 'GetReal'):
        return [float(v.GetReal()), *map(float, v.GetImaginary())]
    return [plain(x) for x in v]


def write(name, v):
    with (OUT/name).open('x') as f:
        json.dump(v, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def xform(p):
    x = UsdGeom.Xformable(p)
    ops = x.GetOrderedXformOps()
    assert all(op.GetNumTimeSamples() == 0 for op in ops), f'Animated transform: {p.GetPath()}'
    return dict(path=str(p.GetPath()), reset_xform_stack=x.GetResetXformStack(),
                order=[op.GetOpName() for op in ops],
                ops=[dict(name=op.GetOpName(), value=plain(op.Get()), inverse=op.IsInverseOp()) for op in ops],
                local_matrix_column=np.asarray(x.GetLocalTransformation(), float).T.tolist())


def rigid(T):
    R = np.asarray(T)[:3, :3]
    return bool(np.max(np.abs(R.T@R-np.eye(3))) < 1e-8 and abs(np.linalg.det(R)-1) < 1e-8)


def frame(pos, quat):
    q = Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:])))
    q.Normalize()
    T = np.eye(4)
    T[:3, :3] = np.asarray(Gf.Matrix3d(q)).T
    T[:3, 3] = pos
    return T


def vec(element, key, default):
    return [float(v) for v in (element.get(key, default) if element is not None else default).split()]


assert not (OUT/'extraction.json').exists(), 'Do not overwrite prior evidence'
source_stat = USD.stat()
sources = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in (
    USD, URDF, PREFLIGHT, OFFSETS, ROOT/'harness/r1pro_kinematics.py', Path(__file__))}
ur = ET.parse(URDF).getroot()
uj = {j.get('name'): j for j in ur.findall('joint')}
child_joint = {j.find('child').get('link'): n for n, j in uj.items()}
needed = set()
for name in LINKS:
    while name in child_joint:
        n = child_joint[name]
        needed.add(n)
        name = uj[n].find('parent').get('link')
# Population excludes the static asset's PhysicsScene; no environment/object stage is opened.
mask = Usd.StagePopulationMask(['/r1pro'])
stage = Usd.Stage.OpenMasked(str(USD), mask, Usd.Stage.LoadNone)
assert stage and stage.GetDefaultPrim().GetPath().pathString == '/r1pro'
assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
assert UsdGeom.GetStageUpAxis(stage) == 'Z'
layers = [l.realPath for l in stage.GetUsedLayers() if l.realPath]
assert layers == [str(USD)], f'Unhashed external composed layer: {layers}'
assert not stage.GetRootLayer().subLayerPaths
root = stage.GetPrimAtPath('/r1pro')
root_xform = xform(root)
assert rigid(root_xform['local_matrix_column']), 'Root scale/shear unsupported'
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
arrays, links, joints, body_metrics = {}, {}, {}, {}
for name in LINKS:
    link = stage.GetPrimAtPath('/r1pro/'+name)
    assert link and link.HasAPI(UsdPhysics.RigidBodyAPI), name
    lx = xform(link)
    assert rigid(lx['local_matrix_column']), f'Nonmetric link: {name}'
    shapes, ignored, all_vertices = [], [], []
    for prim in Usd.PrimRange(link):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            if prim.IsA(UsdGeom.Mesh):
                ignored.append(str(prim.GetPath()))  # Never read visual points.
            continue
        assert prim.IsA(UsdGeom.Mesh), f'Unsupported collision primitive: {prim.GetPath()}'
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        schemas = prim.GetAppliedSchemas()
        assert 'PhysicsCollisionAPI' in schemas and 'PhysicsMeshCollisionAPI' in schemas
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        assert approximation == 'convexHull', approximation
        p = prim
        hierarchy = []
        manual_row = np.eye(4)
        while p != link:
            assert p and not p.IsPseudoRoot(), 'Collision not under selected link'
            assert p == prim or not p.HasAPI(UsdPhysics.RigidBodyAPI), 'Nested rigid body'
            xf = xform(p)
            assert not xf['reset_xform_stack'], 'Reset stack needs explicit handling'
            hierarchy.append(xf)
            manual_row = manual_row @ np.asarray(xf['local_matrix_column']).T
            p = p.GetParent()
        gf_M, reset = cache.ComputeRelativeTransform(prim, link)
        assert not reset
        M = np.asarray(gf_M, dtype=np.float64)
        assert np.max(np.abs(M-manual_row)) < 1e-12
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr()
        assert points.GetNumTimeSamples() == 0
        raw = np.asarray(points.Get(), dtype=np.float64)
        assert raw.ndim == 2 and raw.shape[1] == 3 and len(raw) >= 4 and np.isfinite(raw).all()
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        assert len(indices) == int(counts.sum()) and indices.min() >= 0 and indices.max() < len(raw)
        assert counts.min() >= 3
        local = (np.column_stack((raw, np.ones(len(raw)))) @ M)[:, :3]
        # Independent Gf point transform check, for EVERY input vertex.
        direct = np.array([gf_M.Transform(Gf.Vec3d(*map(float, v))) for v in raw])
        transform_error = float(np.max(np.abs(local-direct)))
        assert transform_error < 1e-12 and np.isfinite(local).all()
        lo = np.nextafter(local.min(axis=0)-1e-9, -np.inf)
        hi = np.nextafter(local.max(axis=0)+1e-9, np.inf)
        inside = np.all((local >= lo) & (local <= hi), axis=1)
        assert inside.all()
        idx = f'{name}__{len(shapes):02d}'
        arrays[idx+'__points_mesh'] = raw
        arrays[idx+'__points_link_m'] = local
        arrays[idx+'__face_vertex_indices'] = indices
        arrays[idx+'__face_vertex_counts'] = counts
        all_vertices.append(local)
        authored_collision = {a.GetName(): plain(a.Get()) for a in prim.GetAuthoredAttributes()
                              if a.GetName().startswith(('physics:', 'physxCollision:', 'physxConvexHullCollision:'))}
        shapes.append(dict(id=idx, prim_path=str(prim.GetPath()), applied_schemas=schemas,
            collision_enabled=True, physics_approximation=approximation,
            approximation_authored=prim.GetAttribute('physics:approximation').HasAuthoredValueOpinion(),
            authored_collision_attributes=authored_collision, vertex_count=len(raw),
            face_count=len(counts), unreferenced_vertex_count=len(raw)-len(np.unique(indices)),
            hierarchy_mesh_to_link=hierarchy, T_link_mesh_column=M.T.tolist(),
            npz_key_points=idx+'__points_link_m', aabb_min_m=lo.tolist(), aabb_max_m=hi.tolist(),
            containment=dict(tested_vertices=len(raw), all_inside=bool(inside.all()),
                outside_count=int((~inside).sum()), outward_numeric_padding_m=1e-9,
                min_boundary_slack_m=float(np.minimum(local-lo, hi-local).min()),
                max_gf_transform_error_m=transform_error,
                max_chain_transform_error=float(np.max(np.abs(M-manual_row))))))
    assert shapes, f'No collision geometry: {name}'
    union = np.concatenate(all_vertices)
    links[name] = dict(usd_link_path=str(link.GetPath()), link_metric_rigid=True,
        # Pose is intentionally not exported: the downstream runtime must use proprio/FK.
        link_xform_order=lx['order'], shapes=shapes, ignored_mesh_paths_without_collision_api=ignored,
        union_aabb_min_m=np.nextafter(union.min(axis=0)-1e-9, -np.inf).tolist(),
        union_aabb_max_m=np.nextafter(union.max(axis=0)+1e-9, np.inf).tolist())

for name in sorted(needed) + [f'{a}_gripper_link_{a}_eef_link_joint' for a in ('left', 'right')]:
    u = uj.get(name)
    parent = u.find('parent').get('link') if u is not None else name.split('_')[0]+'_gripper_link'
    child = u.find('child').get('link') if u is not None else name.split('_')[0]+'_eef_link'
    prim = stage.GetPrimAtPath(f'/r1pro/{parent}/{name}')
    assert prim and prim.IsA(UsdPhysics.Joint), f'Missing static joint {name}'
    j = UsdPhysics.Joint(prim)
    b0, b1 = list(j.GetBody0Rel().GetTargets()), list(j.GetBody1Rel().GetTargets())
    assert list(map(str, b0)) == ['/r1pro/'+parent] and list(map(str, b1)) == ['/r1pro/'+child]
    for b in (b0[0], b1[0]):
        if str(b) not in body_metrics:
            bx = xform(stage.GetPrimAtPath(b))
            assert rigid(bx['local_matrix_column']), f'Nonmetric joint body {b}'
            body_metrics[str(b)] = dict(unit_scale_no_shear=True, xform_order=bx['order'])
    J0 = frame(plain(j.GetLocalPos0Attr().Get()), plain(j.GetLocalRot0Attr().Get()))
    J1 = frame(plain(j.GetLocalPos1Attr().Get()), plain(j.GetLocalRot1Attr().Get()))
    assert j.GetJointEnabledAttr().Get()
    attrs = {a.GetName(): plain(a.Get()) for a in prim.GetAuthoredAttributes()
             if a.GetName().startswith(('physics:', 'state:'))}
    assert all(a.GetNumTimeSamples() == 0 for a in prim.GetAuthoredAttributes())
    axis_attr = prim.GetAttribute('physics:axis')
    axis = axis_attr.Get() if axis_attr else None
    origin = u.find('origin') if u is not None else None
    axis_u = u.find('axis') if u is not None else None
    lim = u.find('limit') if u is not None else None
    urdf = None if u is None else dict(type=u.get('type'), parent=parent, child=child,
        origin_xyz_m=vec(origin, 'xyz', '0 0 0'), origin_rpy_rad=vec(origin, 'rpy', '0 0 0'),
        axis=vec(axis_u, 'xyz', '1 0 0') if u.get('type') != 'fixed' else None,
        limit=None if lim is None else {k: float(v) for k, v in lim.attrib.items()},
        mimic=None if u.find('mimic') is None else dict(u.find('mimic').attrib))
    joints[name] = dict(prim_path=str(prim.GetPath()), usd_type=prim.GetTypeName(),
        parent=parent, child=child, usd_axis_token=axis, usd_authored_attributes=attrs,
        T_parent_joint0_column=J0.tolist(), T_child_joint1_column=J1.tolist(),
        T_parent_child_at_q0_column=(J0@np.linalg.inv(J1)).tolist(), urdf=urdf)

assert USD.stat().st_mtime_ns == source_stat.st_mtime_ns and USD.stat().st_size == source_stat.st_size
np.savez_compressed(OUT/'collision_vertices.npz', **arrays)
result = dict(status='extracted_for_validation', source_files=sources,
    source_unchanged_stat_check=True, units=dict(meters_per_unit=1.0, up_axis='Z',
        source_linear_unit='m', source_angular_joint_unit='degree', runtime_angular_qpos_unit='radian'),
    usd_version=list(Usd.GetVersion()), interpreter=sys.executable,
    used_layers=layers, population_mask=['/r1pro'], source_stage_opens=1,
    links=links, joints=joints, joint_body_metric_checks=body_metrics,
    arrays_file=str(OUT/'collision_vertices.npz'),
    arrays_sha256=sha(OUT/'collision_vertices.npz'),
    prior_fk_evidence=json.loads(PREFLIGHT.read_text()),
    existing_eef_offsets=json.loads(OFFSETS.read_text()),
    resources=dict(cpu_affinity=sorted(os.sched_getaffinity(0)), gpu_used=False,
        simulation_app_started=False, world_or_object_state_read=False, frame_data_read=False,
        elapsed_s=time.monotonic()-START, max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
write('extraction.json', result)
print(json.dumps(dict(status=result['status'], links=len(links), collision_meshes=sum(len(v['shapes']) for v in links.values()),
    vertices=sum(s['vertex_count'] for v in links.values() for s in v['shapes']), joints=len(joints),
    resources=result['resources'], extraction=str(OUT/'extraction.json'))))
