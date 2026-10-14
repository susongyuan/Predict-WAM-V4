#!/usr/bin/env python3
"""CPU-only static ifzxzj audit; no OmniGibson import, simulation, state or downloads.
Run with /home/user/Robotic/project2/.conda-envs/behavior6/bin/python -B <this file>.
Only geometry summaries persist; decrypted USD is private temporary input, removed in finally.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PXR_WORK_THREAD_LIMIT'):
    os.environ[k] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import ast
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path('/home/user/Robotic/project2')
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT/'outputs/validation/task090_completion_20261003_0156'
ASSET = ROOT/'data/omnigibson/behavior-1k-assets/objects/trash_can/ifzxzj'
META = ASSET/'misc/metadata.json'
ENC = ASSET/'usd/ifzxzj.encrypted.usd'
UTIL = ROOT/'third_party/BEHAVIOR-1K/OmniGibson/omnigibson/utils/asset_utils.py'
CONFIG = ROOT/'configs/task090_perception_geometry_v2.json'
COVERAGE = EVIDENCE/'support_coverage/coverage.json'
RESULT = OUT/'scope.json'
START = time.monotonic()
assert OUT == EVIDENCE/'object_geometry_scope'
previous_attempt = json.loads(RESULT.read_text()) if RESULT.exists() else None
assert previous_attempt is None or previous_attempt['static_usd'] is None, 'Preserve completed audit'
os.umask(0o077)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def plain(v):
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else str(v)
    if hasattr(v, 'GetReal'):
        return [float(v.GetReal()), *map(float, v.GetImaginary())]
    try:
        return [plain(x) for x in v]
    except TypeError:
        return str(v)


def bounds(points):
    return {'min_m': points.min(0).tolist(), 'max_m': points.max(0).tolist(),
            'extent_m': np.ptp(points, axis=0).tolist(), 'vertex_count': len(points)}


sources = [META, ENC, UTIL, CONFIG, ROOT/'progress.md', Path(__file__),
           EVIDENCE/'self_geometry_usd/extract_usd.py', EVIDENCE/'self_geometry_usd/extraction.json',
           ROOT/'harness/task090_rigid_observation_v2.py', ROOT/'harness/task090_rigid_observation_v3.py',
           ROOT/'harness/task090_rigid_observation_v4.py', ROOT/'harness/task090_rigid_observation_v5.py', COVERAGE]
source_records = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sources}
meta = json.loads(META.read_text())
prior = json.loads(CONFIG.read_text())['classes']['trash_can']['geometry_prior']
result = {
    'status': 'metadata_only_until_static_parse_succeeds',
    'scope': 'One local static asset, native authored configuration only; no live/dev object state or scene files.',
    'source_files': source_records,
    'metadata': meta,
    'prior': prior,
    'prior_metadata_sha_matches': prior['asset_sha256'] == sha(META),
    'rigid_height_diameter_m': float(np.linalg.norm(prior['native_bbox_m'])),
    'height_code_references': ['harness/task090_rigid_observation_v2.py:65-79,90-91',
                               'harness/task090_rigid_observation_v3.py:144-145'],
    'size_scope': {'native_dimensions_only': True, 'instance_scale': None,
                   'config_scale_assumption': [1., 1., 1.],
                   'config_max_axis_span_is_proposal_gate_not_rigid_diameter': True,
                   'native_bbox_not_certified_collision_or_articulation_enclosure': True},
    'static_usd': None,
    'boundaries': {'current_or_dev_object_state_read': False, 'runtime_changed': False,
                   'new_enclosure_producer': False, 'physical_label': None, 'admission': False,
                   'point_to_link_assignment_verified': False, 'actual_lid_motion_verified': False,
                   'static_pose_injected_into_tracker': False},
}
# Reuse only named, reviewed existing decryption functions. Never import OmniGibson:
# its module initialization has simulation dependencies. No credential value or hash is exported.
phase = 'cpu_usd_import'
temporary_cleaned = True
try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf
    from cryptography.fernet import Fernet
    phase = 'isolated_existing_decrypt'
    module = ast.parse(UTIL.read_text())
    names = {'get_key_path', 'decrypt_file'}
    selected = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    ns = {'os': os, 'gm': SimpleNamespace(DATA_PATH=str(ROOT/'data/omnigibson')), 'Fernet': Fernet}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(UTIL), 'exec'), ns)
    with tempfile.TemporaryDirectory(prefix='.static_decrypt_', dir=OUT) as temporary:
        temporary_cleaned = False
        dec = Path(temporary)/'ifzxzj.usd'
        stage = layer = cache = None
        try:
            ns['decrypt_file'](str(ENC), str(dec))
            phase = 'local_layer_composition_preflight'
            layer = Sdf.Layer.FindOrOpen(str(dec))
            assert layer and not layer.subLayerPaths, 'Sublayers are outside this audit'
            external_arcs = []
            def check_spec(path):
                spec = layer.GetObjectAtPath(path)
                if isinstance(spec, Sdf.PrimSpec):
                    for field in ('references', 'payload'):
                        if spec.HasInfo(field):
                            for item in spec.GetInfo(field).GetAppliedItems():
                                if item.assetPath:
                                    external_arcs.append(str(path))
            layer.Traverse(Sdf.Path.absoluteRootPath, check_spec)
            assert not external_arcs, 'External composition refused; no referenced layers opened'
            assert layer.defaultPrim
            rootpath = '/'+layer.defaultPrim
            phase = 'static_stage_parse'
            stage = Usd.Stage.OpenMasked(layer, Usd.StagePopulationMask([rootpath]), Usd.Stage.LoadNone)
            root = stage.GetPrimAtPath(rootpath)
            assert root
            unit = float(UsdGeom.GetStageMetersPerUnit(stage))
            axis = str(UsdGeom.GetStageUpAxis(stage))
            result['parse_preflight'] = {'default_prim': rootpath, 'meters_per_unit': unit, 'up_axis': axis}
            assert unit > 0 and axis == 'Z', 'Only Z-up support summary is implemented'
            assert [l.realPath for l in stage.GetUsedLayers() if l.realPath] == [str(dec)], 'Unexpected composed layers'
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            def xform(p):
                x = UsdGeom.Xformable(p)
                if not x:
                    return None
                ops = x.GetOrderedXformOps()
                assert all(op.GetNumTimeSamples() == 0 for op in ops)
                return {'order': [op.GetOpName() for op in ops],
                        'reset': x.GetResetXformStack(),
                        'matrix_row_native_units': plain(x.GetLocalTransformation())}
            bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            hierarchy, joints, links = [], [], []
            all_root_points = []
            for p in Usd.PrimRange(root):
                if p == root or p in bodies or p.IsA(UsdPhysics.Joint):
                    hierarchy.append({'path': str(p.GetPath()), 'parent': str(p.GetParent().GetPath()),
                                      'type': p.GetTypeName(), 'schemas': p.GetAppliedSchemas(), 'xform': xform(p)})
                if not p.IsA(UsdPhysics.Joint):
                    continue
                j = UsdPhysics.Joint(p)
                assert all(a.GetNumTimeSamples() == 0 for a in p.GetAuthoredAttributes())
                attrs = {a.GetName(): plain(a.Get()) for a in p.GetAuthoredAttributes()
                         if a.GetName().startswith(('physics:', 'physxJoint:', 'drive:', 'state:'))}
                joints.append({'path': str(p.GetPath()), 'type': p.GetTypeName(),
                               'body0': list(map(str, j.GetBody0Rel().GetTargets())),
                               'body1': list(map(str, j.GetBody1Rel().GetTargets())),
                               'enabled': bool(j.GetJointEnabledAttr().Get()), 'authored_attributes': attrs,
                               'axis_token': plain(p.GetAttribute('physics:axis').Get()) if p.HasAttribute('physics:axis') else None,
                               'lower_limit': plain(p.GetAttribute('physics:lowerLimit').Get()) if p.HasAttribute('physics:lowerLimit') else None,
                               'upper_limit': plain(p.GetAttribute('physics:upperLimit').Get()) if p.HasAttribute('physics:upperLimit') else None,
                               'angular_unit': 'degree for revolute; local positions in stage units',
                               'axis_frame': 'Joint local frame; localRot0/localRot1 must be applied, not a world axis',
                               'authored_state_is_not_runtime_state': True})
            for body in bodies:
                shapes, local_vertices, root_vertices = [], [], []
                for p in Usd.PrimRange(body):
                    if not p.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    owner = p.GetParent()
                    while owner and owner != body and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                        owner = owner.GetParent()
                    assert owner == body, 'Nested rigid body ownership is ambiguous'
                    assert p.IsA(UsdGeom.Mesh), 'Non-mesh collider not implemented'
                    mesh = UsdGeom.Mesh(p)
                    assert mesh.GetPointsAttr().GetNumTimeSamples() == 0
                    raw = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                    inds = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
                    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
                    assert raw.ndim == 2 and raw.shape[1] == 3 and np.isfinite(raw).all()
                    assert len(inds) == counts.sum() and inds.min() >= 0 and inds.max() < len(raw)
                    lb, reset1 = cache.ComputeRelativeTransform(p, body)
                    rb, reset2 = cache.ComputeRelativeTransform(p, root)
                    assert not reset1 and not reset2
                    local = (np.column_stack((raw, np.ones(len(raw)))) @ np.asarray(lb))[:, :3]*unit
                    native = (np.column_stack((raw, np.ones(len(raw)))) @ np.asarray(rb))[:, :3]*unit
                    direct = np.asarray([lb.Transform(Gf.Vec3d(*map(float, v))) for v in raw])*unit
                    assert np.max(np.abs(local-direct)) < 1e-10
                    enabled = bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())
                    attrs = {a.GetName(): plain(a.Get()) for a in p.GetAuthoredAttributes()
                             if a.GetName().startswith(('physics:', 'physxCollision:', 'physxConvexHullCollision:'))}
                    shapes.append({'path': str(p.GetPath()), 'type': p.GetTypeName(),
                                   'schemas': p.GetAppliedSchemas(), 'enabled': enabled,
                                   'attributes': attrs, 'link_local': bounds(local), 'native_root': bounds(native),
                                   'face_count': len(counts), 'unreferenced_vertices': len(raw)-len(np.unique(inds))})
                    local_vertices.append(local)
                    root_vertices.append(native)
                if not local_vertices:
                    links.append({'name': body.GetName(), 'path': str(body.GetPath()), 'shapes': shapes,
                                  'collision_geometry_present': False})
                    continue
                loc = np.concatenate(local_vertices)
                nat = np.concatenate(root_vertices)
                all_root_points.append(nat)
                ref = meta['link_bounding_boxes'].get(body.GetName(), {}).get('collision', {}).get('axis_aligned')
                delta = None
                if ref:
                    center = np.asarray(ref['transform'])[:3, 3]
                    ex = np.asarray(ref['extent'])
                    delta = float(max(np.max(np.abs(loc.min(0)-(center-ex/2))),
                                      np.max(np.abs(loc.max(0)-(center+ex/2)))))
                attrs = {a.GetName(): plain(a.Get()) for a in body.GetAuthoredAttributes()
                         if a.GetName().startswith(('physics:', 'physxRigidBody:'))}
                links.append({'name': body.GetName(), 'path': str(body.GetPath()),
                              'rigid_body_attributes': attrs, 'shapes': shapes,
                              'enabled_collision_count': sum(s['enabled'] for s in shapes),
                              'collision_geometry_present': True,
                              'link_local_collision_union': bounds(loc), 'native_root_collision_union': bounds(nat),
                              'metadata_collision_aabb_max_abs_difference_m': delta})
            assert links and joints
            whole = bounds(np.concatenate(all_root_points))
            minimum = whole['min_m'][2]
            lowest = [{'link': l['name'], 'shape': s['path'], 'min_native_root_z_m': s['native_root']['min_m'][2]}
                      for l in links for s in l['shapes'] if abs(s['native_root']['min_m'][2]-minimum) < 1e-6]
            result['static_usd'] = {'root': rootpath, 'meters_per_unit': unit, 'up_axis': axis,
                                    'usd_version': list(Usd.GetVersion()), 'hierarchy': hierarchy, 'links': links,
                                    'joints': joints, 'whole_declared_collision_native_root': whole,
                                    'lowest_native_authored_configuration': lowest,
                                    'support_scope': 'Declared CollisionAPI mesh vertex minima, INCLUDING any disabled shapes: authored configuration, unit native scale, native +Z up only. Enable flags are separately recorded. Not scene floor/contact, not arbitrary joint angle/body rotation, not a cooked PhysX clearance bound.',
                                    'external_composition_arcs': external_arcs, 'source_layer_count': 1,
                                    'decrypted_usd_sha256': sha(dec), 'decrypted_usd_retained': False}
            result['status'] = 'static_asset_scope_audited_not_physical_admission'
        finally:
            cache = stage = layer = None
            dec.unlink(missing_ok=True)
    temporary_cleaned = not Path(temporary).exists()
except Exception as e:
    # Do not echo exception text: decoder/dependency exceptions can expose private inputs.
    import traceback
    frames = traceback.extract_tb(e.__traceback__)
    result['parse_failure'] = {'phase': phase, 'exception_type': type(e).__name__,
                               'audit_line': frames[-1].lineno,
                               'joint_type_if_not_parsed': None, 'reason': 'Safe metadata-only fallback; no joint guessing.'}
    if 'temporary' in locals():
        temporary_cleaned = not Path(temporary).exists()

# Existing TRAIN sensor landmarks only. Do not open measurements, scene state, poses or dev files.
observations = []
for episode in (18000, 18001):
    bank = EVIDENCE/f'self_geometry_usd/implementation/continuous_train_01/episode_{episode}_original_bank.npz'
    source_records[str(bank)] = {'sha256': sha(bank), 'bytes': bank.stat().st_size}
    with np.load(bank, allow_pickle=False) as data:
        if 'reference' not in data.files:
            observations.append({'episode': episode, 'available_fields': data.files, 'reference_read': False})
            continue
        p = np.asarray(data['reference'], float)
        assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
        observations.append({'episode': episode, 'count': len(p), 'source': str(bank),
                             'observed_reference_z_m': [float(p[:, 2].min()), float(p[:, 2].max())],
                             'z_span_m': float(np.ptp(p[:, 2])),
                             'frame': 'bank reference coordinates, not asset-local; no floor or asset registration assumed'})
coverage = json.loads(COVERAGE.read_text())
result['observations_only'] = {
    'original_bank': observations,
    'existing_rgb_review_reused_not_new_inspection': {
        'source': str(COVERAGE), 'best_navigation_view': coverage['inspection']['best_navigation_view'],
        'operation_window_views': [g for g in coverage['inspection']['sample_groups']
                                   if g.get('episode') == 18001 and g.get('frames') == [840, 870]
                                   or g.get('episode') == 18000 and g.get('frames') == [360, 420, 480]],
        'f724_depth_status_in_existing_review_only': 'not established by that review; no later early-baseline output read'},
    'interpretation': 'Observed point heights and 2D views cannot assign bank points to base_link/link_0 or establish lid motion. Asset and bank coordinates are not aligned. No static pose supplied as an observation answer.'}
result['checkable_conditions'] = {
    'body_only_tracking': [
        'Sensor-supported material-point assignment to the same base_link; immutable IDs and cross-navigation continuity; exclude or separately model link_0 and robot/background.',
        'Enough independently reobserved, spatially spread nondegenerate points; accepted rigid fit with unchanged freshness/residual gates. Fit agreement alone does not identify the tracked link.',
        'Independently validated instance geometry/scale and camera/depth uncertainty; a base_link diameter valid for the tracked complete body, not an automatic swap to the smaller metadata number.',
        'Initial bottom/support relationship and support identity/coverage known, or direct current clearance measurement; a height CHANGE is not absolute clearance.',
        'If anchors track link_0, body-only pose needs a measured joint relation/angle and uncertainty or validated immobility; static joint defaults are not measurements.'
    ],
    'whole_object_clear_of_support': [
        'Independently establish instance model/scale applicability and complete collision/shape coverage including unseen underside, with uncertainty and collision cooking/contact margins.',
        'For every constituent link, measure pose/joint angle or certify a feasible range/immobility; one rigid transform is valid only with independently supported constant relative link transforms throughout the interval.',
        'Bind all relevant support surfaces to the object footprint/side contacts; distant floor plane and a visible front outline alone are insufficient.',
        'Require a conservative positive clearance lower bound for every link against every relevant support, including measurement/shape/joint/scale uncertainty. Missing coverage remains Unknown.',
        'Clearance alone is not grasp, transport, release or task success; separately validate continuous hand-object relations and independent physical outcomes, never synthetic CPU examples as physical labels.'
    ]}
result['resources'] = {'interpreter': sys.executable, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                       'gpu_used': False, 'simulation_started': False, 'downloads': 0, 'agents_spawned': 0,
                       'omni_or_isaac_imported': any(n.split('.')[0] in ('omni', 'isaacsim', 'omnigibson') for n in sys.modules),
                       'elapsed_s': time.monotonic()-START,
                       'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       'decrypted_temporary_files_removed': temporary_cleaned}
result['input_assets_unchanged'] = all(sha(p) == source_records[str(p)]['sha256'] for p in (META, ENC, CONFIG, UTIL))
assert result['input_assets_unchanged'] and temporary_cleaned
if previous_attempt:
    result['prior_parse_attempts'] = previous_attempt.get('prior_parse_attempts', []) + [{
        'failure': previous_attempt.get('parse_failure'), 'preflight': previous_attempt.get('parse_preflight'),
        'resources': previous_attempt['resources']}]
with RESULT.open('w' if previous_attempt else 'x') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    f.write('\n')
print(json.dumps({'status': result['status'], 'result': str(RESULT),
                  'failure': result.get('parse_failure'), 'resources': result['resources']}))
