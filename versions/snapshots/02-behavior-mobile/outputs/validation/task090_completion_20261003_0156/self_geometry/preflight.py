#!/usr/bin/env python3
"""CPU-only stop-condition audit, NOT a geometry tracker or admission check."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import psutil
ROOT = Path('/home/user/Robotic/project2')
sys.path.insert(0, str(ROOT))
from harness.r1pro_kinematics import R1ProKinematics, EEF, q11, quat_to_mat, URDF, OFFSETS
OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent/'continuous_train/extracted/frames.jsonl'
USD = URDF.parent.parent/'usd/r1pro.usda'
START = time.monotonic()
PROCESS = psutil.Process()
PROCESS.cpu_affinity([min(PROCESS.cpu_affinity())])

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write(name, value):
    with (OUT/name).open('x') as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')

def angle(a, b):
    delta = a @ b.T
    sine = np.linalg.norm([delta[2, 1]-delta[1, 2], delta[0, 2]-delta[2, 0], delta[1, 0]-delta[0, 1]])/2
    return float(np.degrees(np.arctan2(sine, (np.trace(delta)-1)/2)))

rows = [json.loads(line) for line in SOURCE.open()]
manifest = json.loads(SOURCE.with_name('manifest.json').read_text())
assert len(rows) == manifest['sample_count'] == 832
assert sha(SOURCE) == manifest['frame_manifest_sha256']
paths = {Path(__file__), SOURCE, SOURCE.with_name('manifest.json'), URDF, OFFSETS, USD,
         ROOT/'harness/r1pro_kinematics.py', ROOT/'harness/task090_rigid_observation_v4.py',
         ROOT/'harness/task090_rigid_observation_v3.py', ROOT/'harness/task090_predicates.py'}
links = {a+'_'+s for a in EEF for s in ('arm_link6', 'arm_link7', 'gripper_link',
         'realsense_link', 'gripper_finger_link1', 'gripper_finger_link2')}
geometry = []
for link in ET.parse(URDF).getroot().findall('link'):
    if link.get('name') not in links:
        continue
    for collision in link.findall('collision'):
        mesh = collision.find('geometry/mesh')
        assert mesh is not None, 'Audit must explicitly handle non-mesh collision geometry'
        path = URDF.parent/mesh.get('filename')
        origin = collision.find('origin')
        geometry.append(dict(link=link.get('name'), collision=collision.get('name'),
            mesh=str(path), exists=path.is_file(), sha256=sha(path) if path.is_file() else None,
            scale=[float(x) for x in mesh.get('scale', '1 1 1').split()],
            origin=dict(origin.attrib) if origin is not None else dict(xyz='0 0 0', rpy='0 0 0')))
        if path.is_file():
            paths.add(path)
assert {x['link'] for x in geometry} == links
# A visual mesh is NOT a replacement for a missing collision asset.
missing = [x['mesh'] for x in geometry if not x['exists']]
assert missing, 'Stop audit is specific to missing collision assets; do not run tracking implicitly'
source_hashes = {str(p): sha(p) for p in sorted(paths)}
kin = R1ProKinematics()
events, summary = [], []
for ep in (18000, 18001):
    ep_rows = [row for row in rows if row['episode'] == ep]
    frames = [row['frame_index'] for row in ep_rows]
    assert frames == list(range(frames[0], frames[-1]+1))
    for row in ep_rows:
        assert row['split'] == 'train' and row['physical_success_label'] is None
        assert row['applied_action_frame'] == row['frame_index']-1 and row['applied_action_verified']
        state = np.asarray(row['state61'], float)
        assert state.shape == (61,) and np.isfinite(state).all()
        for arm in EEF:
            position, orientation = kin.fk(arm, q11(state, arm))
            raw_position, raw_orientation = kin.fk(arm, q11(state, arm), with_offset=False)
            quaternion = state[EEF[arm][1]]
            norm = np.linalg.norm(quaternion)
            assert abs(norm-1) < .001
            measured_orientation = quat_to_mat(quaternion/norm)
            events.append(dict(episode=ep, frame=row['frame_index'], arm=arm,
                position_error_m=float(np.linalg.norm(position-state[EEF[arm][0]])),
                orientation_error_deg=angle(orientation, measured_orientation),
                uncorrected_position_error_m=float(np.linalg.norm(raw_position-state[EEF[arm][0]])),
                uncorrected_orientation_error_deg=angle(raw_orientation, measured_orientation),
                quaternion_norm_error=float(abs(norm-1))))
    for arm in EEF:
        group = [e for e in events if e['episode'] == ep and e['arm'] == arm]
        result = dict(episode=ep, arm=arm, frames=len(ep_rows), frame_range=[frames[0], frames[-1]])
        for key in ('position_error_m', 'orientation_error_deg', 'uncorrected_position_error_m',
                    'uncorrected_orientation_error_deg', 'quaternion_norm_error'):
            result[key] = dict(zip(('min', 'median', 'p99', 'max'), np.quantile([e[key] for e in group], [0, .5, .99, 1]).tolist()))
        summary.append(result)
with (OUT/'fk_audit.jsonl').open('x') as file:
    for event in events:
        file.write(json.dumps(event, allow_nan=False)+'\n')
write('source_sha256.json', source_hashes)
write('preflight.json', dict(status='blocked_missing_urdf_collision_meshes_before_implementation',
    scope='Robot hand/wrist only; never held object; no guessed geometry or radius change',
    missing_collision_meshes=missing, collision_geometry=geometry, independent_train_fk=summary,
    state_mapping=dict(trunk_qpos=[53,57], left_arm_qpos=[3,10], right_arm_qpos=[28,35],
                       left_eef_position=[17,20], left_eef_xyzw=[20,24], left_finger_qpos=[24,26],
                       right_eef_position=[42,45], right_eef_xyzw=[45,49], right_finger_qpos=[49,51]),
    frame_mapping='Robot articulation-root-relative; EEF and RGB-D backprojection use this base frame.',
    offset_evidence=dict(source=str(OFFSETS), fit_task='task001 TRAIN, unchanged',
        static_usd=str(USD), right_eef_joint_lines=[5472,5481], left_eef_joint_lines=[3455,3464],
        translation_m=[0,0,-.06], rotation_wxyz=[0,0,1,0],
        explanation='USD fixed gripper-to-EEF joint agrees with existing fitted offsets. Omitting it yields 6cm / 180deg error; corrected independent task090 TRAIN FK has micrometre-level error. No parameters fitted here.'),
    limits=dict(geometry_located=False, collision_geometry_tested=False, self_filter_implemented=False,
        continuous_tracking_executed=False, continuous_tracking_frames=0, dino_extractions=0,
        physical_success_label=None, candidate_grasp=None, admitted=False, verified=False),
    available_alternative='Static r1pro.usda exists and contains collision prims; its equivalence to missing URDF collision meshes has NOT been verified. No substitute mesh/scale inferred.',
    next_blocker='Restore the authoritative URDF-referenced collision assets, or independently validate and explicitly implement static USD collision-body transforms/units before geometry filtering. Then CPU tests and one 832-frame continuous TRAIN check; not yet executed.',
    unchanged=['all runtime source', 'old 323-frame v4 results', 'progress', 'configs', 'success gates'],
    resources=dict(cpu_affinity=PROCESS.cpu_affinity(), gpu_used=False, wall_seconds=time.monotonic()-START,
                   rss_mib=PROCESS.memory_info().rss/1024**2),
    provenance=str(OUT/'source_sha256.json')))
assert all(sha(p) == digest for p, digest in source_hashes.items()), 'Source changed during audit'
print(json.dumps(dict(status='blocked', missing_collision_meshes=len(missing), train_frames=832,
    tracking_frames_executed=0, preflight=str(OUT/'preflight.json')), ensure_ascii=False))
