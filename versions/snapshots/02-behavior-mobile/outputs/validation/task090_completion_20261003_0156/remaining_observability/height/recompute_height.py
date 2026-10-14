#!/usr/bin/env python3
"""Directed CPU audit: saved fits, exactly three cached depth frames, schema only.
No tracker execution, simulator, GPU, payload parquet reads or threshold search.
All output is restricted to this script's directory.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT/'outputs/validation/task090_completion_20261003_0156/remaining_observability/height'
EVIDENCE = ROOT/'outputs/validation/task090_completion_20261003_0156/self_geometry_usd/implementation/continuous_train_01'
FRAMES = ROOT/'outputs/validation/task090_completion_20261003_0156/continuous_train/extracted/frames.jsonl'
CALIBRATION = ROOT/'configs/task090_perception_geometry_v2.json'
SELECTED = [796, 870]
START = time.monotonic()
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
HASHES = {}


def digest(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = value
    return value


def read_json(path):
    digest(path)
    return json.loads(Path(path).read_text())


def dump(name, value):
    path = OUT/name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value))


def camera(pose):
    pose = np.asarray(pose, float)
    x,y,z,w = pose[3:]/np.linalg.norm(pose[3:])
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return rotation @ np.diag([1.,-1.,-1.]), pose[:3]


def backproject(pose, u, v, depth):
    rotation, translation = camera(pose)
    rays = np.stack([(np.asarray(u, float)-360.)/306.,
                     (np.asarray(v, float)-360.)/306., np.ones(np.shape(u))], -1)
    return translation + (rays @ rotation.T)*np.asarray(depth, float)[...,None]


def floor_from_cached_frame(row):
    path = Path(row['depth_m'])
    assert digest(path) == row['depth_m_sha256']
    depth = np.load(path, allow_pickle=False)
    v,u = np.mgrid[0:depth.shape[0]:8,0:depth.shape[1]:8]
    z = depth[v,u]
    keep = np.isfinite(z)&(z>.05)&(z<6.)
    points = backproject(row['camera_pose7'],u[keep],v[keep],z[keep])
    points = points[np.isfinite(points).all(1)&(np.abs(points[:,2])<.15)]
    A = np.c_[points[:,:2],np.ones(len(points))]
    coeff,_,rank,singular = np.linalg.lstsq(A,points[:,2],rcond=None)
    residual = points[:,2]-A@coeff
    iqr = float(np.percentile(residual,75)-np.percentile(residual,25))
    assert len(points)>=32 and rank==3 and singular[-1]>=1e-3*singular[0]
    assert np.linalg.norm(coeff[:2])<=.2 and iqr<=.03
    return dict(coefficients=coeff, margin_m=3*max(iqr,.002), points=len(points),
                vertical_residual_iqr_m=iqr, max_abs_vertical_fit_residual_m=float(np.max(np.abs(residual))),
                fit_matrix_singular_values=singular), depth


def normal(coeff):
    a,b,c=coeff
    scale=np.sqrt(1+a*a+b*b)
    return np.array([-a,-b,1.])/scale,-c/scale


# Extract only pure functions; importing the runtime is neither needed nor allowed.
source_path=ROOT/'harness/task090_rigid_observation_v2.py'
source=source_path.read_text(); digest(source_path)
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('floor_normal','height_change_bound')]
namespace={'np':np}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source_path),'exec'),namespace)
runtime_formula=namespace['height_change_bound']

cal=read_json(CALIBRATION)
prior=cal['classes']['trash_can']['geometry_prior']
diameter=float(np.linalg.norm(prior['native_bbox_m']))
metadata_path=ROOT/prior['asset']
assert digest(metadata_path)==prior['asset_sha256']
metadata=json.loads(metadata_path.read_text())
provenance=read_json(EVIDENCE/'source_sha256.json')
source_hash_matches={}
for rel in ('harness/task090_rigid_observation_v2.py','harness/task090_rigid_observation_v3.py',
            'harness/task090_rigid_observation_v4.py','harness/task090_perception.py',
            'harness/object_perception.py','configs/task090_perception_geometry_v2.json'):
    path=ROOT/rel
    source_hash_matches[str(path)]=digest(path)==provenance[str(path)]
assert all(source_hash_matches.values())
bank_path=EVIDENCE/'episode_18000_original_bank.npz';digest(bank_path)
with np.load(bank_path,allow_pickle=False) as archive:
    bank={k:archive[k].copy() for k in archive.files}
digest(FRAMES)
rows={}
for line in FRAMES.open():
    row=json.loads(line)
    if row['episode']==18000 and row['frame_index'] in [480]+SELECTED:
        assert row['split']=='train'
        rows[row['frame_index']]=row
reports={}
report_path=EVIDENCE/'episode_18000_measurements.jsonl';digest(report_path)
height_stats=[]
for lineno,line in enumerate(report_path.open(),1):
    report=json.loads(line)
    diag=report['rigid_tracking']
    if 'conditional_min_height_gain_m' in diag:
        height_stats.append((report['observation_id'],diag['conditional_min_height_gain_m'],diag['cumulative_reference_error_m']))
    if report['observation_id'] in [480]+SELECTED:
        reports[report['observation_id']]=(lineno,report)
floor0,_=floor_from_cached_frame(rows[480])
np.testing.assert_array_equal(floor0['coefficients'],bank['origin_floor'])
assert floor0['margin_m']==float(bank['origin_margin'])
n0,d0=normal(floor0['coefficients'])
results=[]
for frame in SELECTED:
    lineno,report=reports[frame]
    diag=report['rigid_tracking'];raw=report['independent_measurement_check']
    a=bank['reference'][raw['original_ids']].astype(float)
    b=np.asarray(raw['current_points_base_m'],float)
    rotation=np.asarray(raw['rotation'],float);translation=np.asarray(raw['translation_m'],float)
    floor1,depth=floor_from_cached_frame(rows[frame]);n1,d1=normal(floor1['coefficients'])
    uv=np.asarray(raw['uv']);u,v=np.rint(uv).astype(int).T
    b_check=backproject(rows[frame]['camera_pose7'],uv[:,0],uv[:,1],depth[v,u])
    np.testing.assert_array_equal(b,b_check)
    residual=np.linalg.norm(a@rotation.T+translation-b,axis=1)
    rms=float(np.sqrt(np.mean(residual**2)))
    spread=float(np.linalg.norm(np.ptp(a,axis=0)))
    gain=float(n1@b.mean(0)+d1-n0@a.mean(0)-d0)
    v_normal=rotation.T@n1-n0
    rotation_allowance=float(diameter*np.linalg.norm(v_normal))
    plane_sum=floor0['margin_m']+floor1['margin_m']
    rms_term=3*rms
    ratio=3*rms/max(spread,.05)
    angular=diameter*min(2.,ratio)
    measurement=plane_sum+.005+rms_term+angular+diag['cumulative_reference_error_m']
    bound=gain-rotation_allowance-measurement
    recomputed={'anchor_height_gain_m':gain,'rotation_allowance_m':rotation_allowance,
        'assumed_measurement_allowance_m':measurement,'conditional_min_height_gain_m':bound,
        'tracked_point_spread_m':spread,'fit_rms_m':rms}
    discrepancies={k:float(value-diag[k]) for k,value in recomputed.items()}
    assert max(abs(value) for value in discrepancies.values())<1e-14
    reference_formula=runtime_formula(a,b,rotation,bank['origin_floor'],floor1['coefficients'],diameter,rms,plane_sum)
    for k,value in reference_formula.items():assert abs(value-diag[k])<1e-14
    results.append(dict(episode=18000,frame=frame,report_path=str(report_path),report_line=lineno,
        original_ids=raw['original_ids'],source_centroid_base_m=a.mean(0),target_centroid_base_m=b.mean(0),
        source_floor=floor0,current_floor=floor1,source_unit_normal=n0,current_unit_normal=n1,
        source_signed_offset_m=d0,current_signed_offset_m=d1,rotation=rotation,translation_m=translation,
        diameter_m=diameter,source_centroid_signed_height_m=float(n0@a.mean(0)+d0),
        current_centroid_signed_height_m=float(n1@b.mean(0)+d1),
        normal_pullback_difference=v_normal,normal_pullback_difference_norm=float(np.linalg.norm(v_normal)),
        source_axis_ranges_m=np.ptp(a,axis=0),
        source_centered_singular_values_m=np.linalg.svd(a-a.mean(0),compute_uv=False),
        independent_residuals_m=residual,maximum_residual_m=float(residual.max()),
        centroid_fit_discrepancy_m=float(np.linalg.norm(a.mean(0)@rotation.T+translation-b.mean(0))),
        components=dict(**recomputed,origin_plane_margin_m=floor0['margin_m'],
            current_plane_margin_m=floor1['margin_m'],plane_margin_sum_m=plane_sum,
            fixed_engineering_margin_m=.005,three_fit_rms_m=rms_term,
            angular_ratio_uncapped=ratio,angular_ratio_capped=min(2.,ratio),
            diameter_times_residual_spread_ratio_m=angular,
            cumulative_reference_error_m=diag['cumulative_reference_error_m'],
            geometric_gain_minus_rotation_only_m=gain-rotation_allowance),
        recomputation_difference_m=discrepancies,
        contact_candidate=diag['gripper_contact_candidate'],candidate_grasp=diag['candidate_grasp'],
        hand_motion_residual_m=diag['hand_motion_residual_m'],hand_motion_gate_m=.02,
        hand_motion_gate_pass=diag['hand_motion_residual_m']<=.02,
        minimum_gain_gate_m=.03,nearest_tracked_point_to_eef_m=diag['nearest_tracked_point_to_eef_m'],
        recovery_frame=diag['recovery_frame'],absolute_support_clearance='unknown'))

# Counterexamples address the UNCERTAINTY interpretation, not the conditional algebra.
a=np.array([[0,0,.1],[.1,0,.1],[0,.1,.1],[.1,.1,.1],[.03,.02,.1],[.07,.08,.1]],float)
b=a+np.array([0,0,.1])
bias=runtime_formula(a,b,np.eye(3),[0,0,0],[0,0,0],1.,0.,.012)
assert bias['conditional_min_height_gain_m']>.03
# Genuine rigid rotation about a retained support vertex; upper points rise, whole object does not.
body=np.array([[0,0,0],[.4,0,0],[0,.1,0],[.4,.1,0],[0,0,.05],[.4,0,.05],[0,.1,.05],[.4,.1,.05]],float)
angle=.3;c=np.cos(angle);s=np.sin(angle)
rotation=np.array([[c,0,-s],[0,1,0],[s,0,c]])
moved=body@rotation.T
upper=[1,3,5,7]
pivot=runtime_formula(body[upper],moved[upper],rotation,[0,0,0],[0,0,0],float(np.linalg.norm(np.ptp(body,axis=0))),0.,0.)
assert min(moved[:,2])==0. and pivot['anchor_height_gain_m']>.1 and pivot['conditional_min_height_gain_m']<=0.
# Algebraic identity across independent initial/current base transforms.
q0=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
q1=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
u0=np.array([.2,-.3,.5]);u1=np.array([-.1,.4,-.2])
n=np.array([0.,0,1.]);r2=q1@rotation@q0.T;t2=u1-r2@u0
p0=body@q0.T+u0;p1=moved@q1.T+u1
nn0=q0@n;nn1=q1@n;dd0=-nn0@u0;dd1=-nn1@u1
v2=r2.T@nn1-nn0
anchor_gain=float(nn1@p1[upper].mean(0)+dd1-nn0@p0[upper].mean(0)-dd0)
identity=anchor_gain+(p0-p0[upper].mean(0))@v2
actual=(p1@nn1+dd1)-(p0@nn0+dd0)
np.testing.assert_allclose(identity,actual,atol=1e-15,rtol=0)
D=float(np.linalg.norm(np.ptp(body,axis=0)))
assert (actual >= anchor_gain-D*np.linalg.norm(v2)-1e-14).all()

tests=dict(
    shared_bias_counterexample=dict(observed_translation_m=[0,0,.1],true_body_height_change_m=0.,
        measured_fit_rms_m=0.,reported_bound=bias,
        meaning='A common target-point bias leaves residual zero; 3*rms is not a physical sensor-error bound. This violates the uncalibrated margin premise, not the conditional geometric theorem.'),
    pivot_contact_counterexample=dict(upper_anchor_gain_m=pivot['anchor_height_gain_m'],
        entire_body_minimum_clearance_m=float(moved[:,2].min()),reported_bound=pivot,
        meaning='Upper surface motion is not whole-object clearance; the support vertex remains at zero.'),
    different_base_coordinate_identity=dict(maximum_identity_error_m=float(np.max(np.abs(identity-actual))),
        diameter_bound_valid=True,meaning='R maps initial base coordinates to current base coordinates. Separate transformed planes make the ideal geometric bound coordinate-correct.'))

# Inspect all locally present Task90 schemas, but no data row groups or parquet payloads.
data_root=ROOT/'data/behavior-1k/2026-challenge-demos'
schemas=[]
for folder in ['data/chunk-090','meta/episodes/chunk-090']:
    for path in sorted((data_root/folder).glob('*.parquet')):
        schema=pq.read_schema(path)
        schemas.append(dict(path=str(path),fields=[dict(name=f.name,type=str(f.type)) for f in schema],
                            schema_sha256=hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
                            payload_read=False))
manifest_path=ROOT/'outputs/training/task090_v3_50k_20260930_s050000/manifest.json'
manifest=read_json(manifest_path)
manifest_inventory=[]
for path in sorted((ROOT/'outputs/training').glob('task090_v3_50k_20260930_s*/manifest.json')):
    item=read_json(path)
    manifest_inventory.append(dict(path=str(path),top_level_keys=list(item),
        offline_proxy_only=item.get('offline_proxy_only'),result_kind=item.get('result_kind'),
        dataset_root=item.get('config',{}).get('dataset',{}).get('root'),
        contract_targets=item.get('config',{}).get('dataset',{}).get('contract_targets'),
        checkpoint_rule=item.get('config',{}).get('manifest',{}).get('checkpoint_rule')))
labels_path=manifest_path.parent/'train_labels.json';labels=read_json(labels_path)
weak_semantics=sorted({label.get('label_semantics','not specified') for rows_ in labels['label_report'].values() for label in rows_})
contract_path=ROOT/'data/skill_vtt_task090/contract_targets.json';contract=read_json(contract_path)
info_path=data_root/'meta/info.json';info=read_json(info_path)
assert 18000 in manifest['train_episodes'] and 18001 in manifest['train_episodes']
schema_audit=dict(parquet_schemas=schemas,parquet_payload_rows_read=0,
    next_fields_present=['next.reward','next.terminated','next.truncated'],
    next_field_semantics='Schema says reward/done, not object-specific support, held, released or clearance labels; values were not read.',
    independent_object_support_outcome_fields=[],independent_object_support_labels_status='MISSING',
    opaque_state_feature=info['features']['observation.state'],
    opaque_state_note='61D list, names=[null]; schema alone cannot infer hidden labels. Existing producer treats it as robot proprioception.',
    manifest_path=str(manifest_path),manifest_top_level_keys=list(manifest),manifest_inventory=manifest_inventory,
    training_split_size=len(manifest['train_episodes']),validation_split_size=len(manifest['validation_episodes']),
    offline_proxy_only=manifest['offline_proxy_only'],checkpoint_rule=manifest['config']['manifest']['checkpoint_rule'],
    training_labels_path=str(labels_path),training_label_semantics=weak_semantics,
    contract_targets_path=str(contract_path),contract_target_training_recipe=contract['training_recipe'],
    contract_target_validation_requirement=contract['validation_requirement'],
    can_calibrate_physical_clearance_from_these_labels=False,runtime_may_use_gt=False)

assumptions=[
    dict(term='anchor_gain',source='task090_rigid_observation_v2.py:69-71',unit='m',
         meaning='Difference of signed distances of the same accepted material-point centroid to two normalized measured planes.',
         dependency='Current RGB-D, initial RGB-D, both camera extrinsics and both plane fits; initial target binding and rigidity not independently admitted.'),
    dict(term='diameter*norm(R.T*n1-n0)',source='task090_rigid_observation_v2.py:72',unit='m',
         meaning='Exact worst-direction diameter relaxation: dh(p)=gain+(R.T*n1-n0).(p-centroid).',
         dependency='A valid upper diameter for the entire same rigid body and all reference anchors belonging to it. Native asset bbox at assumed unit scale is a prior, not observed instance certification; link_0 pose/rigidity is not supplied here.'),
    dict(term='origin/current plane margins',source='task090_perception.py:89-106; task090_rigid_observation_v3.py:415-419',unit='m',
         meaning='Each is 3*max(vertical residual IQR,0.002). Two-frame errors can be summed without independence only IF each is a valid bound; no such calibration is present.',
         dependency='Floor uses every eighth pixel, z-depth in (0.05,6), base |z|<0.15, least squares. No target-support polygon, support identity, intercept error or normal-angle confidence bound. Common/correlated camera and plane errors are not represented by IQR.'),
    dict(term='0.005+3*rms',source='task090_rigid_observation_v2.py:74-75',unit='m',
         meaning='Fixed engineering margin plus final-fit residual, not certified sensor uncertainty or 3-sigma coverage.',
         dependency='RMS is fit to the same retained anchors; coherent pose/camera/depth bias and correspondence ambiguity can leave RMS zero.'),
    dict(term='diameter*min(2,3*rms/max(spread,0.05))',source='task090_rigid_observation_v2.py:73-75',unit='m',
         meaning='Uncalibrated dimensionless orientation-error surrogate times a lever arm. 2 is a normal-difference ceiling; 0.05m is a denominator floor.',
         dependency='spread is norm of coordinate-axis ranges, not rotation-information conditioning. Near-collinear sets can have large spread; the .05 floor is not automatically conservative. The same RMS in two terms is engineering decomposition, not independent statistical errors.'),
    dict(term='reference high water',source='task090_rigid_observation_v4.py:199-207; task090_rigid_observation_v3.py:418-419',unit='m',
         meaning='Exactly zero in this original-only bank and all saved height rows; the old dependent-reference accumulation bug is not present here.',
         dependency='Bank errors initialize zero: no inverse-estimate propagation does not mean zero initial sensor error.'),
    dict(term='whole body clearance',source='task090_rigid_observation_v2.py:1-5; task090_rigid_observation_v3.py:433-449',unit='m',
         meaning='Height change is not absolute support clearance or holding. Positive change plus a verified initial support/nonpenetration relation would be additional evidence, but that relation and support footprint are absent.',
         dependency='Need entire geometry including unseen underside, uncertainty set, support surface identity/domain, physical contact/holding and temporal continuity; contact candidate is commanded-close obstruction, not force/contact sensing.')]
next_plan=dict(
    preferred_implementation='CPU-only whole-boundary/support observability preflight, not a revised positive-height threshold',
    quantity='L_t = inf_{(shape,T,support) in jointly feasible, independently calibrated set} min_{x in complete shape} (n_support dot (T*x) + d_support); check support polygon coverage and all relevant support surfaces; insufficient coverage -> Unknown',
    scope='A support-relative whole-body clearance component only, not a stand-alone pick verdict; retain independent hand/object relation, material identity, freshness and continuous evidence checks.',
    existing_observations=['Initial immutable sparse landmarks, initial/current head RGB-D and camera poses, saved SE(3) estimates with accepted correspondences',
       'Dense initial appearance surface existed in RAM (22357 points at 480) but NPZ saved only sparse bank; original RGB-D can reconstruct a partial surface, not complete shape',
       'Local static native bbox and per-link box metadata, not observed scaled instance shape or full link-pose envelope',
       'Observed near-robot floor sample is recomputable, but is not bound to the object support footprint'],
    missing=['A sensor-derived complete-object outer enclosure with certified bottom/occlusion coverage, or a declared independently validated shape family/size prior aligned to observed instance; partial point-cloud hull is not an outer enclosure',
       'Support identity/domain near the object plus uncertainty for plane normals/offsets and unobserved contact regions',
       'Joint pose/shape/support feasible-set coverage calibrated independently of the same fit residual',
       'Independent TRAIN object/support/held/release outcomes, including positive and negative/Unknown cases'],
    independent_validation=['Keep TRAIN-only fitting and episode-disjoint TRAIN calibration/audit; no dev/test backflow',
       'Do not use VTT phase/contact proxies, reward/done, closed gripper or the same fitted anchors as outcome labels',
       'Independent multi-view/known geometry measurements or audited annotations for complete boundary/support coverage; ambiguous occluded cases remain Unknown',
       'Deterministic CPU ground-truth geometry tests: pivot/rock with bottom contact, slide without lifting, support slope, moving camera, wrong-object track, texture ambiguity, coherent depth bias, missing underside, opening/articulated part; synthetic tests validate math only',
       'For any future GT-supported TRAIN calibration, isolate it from runtime; actual runtime consumes only allowed sensors and validated static calibration'],
    executable_now='Preflight can output measurable/missing fields and test mathematical counterexamples. A physically admitted full-object clearance producer cannot be calibrated from current schemas alone.',
    threshold_changes=0,positive_candidates_created=0,runtime_changed=False)

visual_inspection=[]
for frame in [480,870]:
    rgb_path=Path(rows[frame]['rgb'])
    assert digest(rgb_path)==rows[frame]['rgb_sha256']
    visual_inspection.append(dict(frame=frame,path=str(rgb_path),method='assistant direct image inspection',
        observation='Bin extends beyond lower image boundary; full underside/contact region is not visible.',
        independent_physical_label=False,scope='Only this selected head view; no claim that every other available view is unusable.'))

result=dict(status='directed_cpu_audit_complete_not_admission',source_hash_matches=source_hash_matches,
    selected_view_observability=visual_inspection,
    analyzed_episode=18000,analyzed_frames=SELECTED,depth_frames_recomputed=[480]+SELECTED,
    tracker_frames_rerun=0,parquet_payload_rows_read=0,gpu_used=False,simulator_used=False,
    diameter_prior=dict(diameter_m=diameter,native_bbox_m=prior['native_bbox_m'],asset=str(metadata_path),
                        scale_assumption=cal['geometry_scope']['scale_assumption'],metadata_links=list(metadata['meta_links'])),
    frames=results,
    saved_height_rows=dict(count=len(height_stats),maximum_bound_m=max(x[1] for x in height_stats),
                           maximum_bound_frame=max(height_stats,key=lambda x:x[1])[0],
                           reference_high_water_range_m=[min(x[2] for x in height_stats),max(x[2] for x in height_stats)]),
    confirmed_implementation_error_in_height_arithmetic=False,
    verdict='Recorded key components exactly reproduce. Ideal geometric bound is valid under stated premises; engineering uncertainty margins and complete-object/support premises are uncalibrated. No justified subtraction or shrinkage is established.',
    assumptions=assumptions,counterexamples=tests,schema_audit=schema_audit,next_implementation=next_plan,
    source_sha256=HASHES,
    resources=dict(cpu_affinity=list(os.sched_getaffinity(0)),wall_seconds=time.monotonic()-START,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,threads_requested=1))
result=json.loads(json.dumps(result,default=plain,ensure_ascii=False,allow_nan=False))
dump('components.json',result)
dump('parquet_schema_audit.json',schema_audit)
print(json.dumps(dict(output=str(OUT/'components.json'),frames=[dict(frame=x['frame'],**x['components'],
    hand_motion_gate_pass=x['hand_motion_gate_pass']) for x in result['frames']],schema_status='MISSING',
    resources=result['resources']),ensure_ascii=False,indent=2))
