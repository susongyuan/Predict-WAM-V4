#!/usr/bin/env python3
"""CPU-only conditional visibility for two frozen failure poses; no tracker or model calls."""
import os
# Process-local resource limits only; no runtime/config files are changed.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import psutil
from PIL import Image
from scipy.spatial.transform import Rotation

BASE = Path('/home/user/Robotic/project2/outputs/validation/task090_completion_20261003_0156')
IN = BASE/'dense_tracking/continuous_train_01'
OUT = BASE/'dense_visibility'
TOL = .01
LABELS = ['apparent_nearer_occluder', 'farther_free_space_contradiction', 'depth_compatible', 'outside_image', 'invalid']
TARGETS = [(18000,480,528),(18001,870,940)]
SENSOR_KEYS = ['episode','frame_index','rgb','depth_m','rgb_sha256','depth_m_sha256','camera_pose7','depth_unit','depth_shape']
INPUT_HASHES = {}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def track_file(path):
    path = Path(path)
    INPUT_HASHES[str(path)] = digest(path)
    return path

def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def to_json(x):
    if isinstance(x,np.ndarray): return to_json(x.tolist())
    if isinstance(x,(list,tuple)): return [to_json(v) for v in x]
    if isinstance(x,dict): return {k:to_json(v) for k,v in x.items()}
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(float,np.floating)) and not np.isfinite(x): return None
    if isinstance(x,np.floating): return float(x)
    return x

def camera(row):
    p = np.asarray(row['camera_pose7'],float)
    return Rotation.from_quat(p[3:]/np.linalg.norm(p[3:])).as_matrix() @ np.diag([1.,-1.,-1.]), p[:3]

def count_classes(c, include):
    return {label:int(np.sum(include & (c==i))) for i,label in enumerate(LABELS)}

def compute_stencil(depth, cand, robot, dmax):
    valid = cand & ~robot & np.isfinite(depth) & (depth>.05) & (depth<dmax)
    vv,uu = np.indices(depth.shape)
    keep = valid & ((vv%2)==(uu%2))
    keep[:2,:]=keep[-2:,:]=False
    keep[:,:2]=keep[:,-2:]=False
    for axis in (0,1):
        minus,plus = np.roll(depth,-2,axis),np.roll(depth,2,axis)
        keep &= np.roll(valid,-2,axis) & np.roll(valid,2,axis)
        with np.errstate(divide='ignore',invalid='ignore'):
            pred = 2./(1./minus+1./plus)
        keep &= np.isfinite(pred) & (np.abs(depth-pred)<=TOL)
    return valid,keep

def main():
    began=time.monotonic()
    cv2.setNumThreads(1)
    proc=psutil.Process()
    proc.cpu_affinity([min(proc.cpu_affinity())])
    cpu=psutil.cpu_percent(interval=.1)
    if cpu>=85 or psutil.virtual_memory().percent>=85: raise RuntimeError('CPU/RAM85 guard')
    calpath=track_file(IN/'source/configs/task090_perception_geometry_v2.json')
    cal=json.loads(calpath.read_text())
    dmax=float(cal['depth_max_m'])
    boxes=cal['classes']['trash_can']['hsv_boxes']
    manifest=track_file(BASE/'continuous_train/extracted/frames.jsonl')
    wanted={(e,f) for e,s,t in TARGETS for f in (s,t)}
    sensor={}
    for line in manifest.open():
        r=json.loads(line)
        k=(r['episode'],r['frame_index'])
        if k in wanted: sensor[k]={key:r[key] for key in SENSOR_KEYS}
    assert set(sensor)==wanted
    for name in ['harness/dense_rgbd_registration.py','harness/dense_rgbd_tracking.py',
                 'harness/object_perception.py','harness/task090_rigid_observation_v3.py',
                 'scripts/check_task090_dense_tracking.py']:
        track_file(IN/'source'/name)
    track_file(IN/'summary.json')
    results=[]
    for ep,start,fail in TARGETS:
        if psutil.virtual_memory().percent>=85: raise RuntimeError('RAM85 guard')
        trackpath=track_file(IN/f'episode_{ep}_frames.jsonl')
        rows={}
        for line in trackpath.open():
            r=json.loads(line)
            if r['frame'] in (start,fail): rows[r['frame']]=r
        refpath=track_file(IN/f'episode_{ep}_reference.npz')
        fitpath=track_file(IN/f'episode_{ep}_fits/{fail}.npz')
        with np.load(refpath,allow_pickle=False) as f: ref={k:f[k] for k in f.files if k!='state'}
        with np.load(fitpath,allow_pickle=False) as f: fit={k:f[k] for k in f.files}
        images={}
        for frame in (start,fail):
            sr=sensor[ep,frame]
            assert sr['camera_pose7']==rows[frame]['camera_pose7']
            for key in ('rgb','depth_m'):
                p=track_file(sr[key]); assert INPUT_HASHES[str(p)]==sr[key+'_sha256']
                assert rows[frame]['sensor_source'][key]==str(p)
            with Image.open(sr['rgb']) as im: rgb=np.asarray(im).copy()
            depth=np.load(sr['depth_m'],allow_pickle=False)
            assert rgb.shape==(720,720,3) and depth.shape==(720,720)
            images[frame]=(rgb,depth)
        assert np.array_equal(ref['rgb'],images[start][0]) and np.array_equal(ref['depth'],images[start][1])
        R0,t0=camera(sensor[ep,start]); R,t=camera(sensor[ep,fail])
        assert np.allclose(R0,ref['camera_R'],atol=1e-12) and np.array_equal(t0,ref['camera_t'])
        f,cx,cy=[float(ref['camera_'+key][0]) for key in ('f','cx','cy')]
        rgb,depth=images[fail]; H,W=depth.shape
        pose=np.asarray(rows[fail]['measurement']['transform_source_to_current_base'])
        selected=rows[fail]['measurement']['selected_hypothesis']
        assert np.array_equal(pose,fit['hypotheses'][selected])
        ids=np.flatnonzero(ref['holdout'])
        evaluation_ids=fit['source_hold_indices']
        assert np.isin(evaluation_ids,ids).all()
        evalpos=np.searchsorted(ids,evaluation_ids)
        hd=np.full(len(ids),np.nan); hd[evalpos]=fit['hold_distance_m']
        pd=np.full(len(ids),np.nan); pd[evalpos]=fit['hold_point_distance_m']
        evaluated=np.isfinite(hd); large=evaluated & (hd>TOL)
        p=ref['points'][ids]
        x=p @ pose[:3,:3].T+pose[:3,3]
        pc=(x-t)@R; z=pc[:,2]
        uv=pc[:,:2]/z[:,None]*f+np.array([cx,cy])
        source_uv=ref['uv'][ids]
        assert ((source_uv%2)==1).all()
        back0=(p-t0)@R0
        assert np.allclose(back0[:,:2]/back0[:,2,None]*f+[cx,cy],source_uv,atol=1e-8)
        assert np.allclose(back0[:,2],ref['depth'][source_uv[:,1],source_uv[:,0]],atol=1e-8)
        cand=fit['current_candidate_mask']; robot=fit['current_robot_mask']
        valid,stencil=compute_stencil(depth,cand,robot,dmax)
        savedcentre=np.zeros((H,W),bool)
        savedcentre[fit['current_uv'][:,1],fit['current_uv'][:,0]]=True
        assert np.array_equal(stencil,savedcentre), 'Reconstructed stencil differs from saved accepted centres'
        union=np.zeros(H*W,bool)
        union[fit['current_raw_stencil_indices'][fit['current_holdout']].reshape(-1)]=True
        union=union.reshape(H,W)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        hsvpass=np.zeros((H,W),bool)
        for lower,upper in boxes: hsvpass |= cv2.inRange(hsv,np.asarray(lower,np.uint8),np.asarray(upper,np.uint8))>0
        finiteproj=np.isfinite(uv).all(axis=1)&np.isfinite(z)&(z>0)
        inimage=finiteproj&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        arrays=dict(source_indices=ids,source_uv=source_uv,source_points_base_m=p,
                    source_z_depth_m=back0[:,2],source_raw_stencil_indices=ref['raw_stencil_indices'][ids],
                    source_normals_base=ref['normals'][ids],in_original_held_validation=evaluated,
                    saved_hold_patch_distance_m=hd,saved_hold_point_distance_m=pd,
                    saved_hold_residual_gt_1cm=large,transformed_points_current_base_m=x,
                    transformed_points_current_camera_m=pc,projected_uv=uv,predicted_z_m=z,
                    conditional_pose=pose,current_camera_R=R,current_camera_t=t,
                    current_intrinsics_f_cx_cy=np.array([f,cx,cy]),
                    current_candidate_mask=cand,current_robot_mask=robot,
                    current_valid_mask=valid,current_stencil_mask=stencil,
                    current_saved_surface_centres=savedcentre,current_held_raw_stencil_union=union,
                    current_hsv_pass_mask=hsvpass)
        scopes={}
        for scope in ('held_lattice_raw','full_raw_posthoc'):
            # Nearest measured pixel on a fixed lattice; ties to even lattice index.
            # No depth interpolation or nearest-valid-pixel search is permitted.
            safe=np.where(np.isfinite(uv),uv,0)
            q=(2*np.rint((safe-1)/2)+1).astype(int) if scope=='held_lattice_raw' else np.rint(safe).astype(int)
            qin=inimage&(q[:,0]>=0)&(q[:,0]<W)&(q[:,1]>=0)&(q[:,1]<H)
            qc=np.c_[np.clip(q[:,0],0,W-1),np.clip(q[:,1],0,H-1)]
            def lookup(a):
                out=a[qc[:,1],qc[:,0]].copy()
                if out.dtype==bool: out[~qin]=False
                return out
            obs=lookup(depth).astype(float); obs[~qin]=np.nan
            depthvalid=qin&np.isfinite(obs)&(obs>.05)&(obs<dmax)
            delta=obs-z
            classes=np.full(len(ids),4,np.uint8)
            classes[finiteproj&~qin]=3
            classes[depthvalid&(delta < -TOL)]=0
            classes[depthvalid&(delta > TOL)]=1
            classes[depthvalid&(np.abs(delta)<=TOL)]=2
            v=dict(query_uv=q,query_offset_uv=q-uv,query_in_image=qin,current_z_m=obs,
                   delta_current_minus_predicted_m=delta,raw_depth_valid=depthvalid,category=classes,
                   candidate_mask=lookup(cand),robot_mask=lookup(robot),stencil_mask=lookup(stencil),
                   in_saved_held_raw_stencil=lookup(union),hsv_pass=lookup(hsvpass))
            # This stored self mask is evaluated only inside candidate_mask, not a full-image robot label.
            v['robot_mask_evaluated']=v['candidate_mask']
            v['hsv_excluded']=depthvalid&~v['hsv_pass']
            v['roi_excluded_given_hsv_depth_valid']=depthvalid&v['hsv_pass']&~v['candidate_mask']
            if scope=='held_lattice_raw':
                assert ((q%2)==1).all()
                strict=classes.copy()
                strict[qin&~v['in_saved_held_raw_stencil']]=4
                v['existing_held_stencil_only_category']=strict
                v['existing_held_stencil_unavailable']=qin&~v['in_saved_held_raw_stencil']
                scopes['existing_held_stencil_only']={
                    'all_source_hold':count_classes(strict,np.ones(len(ids),bool)),
                    'original_validation_subset':count_classes(strict,evaluated),
                    'saved_residual_gt_1cm':count_classes(strict,large),
                    'missing_stencil_all':int(v['existing_held_stencil_unavailable'].sum()),
                    'missing_stencil_large':int((v['existing_held_stencil_unavailable']&large).sum())}
            cross=[]
            for flags,n in sorted(Counter((LABELS[int(classes[i])],bool(v['candidate_mask'][i]),
                                            bool(v['robot_mask'][i]),bool(v['stencil_mask'][i]),
                                            bool(v['in_saved_held_raw_stencil'][i]),bool(v['hsv_pass'][i]))
                                           for i in np.flatnonzero(large)).items()):
                cross.append(dict(zip(['category','candidate_mask','robot_mask','stencil_mask','in_saved_held_raw_stencil','hsv_pass'],flags),count=n))
            scopes[scope]={
                'all_source_hold':count_classes(classes,np.ones(len(ids),bool)),
                'original_validation_subset':count_classes(classes,evaluated),
                'saved_residual_gt_1cm':count_classes(classes,large),
                'large_residual_cross_masks':cross,
                'compatible_large_hsv_excluded':int((large&(classes==2)&v['hsv_excluded']).sum()),
                'compatible_large_roi_excluded':int((large&(classes==2)&v['roi_excluded_given_hsv_depth_valid']).sum()),
                'compatible_large_candidate_robot_stencil_missing':int((large&(classes==2)&v['candidate_mask']&~v['robot_mask']&~v['stencil_mask']).sum())}
            arrays.update({scope+'__'+key:value for key,value in v.items()})
        outliers=[]
        for i in np.flatnonzero(large):
            o={key:arrays[key][i] for key in ['source_indices','source_uv','source_points_base_m','projected_uv','predicted_z_m','saved_hold_patch_distance_m']}
            for scope in ('held_lattice_raw','full_raw_posthoc'):
                o[scope]={key:arrays[scope+'__'+key][i] for key in ['query_uv','current_z_m','delta_current_minus_predicted_m','candidate_mask','robot_mask','robot_mask_evaluated','stencil_mask','in_saved_held_raw_stencil','hsv_pass']}
                o[scope]['category']=LABELS[int(arrays[scope+'__category'][i])]
            outliers.append(to_json(o))
        lowids=np.flatnonzero(p[:,2]<.1)
        sampleids=np.unique(np.r_[np.argsort(p[:,2])[:8],np.argsort(p[:,2])[-8:]])
        samples=[to_json(dict(source_index=ids[i],source_uv=source_uv[i],source_xyz_m=p[i],
                              source_rgb=ref['rgb'][source_uv[i,1],source_uv[i,0]],
                              used_by_original_validator=evaluated[i])) for i in sampleids]
        summary=dict(episode=ep,source_frame=start,failure_frame=fail,status='conditional_diagnostic_not_admission',
                     source_hold_all_count=len(ids),saved_validation_hold_count=int(evaluated.sum()),
                     saved_validation_gt_1cm_count=int(large.sum()),
                     saved_validation_p95_m=float(np.percentile(hd[evaluated],95)),
                     initial_all_dense_z_span_m=float(np.ptp(ref['points'][:,2])),
                     initial_held_z_range_m=[float(p[:,2].min()),float(p[:,2].max())],
                     initial_held_z_below_010m_count=len(lowids),source_extreme_samples=samples,
                     selected_hypothesis=selected,scopes=scopes,
                     inputs={'reference_npz':str(refpath),'fit_npz':str(fitpath),'tracking_jsonl':str(trackpath),
                             'source_sensor':sensor[ep,start],'failure_sensor':sensor[ep,fail]},
                     selected_tracking_row_sha256={str(frame):hashlib.sha256(json.dumps(rows[frame],sort_keys=True,separators=(',',':')).encode()).hexdigest() for frame in (start,fail)},
                     all_large_residual_records=outliers,
                     arrays={key:{'shape':list(val.shape),'dtype':str(val.dtype)} for key,val in arrays.items()})
        npath=OUT/f'episode_{ep}_{fail}_visibility.npz'
        with npath.open('xb') as fobj: np.savez_compressed(fobj,**arrays)
        summary['raw_arrays_npz']={'path':str(npath),'sha256':digest(npath)}
        jpath=OUT/f'episode_{ep}_{fail}_visibility.json'
        write_json(jpath,to_json(summary))
        results.append({'episode':ep,'json':str(jpath),'json_sha256':digest(jpath),
                        'npz':str(npath),'npz_sha256':digest(npath),
                        'counts':{k:summary[k] for k in ['source_hold_all_count','saved_validation_hold_count','saved_validation_gt_1cm_count']},
                        'scopes':scopes})
        print(json.dumps({'episode':ep,'counts':results[-1]['counts'],'scopes':scopes,'source_extremes':samples,'outliers':outliers},ensure_ascii=False),flush=True)
    for p,sha in INPUT_HASHES.items(): assert digest(p)==sha,'Input changed during read-only diagnosis'
    report={'status':'conditional_diagnostic_not_admission','verified':False,'admitted':False,'physical_success_label':None,
            'pose_is_truth':False,'progress_version_read':'2026-10-04 04:50 CST',
            'threshold_m':TOL,'depth_max_m':dmax,'class_codes':dict(enumerate(LABELS)),
            'definitions':{
                'delta':'current raw camera z minus predicted camera z; <-.01 apparent nearer; >.01 farther/free-space contradiction; absolute <=.01 compatible',
                'existing_held_stencil_only':'Primary strict scope. Fixed nearest odd/odd query must belong to union of SAVED current held raw five-point stencils; unavailable classified invalid with explicit missing-stencil flag, never occluded.',
                'held_lattice_raw':'Diagnostic extension for mask-deleted locations: fixed nearest odd/odd raw pixel even if absent from saved stencils. No fit-fold depth, interpolation, fallback or pose refinement. Not independent admission.',
                'full_raw_posthoc':'Nearest raw pixel regardless of fold, expressly posthoc and not independent admission.',
                'pixel_rounding':'np.rint ties-to-even; held query within 1px per axis, full within .5px. Offsets saved; axial z test differs from stored 3D patch-distance metric.',
                'all_source_hold':'Every immutable original odd/odd surface point; saved residual available only at pre-existing separated validation indices. NaN means not evaluated, not a passing residual.',
                'stencil_mask':'Reconstructed original pre-normal validity stencil using frozen masks/depth; exact equality asserted against saved current_uv centre set.',
                'robot_mask':'Stored self-geometry exclusion; only evaluated within candidate_mask, false elsewhere means unassessed, not absent robot.',
                'identity':'Depth-compatible is not proof of same physical surface. Nearer is apparent depth ordering conditional on an unverified pose, not a physical occlusion label.'},
            'sensor_frames_read':sorted([list(k) for k in wanted]),'new_windows':0,'fits_run':0,'model_calls':0,'video_decodes':0,'gpu_used':False,
            'runtime_config_tests_progress_changed':False,'inputs_sha256':INPUT_HASHES,
            'script':{'path':str(Path(__file__).resolve()),'sha256':digest(__file__)},'results':results,
            'resources':{'wall_seconds':time.monotonic()-began,'rss_mib':proc.memory_info().rss/1024**2,'cpu_affinity':proc.cpu_affinity(),'initial_system_cpu_percent':cpu,'final_system_ram_percent':psutil.virtual_memory().percent}}
    write_json(OUT/'provenance.json',report)

if __name__=='__main__': main()
