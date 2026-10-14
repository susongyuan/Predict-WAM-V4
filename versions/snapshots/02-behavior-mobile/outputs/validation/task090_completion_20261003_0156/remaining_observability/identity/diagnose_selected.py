#!/usr/bin/env python3
"""Selected-window CPU diagnostic; no production edits and no all-frame replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path('/home/user/Robotic/project2');sys.path.insert(0,str(ROOT))
import cv2,numpy as np,psutil
from PIL import Image
from types import SimpleNamespace
from harness.object_perception import Camera,camera_from_pose
from harness.task090_rigid_observation_v4 import seeded_flow_matches
from harness.task090_rigid_observation_v3 import Task090RigidObservationV3,sample_points,original_anchor_check
from harness.task090_rigid_observation_v2 import robust_rigid,fit_rigid
from harness.robot_self_geometry import RobotSelfGeometry
BASE=ROOT/'outputs/validation/task090_completion_20261003_0156'
OLD=BASE/'self_geometry_usd/implementation/continuous_train_01'
OUT=BASE/'remaining_observability/identity'
cv2.setNumThreads(1);proc=psutil.Process();proc.cpu_affinity([min(proc.cpu_affinity())]);start=time.monotonic()
cal=json.loads((ROOT/'configs/task090_perception_geometry_v2.json').read_text());geometry=RobotSelfGeometry()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,np.ndarray): return clean(x.tolist())
 if isinstance(x,np.generic): return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x): return None
 if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [clean(v) for v in x]
 return x
def write(name,obj):
 with (OUT/name).open('x') as f:json.dump(clean(obj),f,ensure_ascii=False,allow_nan=False,indent=2)
def guard():
 if psutil.virtual_memory().percent>=85 or psutil.cpu_percent(interval=None)>=85: raise RuntimeError('CPU/RAM 85 guard')
rows={(r['episode'],r['frame_index']):r for r in map(json.loads,(BASE/'continuous_train/extracted/frames.jsonl').open())}
oldhash={str(p):digest(p) for p in OLD.glob('*') if p.is_file()}
inputs={};images={}
def load(ep,f):
 if (ep,f) in images:return images[ep,f]
 guard();row=rows[ep,f]
 for key in ('rgb','depth_m'):
  p=row[key];assert digest(p)==row[key+'_sha256'];inputs[p]=row[key+'_sha256']
 rgb=np.asarray(Image.open(row['rgb'])).copy();depth=np.load(row['depth_m']);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
 cam=Camera(*camera_from_pose(np.asarray(row['camera_pose7'])),720,720)
 appearance=Task090RigidObservationV3._appearance(SimpleNamespace(calibration=cal,target='trash_can'),rgb,depth)
 images[ep,f]=dict(rgb=rgb,gray=gray,depth=depth,camera=cam,appearance=appearance,row=row)
 return images[ep,f]
def project(ref,r,t,cam):
 p=(ref@r.T+t-cam.t)@cam.R
 uv=p[:,:2]/p[:,2,None]*cam.f+[cam.cx,cam.cy]
 valid=(p[:,2]>.05)&np.isfinite(uv).all(1)&(uv[:,0]>=1)&(uv[:,0]<719)&(uv[:,1]>=1)&(uv[:,1]<719)
 return uv.astype(np.float32),p[:,2],valid

def flow(old,cur,uv,seeds):
 opt=dict(winSize=(31,31),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
 out,ok,error=cv2.calcOpticalFlowPyrLK(old,cur,np.float32(uv)[:,None].copy(),np.float32(seeds)[:,None].copy(),criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),**opt)
 finite=np.isfinite(out[:,0]).all(1);rin=out.copy();rin[~finite,0]=seeds[~finite]
 back,bok,berr=cv2.calcOpticalFlowPyrLK(cur,old,rin,np.float32(uv)[:,None].copy(),**opt)
 out=out[:,0];fb=np.linalg.norm(back[:,0]-uv,axis=1)
 valid=ok[:,0].astype(bool)&bok[:,0].astype(bool)&finite&np.isfinite(back[:,0]).all(1)&(fb<=1.5)&(out[:,0]>=1)&(out[:,0]<719)&(out[:,1]>=1)&(out[:,1]<719)
 # Independent equality check against the production primitive, no new thresholds.
 actual,av=seeded_flow_matches(old,cur,uv,seeds);np.testing.assert_array_equal(valid,av);np.testing.assert_array_equal(actual,out)
 return out,valid,dict(forward_status=ok[:,0],reverse_status=bok[:,0],fb_error_px=fb,lk_patch_error=error[:,0])

def patch(im,uv):
 u,v=np.rint(uv).astype(int);d=im['depth'];g=im['gray']
 if not(2<=u<718 and 2<=v<718):return dict(depth_m=None,depth_patch_5x5_m=None,depth_span_5x5_m=None)
 dp=d[v-2:v+3,u-2:u+3];gp=g[max(0,v-15):min(720,v+16),max(0,u-15):min(720,u+16)]
 return dict(depth_m=d[v,u],depth_patch_5x5_m=dp,depth_span_5x5_m=float(np.ptp(dp)),gray_std_31px=float(gp.std()),rgb=im['rgb'][v,u])

def fit_detail(ref,points,ids):
 result=dict(status='unknown',input_ids=ids,iterations=[])
 try:
  r,t,rms,keep=robust_rigid(ref,points,.01)
  result['ransac_initial_ids']=ids[keep]
  while True:
   if keep.sum()<6:raise ValueError('too_few_final_consistent_correspondences')
   r,t,_=fit_rigid(ref[keep],points[keep]);res=np.linalg.norm(ref@r.T+t-points,axis=1);new=keep&(res<=.01)
   result['iterations'].append(dict(ids=ids[keep],residual_m=res,retained_ids=ids[new]))
   if np.array_equal(new,keep):break
   keep=new
  ar,at,arms,ak,amax=original_anchor_check(ref,points)
  np.testing.assert_array_equal(ak,keep)
  result.update(status='rigid_motion_measured',accepted_ids=ids[keep],rotation=r,translation_m=t,rms_m=float(np.sqrt(np.mean(res[keep]**2))),max_residual_m=float(res[keep].max()))
 except ValueError as e:result['reason']=str(e)
 return result

def diagnose(ep,frame,bank,source_gray,source_uv,ids,r,t,source_frame):
 im=load(ep,frame);seeds,z,visible=project(bank['reference'],r,t,im['camera']);ids=np.asarray(ids,dtype=int);ids=ids[visible[ids]]
 source_uv=np.asarray(source_uv)[ids]
 uv,valid,metrics=flow(source_gray,im['gray'],source_uv,seeds[ids])
 points=[];passed=[];point_records=[]
 for j,i in enumerate(ids):
  u,v=np.rint(uv[j]).astype(int);inbounds=0<=u<720 and 0<=v<720
  dv=bool(inbounds and np.isfinite(im['depth'][v,u]) and .05<im['depth'][v,u]<cal['depth_max_m'])
  app=bool(inbounds and im['appearance'][v,u]>0)
  pt=sample_points(im['camera'],im['depth'],uv[j:j+1])[0] if dv else np.full(3,np.nan)
  keep=bool(geometry(pt[None],np.asarray(im['row']['state61']))[0]) if dv and valid[j] and app else None
  accepted=bool(valid[j] and dv and app and keep)
  if accepted:points.append(pt);passed.append(i)
  su,sv=np.rint(seeds[i]).astype(int);seed_depth=im['depth'][sv,su]
  point_records.append(dict(id=int(i),initial_uv=bank['uv'][i],source_uv=source_uv[j],seed_uv=seeds[i],seed_z_m=z[i],seed_depth_minus_predicted_z_m=float(seed_depth-z[i]),measured_uv=uv[j],lk_valid=valid[j],depth_valid=dv,appearance_valid=app,self_keep=keep,passed=accepted,current_point_m=pt,measured=patch(im,uv[j]),at_seed=patch(im,seeds[i]),**{k:value[j] for k,value in metrics.items()}))
 passed=np.asarray(passed,dtype=int);points=np.asarray(points).reshape(-1,3)
 result=dict(episode=ep,frame=frame,source_frame=source_frame,source_gray_sha256=hashlib.sha256(source_gray.tobytes()).hexdigest(),projected_ids=ids,not_projected_ids=np.flatnonzero(~visible),lk_pass_count=int(valid.sum()),after_depth_count=sum(p['lk_valid'] and p['depth_valid'] for p in point_records),after_appearance_count=sum(p['lk_valid'] and p['depth_valid'] and p['appearance_valid'] for p in point_records),self_rejected=sum(p['self_keep'] is False for p in point_records),points=point_records,fit=fit_detail(bank['reference'][passed],points,passed))
 return result

all_results=[];short_results=[];initial=[]
windows={18000:(480,793,range(794,800)),18001:(870,945,range(946,952))}
for ep,(init,kf,frames) in windows.items():
 bank=dict(np.load(OLD/f'episode_{ep}_original_bank.npz'))
 records={r['source_frame']['frame']:r for r in map(json.loads,(OLD/f'episode_{ep}_measurements.jsonl').open())}
 im=load(ep,init);assert np.array_equal(im['gray'],bank['gray'])
 np.testing.assert_allclose(sample_points(im['camera'],im['depth'],bank['uv']),bank['reference'],atol=1e-12)
 # Unannotated, native-size source images and unscaled pixel crops only; point indices stay in JSON.
 for f in (init,kf,*frames):
  item=load(ep,f);Image.fromarray(item['rgb']).save(OUT/f'ep{ep}_f{f}_rgb.png')
 Image.fromarray(im['rgb'][480:720,:]).save(OUT/f'ep{ep}_initial_bottom_crop.png')
 initial.append(dict(episode=ep,frame=init,gray_equal=True,anchor_reference_recomputed=True,source_rgb=im['row']['rgb'],point_count=len(bank['uv']),points=[dict(id=i,uv=uv,reference_m=bank['reference'][i],**patch(im,uv)) for i,uv in enumerate(bank['uv'])]))
 for f in frames:
  prev=max(j for j in records if j<f and records[j]['independent_measurement_check'] is not None);m=records[prev]['independent_measurement_check']
  result=diagnose(ep,f,bank,bank['gray'],bank['uv'],np.arange(len(bank['uv'])),np.asarray(m['rotation']),np.asarray(m['translation_m']),init)
  d=records[f]['rigid_tracking'];result['last_accepted_seed_frame']=prev;result['original_record_reason']=d.get('reason');result['original_record_status']=d['status']
  assert result['lk_pass_count']==d['original_anchors_after_lk'],(ep,f,result['lk_pass_count'],d['original_anchors_after_lk'])
  assert result['self_rejected']==d['original_anchors_self_rejected']
  if result['fit']['status']=='unknown':assert d['reason']=='ValueError:'+result['fit']['reason']
  else:np.testing.assert_array_equal(result['fit']['accepted_ids'],records[f]['independent_measurement_check']['original_ids'])
  result['matches_original_record']=True;all_results.append(result)
 # Single short-keyframe counterexample. Only IDs from accepted prior keyframe;
 # original 3D never replaced by inverse pose, and no dependent/new points enter.
 m=records[kf]['independent_measurement_check'];ids=np.asarray(m['original_ids']);uv=np.zeros_like(bank['uv']);uv[ids]=m['uv'];r=np.asarray(m['rotation']);t=np.asarray(m['translation_m']);sg=load(ep,kf)['gray'];sf=kf
 for f in frames:
  result=diagnose(ep,f,bank,sg,uv,ids,r,t,sf);result['diagnostic_only']=True;result['immutable_reference_sha256']=hashlib.sha256(bank['reference'].tobytes()).hexdigest();short_results.append(result)
  if result['fit']['status']=='rigid_motion_measured':
   ids=np.asarray(result['fit']['accepted_ids']);match={p['id']:p['measured_uv'] for p in result['points']};uv[ids]=[match[i] for i in ids];r=np.asarray(result['fit']['rotation']);t=np.asarray(result['fit']['translation_m']);sg=load(ep,f)['gray'];sf=f
  else:break
write('initial_points.json',initial);write('direct_failure_details.json',all_results);write('short_keyframe_counterexample.json',short_results)
summary=dict(status='selected_window_cpu_diagnostic_complete_not_admitted',target='trash_can',windows={ep:list(v[2]) for ep,v in windows.items()},initial_frames={ep:v[0] for ep,v in windows.items()},short_keyframe_start={ep:v[1] for ep,v in windows.items()},rgbd_pairs_loaded=len(images),direct_recomputed=len(all_results),all_direct_results_match_original=True,short_results=[{k:r[k] for k in ('episode','frame','source_frame','lk_pass_count','self_rejected','fit')} for r in short_results],direct_summary=[dict(episode=r['episode'],frame=r['frame'],lk=r['lk_pass_count'],fit=r['fit']) for r in all_results],thresholds=dict(lk_fb_px=1.5,rigid_m=.01,min_inliers=6),threshold_search=False,new_landmarks=False,original_3d_reference_changed=False,runtime_changed=False,policy_or_simulation=False,gpu_used=False,physical_success_label=None,admitted=False,verified=False,seconds=time.monotonic()-start,rss_mib=proc.memory_info().rss/2**20,source_hashes=inputs,old_output_hashes=oldhash)
assert all(digest(p)==h for p,h in oldhash.items());summary['original_outputs_unchanged']=True
summary['diagnostic_script_sha256']=digest(__file__)
write('machine_evidence.json',summary)
print(json.dumps(clean({k:summary[k] for k in ('rgbd_pairs_loaded','direct_recomputed','all_direct_results_match_original','short_results','seconds','rss_mib')}),ensure_ascii=False))
