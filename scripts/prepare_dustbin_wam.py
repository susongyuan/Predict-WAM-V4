#!/usr/bin/env python3
"""Seal official clean50 without relabelling timing/actions or copying episode data."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,yaml
from harness.robotwin_dataset import RoboTwinEpisode,make_split
from harness.robotwin_policy_worker import atomic_json
RAW=ROOT/'data/benchmarks/robotwin2/put_bottles_dustbin_clean50'
OUT=ROOT/'data/benchmarks/robotwin2/put_bottles_dustbin_50_v1'
VERIFY=ROOT/'outputs/verification/dustbin_wam_20261007'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 raw=json.loads((RAW/'SOURCE.json').read_text());assert raw['status']=='complete' and raw['hdf5_files']==50
 files=sorted((RAW/'source').rglob('episode_*.hdf5'));assert len(files)==50
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'data').mkdir();VERIFY.mkdir(parents=True,exist_ok=True)
 rows=[];tiles=[]
 for i,p in enumerate(files):
  ep=RoboTwinEpisode(p);obs=ep.chunk(0,1);dest=OUT/'data'/p.name;os.link(p,dest)
  image=obs['images']['cam_head'];cv2.imwrite(str(VERIFY/(p.stem+'_head.png')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
  tile=cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),(320,240));cv2.putText(tile,p.stem,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);tiles.append(tile)
  rows.append({'source':str(p),'path':str(dest),'sha256':sha(p),'frames':len(ep),'state_dim':ep.state.shape[1],'action_dim':ep.actions.shape[1],'rgb_shapes':{k:list(v.shape) for k,v in obs['images'].items()},'frequency':ep.frequency,'left_gripper_range':[float(ep.state[:,6].min()),float(ep.state[:,6].max())],'right_gripper_range':[float(ep.state[:,13].min()),float(ep.state[:,13].max())]})
 for page in range(5):
  imgs=tiles[page*10:(page+1)*10];grid=np.vstack([np.hstack(imgs[:5]),np.hstack(imgs[5:])]);cv2.imwrite(str(VERIFY/f'first_heads_page{page}.jpg'),grid)
 split=make_split(OUT/'data',seed=20261004,counts=(40,5,5));atomic_json(OUT/'split_manifest.json',split)
 source={'status':'complete','task':'put_bottles_dustbin','task_config':'demo_clean','repo':raw['repo'],'revision':raw['revision'],'data_origin':'official demo_clean v1 files; no conversion, no action shift, no image recompression','raw_source_receipt':str(RAW/'SOURCE.json'),'raw_archive_sha256':raw['sha256'],'raw_source_sha256':sha(RAW/'SOURCE.json'),'episodes':rows,'nominal_frequency':15,'measured_clock_available':False,'state_semantics':'controller_joint_targets_and_gripper_command_not_measured_qpos','embodiment':'aloha-agilex dual-arm fixed-base','limitations':['open trash bin, no lid opening','three rigid bottles, no deformable physics'],'heldout_policy':'fixed40/5/5; heldout not encoded or fitted'}
 atomic_json(OUT/'SOURCE.json',source)
 cfg=yaml.safe_load((ROOT/'configs/fixed_bimanual/robotwin_cabinet_500_wam.yaml').read_text());cfg['dataset'].update(task_name='put_bottles_dustbin',task_config='demo_clean',split_manifest=str((OUT/'split_manifest.json').relative_to(ROOT)),split_sha256=sha(OUT/'split_manifest.json'),feature_cache='data/dustbin_dino_features')
 config=ROOT/'configs/fixed_bimanual/robotwin_dustbin_50_wam.yaml'
 with config.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
 atomic_json(VERIFY/'data_audit.json',{'status':'data_contract_passed_visibility_review_pending','episodes':rows,'total_frames':sum(r['frames'] for r in rows),'split':split,'config':str(config)})
 print(json.dumps({'episodes':len(rows),'frames':sum(r['frames'] for r in rows),'config':str(config)},indent=2))
if __name__=='__main__':main()
