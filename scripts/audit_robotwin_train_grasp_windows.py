#!/usr/bin/env python3
"""TRAIN-only broad gate inspection and images around first closure of each arm."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,h5py,cv2
from PIL import Image,ImageDraw
from omegaconf import OmegaConf
from harness.wam_v3 import create_v3_action_model,WamV3Wrapper,history_frames
from harness.robotwin_eval_evidence import ResourceGuard
from harness.robotwin_dataset import decode_rgb
class GateCaptured(Exception):pass
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(2);guard=ResourceGuard();guard.check()
    base=ROOT/'outputs/validation/target_binding_train_20261006'
    cases=json.loads((base/'selection.json').read_text())['episodes']+json.loads((base/'reserved_selection.json').read_text())['episodes']
    if len(cases)!=len(set(cases)):raise ValueError('duplicate episodes')
    run=ROOT/'outputs/runs/put_object_cabinet/wam_30k_20261005'
    ck=torch.load(run/'checkpoint_step_030000.pt',map_location='cpu',weights_only=False,mmap=True);cfg=ck['contract']['config']
    m=create_v3_action_model(OmegaConf.create(cfg),dino_hidden_size=1024,num_dino_layers=3,task_cond_dim=1024,patch_size=16)
    m.load_state_dict(ck['model_state_dict'],strict=True);del ck
    w=WamV3Wrapper(m,views=cfg['model']['views'],stats=json.loads((run/'temporal_stats.json').read_text()),norm_stats_path=run/'normalization.json',normalization_key='robotwin',proprio_mask='native_all14',proprio_keep_override=np.ones(14),device='cpu').eval()
    captured=[]
    def hook(module,args,value):captured.append(value.detach().softmax(-1).tolist());raise GateCaptured()
    handle=m.gate_head.register_forward_hook(hook);records=[]
    for i,name in enumerate(cases):
        guard.check();cache=next((ROOT/'data/robotwin_features').glob('*/'+name.removesuffix('.hdf5')+'-*/feat.npy'));arr=np.load(cache,mmap_mode='r');pool=np.load(cache.parent/'pool.npy',mmap_mode='r')
        with h5py.File(ROOT/'data/benchmarks/robotwin2/put_object_cabinet_500_v1/data'/name) as f:
            s=np.concatenate([np.asarray(f['state/'+k]) for k in ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')],-1)
            closings={side:int(idx[0]) if len(idx:=np.flatnonzero(s[:,col]<.2)) else None for side,col in [('left',6),('right',13)]}
            ts=sorted({0,*[max(0,min(len(s)-1,t+d)) for t in closings.values() if t is not None for d in (-16,-8,0,8)]})
            case={'episode':name,'first_command_below_0_2':closings,'records':[]}
            sheet=Image.new('RGB',(320*3,264*len(ts)),'white');draw=ImageDraw.Draw(sheet)
            for row,t in enumerate(ts):
                guard.check();b=torch.from_numpy(np.array(arr[t],copy=True).view(np.int16)).view(torch.bfloat16).float()[None]
                state=torch.from_numpy(s[[t]+history_frames(t,0,cfg['model']['v3']['proprio_lags'])])[None]
                mem=torch.from_numpy(np.array(pool[[t]+history_frames(t,0,cfg['model']['v3']['memory_lags'])].transpose(1,0,2),copy=True))[None]
                with torch.inference_mode():
                    try:m(torch.zeros(1),noisy_actions=torch.zeros(1,32,14),qpos_history=w.normalize_state(state),dino_features_list=[b[:,:,j] for j in range(3)],task_cond=torch.zeros(1,1024),memory=w.normalize_memory(mem))
                    except GateCaptured:pass
                case['records'].append({'observation_index':t,'view_coefficients_head_left_right':captured[-1][0],'commanded_grippers':s[t,[6,13]].tolist()})
                for col,view in enumerate(('cam_head','cam_left_wrist','cam_right_wrist')):
                    image=Image.fromarray(decode_rgb(f['vision/'+view+'/colors'][t])).resize((320,240))
                    sheet.paste(image,(col*320,row*264+24));draw.text((col*320+4,row*264+4),f'{view} t={t} gate={captured[-1][0][col]:.3f}',fill='black')
            sheet.save(out/(name.removesuffix('.hdf5')+'.png'));records.append(case)
            with (out/'records.jsonl').open('a') as fp:fp.write(json.dumps(case)+'\n')
            print('CASE',i,name,closings,flush=True)
    handle.remove();r={'scope':'32 TRAIN episodes only; no target visibility labels inferred from gate','checkpoint':'strict 30k','episodes':len(records),'frames':sum(len(r['records']) for r in records),'mean_gate':np.mean([q['view_coefficients_head_left_right'] for r in records for q in r['records']],axis=0).tolist(),'resources':guard.peaks,'CUDA_initialized':torch.cuda.is_initialized(),'limitations':'Gate coefficients are attention bias, not causal utility. Wrist visibility must be manually annotated. Closures are command events, not physical grasp.'}
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
