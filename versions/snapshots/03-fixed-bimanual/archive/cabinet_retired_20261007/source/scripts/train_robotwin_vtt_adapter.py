#!/usr/bin/env python3
"""Bounded VTT condition adaptation with frozen 30k WAM and TRAIN-only VTTs."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,json,argparse,time,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch.utils.data import DataLoader,default_collate
from omegaconf import OmegaConf
from harness.robotwin_vtt_adapter import install
from harness.robotwin_training_data import RobotwinCachedDataset
from harness.robotwin_policy_worker import sha256,verify_gpu_owner,atomic_json
from harness.robotwin_eval_evidence import ResourceGuard
from harness.wam_v3 import WamV3Wrapper,create_v3_action_model
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu-lock-owner',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--validation-samples',type=int,default=256);a=p.parse_args()
    if not 1<=a.steps<=3000 or not 1<=a.batch_size<=16 or not 8<=a.validation_samples<=512:raise ValueError('bounded adaptation only')
    out=Path(a.out).resolve();out.mkdir(exist_ok=False);verify_gpu_owner(a.gpu_lock_owner);torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.cuda.set_per_process_memory_fraction(.8);guard=ResourceGuard();guard.check();run=ROOT/'outputs/runs/put_object_cabinet/wam_30k_20261005';checkpoint=run/'checkpoint_step_030000.pt';base_hash=sha256(checkpoint)
    receipt=json.loads((run/'checkpoint_step_030000.pt.sha256.json').read_text())
    if base_hash!=receipt['sha256']:raise ValueError('base checkpoint content changed')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);cfg=ck['contract']['config'];bank=ROOT/'outputs/validation/cabinet_vtt_bank_20261006';bank_result=json.loads((bank/'result.json').read_text());proto=np.load(bank/'action_vtt_prototype.npy');instances=np.load(bank/'vtt_instances.npy');names=[r['episode'] for r in bank_result['records']];mapping={name:i for i,name in enumerate(names)}
    split=ROOT/cfg['dataset']['split_manifest'];cache=ROOT/cfg['dataset']['feature_cache'];v3=cfg['model']['v3'];ds={s:RobotwinCachedDataset(split,s,cache,v3['proprio_lags'],v3['memory_lags']) for s in ('train','val')}
    if names!=ds['train'].episodes or len(mapping)!=400 or proto.shape!=(1024,) or instances.shape!=(400,1024) or not np.isfinite(instances).all():raise ValueError('VTT TRAIN bank mismatch')
    model=create_v3_action_model(OmegaConf.create(cfg),dino_hidden_size=1024,num_dino_layers=3,task_cond_dim=1024,patch_size=16);model.load_state_dict(ck['model_state_dict'],strict=True);del ck;model.requires_grad_(False)
    random.seed(20261006);np.random.seed(20261006);torch.manual_seed(20261006);adapter=install(model)
    stats=json.loads((run/'temporal_stats.json').read_text());wrapper=WamV3Wrapper(model,views=cfg['model']['views'],stats=stats,norm_stats_path=run/'normalization.json',normalization_key='robotwin',proprio_mask='native_all14',proprio_keep_override=np.ones(14),train_cfg=cfg['training'],device='cuda').to('cuda').eval()
    # Action loss only; frozen future decoder is irrelevant to condition fit.
    model.use_future_feat=False
    trainable=[n for n,q in wrapper.named_parameters() if q.requires_grad]
    if trainable!=['action_model.task_cond_proj.residual.weight']:raise ValueError('unexpected backbone updates')
    val_rng=np.random.default_rng(20261006);indices=val_rng.choice(len(ds['val']),size=a.validation_samples,replace=False).tolist()
    protocol={'base_checkpoint_sha256':base_hash,'strict_load':True,'backbone_frozen':True,'trainable_parameters':trainable,'VTT_source':'TRAIN-only DINOv3 layer-4 head endpoint CLS minus start','VTT_prototype_sha256':sha256(bank/'action_vtt_prototype.npy'),'VTT_instances_sha256':sha256(bank/'vtt_instances.npy'),'steps':a.steps,'batch_size':a.batch_size,'accumulation':2,'lr':1e-4,'weight_decay':.01,'seed':20261006,'instance_condition_probability':.2,'validation_indices':indices,'heldout_used':False,'baseline_and_adapter_noise_paired':True,'admission':'val paired flow MSE <=1.05*base and finite output; compatibility only, physical success requires actual tests','source_sha256':{str(p):sha256(p) for p in (Path(__file__),ROOT/'harness/robotwin_vtt_adapter.py')}}
    atomic_json(out/'protocol.json',protocol);atomic_json(out/'status.json',{'status':'initializing','optimizer_step':0,'target_steps':a.steps})
    def condition(batch,fit=False,rng=None):
        c=[]
        for ep in batch['episode']:
            if fit and rng.random()<.2:c.append(instances[mapping[ep]])
            else:c.append(proto)
        batch['task_cond']=torch.from_numpy(np.stack(c).astype(np.float32));return batch
    def validate():
        values={'base':[],'adapter':[]};adapter.eval()
        with torch.inference_mode():
            for lo in range(0,len(indices),a.batch_size):
                guard.check();verify_gpu_owner(a.gpu_lock_owner);batch=condition(default_collate([ds['val'][i] for i in indices[lo:lo+a.batch_size]]))
                for key in values:
                    torch.manual_seed(20261006+lo);adapter.enabled=key=='adapter';loss,info,_=wrapper(batch,{})
                    if not torch.isfinite(loss):raise ValueError('nonfinite validation')
                    values[key].append((info['mse'],len(batch['episode'])))
        adapter.enabled=True
        return {key:sum(v*n for v,n in rows)/sum(n for _,n in rows) for key,rows in values.items()}
    before=validate();atomic_json(out/'validation_before.json',before)
    if before['base']!=before['adapter']:raise ValueError('zero initialized adapter altered base')
    optimizer=torch.optim.AdamW(adapter.residual.parameters(),lr=1e-4,weight_decay=.01);rng=np.random.default_rng(20261006);generator=torch.Generator().manual_seed(20261006);loader=DataLoader(ds['train'],batch_size=a.batch_size,shuffle=True,generator=generator,num_workers=0,pin_memory=True);stream=iter(loader);start=time.monotonic();torch.manual_seed(20261006)
    for step in range(1,a.steps+1):
        guard.check();verify_gpu_owner(a.gpu_lock_owner);optimizer.zero_grad();loss_value=0.;adapter.train()
        for _ in range(2):
            try:batch=next(stream)
            except StopIteration:stream=iter(loader);batch=next(stream)
            batch=condition(batch,True,rng);loss,info,_=wrapper(batch,{})
            if not torch.isfinite(loss):raise ValueError('nonfinite adaptation loss')
            (loss/2).backward();loss_value+=float(loss.detach())/2
        grad=torch.nn.utils.clip_grad_norm_(adapter.residual.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if step%25==0 or step==a.steps:
            status={'status':'training','optimizer_step':step,'target_steps':a.steps,'loss':loss_value,'grad_norm':float(grad),'elapsed_seconds':time.monotonic()-start,'resources':guard.peaks};atomic_json(out/'status.json',status);print('UPDATE',step,'LOSS',loss_value,'ELAPSED',status['elapsed_seconds'],flush=True)
        if step%100==0 or step==a.steps:
            torch.save({'format':'robotwin-vtt-residual-adapter-v1','base_checkpoint_sha256':base_hash,'state_dict':{k:v.detach().cpu() for k,v in adapter.residual.state_dict().items()},'prototype':torch.from_numpy(proto.copy()),'steps':step,'protocol':protocol},out/f'adapter_step_{step:06d}.pt')
    after=validate();admitted=after['adapter']<=1.05*after['base'];result={'status':'completed','steps':a.steps,'validation_before':before,'validation_after':after,'offline_compatibility_admitted':admitted,'physical_success_tested':False,'heldout_used':False,'train_elapsed_seconds':time.monotonic()-start,'resources':guard.peaks}
    atomic_json(out/'result.json',result);atomic_json(out/'status.json',result);print('RESULT',result,flush=True)
if __name__=='__main__':main()
