"""CPU-only bounded prefix diagnostic. No trainer import, training, or simulator."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['P2_TASK_ID'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import sys, json, hashlib, time, signal
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

def angle(a, b):
    for r in (a,b):
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):
            raise ValueError('not SO(3)')
    return float(np.arccos(np.clip((np.trace(a.T @ b)-1)/2,-1,1)))

def poses(kin, chunk):
    return [[kin.fk(hand,np.r_[a[3:7],a[7:14] if hand=='left' else a[15:22]]) for hand in ('left','right')] for a in chunk]

def errors(kin, pred, target):
    pp, tt = poses(kin,pred), poses(kin,target)
    out=[]
    for p,t in zip(pp,tt):
        pr = (p[0][1].T@(p[1][0]-p[0][0]),p[0][1].T@p[1][1])
        tr = (t[0][1].T@(t[1][0]-t[0][0]),t[0][1].T@t[1][1])
        out.append({'left_m':float(np.linalg.norm(p[0][0]-t[0][0])), 'right_m':float(np.linalg.norm(p[1][0]-t[1][0])),
          'left_rad':angle(p[0][1],t[0][1]),'right_rad':angle(p[1][1],t[1][1]),
          'relative_m':float(np.linalg.norm(pr[0]-tr[0])),'relative_rad':angle(pr[1],tr[1])})
    return out

def gripper(chunk, n):
    # Sign classes indicate commanded side of half-aperture, NOT contact/grasp events.
    out={}
    for hand,col in [('left',14),('right',22)]:
        x=np.clip(chunk[:n,col],-1,1); s=np.sign(x).astype(int)
        out[hand]={'commands':x.tolist(),'finger_target_m':((x+1)*.025).tolist(),
          'initial_sign':int(s[0]),'internal_sign_transitions':[{'index':i,'from':int(s[i-1]),'to':int(s[i])} for i in range(1,n) if s[i]!=s[i-1]]}
    return out

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    import torch, psutil
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from harness.wam_v3 import WamV3Dataset, create_v3_action_model, WamV3Wrapper
    from harness.wam_behavior import hold_baseline
    from harness.r1pro_kinematics import R1ProKinematics
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    # Limit diagnostic CPU affinity to four cores, well below 85% machine capacity here.
    cpus=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,cpus[:max(1,min(4,int(len(cpus)*.5)))])
    def check():
        if psutil.virtual_memory().percent>=85: raise RuntimeError('RAM >=85%')
        if psutil.cpu_percent(interval=.1)>=85: raise RuntimeError('sampled global CPU >=85%')
    def alarm(*_): raise TimeoutError('300 second diagnostic walltime reached')
    signal.signal(signal.SIGALRM,alarm); signal.alarm(300)
    out=ROOT/'outputs/validation/task0_prefix4_diagnostic_20260930'
    out.mkdir(parents=True,exist_ok=True)
    def save(name,obj):
        with (out/name).open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)
    start=time.monotonic(); check()
    run=ROOT/'outputs/training/task0_v3_30k_20260929_s030000'
    cfg=json.loads((run/'config.json').read_text()); plan=json.loads((run/'validation_plan.json').read_text())
    chosen=[]
    for skill in sorted({r['skill'] for r in plan['fine']}):
        chosen.append(sorted((r for r in plan['fine'] if r['skill']==skill),key=lambda r:(r['episode'],r['anchor']))[0])
    assert 0<len(chosen)<=8 and plan['split']=='validation'
    def seed(r):
        s=f"wam-v3-v1:sample-k1:{plan['seed']}:{r['episode']}:{r['anchor']}"
        return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'little')%(2**63-1)
    ck=run/'checkpoint_step_030000.pt'
    source_files=[Path(__file__),run/'config.json',run/'validation_plan.json',ROOT/'harness/wam_v3.py',ROOT/'harness/wam_behavior.py',ROOT/'harness/r1pro_kinematics.py',ROOT/cfg['model']['v3']['stats_path'],ROOT/cfg['dataset']['stats_path'],ROOT/'configs/r1pro_eef_offsets.json']
    sources={str(p):sha(p) for p in source_files}
    save('manifest.json',{'samples':[{**r,'noise_seed':seed(r)} for r in chosen], 'selection':'first (episode,anchor) per skill from existing fine validation plan, chosen before inference',
       'checkpoint':str(ck),'checkpoint_sha256_sidecar':json.loads(Path(str(ck)+'.sha256.json').read_text()),'checkpoint_hash_recomputed':False,
       'sources':sources,'split':'validation; no development samples','same_noise':'one K1 full chunk reused for both prefixes',
       'prefix':'indices 0..3 vs 0..7 of same 32-step absolute command chunk, not temporal subsampling; online replans after H actions unless interrupted',
       'gripper_units':'dimensionless smooth command clipped [-1,+1], -1 closed/+1 open; each finger target (u+1)*0.025 m, zero half-open; sign transitions are not contact events',
       'geometry':'FK commanded joint targets in base frame, not realized motion; relative right pose expressed in left hand frame; SO3 geodesic radians',
       'scope':'CPU float32 K1, 10 Euler steps, cached frozen features; small interface diagnostic, not a checkpoint or exec4 capability evaluation',
       'walltime_limit_s':300,'threads':4,'affinity':sorted(os.sched_getaffinity(0))})
    inventories=[]
    for step in ('025000','030000'):
        rr=ROOT/f'outputs/training/task0_v3_30k_20260929_s{step}'
        vv=json.loads((rr/f'validation_{step}.json').read_text())
        inventories.append({'path':str(rr),'files':[p.name for p in sorted(rr.iterdir()) if p.is_file()], 'validation_row_keys':list(vv['rows'][0]),'full_predictions_retained':False})
    timing=[]
    for step in ('025000','030000'):
        rr=ROOT/f'outputs/behavior/closed_loop/task0_v3_30k_20260929_s{step}_dev10_k1'
        for p in rr.glob('policy_trace*.jsonl'):
            rows=[json.loads(l) for l in p.open()]
            entry={'source':str(p),'sha256':sha(p),'full_chunk_retained':False}
            for event,field in [('chunk','infer_ms'),('v3_context','total_vision_ms')]:
                a=[r[field] for r in rows if r['event']==event and field in r]
                entry[field]={'n':len(a),'mean':float(np.mean(a)),'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95)),'max':max(a)} if a else None
            entry['note']='infer_ms is chunk-triggering policy call elapsed time including other work, not isolated diffusion time; vision is per frame, do not add these overlapping metrics or infer exec4 throughput'
            timing.append(entry)
    save('existing_artifact_audit.json',{'training':inventories,'timing':timing})
    d,v=cfg['dataset'],cfg['model']['v3']
    ds=WamV3Dataset('validation',feature_cache=ROOT/d['feature_cache'],views=cfg['model']['views'],proprio_lags=v['proprio_lags'],memory_lags=v['memory_lags'],contract_targets=str(ROOT/d['contract_targets']),contract_targets_sha256=d['contract_targets_sha256'],seed=cfg['training']['seed'])
    check()
    ec=json.loads((ROOT/cfg['model']['vision_encoder']['checkpoint_path']/'config.json').read_text())
    model=create_v3_action_model(OmegaConf.create(cfg),dino_hidden_size=ec['hidden_size'],num_dino_layers=len(cfg['model']['vision_encoder']['feat_layers']),task_cond_dim=ec['hidden_size'],patch_size=ec['patch_size'])
    checkpoint=torch.load(ck,map_location='cpu',mmap=True,weights_only=False)
    assert checkpoint['step']==30000
    model.load_state_dict(checkpoint['model_state_dict'],strict=True); del checkpoint
    w=WamV3Wrapper(model,views=cfg['model']['views'],stats=json.loads((ROOT/v['stats_path']).read_text()),norm_stats_path=ROOT/d['stats_path'],proprio_mask=cfg['model']['proprio_mask'],train_cfg=cfg['training'],vision_encoder=None,feat_layers=cfg['model']['vision_encoder']['feat_layers'],num_register_tokens=ec['num_register_tokens'],device='cpu').eval()
    kin=R1ProKinematics(); results=[]
    for i,r in enumerate(chosen):
        check(); t=time.monotonic(); b=default_collate([ds[r['index']]])
        assert (int(b['episode'][0]),int(b['anchor'][0]),int(b['phase'][0]),int(b['active_hand'][0]))==(r['episode'],r['anchor'],r['phase'],r['active_hand'])
        f=b['feats'].view(torch.bfloat16).float()
        with torch.inference_mode():
            p,_=w.sample_actions([f[:,:,j] for j in range(f.shape[2])],b['state'],b['pool'],b['task_cond'],num_steps=10,generator=torch.Generator().manual_seed(seed(r)))
            held=hold_baseline(b['state'],32)[0].numpy()
        p=p[0].numpy(); target=b['action_sequence'][0].numpy()
        assert p.shape==target.shape==(32,23) and np.isfinite(p).all()
        # Save all raw outputs before metrics; never overwrite prior runs.
        with (out/f'anchor_{i}_full_chunk.npz').open('xb') as fh: np.savez(fh,pred=p,target=target,hold=held,state=b['state'].numpy())
        e=errors(kin,p,target); h=errors(kin,held,target)
        row={**r,'cpu_seconds':time.monotonic()-t,'steps':e,'hold_steps':h,'prefix':{}}
        for n in (4,8,32):
            row['prefix'][str(n)]={'pred':{k:float(np.mean([x[k] for x in e[:n]])) for k in e[0]},'hold':{k:float(np.mean([x[k] for x in h[:n]])) for k in h[0]},
              'gripper_pred':gripper(p,n),'gripper_demo':gripper(target,n),'gripper_hold':gripper(held,n),
              'gripper_command_mae':np.mean(np.abs(np.clip(p[:n,[14,22]],-1,1)-np.clip(target[:n,[14,22]],-1,1)),axis=0).tolist()}
        save(f'anchor_{i}_metrics.json',row); results.append(row)
        print(json.dumps({'anchor':i,'skill':r['skill'],'seconds':row['cpu_seconds'],'prefix':{n:row['prefix'][n]['pred'] for n in ('4','8')}}),flush=True)
    check()
    save('result.json',{'completed':len(results),'seconds':time.monotonic()-start,'cuda_initialized':torch.cuda.is_initialized(),'max_rss_kib':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
       'aggregate':{str(n):{mode:{k:float(np.mean([r['prefix'][str(n)][mode][k] for r in results])) for k in results[0]['steps'][0]} for mode in ('pred','hold')} for n in (4,8,32)},
       'limitations':['Three fine validation anchors are not a sufficient capability estimate.','Early-prefix error can be smaller simply because demonstration motion is smaller; compare hold, not only prefix means.','Same-anchor prefix comparison never observes the new state or new noise after exec4 replans.','No physical execution, press success, grasp contact or throughput was tested.','CPU float32 is not exact GPU bf16 reproduction.','Live clipping, base hold, arm gain and contracts can alter these raw model targets.','SO3 and relative-pose measures are command FK, not physical tracking error.']})
    assert sources=={str(p):sha(p) for p in source_files},'stable dependency changed during diagnostic'
    signal.alarm(0)

if __name__=='__main__': main()
