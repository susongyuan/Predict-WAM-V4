#!/usr/bin/env python3
"""CPU preflight for V3. Reports integrity/contract checks, NOT robot success."""
import argparse, hashlib, json, os, sys, time, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('P2_TASK_ID', '0')
import numpy as np
import torch
from harness import task_profile
from harness.wam_v3 import (V3Context, history_frames, v3_model_class, load_v3_state,
    WamV3Dataset, episode_phase_labels, gate_prior, bf16_from_bits, pool_from_hidden)
from harness.wam_behavior import model_views, multiview_model_class, SkillTable

class UnitTests(unittest.TestCase):
    def test_history_bounded_exact(self):
        ctx = V3Context([4,8,16,32,64],[8,32],['h','l','r'])
        with self.assertRaises(RuntimeError): ctx.inputs()
        for t in range(333):
            ctx.push(np.full(61,t), np.full((3,2,8),t))
            s,p=ctx.inputs()
            np.testing.assert_array_equal(s[:,0],[t]+history_frames(t,0,[4,8,16,32,64]))
            np.testing.assert_array_equal(p[0,:,0],[t]+history_frames(t,0,[8,32]))
            self.assertLessEqual(len(ctx.states),65)
        ctx.reset(); ctx.push(np.ones(61)*999,np.ones((3,2,8))*999)
        self.assertEqual(ctx.steps,1); self.assertTrue(np.all(ctx.inputs()[0]==999))
        with self.assertRaises(ValueError): history_frames(5,6,[4])
        with self.assertRaises(ValueError): history_frames(5,0,[-4])

    def test_phase_and_no_missing_label_fabrication(self):
        a=np.ones((120,23)); s=np.zeros((120,61)); s[:,23]=1; s[:,48]=1; a[30:,22]=-1; a[90:,22]=1
        s[:,42]=np.maximum(0,30-np.arange(120))*.01
        segs=[dict(skill='pick up from',start=0,end=50),dict(skill='press',start=50,end=80),dict(skill='place on',start=80,end=120)]
        ph,di,ha,rep=episode_phase_labels(segs,a,s,60)
        self.assertEqual(rep[0]['hand'],'right');self.assertEqual(rep[1]['hand'],'left');self.assertEqual(rep[2]['event'],90)
        self.assertEqual(ph[0],0);self.assertEqual(ph[25],1);self.assertEqual(ph[30],2)
        ph2,_,_,_=episode_phase_labels(segs,a,s,None)
        self.assertTrue(np.all(ph2[50:80]==-1))

    def test_phase_consistent_frame(self):
        from harness.wam_v3 import phase_positions
        s=np.zeros((31,61));s[:,23]=s[:,48]=1
        s[:,0]=.3;s[:,17]=1;s[:,42]=.5
        p=phase_positions(s,0,31,"left")
        self.assertAlmostEqual(p[-1,0]-p[0,0],.3,places=6)
        s[:,17]-=np.arange(31)*.01
        p=phase_positions(s,0,31,"left")
        np.testing.assert_allclose(p[:,0],1,atol=1e-8)
        s[:,17]=s[:,42]+.2
        p=phase_positions(s,0,31,"left","right")
        np.testing.assert_allclose(p,np.tile([.2,0,0],(31,1)))
        # Co-rotating both hands leaves the holding-frame coordinate invariant.
        theta=np.linspace(0,1,31);s[:,45:49]=np.stack([theta*0,theta*0,np.sin(theta/2),np.cos(theta/2)],axis=1)
        s[:,17]=s[:,42]+.2*np.cos(theta);s[:,18]=.2*np.sin(theta)
        np.testing.assert_allclose(phase_positions(s,0,31,"left","right"),np.tile([.2,0,0],(31,1)),atol=1e-8)

    def test_memory_bits_pool(self):
        h=torch.randn(2,9,8).bfloat16()
        bits=h.view(torch.int16).numpy().view(np.uint16)
        self.assertTrue(torch.equal(h,bf16_from_bits(bits)))
        pool=pool_from_hidden(h,4)
        self.assertTrue(torch.equal(pool[:,1],h[:,5:].float().mean(1)))

    def test_real_blocks_backward_gate_and_strict_migration(self):
        torch.manual_seed(51); torch.set_num_threads(2)
        kw=dict(action_dim=23,proprio_dim=61,hidden_dim=32,num_heads=4,depth=2,action_len=4,
            proprio_len=1,num_registers=2,dino_feat_dims=(8,8,8),vlm_num_queries=3,adapter_depth=1,
            task_cond_dim=8,fusion_mode='concat',use_future_feat=True,future_feat_num_queries=4,
            future_feat_out_dim=8,future_feat_depth=1,future_feat_heads=4)
        old=multiview_model_class()(3,**kw)
        kw['proprio_len']=7
        m=v3_model_class()(3,3,memory_dim=16,phase_hidden=16,**kw)
        notes=load_v3_state(m,old.state_dict())
        self.assertEqual(notes['future_feat_decoder.query_embed'],'head queries tiled x3')
        bad=dict(m.state_dict());del bad['mem_proj.weight']
        with self.assertRaises(RuntimeError):load_v3_state(m,bad)
        load_v3_state(m,m.state_dict())
        # Trained upstream DiT gates are not zero; model scratch initialization zeros them by design.
        for b in m.blocks: torch.nn.init.normal_(b.adaLN_modulation[-1].weight,std=.05)
        m.eval()
        args=dict(t=torch.tensor([.5,.8]),noisy_actions=torch.randn(2,4,23),qpos_history=torch.randn(2,7,61),
            dino_features_list=[torch.randn(2,3,9,8) for _ in range(3)],task_cond=torch.randn(2,8),memory=torch.randn(2,3,3,16))
        out=m(**args);torch.testing.assert_close(out['gate'],torch.full((2,3),1/3))
        loss=out['final_pred'].square().mean()+out['phase_logits'].square().mean()+out['log_dist'].square().mean()
        loss=loss+torch.nn.functional.kl_div(out['gate'].log(),torch.tensor([[.6,.1,.3],[.2,.6,.2]]),reduction='batchmean')
        loss=loss+m.future_feat_decoder(out['cond_tokens']).square().mean();loss.backward()
        for name in ['mem_proj.weight','phase_net.1.weight','gate_head.weight','phase_head.weight','dist_head.weight']:
            grad=dict(m.named_parameters())[name].grad
            self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name);self.assertGreater(grad.abs().sum().item(),0,name)
        with torch.no_grad():
            m.gate_head.bias.copy_(torch.tensor([-10000.,0.,10000.]))
            self.assertTrue(torch.isfinite(m(**args)['final_pred']).all())


def audit(config,out):
    from omegaconf import OmegaConf
    from scripts.cache_dino_features import manipulation_frames
    from harness.wam_v3 import reward_frames
    cfg=OmegaConf.load(config); views=model_views(cfg);skills=SkillTable()
    index=json.loads((task_profile.LOWDIM/'index.json').read_text())
    split=json.loads(task_profile.SPLIT.read_text())
    tr=set(split['train_episode_indices']);va=set(split['validation_episode_indices'])
    assert not tr & va
    rows=0; total=0; files=0; signatures=[]
    for ep0,entry in index['episodes'].items():
        ep=int(ep0); expected=manipulation_frames(skills.episodes[ep]['segments'],entry['length'])
        for view in views:
            base=ROOT/cfg.dataset.feature_cache/view/f'episode_{ep}'
            meta=json.loads(Path(str(base)+'.meta.json').read_text())
            assert meta['complete'] and meta['episode']==ep and meta['view']==view
            assert meta['feat_layers']==list(cfg.model.vision_encoder.feat_layers)
            frames=np.load(str(base)+'.frames.npy');np.testing.assert_array_equal(frames,expected)
            ft=np.load(str(base)+'.feat.npy',mmap_mode='r');pool=np.load(str(base)+'.pool.npy',mmap_mode='r')
            assert ft.shape==(len(frames),3,201,1024) and ft.dtype==np.uint16
            assert pool.shape==(len(frames),2,1024) and pool.dtype==np.float32
            for arr,suffix in [(ft,'.feat.npy'),(pool,'.pool.npy')]:
                size=Path(str(base)+suffix).stat().st_size
                assert size==arr.offset+arr.nbytes
                total+=size
            src=task_profile.FRAMES/view/f'episode_{ep}.meta.json'
            assert hashlib.sha256(src.read_bytes()).hexdigest()==meta['source_frame_meta_sha256']
            for i in {0,len(frames)//2,len(frames)-1}:
                h=bf16_from_bits(np.array(ft[i,-1],copy=True))
                assert torch.isfinite(h).all()
                # Cache pooled on CUDA; CPU reduction has different FP32 summation order.
                np.testing.assert_allclose(pool_from_hidden(h).numpy(),pool[i],rtol=3e-7,atol=2e-5)
            signatures.append(meta);files+=1;rows+=len(frames)
            del ft,pool
    rewards=reward_frames()
    dsargs=dict(feature_cache=ROOT/cfg.dataset.feature_cache,views=views,proprio_lags=cfg.model.v3.proprio_lags,
       memory_lags=cfg.model.v3.memory_lags,contract_targets=ROOT/cfg.dataset.contract_targets,
       contract_targets_sha256=cfg.dataset.contract_targets_sha256,rewards=rewards)
    datasets={name:WamV3Dataset(name,**dsargs) for name in ['train','validation']}
    labels={}
    for name,ds in datasets.items():
        # All cached history/current/future indices must exist and remain inside assigned contract.
        for ep,t in ds.anchors:
            seg=ds.skills.segment(ep,t);s=int(seg['start']);e=min(seg['end'],seg['fill_end'])
            assert s<=t and t+32<e
            mf=[t,t+32]+history_frames(t,s+(s%2),ds.memory_lags)
            for f in mf:ds.cache.row(views[0],ep,f)
        labels[name]=ds.label_summary()
        item=ds[0];assert item['state'].shape==(7,61) and item['pool'].shape==(3,4,2048)
    stats=json.loads((ROOT/cfg.model.v3.stats_path).read_text())
    assert stats['split_sha256']==hashlib.sha256(task_profile.SPLIT.read_bytes()).hexdigest()
    assert stats['split']=='train only' and stats['train_episodes']==len(tr)
    result={'kind':'CPU cache/contract preflight; not policy or physical success','complete':True,'episodes':len(index['episodes']),
        'view_episode_packs':files,'cached_images':rows,'feature_pool_bytes':total,
        'content_check':'first/middle/last rows per pack finite + pooled CPU tolerance (rtol=3e-7, atol=2e-5); full payload not hashed',
        'metadata_sha256':hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest(),
        'anchors':{k:len(d) for k,d in datasets.items()},'phase_labels':labels,'train_validation_disjoint':True}
    out.mkdir(parents=True,exist_ok=True);(out/'cache_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='phase_labels'},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit',action='store_true');ap.add_argument('--config',default='configs/lilawam_task000_multiview_manip_v3_train.yaml');ap.add_argument('--out',default='outputs/validation/task0_v3_audit_20260928');args=ap.parse_args()
    if args.audit:audit(args.config,Path(args.out))
    else:
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(UnitTests));sys.exit(not r.wasSuccessful())
