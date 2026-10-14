#!/usr/bin/env python3
"""Small real-DiT CPU tests, never a robot-policy success claim or GPU run."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from torch import nn
from harness.wam_v3 import v3_model_class,WamV3Wrapper
from harness.wam_hjepa import (upgrade_wrapper,variant_settings,future_with_offset,inverse_loss,
                              gradient_evidence,HjepaAuxHeads)


def fixture():
    torch.manual_seed(20261009)
    model=v3_model_class()(num_views=3,num_memory=1,memory_dim=16,phase_hidden=16,
        action_dim=16,proprio_dim=16,hidden_dim=32,num_heads=4,depth=2,action_len=32,
        proprio_len=2,num_registers=2,dino_feat_dims=(8,8,8),vlm_num_queries=4,adapter_depth=1,
        state_inject_start=1,task_cond_dim=8,fusion_mode='concat',concat_out_dim=8,
        use_future_feat=True,future_feat_num_queries=2,future_feat_out_dim=8,future_feat_depth=1,future_feat_heads=4)
    stats={'views':['head','left','right'],'proprio_diff_std':[[1.]*16],
           'memory_diff_std':[[[1.]*16] for _ in range(3)]}
    with tempfile.TemporaryDirectory() as directory:
        norm=Path(directory)/'norm.json'
        norm.write_text(json.dumps({'action_repr':'absolute','native16':{
            'action':{'min':[-2.]*16,'max':[2.]*16},'state':{'min':[-2.]*16,'max':[2.]*16}}}))
        wrapper=WamV3Wrapper(model,views=stats['views'],stats=stats,norm_stats_path=norm,
            proprio_mask='native_all16',proprio_keep_override=[1.]*16,normalization_key='native16',
            device='cpu',vision_encoder=nn.Linear(1,1))
    b=2
    batch=dict(feats=torch.randn(b,3,3,7,8),state=torch.randn(b,2,16)*.2,
        pool=torch.randn(b,3,2,16),action_sequence=torch.randn(b,32,16)*.2,
        action_valid_mask=torch.ones(b,32,dtype=torch.bool),future_valid_mask=torch.ones(b,dtype=torch.bool),
        future_feats=torch.randn(b,3,2,8),task_cond=torch.randn(b,8),phase=torch.full((b,),-1),
        active_hand=torch.full((b,),-1),dist=torch.zeros(b),
        future_feats_8=torch.randn(b,3,2,8),future_valid_mask_8=torch.ones(b,dtype=torch.bool),
        idm_future_feats=torch.randn(b,4,3,3,7,8),idm_target_raw=torch.randn(b,4,16)*.1,
        idm_valid_mask=torch.ones(b,dtype=torch.bool))
    return wrapper,batch


def run(wrapper,batch,seed=42):
    torch.manual_seed(seed)
    return wrapper(batch)


class WamHjepaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_baseline_matches_original_loss_output_and_rng(self):
        base,b=fixture(); original=copy.deepcopy(base)
        wrapped=upgrade_wrapper(base,variant='baseline',idm_scale=[1.]*16,idm_hidden=32)
        torch.manual_seed(42); expected,_,out=original(b,{'future':.5,'phase_ce':0.,'gate_kl':0.,'log_dist':0.});rng=torch.get_rng_state()
        actual,_,new=run(wrapped,b)
        self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(out['final_pred'],new['final_pred']))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_upgrade_preserves_original_weights_and_rng(self):
        base,_=fixture(); state=copy.deepcopy(base.action_model.state_dict());rng=torch.get_rng_state()
        wrapped=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        after=wrapped.action_model.state_dict()
        self.assertTrue(all(torch.equal(v,after[k]) for k,v in state.items()))
        self.assertTrue(all(k.startswith('hjepa_aux.') for k in set(after)-set(state)))

    def test_aux_initialization_does_not_reseed_cuda(self):
        base,_=fixture();rng=torch.get_rng_state()
        with mock.patch('torch.cuda.manual_seed_all') as seed_cuda:
            upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        seed_cuda.assert_not_called()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_zero_offset_is_original_decoder_mapping(self):
        base,_=fixture();dec=base.action_model.future_feat_decoder;cond=torch.randn(2,10,32)
        self.assertTrue(torch.equal(dec(cond),future_with_offset(dec,cond,torch.zeros(32))))

    def test_four_variants_objective_and_shapes(self):
        for variant in ('baseline','multiscale','idm','combined'):
            with self.subTest(variant=variant):
                base,b=fixture();w=upgrade_wrapper(base,variant=variant,idm_scale=[1.]*16,idm_hidden=32)
                total,info,out=run(w,b)
                self.assertEqual(out['final_pred'].shape,(2,32,16));self.assertTrue(torch.isfinite(total))
                s=variant_settings(variant)
                expected=info['mse']+s['future32_weight']*info['future32']+s['future8_weight']*info.get('future8',0)+s['idm_weight']*info.get('idm',0)
                self.assertAlmostEqual(float(total),expected,places=5)
                if s['use_idm']:self.assertEqual(out['idm_pred'].shape,(2,4,16))

    def test_future_images_and_labels_cannot_change_current_policy(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        _,_,a=run(w,b)
        altered=copy.deepcopy(b)
        altered['future_feats']*=7;altered['future_feats_8']*=-2
        altered['idm_future_feats']=torch.randn_like(altered['idm_future_feats'])*3
        altered['idm_target_raw']+=100
        altered['future_proprio']=torch.full((2,4,16),1e6)
        _,_,z=run(w,altered)
        self.assertTrue(torch.equal(a['final_pred'],z['final_pred']))
        self.assertFalse(torch.allclose(a['idm_pred'],z['idm_pred']))

    def test_idm_does_not_read_actions_vtt_history_or_future_proprio(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        _,_,a=run(w,b)
        modified=copy.deepcopy(b);modified['action_sequence']+=.5;modified['task_cond']*=11
        modified['state'][:,1:]+=2;modified['pool']*=12;modified['future_proprio']=torch.randn(2,4,16)
        _,_,z=run(w,modified)
        self.assertTrue(torch.equal(a['idm_pred'],z['idm_pred']))

    def test_idm_only_gradients_reach_shared_adapters_not_action_mixed_layers(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        b['idm_future_feats'].requires_grad_();b['idm_target_raw'].requires_grad_();b['feats'].requires_grad_()
        _,_,out=run(w,b);out['aux_losses']['idm'].backward()
        evidence=gradient_evidence(w)
        self.assertGreater(evidence['shared_visual_grad_norm'],0);self.assertGreater(evidence['idm_head_grad_norm'],0)
        self.assertFalse(evidence['dino_has_grad']);self.assertFalse(evidence['idm_scale_requires_grad'])
        for group in (w.action_model.action_proj,w.action_model.output_proj,w.action_model.blocks,w.action_model.future_feat_decoder):
            self.assertTrue(all(p.grad is None for p in group.parameters()))
        for name in ('idm_future_feats','idm_target_raw','feats'):
            self.assertIsNone(b[name].grad,name)

    def test_current_and_each_future_adapter_call_remain_differentiable(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='idm',idm_scale=[1.]*16,idm_hidden=32)
        stored=[]
        def hook(m,i,o):o.retain_grad();stored.append(o)
        handle=w.action_model.dino_adapters[0].register_forward_hook(hook)
        try:
            _,_,out=run(w,b);out['aux_losses']['idm'].backward()
        finally:handle.remove()
        self.assertEqual(len(stored),2)  # current B*V, then independent (B*4)*V
        self.assertGreater(float(stored[0].grad.abs().sum()),0)
        grad=stored[1].grad.reshape(2,4,3,4,32)
        self.assertTrue(all(float(grad[:,k].abs().sum())>0 for k in range(4)))

    def test_future_teacher_values_are_stopped(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        b['future_feats'].requires_grad_();b['future_feats_8'].requires_grad_()
        loss,_,_=run(w,b);loss.backward()
        self.assertIsNone(b['future_feats'].grad);self.assertIsNone(b['future_feats_8'].grad)
        self.assertGreater(gradient_evidence(w)['future8_offset_grad_norm'],0)

    def test_extra_branches_do_not_shift_next_flow_rng(self):
        base,b=fixture();a=upgrade_wrapper(copy.deepcopy(base),variant='baseline',idm_scale=[1.]*16,idm_hidden=32)
        d=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        run(a,b);ra=torch.get_rng_state();next_a=torch.randn(5)
        run(d,b);rd=torch.get_rng_state();next_d=torch.randn(5)
        self.assertTrue(torch.equal(ra,rd));self.assertTrue(torch.equal(next_a,next_d))

    def test_five_frames_include_intermediate_transitions(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='idm',idm_scale=[1.]*16,idm_hidden=32)
        _,_,a=run(w,b);modified=copy.deepcopy(b)
        modified['idm_future_feats'][:,1]=torch.randn_like(modified['idm_future_feats'][:,1])
        _,_,z=run(w,modified)
        self.assertFalse(torch.allclose(a['idm_pred'],z['idm_pred']))
        self.assertTrue(torch.equal(a['final_pred'],z['final_pred']))

    def test_all_invalid_aux_is_zero_and_ignores_padding_nan(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        b['future_valid_mask_8'].zero_();b['idm_valid_mask'].zero_()
        b['future_feats_8'].fill_(float('nan'));b['idm_future_feats'].fill_(float('nan'));b['idm_target_raw'].fill_(float('nan'))
        loss,info,_=run(w,b)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(info['future8'],0);self.assertEqual(info['idm'],0)
        loss.backward()

    def test_partial_invalid_sample_does_not_contaminate_valid_loss(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        b['future_valid_mask_8'][1]=False;b['idm_valid_mask'][1]=False
        b['future_feats_8'][1]=float('nan');b['idm_future_feats'][1]=float('nan');b['idm_target_raw'][1]=float('nan')
        loss,info,out=run(w,b)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(info['idm_valid_samples'],1)
        self.assertEqual(out['idm_pred'].shape,(1,4,16))

    def test_aux_mask_dtype_and_shape_fail_closed(self):
        for key in ('idm_valid_mask','future_valid_mask_8'):
            base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
            b[key]=torch.ones(2)
            with self.assertRaises(ValueError):run(w,b)

    def test_loss_step_reduction_and_arm_gripper_balance(self):
        pred=torch.zeros(2,4,16);target=torch.ones_like(pred)
        self.assertEqual(float(inverse_loss(pred,target,torch.ones(16))),.5)
        target.zero_();target[:,:,[7,15]]=1
        self.assertEqual(float(inverse_loss(pred,target,torch.ones(16))),.25)

    def test_reject_bad_scale_variant_weights_and_double_upgrade(self):
        base,b=fixture()
        for scale in ([0.]*16,[1.]*14,[float('nan')]*16):
            with self.assertRaises(ValueError):upgrade_wrapper(base,variant='combined',idm_scale=scale)
        with self.assertRaises(ValueError):upgrade_wrapper(base,variant='unknown',idm_scale=[1.]*16)
        w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        with self.assertRaises(ValueError):w(b,{'future':.5})
        with self.assertRaises(ValueError):upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16)

    def test_same_aux_state_keys_across_all_four_groups(self):
        states=[]
        for variant in ('baseline','multiscale','idm','combined'):
            base,_=fixture();w=upgrade_wrapper(base,variant=variant,idm_scale=[1.]*16,idm_hidden=32)
            states.append(w.action_model.state_dict())
        for state in states[1:]:
            self.assertEqual(set(state),set(states[0]));self.assertTrue(all(torch.equal(state[k],states[0][k]) for k in state))

    def test_checkpoint_full_state_roundtrip_and_old_strict_rejects(self):
        base,b=fixture();original=copy.deepcopy(base)
        w=upgrade_wrapper(base,variant='combined',idm_scale=[.1]*16,idm_hidden=32)
        state=copy.deepcopy(w.action_model.state_dict())
        with self.assertRaises(RuntimeError):original.action_model.load_state_dict(state,strict=True)
        new_base,_=fixture();new_base.action_model.load_state_dict({k:v for k,v in state.items() if not k.startswith('hjepa_aux.')},strict=True)
        new=upgrade_wrapper(new_base,variant='combined',idm_scale=[.1]*16,idm_hidden=32)
        new.action_model.load_state_dict(state,strict=True)
        self.assertTrue(torch.equal(run(w,b)[2]['final_pred'],run(new,b)[2]['final_pred']))

    def test_online_sampler_never_calls_auxiliary_head(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='combined',idm_scale=[1.]*16,idm_hidden=32)
        calls=[];h=w.action_model.hjepa_aux.idm_head.register_forward_hook(lambda m,i,o:calls.append(1))
        try:
            action,info=w.sample_actions([b['feats'][:,:,i] for i in range(3)],b['state'],b['pool'],b['task_cond'],num_steps=2,generator=torch.Generator().manual_seed(7))
        finally:h.remove()
        self.assertEqual(action.shape,(2,32,16));self.assertEqual(calls,[])

    def test_hook_cleanup_on_base_error(self):
        base,b=fixture();w=upgrade_wrapper(base,variant='idm',idm_scale=[1.]*16,idm_hidden=32)
        b['action_sequence']=torch.zeros(2,31,16)
        with self.assertRaises(ValueError):run(w,b)
        self.assertEqual(len(w.action_model.dino_adapters[0]._forward_hooks),0)

    def test_no_gpu_context(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
