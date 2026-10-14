#!/usr/bin/env python3
"""CPU wiring/continuation regressions; no GPU, production training or simulator."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from harness import zhuhai_precision_v2_training as p,zhuhai_hjepa_training as old,zhuhai_training as zt
from harness.wam_event_clock import EventClock,episode_event_features
from scripts import train_zhuhai_wam_precision_v2 as train,eval_zhuhai_wam_precision_v2 as evaluate


class BaseDataset:
    def __init__(self,episodes):
        self.episodes=episodes;self.ends=np.cumsum([len(ep.state) for ep in episodes]).tolist()
        self.condition=None
    def __len__(self):return self.ends[-1]
    def __getitem__(self,index):
        from bisect import bisect_right
        i=bisect_right(self.ends,index);t=index-(self.ends[i-1] if i else 0)
        return dict(anchor=t,state=self.episodes[i].state[t:t+1].copy())


class PrecisionWiringTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory(prefix='precision-wiring-cpu-')
        self.addCleanup(self.tmp.cleanup);self.directory=Path(self.tmp.name)
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized());self.assertNotIn('isaacsim',sys.modules)
    def dataset(self,state,action,initial=None,name='episode'):
        path=self.directory/(name+'.hdf5')
        initial=state[0] if initial is None else initial
        with h5py.File(path,'x') as f:f.create_dataset('target_command',data=np.asarray([initial]))
        return p.PrecisionDataset(BaseDataset([SimpleNamespace(path=path,state=state,action=action)]))
    def test_bounded_suffix_is_exact_for_saturation_and_recent_transitions(self):
        n=480;state=np.zeros((n,16),np.float32);action=state.copy()
        state[:,[7,15]]=1
        action[210:267,7]=1;action[299:,15]=1
        state[170:176,0]=np.arange(6)*.003
        state[176:,0]=.015;state[330:339,8]=np.arange(9)*.001
        state[339:,8]=.008
        initial=action[0].copy();ds=self.dataset(state,action,initial)
        expected=episode_event_features(state,action,initial_target=initial,**p.CLOCK_CONFIG)
        for t in range(n):np.testing.assert_array_equal(ds[t]['event_state'],expected[t])
        self.assertEqual(float(ds[150]['event_state'][0]),1.)
        self.assertEqual(float(ds[150]['event_state'][1]),1.)
        self.assertEqual(sum(v.nbytes for v in ds.initial_targets),64)
    def test_random_windows_online_offline_same_and_future_commands_do_not_leak(self):
        rng=np.random.default_rng(91);n=325
        state=np.cumsum(rng.normal(0,.0007,(n,16)),axis=0).astype(np.float32)
        state[:,[7,15]]=1;action=state.copy();action[:,[7,15]]=rng.integers(0,2,(n,2))
        initial=action[0].copy();initial[[7,15]]=0
        ds=self.dataset(state,action,initial)
        clock=EventClock(**p.CLOCK_CONFIG)
        for t in range(n):np.testing.assert_array_equal(ds[t]['event_state'],clock.update(state[t],initial if t==0 else action[t-1]))
        t=123;before=ds[t]['event_state'].copy();action[t:]=1000
        np.testing.assert_array_equal(ds[t]['event_state'],before)
        ds.condition=np.ones(3);self.assertIs(ds.condition,ds.base.condition)
    def test_make_plan_preserves_old_frozen_feature_binding_and_statistics_identity(self):
        args=train.parser().parse_args(['--out',str(self.directory/'out')])
        binding=dict(format=old.FORMAT+'/feature-preparation',labels=old.LABEL_CONTRACT,source_files={'frozen':'a'*64})
        contract=dict(format=old.FORMAT,config=dict(model={}),hjepa=dict(cache_binding=binding),source_files={},cache=dict(policy='frozen producer'))
        original=copy.deepcopy(contract)
        with patch.object(old,'make_plan',return_value=(dict(status='cpu_preflight_completed',contract=contract),{'checkpoint':'same'})):
            result,checkpoint=p.make_plan(args)
        self.assertEqual(contract,original);self.assertEqual(checkpoint,{'checkpoint':'same'})
        new=result['contract'];self.assertEqual(new['hjepa']['cache_binding'],binding)
        self.assertEqual(new['precision_cache_compatibility']['raw_feature_binding'],binding)
        self.assertEqual(new['format'],p.FORMAT)
        self.assertEqual(new['hjepa']['code_hashes'],new['source_files'])
        self.assertTrue(all(str(ROOT/name) in new['source_files'] for name in p.EXTENSION_SOURCES))
        contract['source_files']={str(ROOT/'harness/zhuhai_dataset.py'):'0'*64}
        with patch.object(old,'make_plan',return_value=(dict(status='cpu_preflight_completed',contract=contract),{})):
            with self.assertRaisesRegex(ValueError,'Inherited source changed'):p.make_plan(args)
    def test_old_train_preparation_commit_reused_after_precision_upgrade(self):
        from scripts.test_zhuhai_preparation_reuse import config,source,ReuseTests
        from scripts.test_zhuhai_hjepa_dataset import Episode,ForbiddenStore
        cfg=config();ds=source(self.directory,cfg,p.TASK,101);original=ds.episodes[0]
        ep=Episode(n=len(original.state),state=original.state,action=original.action)
        vars(ep).update(vars(original));ds.task=p.TASK;ds.episodes=[ep];ds.store=ForbiddenStore()
        datasets=dict(train={p.TASK:ds},val={},heldout={})
        feature=dict(encoder_sha256='a'*64,preprocess_sha256='b'*64)
        binding=dict(labels=copy.deepcopy(old.LABEL_CONTRACT),format=old.FORMAT+'/feature-preparation')
        helper=SimpleNamespace(counts=dict(statistics=0,conditions=0))
        stats_builder=lambda *args,**kw:ReuseTests.stats_builder(helper,*args,**kw)
        bank_builder=lambda *args,**kw:ReuseTests.bank_builder(helper,*args,**kw)
        before=old.preparation_cache(self.directory/'preparation',binding)
        statistics=before.statistics(datasets['train'],cfg,feature_identity=feature,builder=stats_builder)
        bank=before.conditions(datasets,cfg,statistics,builder=bank_builder)
        idm=before.idm_statistics(datasets['train'],cfg,feature_identity=feature)
        upgraded=copy.deepcopy(cfg);upgraded['model']['precision']=copy.deepcopy(p.PRECISION_SPEC)
        after=p.preparation_cache(self.directory/'preparation',binding)
        forbid=Mock(side_effect=AssertionError('identical TRAIN content must not be recomputed'))
        with patch('harness.zhuhai_hjepa_dataset.fit_idm_statistics',forbid):
            self.assertEqual(statistics,after.statistics(datasets['train'],upgraded,feature_identity=feature,builder=forbid))
            self.assertEqual(bank,after.conditions(datasets,upgraded,statistics,builder=forbid))
            self.assertEqual(idm,after.idm_statistics(datasets['train'],upgraded,feature_identity=feature))
        self.assertEqual([op['outcome'] for op in after.receipt()['operations']],['hit','hit','hit'])
        self.assertEqual([op['key'] for op in before.receipt()['operations']],[op['key'] for op in after.receipt()['operations']])
    def test_combined_only_and_plan_cannot_execute(self):
        args=train.parser().parse_args(['--out',str(self.directory/'out')])
        p.validate_request(args)
        self.assertEqual((args.mode,args.steps,args.batch_size,args.save_every),('plan',30000,4,5000))
        for changes in ({'variant':'baseline'},{'execute':True},{'batch_size':8},{'randomized_count':200}):
            bad=copy.deepcopy(args);vars(bad).update(changes)
            with self.assertRaises(ValueError):p.validate_request(bad)
    def test_architecture_clock_and_compatibility_tampering_rejected(self):
        from harness.wam_precision_v2 import PRECISION_VERSION,REFINEMENT_VIEW_INDICES
        self.assertEqual(p.PRECISION_SPEC['checkpoint_model_version'],PRECISION_VERSION)
        self.assertEqual(p.PRECISION_SPEC['refinement_view_indices'],list(REFINEMENT_VIEW_INDICES))
        binding=dict(labels=old.LABEL_CONTRACT)
        contract=dict(format=p.FORMAT,precision=copy.deepcopy(p.PRECISION_SPEC),
            config=dict(model=dict(precision=copy.deepcopy(p.PRECISION_SPEC))),hjepa=dict(cache_binding=binding),
            precision_cache_compatibility=dict(raw_feature_binding=binding,normalization_and_vtt_unchanged=True),
            source_files={str(ROOT/name):'a'*64 for name in p.EXTENSION_SOURCES})
        with patch.object(old,'validate_training_contract') as validate:
            self.assertIs(p.validate_training_contract(contract),contract)
            self.assertEqual(validate.call_args.args[0]['format'],old.FORMAT)
            for field in ('clock','model'):
                bad=copy.deepcopy(contract);bad['precision'][field]={}
                with self.assertRaises(ValueError):p.validate_training_contract(bad)
            bad=copy.deepcopy(contract);bad['precision_cache_compatibility']['raw_feature_binding']={}
            with self.assertRaises(ValueError):p.validate_training_contract(bad)
    def test_full_self_contained_model_reconstruction_and_strict_loading(self):
        from scripts.test_wam_precision import fixture
        hj,batch=fixture(queries=64);raw=copy.deepcopy(hj.base.action_model);del raw.hjepa_aux
        base=hj.base
        cfg=dict(model=dict(views=base.views),training={})
        moments=dict(min=[-2.]*16,max=[2.]*16)
        stats=dict(normalization={'action_repr':'absolute',zt.NORMALIZATION_KEY:dict(state=moments,action=moments)},
            temporal_stats=dict(views=base.views,proprio_diff_std=base.prop_std.tolist(),memory_diff_std=base.mem_std.tolist()),
            proprio_keep_mask=base.proprio_keep.tolist())
        with patch.object(old,'build_wrapper',return_value=(hj,{})),patch.object(p,'AUX_CONFIG',dict(idm_hidden=32,step_embedding_dim=16,aux_init_seed=20261009)), \
             patch.object(zt,'validate_statistics'),patch.object(zt,'create_action_model',side_effect=lambda *a,**kw:copy.deepcopy(raw)):
            wrapper,_=p.build_wrapper(cfg,stats,variant='combined',idm_statistics={'scale16':[1.]*16},initialization={})
            state=copy.deepcopy(wrapper.action_model.state_dict())
            loaded,_=p.build_wrapper(cfg,stats,variant='combined',idm_statistics={'scale16':[1.]*16},state_dict=state)
            for key in state:torch.testing.assert_close(state[key],loaded.action_model.state_dict()[key],rtol=0,atol=0)
            bad=copy.deepcopy(state);bad['hjepa_aux.idm_scale']*=2
            with self.assertRaisesRegex(ValueError,'IDM scale'):p.build_wrapper(cfg,stats,variant='combined',idm_statistics={'scale16':[1.]*16},state_dict=bad)
            bad=copy.deepcopy(state);bad['precision.version'].fill_(1)
            with self.assertRaisesRegex(RuntimeError,'precision.version'):p.build_wrapper(cfg,stats,variant='combined',idm_statistics={'scale16':[1.]*16},state_dict=bad)
            bad=copy.deepcopy(state);bad.pop('precision.residual.bias')
            with self.assertRaises(RuntimeError):p.build_wrapper(cfg,stats,variant='combined',idm_statistics={'scale16':[1.]*16},state_dict=bad)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(11);first=wrapper(batch)[2]['final_pred']
                torch.random.default_generator.manual_seed(11);second=loaded(batch)[2]['final_pred']
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            # Receipt/task/format gates exercise the public V2 loader, while only
            # the unrelated full production provenance is replaced by a fixture.
            contract=dict(config=cfg,statistics=stats,idm_statistics={'scale16':[1.]*16},
                hjepa={'variant':'combined'},conditions={p.TASK:[0.]*8},source_files=p.source_hashes()[0])
            with patch.object(p,'validate_training_contract',side_effect=lambda c:c):
                path=train.save_checkpoint(self.directory,wrapper,contract,20000)
                receipt_path=path.with_name(path.name+'.sha256.json');receipt=p.read_json(receipt_path)
                with patch.object(zt,'checkpoint_header',side_effect=AssertionError('self-contained loader must not read 50k')), \
                     patch.object(old,'build_wrapper',side_effect=AssertionError('self-contained loader must not migrate')):
                    policy,condition,actual=p.load_policy(path,p.TASK,device='cpu')
                self.assertEqual(actual,contract);self.assertEqual(condition.dtype,np.float32)
                self.assertEqual(int(policy.action_model.precision.version),2);self.assertFalse(policy.training)
                with self.assertRaises(ValueError):p.load_policy(path,'zhuhai_task1',device='cpu')
                for field,value in (('step',25000),('size',receipt['size']+1),('sha256','0'*64),('model_only',False)):
                    p.write_json(receipt_path,dict(receipt,**{field:value}))
                    with self.assertRaises(ValueError):p.load_policy(path,p.TASK,device='cpu')
                p.write_json(receipt_path,receipt)

    def test_native14_migration_resets_only_projections_before_v2_upgrade(self):
        from scripts.test_wam_precision import fixture
        from harness.wam_precision import GATE_MODULES
        hj,_=fixture(queries=64);raw=copy.deepcopy(hj.action_model);del raw.hjepa_aux
        source=copy.deepcopy(raw)
        source.action_proj=torch.nn.Linear(14,32);source.proprio_proj=torch.nn.Linear(14,32)
        source.output_proj=torch.nn.Linear(32,14)
        with torch.no_grad():
            source.action_proj.bias.fill_(7.);source.proprio_proj.bias.fill_(8.);source.output_proj.bias.fill_(9.)
        initial=dict(format=zt.INITIALIZATION_FORMAT,step=50000,model_only=True,model_state_dict=source.state_dict())
        cfg=dict(**zt.protocol_binding(),model=dict(views=hj.views),training={})
        moments=dict(min=[-2.]*16,max=[2.]*16)
        stats=dict(normalization={'action_repr':'absolute',zt.NORMALIZATION_KEY:dict(state=moments,action=moments)},
            temporal_stats=dict(views=hj.views,proprio_diff_std=hj.prop_std.tolist(),memory_diff_std=hj.mem_std.tolist()),
            proprio_keep_mask=hj.proprio_keep.tolist())
        with patch.object(zt,'validate_statistics'),patch.object(zt,'create_action_model',side_effect=lambda *a,**k:copy.deepcopy(raw)):
            wrapper,notes=p.build_wrapper(cfg,stats,variant='combined',idm_statistics={'scale16':[1.]*16},initialization=initial)
            bad=copy.deepcopy(initial)
            # Use a guaranteed non-projection key for the strict migration gate.
            bad['model_state_dict'].pop('mem_proj.weight')
            with self.assertRaises(RuntimeError):p.build_wrapper(cfg,stats,variant='combined',idm_statistics={'scale16':[1.]*16},initialization=bad)
        reset={name+'.'+part for name in ('action_proj','proprio_proj','output_proj') for part in ('weight','bias')}
        self.assertEqual(set(notes['reset']),reset);self.assertFalse(notes['numerical_normalization_reused'])
        state=wrapper.action_model.state_dict()
        for key,value in initial['model_state_dict'].items():
            if key in reset or any(key.startswith(name+'.') for name in GATE_MODULES):continue
            if key=='dino_adapters.0.query_embed':value=value.reshape(1,16,4,32).mean(dim=2)
            torch.testing.assert_close(state[key],value,rtol=0,atol=0)
        self.assertEqual(tuple(state['action_proj.weight'].shape),(32,16))
        self.assertEqual(tuple(state['proprio_proj.weight'].shape),(32,16))
        self.assertEqual(tuple(state['output_proj.weight'].shape),(16,32))
        self.assertFalse(torch.all(state['action_proj.bias']==7.));self.assertFalse(torch.all(state['proprio_proj.bias']==8.))
        self.assertEqual(int(state['precision.version']),2)

    def test_new_tiny_gradient_admission_first_and_second_update(self):
        from scripts.test_wam_precision import fixture
        from harness.wam_precision_v2 import upgrade_precision_v2 as upgrade_precision
        hj,batch=fixture(queries=64);wrapper=upgrade_precision(hj)
        wrapper(batch)[0].backward();first=train.gradient_evidence(wrapper)
        train.validate_tiny_gradients(first,wrapper.settings,0)
        self.assertEqual(first['precision_attention_grad_norm'],0.)
        with self.assertRaises(ValueError):train.validate_tiny_gradients(first,wrapper.settings,1)
        wrapper.zero_grad(set_to_none=True)
        with torch.no_grad():wrapper.action_model.precision.residual.weight.fill_(.03)
        wrapper(batch)[0].backward();second=train.gradient_evidence(wrapper)
        train.validate_tiny_gradients(second,wrapper.settings,1)
        bad=dict(second,precision_clock_grad_norm=float('nan'))
        with self.assertRaises(ValueError):train.validate_tiny_gradients(bad,wrapper.settings,1)
    def test_execute_cpu_mock_runtime_reaches_two_updates_and_releases_sample_copies(self):
        import hashlib,weakref
        from harness.wam_precision_v2 import upgrade_precision_v2
        from harness.robotwin_multitask_training import install_joint_projection
        from scripts.test_wam_precision import fixture
        hj,batch=fixture(queries=64)
        hj.action_model.task_cond_proj=torch.nn.Linear(1024,32);batch['task_cond']=torch.randn(2,1024)
        install_joint_projection(hj.action_model)
        wrapper=upgrade_precision_v2(hj);refs=[]
        class Dataset:
            def __len__(self):return 1
            def __getitem__(self,index):
                assert all(v.grad is None for v in wrapper.action_model.parameters()), 'previous-update gradients survived into feature loading'
                sample={k:v[0].clone() for k,v in batch.items()}
                refs.append(weakref.ref(sample['idm_future_feats']))
                return sample
        def check_lifetime(module,inputs):
            self.assertTrue(all(ref() is None for ref in refs),'individual dense samples survived collation')
        hook=wrapper.register_forward_pre_hook(check_lifetime);self.addCleanup(hook.remove)
        args=train.parser().parse_args(['--out',str(self.directory/'tiny'),'--mode','tiny',
            '--cache-root',str(self.directory/'cache'),'--preparation-root',str(self.directory/'prep')])
        encoder_dir=self.directory/'encoder';encoder_dir.mkdir();(encoder_dir/'model.safetensors').write_bytes(b'cpu-fixture')
        training=dict(max_steps=2,save_every=2,warmup_steps=0,learning_rate=1e-5,betas=[.9,.999],weight_decay=.01,
            grad_clip_norm=1.,auxiliary_loss=dict(future=.25,future8=.25,idm=.1,gate_kl=0.,phase_ce=0.,log_dist=0.))
        cfg=dict(model=dict(vision_encoder=dict(checkpoint_path=str(encoder_dir))),training=training)
        base=dict(config=cfg,cache=dict(encoder_sha256=hashlib.sha256(b'cpu-fixture').hexdigest()),splits={},
            source_files={str(ROOT/'harness/wam_precision_v2.py'):p.sha256_file(ROOT/'harness/wam_precision_v2.py')},
            hjepa=dict(cache_binding={}))
        datasets=dict(train={p.TASK:Dataset()},val={p.TASK:Dataset()},heldout={})
        encoder=torch.nn.Linear(1,1).requires_grad_(False)
        runner=SimpleNamespace(ModelFactory=SimpleNamespace(create_vision_encoder=lambda *a,**k:(encoder,1024,4,16)))
        prep=SimpleNamespace(statistics=lambda *a,**k:{},conditions=lambda *a,**k:{'conditions':{p.TASK:[1.]*1024}},
            idm_statistics=lambda *a,**k:{'scale16':[1.]*16},receipt=lambda:{'cpu_fixture':True})
        with patch.object(train,'reserve'),patch.object(train,'log') as logger,patch.object(train.signal,'signal'), \
             patch('harness.wam_behavior.load_upstream',return_value=(None,runner)), \
             patch.object(p,'build_datasets',return_value=datasets),patch.object(p,'preparation_cache',return_value=prep), \
             patch.object(p,'build_wrapper',return_value=(wrapper,{})),patch.object(p,'validate_training_contract'), \
             patch('torch.set_num_threads'),patch('torch.cuda.set_per_process_memory_fraction'), \
             patch('torch.cuda.max_memory_allocated',return_value=0),patch('torch.cuda.max_memory_reserved',return_value=0), \
             patch('torch.cuda.get_rng_state_all',return_value=[]),patch('builtins.print'):
            self.assertEqual(train.execute(args,{'contract':base},{}),0)
        self.assertEqual(logger.call_count,1)
        status=p.read_json(args.out/'status.json');self.assertEqual(status['step'],2)
        self.assertTrue(status['precision_gradient_admission_passed'])  # Mock CPU receipt, not real GPU evidence.
        checkpoint=torch.load(args.out/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        self.assertTrue(checkpoint['optimizer_state_dict']['state'])
        self.assertEqual(checkpoint['step'],2)
        self.assertTrue((args.out/'initial_diagnostic.json').is_file())
        self.assertTrue((args.out/'diagnostic_000002.json').is_file())

    def test_completion_receipt_only_admits_two_completed_tiny_updates(self):
        import ast,inspect
        node=ast.parse(inspect.getsource(train.execute))
        self.assertTrue(any(isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='completion_status' for call in ast.walk(node)))
        for mode,step,target,admitted in (('tiny',2,2,True),('tiny',1,2,False),('train',20000,30000,False),('train',30000,30000,False)):
            value=train.completion_status(mode,step,target)
            self.assertIs(value['precision_gradient_admission_passed'],admitted)
            self.assertIs(value['auxiliary_gradient_admission_passed'],admitted)
    def test_diagnostic_preserves_rng_and_training_mode(self):
        import random
        from harness.wam_precision_v2 import upgrade_precision_v2
        from scripts.test_wam_precision import fixture
        hj,batch=fixture(queries=64);wrapper=upgrade_precision_v2(hj)
        sample={k:v[0] for k,v in batch.items()}
        torch_before=torch.get_rng_state().clone();np_before=np.random.get_state();py_before=random.getstate()
        for training in (True,False):
            wrapper.train(training)
            result=train.diagnostic(wrapper,{p.TASK:[sample]})
            self.assertTrue(np.isfinite(result[p.TASK]['loss']))
            self.assertIs(wrapper.training,training)
            torch.testing.assert_close(torch.get_rng_state(),torch_before,rtol=0,atol=0)
            self.assertEqual(np.random.get_state()[0],np_before[0])
            np.testing.assert_array_equal(np.random.get_state()[1],np_before[1])
            self.assertEqual(np.random.get_state()[2:],np_before[2:])
            self.assertEqual(random.getstate(),py_before)
        invalid=dict(sample);invalid.pop('event_state')
        with self.assertRaises(ValueError):train.diagnostic(wrapper,{p.TASK:[invalid]})
        self.assertFalse(wrapper.training)
        torch.testing.assert_close(torch.get_rng_state(),torch_before,rtol=0,atol=0)

    def test_public_source_hashes_bind_v2_and_runtime_rng_helper(self):
        hashes,missing=p.source_hashes()
        self.assertEqual(missing,[])
        for rel in (*p.EXTENSION_SOURCES,'scripts/train_wam_v3.py'):
            self.assertIn(str(ROOT/rel),hashes)
            self.assertEqual(hashes[str(ROOT/rel)],p.sha256_file(ROOT/rel))

    def test_resume_rng_and_optimizer_metadata_without_optimizer_updates(self):
        model=torch.nn.Linear(2,2);wrapper=SimpleNamespace(action_model=model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        rng=np.random.default_rng(41);rng.integers(999,size=13)
        torch.random.default_generator.manual_seed(65);torch.randn(7)
        with patch.object(p,'validate_training_contract'),patch('torch.cuda.get_rng_state_all',return_value=[]):
            path=train.save_checkpoint(self.directory,wrapper,{},20000,optimizer=optimizer,rng=rng.bit_generator.state)
        expected_np=rng.integers(999,size=12);expected_torch=torch.randn(12)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        rng=np.random.default_rng(9);torch.random.default_generator.manual_seed(12)
        cuda=Mock();api=SimpleNamespace(set_rng_state=torch.set_rng_state,cuda=SimpleNamespace(set_rng_state_all=cuda))
        self.assertEqual(train.restore_rng(saved,rng,api),20000)
        np.testing.assert_array_equal(rng.integers(999,size=12),expected_np)
        torch.testing.assert_close(torch.randn(12),expected_torch,rtol=0,atol=0);cuda.assert_called_once_with([])
        self.assertFalse(saved['model_only']);self.assertEqual(saved['format'],p.FORMAT)
        self.assertEqual(saved['optimizer_state_dict'],optimizer.state_dict())
    def test_nonempty_adamw_rng_exact_through_20k_25k_30k_boundaries(self):
        # Three scalar toy parameters only; this is a serialization regression,
        # not a WAM training run or GPU admission.
        cfg=dict(max_steps=30000,warmup_steps=100,learning_rate=1e-5)
        def run(staged):
            torch.random.default_generator.manual_seed(73);rng=np.random.default_rng(81)
            model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
            for update in range(30000):
                for group in optimizer.param_groups:group['lr']=train.learning_rate_for_update(cfg,update)
                for parameter in model.parameters():
                    parameter.grad=torch.randn_like(parameter)*float(rng.uniform(.5,1.5))
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if staged and update+1 in (20000,25000):
                    path=train.save_checkpoint(self.directory,SimpleNamespace(action_model=model),{},update+1,
                        optimizer=optimizer,rng=rng.bit_generator.state)
                    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                    self.assertTrue(saved['optimizer_state_dict']['state'])
                    self.assertTrue(all(int(value['step'])==update+1 for value in saved['optimizer_state_dict']['state'].values()))
                    model=torch.nn.Linear(2,1);model.load_state_dict(saved['model_state_dict'],strict=True)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=1.)
                    optimizer.load_state_dict(saved['optimizer_state_dict'])
                    rng=np.random.default_rng(0)
                    self.assertEqual(train.restore_rng(saved,rng,torch),update+1)
            return model.state_dict(),optimizer.state_dict(),rng.integers(999,size=9),torch.randn(9)
        with patch.object(p,'validate_training_contract'),patch('torch.cuda.get_rng_state_all',return_value=[]), \
             patch('torch.cuda.set_rng_state_all') as cuda:
            first=run(False);second=run(True)
        self.assertEqual(cuda.call_count,2)
        for key in first[0]:torch.testing.assert_close(first[0][key],second[0][key],rtol=0,atol=0)
        self.assertEqual(first[1]['param_groups'],second[1]['param_groups'])
        for key,value in first[1]['state'].items():
            for name,tensor in value.items():torch.testing.assert_close(tensor,second[1]['state'][key][name],rtol=0,atol=0)
        np.testing.assert_array_equal(first[2],second[2]);torch.testing.assert_close(first[3],second[3],rtol=0,atol=0)

    def test_lr_uses_fixed_30k_schedule_at_both_resumes(self):
        cfg=dict(max_steps=30000,warmup_steps=100,learning_rate=1e-5)
        continuous=[train.learning_rate_for_update(cfg,i) for i in range(30000)]
        staged=[train.learning_rate_for_update(cfg,i) for a,b in ((0,20000),(20000,25000),(25000,30000)) for i in range(a,b)]
        self.assertEqual(continuous,staged)
        self.assertLess(continuous[20000],1e-5);self.assertLess(continuous[25000],continuous[20000])
        self.assertAlmostEqual(continuous[0],1e-7);self.assertAlmostEqual(continuous[-1],1e-6,places=10)
    def test_eval_blocks_precision_and_hjepa_trainers(self):
        for script in ('train_zhuhai_wam.py','train_zhuhai_wam_hjepa.py','train_zhuhai_wam_precision.py','train_zhuhai_wam_precision_v2.py',
                       'scripts.train_zhuhai_wam_precision_v2','scripts.train_zhuhai_wam_precision'):
            argv=('python\0-m\0'+script+'\0') if script.startswith('scripts.') else ('python\0/home/user/'+script+'\0')
            entry=Mock();entry.name='999999';entry.__truediv__=Mock(return_value=SimpleNamespace(read_bytes=lambda:argv.encode()))
            with patch.object(p,'verify_owner'),patch.object(evaluate.Path,'iterdir',return_value=[entry]):
                with self.assertRaisesRegex(evaluate.base.InfrastructureError,'training process'):evaluate.verify_owner('cpu-test-owner')
    def test_online_policy_updates_clock_on_queued_steps_and_resets(self):
        from harness import zhuhai_robot_spec as robot
        from harness.zhuhai_action_smoothing import protocol
        state=np.asarray(robot.HOME_ACTION,np.float32);state[[7,15]]=0
        class Context:
            def reset(self):pass
            def push(self,*args):pass
            def inputs(self):return None
        sampled=[]
        def sample(feats,inputs,seed,step,event):
            sampled.append((step,event.copy()))
            return np.repeat(state[None],32,axis=0)
        policy=evaluate.clocked_policy(Context(),lambda rgb:(None,None),sample,protocol())
        observation=dict(measured_joint_gripper_state=state.copy(),commanded_joint_gripper_state=state.copy(),
            rgb={v:np.zeros((2,2,3),np.uint8) for v in evaluate.base.VIEWS})
        policy.reset('episode-1',5)
        clock=EventClock(**p.CLOCK_CONFIG)
        for t in range(17):
            expected=clock.update(state,state);policy.observe('episode-1',t,observation)
            np.testing.assert_array_equal(policy.event_state,expected)
        self.assertEqual([step for step,event in sampled],[0,8,16])
        np.testing.assert_array_equal(sampled[-1][1],policy.event_state)
        before=policy.event_state.copy()
        with self.assertRaises(ValueError):policy.observe('episode-1',16,observation)
        np.testing.assert_array_equal(policy.event_state,before)
        policy.reset('episode-2',6);policy.observe('episode-2',0,observation)
        np.testing.assert_array_equal(policy.event_state,[0,0,1,0,0,1])


if __name__=='__main__':unittest.main(verbosity=2)
