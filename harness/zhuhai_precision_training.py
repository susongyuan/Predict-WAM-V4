"""Precision-WAM wiring. Reuse frozen DINO data/normalization, not old model semantics."""
from __future__ import annotations
from bisect import bisect_right
import copy
import math
from pathlib import Path
import tempfile
from harness import zhuhai_hjepa_training as old
from harness.zhuhai_hjepa_training import *

FORMAT='zhuhai-precision-wam-native16-measured-10hz-v1'
PRECISION_CONFIG=dict(queries_per_view=16,precision_steps=8,aux_seed=20261009)
CLOCK_CONFIG=dict(dt=.1,still_velocity_rad_s=.02,max_seconds=10.,close_threshold=.5)
PRECISION_SPEC=dict(schema='spatially_dense_near_term_precision_v1',model=PRECISION_CONFIG,clock=CLOCK_CONFIG,
    clock_layout=['left_close_elapsed','left_still_elapsed','left_closed_target','right_close_elapsed','right_still_elapsed','right_closed_target'],
    clock_semantics='past commanded closure and measured arm stillness; NOT verified object grasp duration',
    clock_input='q[t], target_command[t]=action[t-1]; original target_command[0] at reset',
    clock_replay='bounded past-only ceil(max_seconds/dt)+2 suffix; exact after timer saturation',
    visual_queries_per_view=16,dit_tokens=100,removed=['phase_net','gate_head','phase_head','dist_head'],
    dense_memory='CURRENT left/right wrist fused DINO patches + 2D patch coordinates; no new DINO calls',
    refined_flow_prefix=8,action_chunk=32,execution_prefix=8,zero_initialized_refinement=True,
    tiny_gradient_admission='update1: residual+clock+HJEPAs; update2: plus dense patch projection and attention',
    online_future_inputs=False,original_frozen_code_modified=False)
EXTENSION_SOURCES=('harness/wam_precision.py','harness/wam_event_clock.py','harness/zhuhai_precision_training.py',
                   'scripts/train_zhuhai_wam_precision.py','scripts/eval_zhuhai_wam_precision.py',
                   'scripts/run_zhuhai_wam_precision_pipeline.py','scripts/test_wam_precision.py',
                   'scripts/test_wam_event_clock.py','scripts/test_zhuhai_precision_training.py',
                   'scripts/test_zhuhai_wam_precision_pipeline.py')


def validate_request(args):
    old.validate_request(args)
    if args.variant!='combined':raise ValueError('This first Precision-WAM run keeps the existing combined objective')


def make_plan(args):
    validate_request(args);plan,checkpoint=old.make_plan(args)
    if plan['status']!='cpu_preflight_completed':return plan,checkpoint
    missing=[str(ROOT/p) for p in EXTENSION_SOURCES if not (ROOT/p).is_file()]
    if missing:return dict(status='waiting_for_code',missing_code=missing,plan_frozen=False,cuda_initialized=False),None
    base=plan['contract'];binding=copy.deepcopy(base['hjepa']['cache_binding'])
    contract=copy.deepcopy(base);contract['format']=FORMAT
    contract['precision']=copy.deepcopy(PRECISION_SPEC)
    contract['config']['model']['precision']=copy.deepcopy(PRECISION_SPEC)
    contract['source_files'].update({str(ROOT/p):sha256_file(ROOT/p) for p in EXTENSION_SOURCES})
    contract['hjepa']['code_hashes']=copy.deepcopy(contract['source_files'])
    # DINO producer, source bytes, layers and histories are unchanged. Do not
    # invalidate already generated frozen features merely because the policy changed.
    contract['precision_cache_compatibility']=dict(raw_feature_binding=binding,normalization_and_vtt_unchanged=True,
        scope='only raw frozen DINO and original TRAIN numeric moments; no trainable adapter/model values reused')
    contract['cache']['policy']+='; raw features retain their original frozen producer binding across policy-only upgrades'
    plan=dict(plan,contract=contract,contract_sha256=digest(contract))
    return plan,checkpoint


class PrecisionDataset:
    """Add six causal clock inputs without retaining another unbounded RAM cache."""
    def __init__(self,dataset):
        import h5py
        self.base=dataset;self.initial_targets=[]
        for ep in dataset.episodes:
            with h5py.File(ep.path,'r') as f:self.initial_targets.append(f['target_command'][0].copy())
    def __getattr__(self,name):return getattr(self.base,name)
    def __len__(self):return len(self.base)
    @property
    def condition(self):return self.base.condition
    @condition.setter
    def condition(self,value):self.base.condition=value
    def __getitem__(self,index):
        from harness.wam_event_clock import episode_event_features
        sample=self.base[index];i=bisect_right(self.base.ends,index);ep=self.base.episodes[i];t=sample['anchor']
        # Two extra intervals account for absent pre-suffix velocity/flags. Any
        # duration whose event predates this window is saturated at the anchor.
        start=max(0,t-math.ceil(CLOCK_CONFIG['max_seconds']/CLOCK_CONFIG['dt'])-2)
        initial=self.initial_targets[i] if start==0 else ep.action[start-1]
        values=episode_event_features(ep.state[start:t+1],ep.action[start:t+1],initial_target=initial,**CLOCK_CONFIG)
        sample['event_state']=values[-1].copy()
        return sample


def build_datasets(splits,store,cfg,*,variant):
    datasets=old.build_datasets(splits,store,cfg,variant=variant)
    return {split:{task:PrecisionDataset(ds) for task,ds in tasks.items()} for split,tasks in datasets.items()}


def build_wrapper(cfg,statistics,*,variant,idm_statistics,device='cpu',initialization=None,state_dict=None):
    from harness import zhuhai_training as zt
    from harness.wam_v3 import WamV3Wrapper
    from harness.wam_hjepa import upgrade_wrapper
    from harness.wam_precision import upgrade_precision
    if (initialization is None)==(state_dict is None):raise ValueError('Supply exactly one initializer or Precision-WAM state')
    if initialization is not None:
        wrapper,notes=old.build_wrapper(cfg,statistics,variant=variant,idm_statistics=idm_statistics,device=device,initialization=initialization)
    else:
        # Construct the declared base, install BOTH upgrades, then strict-load the
        # self-contained checkpoint. Loading does not depend on the old 50k file.
        zt.validate_statistics(statistics,cfg);model=zt.create_action_model(cfg,device=device)
        with tempfile.TemporaryDirectory(prefix='precision-wam-norm-') as directory:
            path=Path(directory)/'norm.json';path.write_text(json.dumps(statistics['normalization'],allow_nan=False))
            base=WamV3Wrapper(model,views=cfg['model']['views'],stats=statistics['temporal_stats'],norm_stats_path=path,
                normalization_key=zt.NORMALIZATION_KEY,proprio_mask='native_all16',proprio_keep_override=statistics['proprio_keep_mask'],
                train_cfg=cfg['training'],vision_encoder=None,feat_layers=(-12,-8,-4),num_register_tokens=4,device=device).to(device)
        wrapper=upgrade_wrapper(base,variant=variant,idm_scale=idm_statistics['scale16'],**AUX_CONFIG)
        notes=dict(initialization='strict_self_contained_precision_native16',projections_reset=False)
    wrapper=upgrade_precision(wrapper,**PRECISION_CONFIG)
    if state_dict is not None:
        import torch
        wrapper.action_model.load_state_dict(state_dict,strict=True)
        expected=torch.tensor(idm_statistics['scale16'],dtype=torch.float32,device=device)
        if not torch.equal(wrapper.action_model.hjepa_aux.idm_scale,expected):raise ValueError('IDM scale differs from TRAIN statistics')
    notes.update(precision=copy.deepcopy(PRECISION_SPEC),base_config_queries_before_upgrade=64,
                 effective_visual_queries_per_view=16,old_gate_removed=True,
                 action_model_parameters=sum(p.numel() for p in wrapper.action_model.parameters()))
    return wrapper,notes


def validate_training_contract(contract):
    if contract.get('format')!=FORMAT:raise ValueError('Precision-WAM checkpoint required')
    if contract.get('precision')!=PRECISION_SPEC or contract['config']['model'].get('precision')!=PRECISION_SPEC:
        raise ValueError('Precision architecture/clock contract mismatch')
    compatibility=contract.get('precision_cache_compatibility',{})
    if compatibility.get('raw_feature_binding')!=contract['hjepa']['cache_binding'] or compatibility.get('normalization_and_vtt_unchanged') is not True:
        raise ValueError('Frozen feature reuse identity mismatch')
    for rel in EXTENSION_SOURCES:
        if str(ROOT/rel) not in contract['source_files']:raise ValueError('Missing Precision implementation binding: '+rel)
    old.validate_training_contract(dict(contract,format=old.FORMAT))
    return contract


def load_policy(path,task,device='cuda'):
    import torch
    import numpy as np
    path=resolve(path);receipt=read_json(path.with_name(path.name+'.sha256.json'))
    if receipt.get('size')!=path.stat().st_size or receipt.get('sha256')!=sha256_file(path):raise ValueError('Precision checkpoint receipt mismatch')
    ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if ck.get('format')!=FORMAT or ck.get('model_only') is not True or receipt.get('model_only') is not True or receipt.get('step')!=ck.get('step'):
        raise ValueError('Immutable Precision milestone required')
    contract=validate_training_contract(ck['contract'])
    if ck.get('contract_sha256')!=digest(contract) or task!=TASK or task not in contract['conditions']:
        raise ValueError('Precision task/contract mismatch')
    verify_code_hashes(contract['source_files'])
    wrapper,_=build_wrapper(contract['config'],contract['statistics'],variant=contract['hjepa']['variant'],
        idm_statistics=contract['idm_statistics'],device=device,state_dict=ck['model_state_dict'])
    wrapper.eval();return wrapper,np.asarray(contract['conditions'][task],dtype=np.float32),contract
