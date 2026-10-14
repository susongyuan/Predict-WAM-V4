"""V4 dataset, checkpoint migration and strict self-contained inference wiring."""
from __future__ import annotations

from bisect import bisect_right
import copy

import numpy as np

from harness import zhuhai_precision_v2_training as v2
from harness import zhuhai_precision_v3_training as v3
from harness import zhuhai_precision_v3_evaluation_v42 as v3_eval
from harness.zhuhai_hjepa_dataset import _last_real_observation
from harness.wam_precision_v4 import HORIZON, CHUNK_WEIGHT, COSINE_WEIGHT, COSINE_EPS, upgrade_precision_v4

FORMAT = 'zhuhai-precision-wam-native16-measured-10hz-v4'
TASK = v2.TASK
CLOCK_CONFIG = v2.CLOCK_CONFIG
PRECISION_V4_SPEC = dict(schema='causal_chunk_vtt16_v4', horizon=HORIZON,
    input_dim=7184, hidden_dims=[512, 512], output_shape=[3, 1024], projection_shape=[3072, 768],
    current='frozen DINO layer -4 mean patches, per-view non-affine LayerNorm',
    history='mean-patch current minus max(t-16,0), original TRAIN memory std and clip +/-5',
    state='original normalized measured current native16', task='TRAIN-only Task-VTT, non-affine LayerNorm',
    target='frozen DINO layer -4 mean patches at t+16 minus t; raw feature difference',
    huber_beta=1., cosine_weight=COSINE_WEIGHT, cosine_target_norm_epsilon=COSINE_EPS,
    auxiliary_weight=CHUNK_WEIGHT, fusion='existing projected Task token + alpha * Linear(chunk_vtt16)',
    alpha_initial=0., predictions_per_replan=1, euler_steps=10, execution_prefix=8,
    future_auxiliary_horizons_unchanged=[8, 32], idm_horizon_unchanged=4,
    global_visual_tokens=48, dense_patches=588, refine_steps=32, hidden_dim=768,
    online_future_inputs=False)
EXT_V4 = ('harness/wam_precision_v4.py', 'harness/zhuhai_precision_v4_training.py',
          'harness/zhuhai_precision_v3_evaluation_v42.py',
          'scripts/test_wam_precision_v4.py')


class ChunkVTTDataset:
    def __init__(self, dataset):
        self.base = dataset

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    @property
    def condition(self):
        return self.base.condition

    @condition.setter
    def condition(self, value):
        self.base.condition = value

    def __getitem__(self, index):
        sample = self.base[index]
        episode = self.base.episodes[bisect_right(self.base.ends, index)]
        future = int(sample['anchor']) + HORIZON
        valid = future <= _last_real_observation(episode)
        current = sample['pool'][:, 0, 1024:]
        sample['chunk_vtt_target'] = (self.base.store.get(episode, future)[1][:, 1024:] - current
                                      if valid else np.zeros((3, 1024), np.float32)).astype(np.float32, copy=True)
        sample['chunk_vtt_valid_mask'] = np.bool_(valid)
        return sample


def build_datasets(splits, store, cfg, *, variant):
    datasets = v2.build_datasets(splits, store, cfg, variant=variant)
    return {split: {task: ChunkVTTDataset(ds) for task, ds in tasks.items()}
            for split, tasks in datasets.items()}


def new_contract(base):
    contract = copy.deepcopy(base)
    contract['format'] = FORMAT
    contract['precision_v3'] = copy.deepcopy(v3.PRECISION_V3_SPEC)
    contract['config']['model']['precision_v3'] = copy.deepcopy(v3.PRECISION_V3_SPEC)
    contract['precision_v4'] = copy.deepcopy(PRECISION_V4_SPEC)
    contract['config']['model']['precision_v4'] = copy.deepcopy(PRECISION_V4_SPEC)
    for rel in (*v3_eval.V3_SOURCES, *EXT_V4):
        contract['source_files'][str(v2.ROOT / rel)] = v2.sha256_file(v2.ROOT / rel)
    contract['hjepa']['code_hashes'] = copy.deepcopy(contract['source_files'])
    return contract


def build_wrapper(cfg, statistics, *, variant, idm_statistics, device='cpu',
                  state_dict=None, initialization=None):
    if (state_dict is None) == (initialization is None):
        raise ValueError('Supply exactly one initializer or state dictionary')
    if initialization is not None:
        raise ValueError('V4 starts from verified Precision V2/V3 weights, not a new legacy initializer')
    is_v4 = any(key.startswith('chunk_vtt16.') for key in state_dict)
    if is_v4:
        validate_model_identity(state_dict)
    version = int(state_dict['precision.version'])
    if version == 2 and not is_v4:
        wrapper, notes = v3.build_wrapper(cfg, statistics, variant=variant,
            idm_statistics=idm_statistics, device=device, state_dict=state_dict)
        origin = 'strict V2 80k, then existing V3 full32 warm-start'
    elif version == 3:
        backbone = {key: value for key, value in state_dict.items() if not key.startswith('chunk_vtt16.')}
        wrapper = v3_eval.build_wrapper(cfg, statistics, variant=variant,
            idm_statistics=idm_statistics, device=device, state_dict=backbone)
        notes, origin = {}, 'strict self-contained V4' if is_v4 else 'strict V3'
    else:
        raise ValueError('V4 requires Precision V2 or V3 backbone weights')
    wrapper = upgrade_precision_v4(wrapper, memory_lags=cfg['model']['v3']['memory_lags'])
    if is_v4:
        wrapper.action_model.load_state_dict(state_dict, strict=True)
    added = sum(p.numel() for p in wrapper.action_model.chunk_vtt16.parameters())
    notes.update(initialization=origin, precision_v4=copy.deepcopy(PRECISION_V4_SPEC),
                 added_parameters=added, action_model_parameters=sum(p.numel() for p in wrapper.action_model.parameters()))
    return wrapper, notes


def validate_training_contract(contract):
    if not isinstance(contract, dict) or contract.get('format') != FORMAT:
        raise ValueError('Precision V4 checkpoint format required')
    if (contract.get('precision_v4') != PRECISION_V4_SPEC
            or contract.get('config', {}).get('model', {}).get('precision_v4') != PRECISION_V4_SPEC):
        raise ValueError('Precision V4 architecture/target contract mismatch')
    v3_eval.validate_training_contract(dict(contract, format=v3.FORMAT))
    if any(str(v2.ROOT / rel) not in contract['source_files'] for rel in EXT_V4):
        raise ValueError('Precision V4 implementation binding missing')
    return contract


def validate_model_identity(state_dict):
    import torch
    v3_eval.validate_model_identity(state_dict)
    expected = torch.tensor([4, 16, 1024, 16, 1024, 768, 512])
    config = state_dict.get('chunk_vtt16.configuration')
    if not isinstance(config, torch.Tensor) or config.dtype != torch.int64 or not torch.equal(config.cpu(), expected):
        raise ValueError('Precision V4 Chunk-VTT identity mismatch')
    shapes = {'predictor.0.weight': (512, 7184), 'predictor.0.bias': (512,),
              'predictor.2.weight': (512, 512), 'predictor.2.bias': (512,),
              'predictor.4.weight': (3072, 512), 'predictor.4.bias': (3072,),
              'projection.weight': (768, 3072), 'projection.bias': (768,), 'alpha': ()}
    for key, shape in shapes.items():
        value = state_dict.get('chunk_vtt16.' + key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError('Precision V4 Chunk-VTT tensor mismatch: ' + key)
    return dict(version=4, horizon=16, input_dim=7184, output_dim=3072, added_parameters=7877377)


def _load_verified_checkpoint(path):
    import torch
    path = v2.no_symlink(v2.resolve(path))
    receipt = v2.read_json(v2.no_symlink(path.with_name(path.name + '.sha256.json')))
    if (type(receipt.get('size')) is not int or receipt['size'] != path.stat().st_size
            or receipt.get('sha256') != v2.sha256_file(path)):
        raise ValueError('Precision V4 checkpoint receipt mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (ck.get('format') != FORMAT or ck.get('model_only') is not True
            or receipt.get('model_only') is not True or type(ck.get('step')) is not int
            or ck['step'] < 0 or type(receipt.get('step')) is not int or receipt['step'] != ck['step']):
        raise ValueError('Immutable Precision V4 milestone required')
    contract = validate_training_contract(ck['contract'])
    if ck.get('contract_sha256') != v2.digest(contract):
        raise ValueError('Precision V4 contract SHA mismatch')
    v2.verify_code_hashes(contract['source_files'])
    validate_model_identity(ck['model_state_dict'])
    return path, receipt, ck


def read_contract(path):
    path, receipt, ck = _load_verified_checkpoint(path)
    manifest = v2.read_json(v2.no_symlink(path.parent / 'manifest.json'))
    if manifest.get('contract') != ck['contract'] or manifest.get('contract_sha256') != ck['contract_sha256']:
        raise ValueError('Precision V4 checkpoint/manifest contract mismatch')
    contract = ck['contract']
    for source in contract['sources'].values():
        if source.get('engine', {}).get('name') != 'Isaac Sim':
            raise ValueError('Precision V4 requires Isaac Sim collection provenance')
        scene = source['scene']
        if v2.sha256_file(v2.no_symlink(scene['path'])) != scene['sha256']:
            raise ValueError('Precision V4 collection scene changed')
        for row in source['manifests']:
            if v2.sha256_file(v2.no_symlink(row['path'])) != row['sha256']:
                raise ValueError('Precision V4 collection manifest changed')
            for field in ('config_hashes', 'scene_dependency_sha256'):
                if row['contract'].get(field):
                    v2.verify_code_hashes(row['contract'][field])
    from harness.zhuhai_action_smoothing import protocol, protocol_hash
    v2.validate_postprocessing(contract['action_postprocessing'])
    if (contract['action_postprocessing'] != protocol()
            or contract.get('action_postprocessing_sha256') != protocol_hash(protocol())):
        raise ValueError('Precision V4 frozen TRAIN action postprocessing changed')
    return ck['contract'], receipt


def load_policy(path, task=TASK, device='cpu'):
    path, receipt, ck = _load_verified_checkpoint(path)
    contract = ck['contract']
    if task != TASK or task not in contract['conditions']:
        raise ValueError('Precision V4 task/condition mismatch')
    wrapper, notes = build_wrapper(contract['config'], contract['statistics'],
        variant=contract['hjepa']['variant'], idm_statistics=contract['idm_statistics'],
        device=device, state_dict=ck['model_state_dict'])
    wrapper.eval()
    wrapper.precision_v4_load_receipt = dict(checkpoint_path=str(path), checkpoint_sha256=receipt['sha256'],
        step=ck['step'], strict_load=True, self_contained=True, **notes)
    return wrapper, np.asarray(contract['conditions'][task], np.float32), contract
