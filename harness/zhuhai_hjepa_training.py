"""Independent task2 H-JEPA-inspired training sidecar; never auto-evaluate.

The original native16 migration, normalization, VTT, family split, cache engine,
optimizer equation and sample ordering remain authoritative. Only auxiliary
labels/wrapper and the new, incompatible artifact contract are added here.
Imports are lazy: importing this module or asking for CLI help starts no model.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'zhuhai-hjepa-native16-measured-10hz-v1'
VARIANTS = ('baseline', 'multiscale', 'idm', 'combined')
TASK = 'zhuhai_task2'
DEFAULT_STEPS = 30000
BATCH_SIZE = 4
GIB = 1024**3
AUX_PREFIX = 'hjepa_aux.'
AUX_CONFIG = dict(idm_hidden=512, step_embedding_dim=16, aux_init_seed=20261009)
INITIALIZATION = ROOT/'outputs/runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/train/checkpoint_step_050000.pt'
INITIALIZATION_SHA = '8960be6ceb4d5293d2c5455ac4308421f64f1e31bf64f0646b7208735f63a5f4'
DEFAULT_TASKS = ROOT/'outputs/zhuhai2026_tasks12/production_task2_collection_only_20261009_v11/tasks.json'
NEW_SOURCES = ('harness/wam_hjepa.py', 'harness/zhuhai_hjepa_dataset.py',
               'harness/zhuhai_hjepa_training.py', 'scripts/train_zhuhai_wam_hjepa.py',
               'scripts/eval_zhuhai_wam_hjepa.py', 'scripts/test_zhuhai_hjepa_training.py')
BASE_SOURCES = ('harness/zhuhai_training.py', 'scripts/train_zhuhai_wam.py',
                'harness/zhuhai_dataset.py', 'harness/zhuhai_preparation_reuse.py',
                'harness/zhuhai_hybrid_cache.py', 'harness/zhuhai_demo_recording.py',
                'harness/zhuhai_robot_spec.py', 'harness/zhuhai_action_smoothing.py',
                'harness/wam_v3.py', 'harness/wam_behavior.py',
                'harness/robotwin_training_data.py', 'harness/robotwin_multitask_training.py',
                'harness/robotwin_vtt_adapter.py', 'third_party/LiLa-WAM/robotwin_infer.py',
                'scripts/gpu_lock.py')
LABEL_CONTRACT = dict(control_hz=10., idm_steps=4, short_future_offset=8, long_future_offset=32,
    action_chunk_steps=32, execution_steps=8, future_target_layer=-4, future_teacher_stop_gradient=True,
    idm_visual_observation_offsets=[0, 1, 2, 3, 4], idm_target_shape=[4, 16],
    idm_target_dtype='float32', idm_anchor='fixed_current_measured_q_t',
    idm_archive_formula='action[t+k]-state[t], k=0..3; archive action[t]=command[t+1]',
    idm_normalization='raw_div_scale_no_center_no_clip', invalid_windows='mask; no clamping/cross-episode padding as valid',
    future_is_policy_input=False, causal_prefix_claim=False)


def _zt():
    from harness import zhuhai_training
    return zhuhai_training


def protocol_binding():
    return _zt().protocol_binding()


def robot_contract():
    return _zt().robot_contract()


def validate_postprocessing(settings):
    return _zt().validate_postprocessing(settings)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def resolve(path):
    return (ROOT/Path(path)).resolve()


def no_symlink(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'Symlink path forbidden: {path}')
    return path


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value, *, immutable=False):
    path = no_symlink(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f'Immutable evidence differs: {path}; choose an independent output')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.'+path.name+'-', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            temporary.replace(path)
        _sync(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(out):
    out = no_symlink(out); out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out/'.hjepa_training.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another writer owns this independent HJEPA output') from exc
        yield
    finally:
        os.close(fd)


def verify_owner(owner):
    from scripts.gpu_lock import owner as current
    if not owner or current() != owner:
        raise RuntimeError('Matching externally owned GPU lock required; sidecar never acquires/releases it')


def check_disk(path, required=0, *, minimum=100*GIB):
    from shutil import disk_usage
    path = Path(path)
    while not path.exists():
        path = path.parent
    if disk_usage(path).free < minimum + required:
        raise RuntimeError('100GiB disk reserve breached; preserve output and stop')


def variant_settings(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown HJEPA variant')
    multi, inverse = variant in ('multiscale', 'combined'), variant in ('idm', 'combined')
    return dict(variant=variant, multi_horizon=multi, use_idm=inverse,
                future32_weight=.25 if multi else .5, future8_weight=.25 if multi else 0.,
                idm_weight=.1 if inverse else 0.)


def source_hashes():
    paths = [ROOT/name for name in (*NEW_SOURCES, *BASE_SOURCES)]
    missing = [str(path) for path in paths if not path.is_file()]
    return {str(no_symlink(path)): sha256_file(path) for path in paths if path.is_file()}, missing


def verify_code_hashes(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Missing code hashes')
    for path, expected in mapping.items():
        if sha256_file(no_symlink(path)) != expected:
            raise ValueError(f'Frozen source changed: {path}; no silent resume/code upgrade')


def validate_request(args):
    for key in ('steps', 'batch_size', 'save_every', 'seed'):
        value = getattr(args, key)
        if type(value) is not int or value <= 0:
            raise ValueError('Positive integer '+key+' required')
    if args.batch_size != BATCH_SIZE:
        raise ValueError('B4 is frozen; B8 requires a separately admitted new contract')
    if (args.source_count, args.randomized_count) != (100, 100) or any(
            type(getattr(args, key)) is not int for key in ('source_count', 'randomized_count')):
        raise ValueError('Task2 requires exactly 100 source + 100 DR, including tiny mode')
    variant_settings(args.variant)
    if args.mode not in ('plan', 'tiny', 'train') or (args.mode == 'plan' and args.execute):
        raise ValueError('Default plan is CPU-only; explicitly select tiny/train plus --execute')
    if type(args.cache_budget_bytes) is not int or not 0 <= args.cache_budget_bytes <= 10**12:
        raise ValueError('1TB cache cap cannot be increased')
    if type(args.cache_ram_budget_bytes) is not int or not 0 <= args.cache_ram_budget_bytes <= 2*GIB:
        raise ValueError('2GiB RAM cache cap cannot be increased')
    if type(args.minimum_free_bytes) is not int or args.minimum_free_bytes < 100*GIB:
        raise ValueError('100GiB disk reserve cannot be weakened')
    if args.max_updates is not None and (type(args.max_updates) is not int or args.max_updates <= 0):
        raise ValueError('max-updates is a positive absolute stopping boundary, not a new schedule')
    if args.mode == 'tiny' and args.max_updates not in (None, 2):
        raise ValueError('Tiny always performs exactly two updates; use an independent output')
    if args.initialization_sha256 != INITIALIZATION_SHA or resolve(args.initialization) != INITIALIZATION.resolve():
        raise ValueError('Original frozen native14 50k initialization is required')
    for path in (args.out, args.cache_root, args.preparation_root):
        no_symlink(path)
    out = resolve(args.out)
    protected = [ROOT/'harness', ROOT/'scripts', INITIALIZATION.parent, resolve(args.cache_root), resolve(args.preparation_root)]
    if any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError('Independent output may not overlap code, frozen weights or cache roots')


def admit_tasks(path):
    """Current scope is task2 only; do not wait for or silently include garbage."""
    zt = _zt(); path = no_symlink(resolve(path)); supplied = read_json(path)
    specs = supplied.get('tasks')
    if not isinstance(specs, list) or len(specs) != 1 or specs[0].get('task') != TASK:
        raise ValueError('This run requires a task2-only task list; task1 is not authorized')
    from scripts import collect_isaac_zhuhai as source_protocol
    allowed = {'collect_isaac_zhuhai_tasks12_repair', 'collect_isaac_zhuhai_task2_repair',
               'collect_isaac_zhuhai_recycle', 'collect_isaac_zhuhai_100_parallel'}
    source = no_symlink(resolve(specs[0]['source_manifest']))
    randomized = no_symlink(resolve(specs[0]['randomized_manifest']))
    doc = read_json(randomized); implementation = doc['contract']['collection_implementation']
    collector_path = no_symlink(resolve(implementation['path']))
    if collector_path.parent != ROOT/'scripts' or collector_path.stem not in allowed or sha256_file(collector_path) != implementation['sha256']:
        raise ValueError('Unsupported or changed task2 collection implementation')
    collector = importlib.import_module('scripts.'+collector_path.stem)
    if (doc.get('status') != 'complete' or doc['contract'].get('source_probe') is not False or
            doc['contract'].get('allow_unvalidated_probe') is not False or doc['contract'].get('collection_scope') != 'production'):
        raise ValueError('Completed production task2 DR required; probes/partial cannot enter any mode')
    source_rows = source_protocol.validate_manifest(source, task=TASK, mode='source', verify_files=True)
    dr_rows = collector.validate_manifest(randomized, task=TASK, mode='randomized', verify_files=True)
    if len(source_rows) != 100 or len(dr_rows) != 100:
        raise ValueError('Full task2 100+100 quota required before even a plan/tiny admission')
    parents = {r['family_id']: r['seed'] for r in source_rows}
    if len(parents) != 100 or len({r['family_id'] for r in dr_rows}) != 100 or any(
            parents.get(r['family_id']) != r.get('source_parent') for r in dr_rows):
        raise ValueError('Exactly one DR per original source family required')
    rows = [*source_rows, *dr_rows]
    for key in ('seed', 'data_sha256', 'data_path'):
        values = [str(resolve(r[key])) if key == 'data_path' else r[key] for r in rows]
        if len(set(values)) != 200:
            raise ValueError('Duplicate task2 source/DR '+key)
    splits, sources = zt.task_sources(path, allow_small=False, source_count=100, randomized_count=100)
    validate_splits(splits, sources)
    return splits, sources


def validate_splits(splits, sources):
    if set(splits) != {TASK} or set(sources) != {TASK}:
        raise ValueError('Frozen task2-only scope required')
    if sources[TASK].get('quotas') != dict(source=100, randomized=100):
        raise ValueError('No relaxed quota, even for tiny')
    groups, seen_paths, seen_hashes, seen_seeds = {}, set(), set(), set()
    if set(splits[TASK]) != {'train', 'val', 'heldout'}:
        raise ValueError('Exactly TRAIN/VAL/heldout split containers required')
    for split, expected in (('train', 80), ('val', 10), ('heldout', 10)):
        rows = splits[TASK][split]
        families = {}
        for row in rows:
            if row.get('split') != split or row.get('domain') not in ('source', 'randomized'):
                raise ValueError('Split/domain provenance mismatch')
            family = str(row['family_id']); families.setdefault(family, []).append(row['domain'])
            if family in groups and groups[family] != split:
                raise ValueError('Source family crossed split')
            groups[family] = split
            path = str(resolve(row['path']))
            if path in seen_paths or row['sha256'] in seen_hashes or row['seed'] in seen_seeds:
                raise ValueError('Duplicated/leaked episode across splits')
            seen_paths.add(path); seen_hashes.add(row['sha256']); seen_seeds.add(row['seed'])
        if len(rows) != 2*expected or len(families) != expected or any(sorted(v) != ['randomized', 'source'] for v in families.values()):
            raise ValueError('Exact family80/10/10, each with one source and one DR, required')
    return splits


def build_datasets(splits, store, cfg, *, variant):
    """Original iteration/anchor order, with only the dataset class replaced."""
    zt = _zt()
    from harness.zhuhai_hjepa_dataset import HybridHjepaTaskDataset
    zt.validate_config(cfg); settings = variant_settings(variant)
    result = {split: {} for split in ('train', 'val', 'heldout')}
    seen_paths, seen_hashes = set(), set()
    for task, task_splits in splits.items():
        family_splits = {}
        for split, rows in task_splits.items():
            if split not in result:
                raise ValueError('Unknown split')
            for row in rows:
                zt.validate_binding(row)
                if row.get('split') != split:
                    raise ValueError('Episode split mismatch')
                path = str(resolve(row['path']))
                if path in seen_paths or row['sha256'] in seen_hashes:
                    raise ValueError('Duplicate/leaked episode across splits')
                seen_paths.add(path); seen_hashes.add(row['sha256'])
                family = str(row['family_id'])
                if family in family_splits and family_splits[family] != split:
                    raise ValueError('Source family leaked across splits')
                family_splits[family] = split
    # Reject all metadata leaks before constructing (and reading) any archive.
    for task, task_splits in splits.items():
        for split, rows in task_splits.items():
            if rows and split != 'heldout':
                ds = HybridHjepaTaskDataset(task, rows, store,
                    proprio_lags=cfg['model']['v3']['proprio_lags'], memory_lags=cfg['model']['v3']['memory_lags'],
                    chunk=32, future_offset=32, multi_horizon=settings['multi_horizon'], use_idm=settings['use_idm'])
                if len(ds) != sum(len(ep) for ep in ds.episodes) or [str(ep.path.resolve()) for ep in ds.episodes] != [str(resolve(r['path'])) for r in rows]:
                    raise ValueError('Auxiliary dataset must not filter/reorder original anchors')
                for ep in ds.episodes:
                    zt.validate_episode_binding(ep)
                result[split][task] = ds
    return result


def validate_idm_statistics(value, provenance):
    import numpy as np
    from harness.zhuhai_hjepa_dataset import IDM_STATISTICS_SCHEMA
    fields = dict(schema=IDM_STATISTICS_SCHEMA, fit_split='train', heldout_used=False, horizon_steps=4, action_dim=16,
        target_dtype='float32', accumulator_dtype='float64', variance='population',
        normalization='raw_div_scale_no_center_no_clip', center=False, clip=False,
        label_formula='u[t+k]-q[t], k=0..3; fixed measured q[t]',
        scale_floor=dict(train_action_range_multiplier=.01, absolute=1e-6))
    if any(value.get(k) != v for k, v in fields.items()):
        raise ValueError('IDM statistics schema/label/TRAIN-only contract changed')
    if value.get('statistics_sha256') != canonical_digest({k:v for k,v in value.items() if k != 'statistics_sha256'}):
        raise ValueError('IDM statistics self hash mismatch')
    sources = value.get('train_sources', [])
    if not sources or value.get('train_sources_sha256') != canonical_digest(sources):
        raise ValueError('IDM TRAIN source hash mismatch')
    def key(row):
        return (row['task'], str(resolve(row['path'])), row['sha256'], row['split'], row['family_id'], row['seed'], row['frames'])
    expected, actual = {key(row) for row in provenance['records']}, {key(row) for row in sources}
    if actual != expected or len(actual) != len(sources) or any(row['split'] != 'train' for row in sources):
        raise ValueError('IDM scale must use exactly the original successful TRAIN records')
    windows = value.get('valid_windows')
    if (type(windows) is not int or windows <= 0 or value.get('valid_label_vectors') != 4*windows or
            any(type(row.get('valid_windows')) is not int or row['valid_windows'] < 0 for row in sources) or
            windows != sum(row['valid_windows'] for row in sources)):
        raise ValueError('IDM complete four-step TRAIN windows required')
    arrays = [np.asarray(value.get(key), dtype=np.float64) for key in ('scale16', 'std16', 'action_range16')]
    if any(x.shape != (16,) or not np.isfinite(x).all() or (x < 0).any() for x in arrays) or (arrays[0] <= 0).any():
        raise ValueError('Finite positive TRAIN IDM scale16 required')
    expected_scale = np.maximum(np.maximum(arrays[1], .01*arrays[2]), 1e-6)
    if not np.array_equal(arrays[0], expected_scale):
        raise ValueError('IDM scale must equal max(TRAIN std, .01*TRAIN action range, 1e-6)')
    return value


class SourceBoundFeatureCache:
    """Keep the original bounded cache engine; extend keys, not feature math."""
    def __init__(self, cache, binding):
        self.cache, self.binding = cache, copy.deepcopy(binding)

    def read_bytes(self, identity, generator):
        return self.cache.read_bytes(dict(identity, hjepa_binding=self.binding), generator)

    def __getattr__(self, name):
        return getattr(self.cache, name)


def preparation_cache(root, binding):
    from harness.zhuhai_preparation_reuse import PreparationReuse
    from harness.zhuhai_hjepa_dataset import fit_idm_statistics, _train_entries
    class HjepaPreparation(PreparationReuse):
        def _identity(self, datasets, cfg, feature_identity, check):
            _train_entries(datasets)
            original = super()._identity(datasets, cfg, feature_identity, check)
            return dict(original, hjepa_binding=copy.deepcopy(binding))

        def idm_statistics(self, datasets, cfg, *, feature_identity, check=lambda: None):
            with self._locked('idm_statistics', check) as operation:
                identity = self._identity(datasets, cfg, feature_identity, check)
                provenance = identity['train_provenance']
                def recheck():
                    if self._identity(datasets, cfg, feature_identity, check) != identity:
                        raise ValueError('IDM TRAIN/code identity changed during fit')
                return self._obtain('idm_statistics', identity,
                    lambda value: validate_idm_statistics(value, provenance),
                    lambda: fit_idm_statistics(datasets), recheck, check, operation)
    return HjepaPreparation(root)


def make_plan(args):
    """No GPU calls or source fitting. Data/code must be complete before freezing."""
    validate_request(args); splits, sources = admit_tasks(args.tasks)
    hashes, missing = source_hashes()
    if missing:
        return dict(status='waiting_for_code', missing_code=missing, plan_frozen=False, cuda_initialized=False), None
    zt = _zt()
    ck, initialization = zt.checkpoint_header(args.initialization, expected_sha=args.initialization_sha256)
    kind = 'tiny' if args.mode == 'tiny' else 'train'
    steps = 2 if kind == 'tiny' else args.steps
    cfg = zt.config_for(ck, steps=steps, seed=args.seed, batch=args.batch_size)
    settings = variant_settings(args.variant)
    cfg['training']['auxiliary_loss'] = dict(future=settings['future32_weight'],
        future8=settings['future8_weight'], idm=settings['idm_weight'], gate_kl=0., phase_ce=0., log_dist=0.)
    cfg['training'].update(save_every=2 if kind == 'tiny' else args.save_every,
        init_from=str(resolve(args.initialization)), init_from_sha256=args.initialization_sha256)
    from harness.zhuhai_action_smoothing import protocol, protocol_hash
    from harness.zhuhai_dataset import PREPROCESS_SHA256
    smoothing = zt.validate_postprocessing(protocol()); cfg.setdefault('inference', {})['action_postprocessing'] = smoothing
    encoder_path = resolve(cfg['model']['vision_encoder']['checkpoint_path'])
    provenance = read_json(encoder_path/'DOWNLOAD_PROVENANCE.json')
    encoder_sha = provenance['files']['model.safetensors']['sha256']
    binding = dict(format=FORMAT+'/feature-preparation', source_files={str(ROOT/p):hashes[str(ROOT/p)] for p in NEW_SOURCES},
                   labels=copy.deepcopy(LABEL_CONTRACT), variants_share_base_features_and_TRAIN_statistics=True)
    contract = dict(format=FORMAT, mode=kind, config=cfg, sources=sources, splits=splits, initialization=initialization,
        time=zt.time_contract(cfg), **zt.protocol_binding(), overfit_anchors=0,
        numerical_normalization='new successful TRAIN-only native16 measured-state/target-action statistics; native14 normalization forbidden',
        cache=dict(root=str(resolve(args.cache_root)), disk_budget_bytes=args.cache_budget_bytes,
            ram_budget_bytes=args.cache_ram_budget_bytes, minimum_free_bytes=args.minimum_free_bytes,
            encoder_sha256=encoder_sha, preprocess_sha256=PREPROCESS_SHA256, raw_data_evicted=False,
            policy='original HybridFeatureCache + BoundedRamFeatures; new source/label identity; no decoded RGB cache',
            ram_scope='derived features and index; model/optimizer/minibatch transient RAM excluded',
            estimated_task2_train_val_raw_dino_bytes=650*10**9, estimate_is_not_measured=True),
        protocol=dict(record_video=False, third_view_camera_created=False, pure_wam=True, vlm=False,
            high_level_planner=False, ask=False, lifelong_memory_writes=False, capacity_is_not_task_success=True),
        action_postprocessing=smoothing, action_postprocessing_sha256=protocol_hash(smoothing), source_files=hashes,
        automatic_policy_evaluation=False,
        hjepa=dict(schema=FORMAT+'/contract', variant=args.variant, settings=settings,
            labels=copy.deepcopy(LABEL_CONTRACT), auxiliary=copy.deepcopy(AUX_CONFIG),
            code_hashes=hashes, cache_binding=binding, preparation_root=str(resolve(args.preparation_root)),
            candidate_steps=DEFAULT_STEPS, schedule_steps=steps, requested_steps=args.steps,
            task_scope=[TASK], sample_order='original task-uniform then anchor-uniform NumPy Generator; no filtering/reordering',
            sample_order_sha256=digest(splits), tiny_updates=2 if kind == 'tiny' else None,
            suggested_test_milestones=([step for step in (steps-10000, steps-5000, steps) if step > 0]
                                       if kind == 'train' else []), automatic_evaluation=False))
    return dict(status='cpu_preflight_completed', contract=contract, contract_sha256=digest(contract),
                cuda_initialized=False, frame_counts_unknown_until_hdf_validation=True,
                statistics='pending TRAIN-only preparation; not training step0'), ck




def validate_training_contract(contract):
    zt = _zt()
    if contract.get('format') != FORMAT:
        raise ValueError('Independent H-JEPA native16 checkpoint required')
    zt.validate_training_contract(dict(contract, format=zt.FORMAT))
    validate_splits(contract['splits'], contract['sources'])
    h = contract['hjepa']; settings = variant_settings(h['variant'])
    if (h.get('schema') != FORMAT+'/contract' or h.get('settings') != settings
            or h.get('labels') != LABEL_CONTRACT or h.get('auxiliary') != AUX_CONFIG
            or h.get('task_scope') != [TASK] or h.get('automatic_evaluation') is not False):
        raise ValueError('H-JEPA model/label/task contract changed')
    expected = dict(future=settings['future32_weight'], future8=settings['future8_weight'],
                    idm=settings['idm_weight'], gate_kl=0., phase_ce=0., log_dist=0.)
    if contract['config']['training']['auxiliary_loss'] != expected:
        raise ValueError('H-JEPA loss weights changed')
    if h.get('code_hashes') != contract['source_files']:
        raise ValueError('H-JEPA source aliases differ')
    if (contract.get('automatic_policy_evaluation') is not False or
            any(contract['protocol'].get(k) is not False for k in
                ('record_video', 'third_view_camera_created', 'vlm', 'ask', 'lifelong_memory_writes'))):
        raise ValueError('Unexpected recording/online-system protocol')
    validate_idm_statistics(contract['idm_statistics'], contract['statistics']['normalization']['provenance'])
    if contract.get('idm_statistics_sha256') != digest(contract['idm_statistics']):
        raise ValueError('IDM statistics alias hash mismatch')
    return contract


def build_wrapper(cfg, statistics, *, variant, idm_statistics, device='cpu', initialization=None, state_dict=None):
    from harness.wam_hjepa import upgrade_wrapper
    base_state = None if state_dict is None else {k:v for k,v in state_dict.items() if not k.startswith(AUX_PREFIX)}
    base, notes = _zt().build_wrapper(cfg, statistics, device=device, initialization=initialization, state_dict=base_state)
    wrapper = upgrade_wrapper(base, variant=variant, idm_scale=idm_statistics['scale16'], **AUX_CONFIG)
    if state_dict is not None:
        wrapper.action_model.load_state_dict(state_dict, strict=True)
        import torch
        expected = torch.tensor(idm_statistics['scale16'], dtype=torch.float32, device=device)
        if not torch.equal(wrapper.action_model.hjepa_aux.idm_scale, expected):
            raise ValueError('Checkpoint IDM scale differs from TRAIN statistics')
    notes.update(hjepa_variant=variant, auxiliary_parameters=sum(p.numel() for p in wrapper.action_model.hjepa_aux.parameters()),
                 future_is_policy_input=False, frozen_original_checkpoint_modified=False)
    return wrapper, notes


def load_policy(path, task, device='cuda'):
    import numpy as np
    import torch
    path = resolve(path); receipt = read_json(path.with_name(path.name+'.sha256.json'))
    if receipt.get('size') != path.stat().st_size or receipt.get('sha256') != sha256_file(path):
        raise ValueError('H-JEPA checkpoint content/receipt mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (ck.get('format') != FORMAT or ck.get('model_only') is not True
            or receipt.get('model_only') is not True or receipt.get('step') != ck.get('step')):
        raise ValueError('Evaluation requires an immutable H-JEPA milestone')
    contract = validate_training_contract(ck['contract'])
    if ck.get('contract_sha256') != digest(contract) or task != TASK or task not in contract['conditions']:
        raise ValueError('H-JEPA checkpoint task/contract mismatch')
    verify_code_hashes(contract['source_files'])
    wrapper, _ = build_wrapper(contract['config'], contract['statistics'], device=device,
        variant=contract['hjepa']['variant'], idm_statistics=contract['idm_statistics'], state_dict=ck['model_state_dict'])
    wrapper.eval()
    return wrapper, np.asarray(contract['conditions'][task], dtype=np.float32), contract
