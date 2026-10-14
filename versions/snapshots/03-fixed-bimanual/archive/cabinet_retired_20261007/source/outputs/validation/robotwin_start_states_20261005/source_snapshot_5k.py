#!/usr/bin/env python3
"""Fixed 5k CPU frame-zero diagnostic. Only this script and OUT are writable.

K=1, ten Euler steps, seed=20261004+sorted VAL episode index. No retries,
checkpoint selection, encoder, simulator, future visual features, or labels.
Uses the exact dataset history rule, cache validator, wrapper sampler and metrics.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import math
import signal
import time
import traceback
from datetime import datetime, timezone

import h5py
import numpy as np
import psutil
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_robotwin_wam import build_wrapper, action_metrics, digest, file_identity
from harness.robotwin_training_data import (validate_payload, episode_directory,
    resolve_cache, digest as data_digest)
from harness.robotwin_dataset import JOINT_FIELDS
from harness.wam_v3 import history_frames, bf16_from_bits

RUN = ROOT / 'outputs/runs/put_object_cabinet/wam_30k_20261005'
OUT = ROOT / 'outputs/validation/robotwin_start_states_20261005'
CONFIG = ROOT / 'configs/fixed_bimanual/robotwin_cabinet_500_wam.yaml'
CKPT = RUN / 'checkpoint_step_005000.pt'
SEMANTICS = 'controller_joint_targets_and_gripper_command_not_measured_qpos'
GROUPS = ('left_arm_rad', 'right_arm_rad', 'left_gripper_units', 'right_gripper_units')
WINDOWS = {'chunk32': 32, 'exec8': 8}
SEED = 20261004
START = time.monotonic()
PROCESS = psutil.Process()
PEAKS = {'system_cpu_percent': 0., 'system_ram_percent': 0., 'process_rss_bytes': 0,
         'process_cpu_percent_one_core_100': 0.}
READS = []
SOURCES = {}
ROWS = []
psutil.cpu_percent(None)
PROCESS.cpu_percent(None)


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def write_json(name, value):
    require(Path(name).name == name, 'write outside assigned directory')
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def runtime():
    return {'elapsed_seconds': time.monotonic() - START, 'sampled_resource_peaks': dict(PEAKS),
            'process_cpu_seconds': PROCESS.cpu_times().user + PROCESS.cpu_times().system,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'cuda_initialized': torch.cuda.is_initialized(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'device': 'cpu', 'python': sys.version, 'executable': sys.executable,
            'torch': str(torch.__version__), 'numpy': np.__version__, 'h5py': h5py.__version__,
            'resource_sample_period_seconds': 1, 'abort_cpu_ram_percent': 85}


def check(*unused):
    values = {'system_cpu_percent': psutil.cpu_percent(None),
              'system_ram_percent': psutil.virtual_memory().percent,
              'process_rss_bytes': PROCESS.memory_info().rss,
              'process_cpu_percent_one_core_100': PROCESS.cpu_percent(None)}
    for k, v in values.items():
        PEAKS[k] = max(PEAKS[k], v)
    require(values['system_ram_percent'] < 85 and values['system_cpu_percent'] < 85,
            f'CPU/RAM85 guard: {values}')
    require(time.monotonic() - START < 1800, '30-minute total diagnostic time limit')
    require(not torch.cuda.is_initialized(), 'CUDA must not initialize')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def register(path, fresh_sha=True):
    path = Path(path).resolve()
    identity = file_identity(path)
    if fresh_sha:
        identity['sha256'] = sha(path)
    SOURCES[str(path)] = identity
    return identity


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def summarize(items):
    n = sum(x['valid_action_rows'] for x in items)
    out = {k: sum(x[k] * x['valid_action_rows'] for x in items) / n
           for k in items[0] if k != 'valid_action_rows'}
    for g in GROUPS:
        out[g + '_rmse'] = math.sqrt(out[g + '_mse'])
    return {'samples': len(items), 'valid_action_rows': n,
            'normalized_scalar_denominator': n * 14, **out}


def bounds(values):
    out = {}
    for name, x in [('left_col6', values[:, 6]), ('right_col13', values[:, 13]),
                    ('both', values[:, [6, 13]].reshape(-1))]:
        x = x.astype(np.float64)
        violation = np.maximum(np.maximum(-x, x - 1), 0)
        count = int(np.count_nonzero(violation))
        out[name] = {'scalar_count': len(x), 'below_zero_count': int((x < 0).sum()),
                     'above_one_count': int((x > 1).sum()), 'out_of_range_count': count,
                     'out_of_range_fraction': count / len(x), 'raw_min': float(x.min()),
                     'raw_max': float(x.max()), 'max_exceedance_units': float(violation.max()),
                     'mean_exceedance_all_units': float(violation.mean()),
                     'mean_exceedance_violations_units': float(violation.sum() / count) if count else 0.}
    return out


def read_anchor(identity, cfg, cache_root, frozen_cache):
    name, n = identity['episode'], identity['length']
    path = Path(cfg['data_root']) / name
    require(path.parent == Path(cfg['data_root']) and not path.is_symlink(), 'unexpected episode path')
    hdf_stat = register(path, False)
    valid_n = min(32, n)
    with h5py.File(path, 'r') as f:
        require(str(f.attrs.get('state_semantics', '')) == SEMANTICS, 'state is not commanded targets')
        values = {}
        for group, count in [('state', 1), ('action', valid_n)]:
            columns = []
            for field, width in JOINT_FIELDS:
                ds = f[f'{group}/{field}']
                require(ds.shape == (n, width) and ds.dtype.kind in 'fiu', 'bad lowdim shape/dtype')
                a = ds[:count]
                require(np.isfinite(a).all() and (np.abs(a) <= np.finfo(np.float32).max).all(), 'bad lowdim values')
                columns.append(a.astype(np.float32))
            values[group] = np.concatenate(columns, axis=1)
    directory = episode_directory(cache_root, identity)
    cache_stats = {}
    for filename in ('meta.json', 'feat.npy', 'pool.npy'):
        p = directory / filename
        got = register(p, filename == 'meta.json')
        require(got == frozen_cache[str(p)], f'frozen cache identity changed: {p}')
        cache_stats[filename] = got
    arrays = validate_payload(cache_root, identity)
    # Only row zero is copied: unlike Dataset.__getitem__, never read future target features.
    feat_bits = arrays['feat'][0].copy()
    pool0 = arrays['pool'][0:1].copy()
    require(not np.any((feat_bits & 0x7f80) == 0x7f80) and np.isfinite(pool0).all(), 'nonfinite cache')
    pframes = [0] + history_frames(0, 0, [2, 4, 8, 16, 32])
    mframes = [0] + history_frames(0, 0, [4, 16])
    state = torch.from_numpy(values['state'][pframes].copy())[None]
    pool = torch.from_numpy(pool0[mframes].transpose(1, 0, 2).copy())[None]
    feats = bf16_from_bits(torch.from_numpy(feat_bits)[None]).float()
    require(feats.dtype == torch.float32 and feats.shape == (1, 3, 3, 201, 1024), 'CPU feature contract')
    # Floating tensors must remain values when the sampler calls bf16_from_bits again.
    require(bf16_from_bits(feats).data_ptr() == feats.data_ptr(), 'floating features reinterpreted')
    require(state.shape == (1, 7, 14) and pool.shape == (1, 3, 4, 2048), 'history shape')
    target = torch.zeros((1, 32, 14), dtype=torch.float32)
    target[0, :valid_n] = torch.from_numpy(values['action'])
    valid = (torch.arange(32) < valid_n)[None]
    entry = {'episode': name, 'hdf': hdf_stat, 'state_semantics': SEMANTICS,
             'whole_hdf_sha256_run_receipt_not_freshly_rehashed': identity['hdf_sha256'],
             'hdf_read_ranges': {'state/all_four_joint_fields': [0, 1], 'action/all_four_joint_fields': [0, valid_n]},
             'cache_files': cache_stats, 'cache_payload_frame_ranges': {'feat.npy': [0, 1], 'pool.npy': [0, 1]},
             'fresh_read_array_sha256': {**{k + '_float32': array_sha(v) for k, v in values.items()},
                                       'feat_frame0_uint16_bf16bits': array_sha(feat_bits), 'pool_frame0_float32': array_sha(pool0)},
             'hash_encoding': 'C order native little endian, exact dtype named in key',
             'state_history_frames': pframes, 'pool_history_frames': mframes}
    READS.append(entry)
    write_json('read_ledger.json', READS)
    return feats, state, pool, target, valid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / 'result.json').exists() and not (OUT / 'per_episode.json').exists(),
            'refuse rerun or overwrite previous diagnostic')
    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, 1., 1.)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(RuntimeError(f'signal {signum}')))
    try:
        check()
        manifest = json.loads((RUN / 'manifest.json').read_text())
        contract = manifest['contract']
        cfg = contract['config']
        require(digest(contract) == manifest['contract_sha256'], 'run contract digest mismatch')
        require(register(CONFIG)['sha256'] == contract['config_sha256'], 'frozen config changed')
        split_path = ROOT / cfg['dataset']['split_manifest']
        require(register(split_path)['sha256'] == cfg['dataset']['split_sha256'] == contract['split_sha256'], 'split mismatch')
        split = json.loads(split_path.read_text())
        names = sorted(split['splits']['val'])
        require(len(names) == 50 and len(set(names)) == 50, 'require every one of 50 VAL episodes')
        require(not set(names) & (set(split['splits']['train']) | set(split['splits']['heldout'])), 'split overlap')
        require(cfg['common']['action_chunk_size'] == 32 and cfg['common']['action_execution_horizon'] == 8
                and cfg['common']['proprio_len'] == 7 and cfg['dataset']['cond_mode'] == 'zero_single_task'
                and cfg['model']['v3']['proprio_lags'] == [2,4,8,16,32]
                and cfg['model']['v3']['memory_lags'] == [4,16], 'unexpected recipe')
        protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split': 'val', 'frame': 0,
                    'checkpoint': str(CKPT), 'checkpoint_step': 5000, 'samples_per_episode_K': 1,
                    'Euler_steps': 10, 'noise_seed_rule': '20261004 + zero-based lexicographically sorted VAL episode index',
                    'episode_plan': [{'episode': n, 'ordinal': i, 'seed': SEED+i} for i,n in enumerate(names)],
                    'policy_inputs': ['cached frame0 three-view DINO features', 'commanded state frame0 repeated seven times',
                                      'cached frame0 pool repeated four times per view', 'task_cond=zeros(1024)'],
                    'excluded_inputs': ['future features', 'future/current expert action targets', 'skill', 'phase', 'dist', 'active_hand'],
                    'target_only': 'stored action[0:32], no additional shift; masks for short episodes',
                    'windows': WINDOWS, 'clip_model_predictions': False, 'precision': 'CPU float32 (cached uint16 BF16 bits decoded once)',
                    'cpu_threads': 2, 'abort_cpu_ram_percent': 85, 'time_limit_seconds': 1800,
                    'first_anchor_time_limit_seconds': 120, 'selection': 'none; no retries, no favorable samples',
                    'boundary': 'offline first-command diagnostic, not simulator reset/closed-loop/physical success or visual causality'}
        write_json('protocol.json', protocol)
        for path, expected in contract['source_hashes'].items():
            require(register(path)['sha256'] == expected, f'frozen source changed: {path}')
        register(__file__)
        register(RUN / 'manifest.json')
        paths = {k: str(RUN / f'{k}.json') for k in ('normalization', 'temporal_stats')}
        stats = {k: json.loads(Path(p).read_text()) for k,p in paths.items()}
        for k, p in paths.items():
            require(register(p)['sha256'] == contract['stats'][k]['sha256'], 'statistics changed')
            pr = stats[k]['provenance']
            require(pr['fit_split'] == 'train' and pr['episodes'] == contract['episodes']['train']
                    and pr['episode_count'] == len(split['splits']['train'])
                    and pr['split_manifest_sha256'] == data_digest(split)
                    and pr['cache_config_id'] == contract['cache_config_id'], 'TRAIN stats provenance mismatch')
        cache_root, config_id = resolve_cache(split, ROOT / cfg['dataset']['feature_cache'])
        require(config_id == contract['cache_config_id'], 'encoder/cache provenance mismatch')
        require(register(cache_root / 'cache_config.json') == contract['cache_config'], 'cache config changed')
        frozen_cache = {x['path']: x for x in contract['cache_files']}
        identities = {x['episode']: x for x in contract['episodes']['val']}
        require(set(identities) == set(names), 'VAL membership differs from run')
        receipt_path = CKPT.with_name(CKPT.name + '.sha256.json')
        register(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        ck_identity = register(CKPT)
        require(ck_identity['sha256'] == receipt['sha256'] and ck_identity['size'] == receipt['size']
                and receipt['step'] == 5000 and receipt['model_only'] is True, 'checkpoint receipt mismatch')
        checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
        require(checkpoint['step'] == 5000 and checkpoint['model_only'] is True
                and checkpoint['contract'] == contract and checkpoint['contract_sha256'] == manifest['contract_sha256'],
                'not the fixed 5k checkpoint and exact run contract')
        wrapper, load_notes = build_wrapper(cfg, paths, stats, 'cpu', checkpoint=checkpoint)
        wrapper.eval()
        del checkpoint
        require(load_notes == {'strict_resume': True, 'step': 5000} and wrapper.vision_encoder is None, 'strict model-only load required')
        require(all(p.device.type == 'cpu' for p in wrapper.parameters()), 'non-CPU model')
        hook_counts = {'action_model_forward': 0, 'future_decoder_forward': 0}
        def model_check(module, args, kwargs):
            check()
            require(set(kwargs) == {'noisy_actions', 'qpos_history', 'dino_features_list', 'task_cond', 'memory'}, 'unexpected conditioning')
            require(bool((kwargs['task_cond'] == 0).all()), 'task condition must be zero')
            hook_counts['action_model_forward'] += 1
        wrapper.action_model.register_forward_pre_hook(model_check, with_kwargs=True)
        def forbid_future(*args):
            hook_counts['future_decoder_forward'] += 1
            raise RuntimeError('future decoder must not run during sampling')
        wrapper.action_model.future_feat_decoder.register_forward_pre_hook(forbid_future)
        predictions, targets, masks, commands, timings = [], [], [], [], []
        write_json('provenance.json', {'source_files': SOURCES, 'run_contract_sha256': manifest['contract_sha256'],
                   'cache_config_id': config_id, 'load_notes': load_notes, 'TRAIN_provenance': stats['normalization']['provenance']})
        for i, name in enumerate(names):
            check()
            tick = time.monotonic()
            feats, state, pool, target, valid = read_anchor(identities[name], split, cache_root, frozen_cache)
            require(not wrapper.normalize_state(state)[:, 1:].any() and not wrapper.normalize_memory(pool).any(), 'nonzero reset history differences')
            pred, info = wrapper.sample_actions([feats[:, :, layer] for layer in range(3)], state, pool,
                torch.zeros((1,1024), dtype=torch.float32), num_steps=10,
                generator=torch.Generator(device='cpu').manual_seed(SEED+i))
            require(pred.shape == (1,32,14) and pred.dtype == torch.float32 and torch.isfinite(pred).all(), 'bad prediction shape/dtype/values')
            hold = state[:, :1].repeat(1,32,1)
            row = {'episode': name, 'ordinal': i, 'anchor': 0, 'seed': SEED+i,
                   'gate': info['gate'].tolist(), 'metrics': {}, 'raw_prediction_gripper_bounds': {}}
            for label,h in WINDOWS.items():
                row['metrics'][label] = {kind: action_metrics(p[:, :h], target[:, :h], valid[:, :h], wrapper)
                                        for kind,p in [('model',pred),('hold',hold)]}
                row['raw_prediction_gripper_bounds'][label] = bounds(pred[:, :h][valid[:, :h]].numpy())
            row['elapsed_seconds'] = time.monotonic() - tick
            timings.append(row['elapsed_seconds'])
            predictions.append(pred.numpy()[0]); targets.append(target.numpy()[0]); masks.append(valid.numpy()[0]); commands.append(state.numpy()[0,0])
            ROWS.append(row)
            write_json('per_episode.json', ROWS)
            np.savez(OUT / 'predictions.npz', episode=np.array(names[:i+1]), prediction=np.stack(predictions),
                     target=np.stack(targets), valid=np.stack(masks), current_command=np.stack(commands),
                     seeds=np.arange(SEED,SEED+i+1,dtype=np.int64))
            if i == 0:
                write_json('first_anchor.json', {'passed': True, 'episode': name, 'shape': list(pred.shape),
                    'feature_shape': list(feats.shape), 'state_shape': list(state.shape), 'pool_shape': list(pool.shape),
                    'dtype': str(pred.dtype), 'elapsed_seconds': timings[0], 'resource_snapshot': runtime(),
                    'continuation_allowed': timings[0] < 120, 'metrics': row['metrics']})
                print(json.dumps({'event':'first_anchor', 'episode':name, 'seconds':timings[0], 'shape':list(pred.shape), 'metrics':row['metrics']}), flush=True)
                require(timings[0] < 120, 'first anchor exceeds 120 seconds; no continuation')
            elif (i+1) % 10 == 0:
                print(json.dumps({'event':'progress', 'episodes':i+1, 'elapsed_seconds':time.monotonic()-START}), flush=True)
        prediction, target, valid = np.stack(predictions), np.stack(targets), np.stack(masks)
        metrics, violations, comparisons = {}, {}, {}
        for label,h in WINDOWS.items():
            metrics[label] = {kind:summarize([r['metrics'][label][kind] for r in ROWS]) for kind in ('model','hold')}
            violations[label] = bounds(prediction[:, :h][valid[:, :h]])
            m,hb = metrics[label]['model']['normalized_mse'],metrics[label]['hold']['normalized_mse']
            comparisons[label] = {'normalized_mse_model_over_hold': m / hb, 'normalized_mse_relative_reduction': 1-m/hb,
                'episodes_model_better':sum(r['metrics'][label]['model']['normalized_mse'] < r['metrics'][label]['hold']['normalized_mse'] for r in ROWS),
                'episodes_tied':sum(r['metrics'][label]['model']['normalized_mse'] == r['metrics'][label]['hold']['normalized_mse'] for r in ROWS)}
        for path, before in SOURCES.items():
            require(file_identity(path) == {k:v for k,v in before.items() if k != 'sha256'}, f'input stat changed: {path}')
            if 'sha256' in before and path != str(CKPT):
                require(sha(path) == before['sha256'], f'input content changed: {path}')
        require(hook_counts == {'action_model_forward':500,'future_decoder_forward':0}, 'unexpected forward count')
        check()
        write_json('provenance.json', {'source_files': SOURCES, 'run_contract_sha256': manifest['contract_sha256'],
                   'cache_config_id': config_id, 'load_notes': load_notes, 'TRAIN_provenance': stats['normalization']['provenance'],
                   'verification': 'fresh SHA of checkpoint and code/config/stats/metadata; whole HDF SHA inherited from frozen run, not rehashed; fresh hashes of only read arrays; cache payload files stat/headers checked; no whole cache hash'})
        result = {'status':'complete', 'checkpoint_step':5000, 'episodes':len(ROWS), 'K':1,'Euler_steps':10,
                  'metrics':metrics, 'comparisons':comparisons, 'raw_prediction_gripper_bounds':violations,
                  'state_semantics':SEMANTICS, 'runtime':runtime(),
                  'per_episode_seconds':{'first':timings[0], 'min':min(timings), 'median':float(np.median(timings)), 'max':max(timings), 'sum':sum(timings)},
                  'forward_counts':hook_counts,
                  'read_counts':{'val_hdf':50, 'train_hdf':0, 'heldout_hdf':0, 'image_payloads':0,'future_feature_rows':0,'encoder_weight_loads':0,'encoder_forward':0},
                  'metric_semantics': 'exact trainer action_metrics CPU float32; row-weighted aggregation; per-group RMSE=sqrt(aggregated MSE); normalized TRAIN minmax MSE denominator=valid rows*14; no action output clipping',
                  'bound_semantics':'zero-based gripper cols6/13 compared strictly to [0,1]; exceedance=max(-x,x-1,0); not proof of physical failure; explicit controller bounds could be evaluated separately',
                  'boundaries':['Frame0 is recorded episode start, not a simulator reset rollout.', 'current_state is commanded, not measured.',
                                'No expert prehistory: all history clamped to frame0; no future visual/skill/phase inputs.',
                                'No closed-loop/physical success or visual causality claim.', 'No training, checkpoint selection, or change to established 30k budget.',
                                'One CPU noise sample per episode; not directly paired to existing CUDA val512 noise/anchors.'],
                  'artifacts':{p.name:{'path':str(p),'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}}
        write_json('result.json',result)
        print(json.dumps({'result':str(OUT/'result.json'),'metrics':metrics,'comparisons':comparisons,
                          'gripper_bounds':violations,'runtime':result['runtime']}),flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        write_json('failure.json', {'status':'stopped','reason':str(exc),'traceback':traceback.format_exc(),
                                   'completed_episodes':len(ROWS),'runtime':runtime(),'source_files':SOURCES,'read_ledger':READS})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    main()
