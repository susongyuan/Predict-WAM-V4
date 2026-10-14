#!/usr/bin/env python3
"""Paired, inference-only first-closure probes on all original TRAIN/VAL episodes."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import zhuhai_precision_v2_training as h
from scripts import gpu_lock


def array_digest(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check_checkpoint(path):
    import torch
    path = Path(path).resolve()
    receipt = h.read_json(path.with_name(path.name + '.sha256.json'))
    if receipt['size'] != path.stat().st_size or receipt['sha256'] != h.sha256_file(path):
        raise ValueError('Checkpoint receipt mismatch: ' + str(path))
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    contract = h.validate_training_contract(ck['contract'])
    h.verify_code_hashes(contract['source_files'])
    if (ck.get('model_only') is not True or receipt.get('model_only') is not True
            or ck['step'] != receipt['step'] or ck['contract_sha256'] != h.digest(contract)):
        raise ValueError('Model-only milestone required')
    return ck, receipt


def metrics(prediction, target, lo, hi):
    import numpy as np
    arm = [*range(7), *range(8, 15)]
    scale = np.where(hi - lo < 1e-6, 1, hi - lo)
    error = prediction - target
    result = {'windows': len(target), 'normalized_mse': float(np.mean((2 * error / scale) ** 2)),
              'arm_mae_rad': float(np.mean(np.abs(error[..., arm]))),
              'left_arm_mae_rad': float(np.mean(np.abs(error[..., :7]))),
              'right_arm_mae_rad': float(np.mean(np.abs(error[..., 8:15])))}
    for name, dim in [('left', 7), ('right', 15)]:
        p, t = prediction[..., dim], target[..., dim]
        true_close, pred_close = t < .2, p < .2
        true_open, pred_open = t > .9, p > .9
        result[name + '_gripper'] = {
            'mae_fraction': float(np.mean(np.abs(p - t))),
            'target_close_frames': int(true_close.sum()),
            'predicted_close_frames': int(pred_close.sum()),
            'close_recall': float((true_close & pred_close).sum() / true_close.sum()) if true_close.any() else None,
            'target_open_frames': int(true_open.sum()),
            'predicted_open_frames': int(pred_open.sum()),
            'missed_close_stays_open': int((true_close & pred_open).sum()),
            'false_close_on_open': int((true_open & pred_close).sum()),
            'predicted_mean_when_target_closed': float(p[true_close].mean()) if true_close.any() else None,
            'predicted_mean_when_target_open': float(p[true_open].mean()) if true_open.any() else None,
        }
    return result


def execute(args):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from harness import zhuhai_training as zt
    from harness.wam_behavior import load_upstream
    from harness.zhuhai_dataset import FrozenDinoProducer, FrameFeatureStore, HybridTaskDataset
    from harness.zhuhai_hybrid_cache import HybridFeatureCache
    from harness.zhuhai_action_smoothing import smooth_chunk
    from scripts.progress1_log import log

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def status(phase, **values):
        h.write_json(out / 'status.json', dict(status=phase, elapsed_seconds=time.monotonic() - started,
                                             optimizer_updates=0, **values))

    status('verifying_checkpoints')
    checkpoints = []
    base = None
    for path in args.checkpoint:
        ck, receipt = check_checkpoint(path)
        contract = ck['contract']
        if base is None:
            base = contract
        for key in ('splits', 'statistics', 'conditions', 'precision', 'idm_statistics'):
            if contract[key] != base[key]:
                raise ValueError('Comparison contract changed: ' + key)
        if (contract['config']['model'] != base['config']['model']
                or contract['hjepa']['cache_binding'] != base['hjepa']['cache_binding']
                or contract['action_postprocessing'] != base['action_postprocessing']):
            raise ValueError('Architecture/cache/postprocessing differs')
        checkpoints.append(dict(path=str(Path(path).resolve()), **receipt))
        del ck
    if len({c['step'] for c in checkpoints}) != len(checkpoints):
        raise ValueError('Duplicate checkpoints')
    status('loading_model_and_frozen_encoder')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.90)
    wrapper, condition, actual = h.load_policy(checkpoints[0]['path'], h.TASK, 'cuda')
    if actual != base:
        raise ValueError('Loaded base contract differs')
    encoder_path = h.resolve(base['config']['model']['vision_encoder']['checkpoint_path'])
    if h.sha256_file(encoder_path / 'model.safetensors') != base['cache']['encoder_sha256']:
        raise ValueError('Frozen encoder changed')
    _, runner = load_upstream()
    encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
        str(encoder_path), dtype=torch.bfloat16, device='cuda')
    if (hidden, registers, patch) != (1024, 4, 16):
        raise ValueError('Encoder geometry changed')
    cache = HybridFeatureCache(base['cache']['root'], budget_bytes=base['cache']['disk_budget_bytes'],
                               min_free_bytes=base['cache']['minimum_free_bytes'])
    disk = FrameFeatureStore(h.SourceBoundFeatureCache(cache, base['hjepa']['cache_binding']),
                             FrozenDinoProducer(zt.FrozenFeatureEncoder(encoder, device='cuda'), encoder),
                             encoder_sha256=base['cache']['encoder_sha256'])
    store = zt.BoundedRamFeatures(disk, budget_bytes=base['cache']['ram_budget_bytes'])
    v3 = base['config']['model']['v3']
    datasets, probes = {}, []
    status('admitting_original_train_val')
    for split in ('train', 'val'):
        rows = base['splits'][h.TASK][split]
        ds = h.PrecisionDataset(HybridTaskDataset(h.TASK, rows, store,
            proprio_lags=v3['proprio_lags'], memory_lags=v3['memory_lags'], condition=condition))
        datasets[split] = ds
        for i, ep in enumerate(ds.episodes):
            closed = np.flatnonzero(ep.action[:, 7] < .2)
            if not len(closed):
                raise ValueError('Original episode lacks first left closure')
            close = int(closed[0])
            for phase, anchor in [('reset', 0), ('before_close_16', close-16),
                                  ('before_close_4', close-4), ('after_close_4', close+4)]:
                if not 0 <= anchor <= len(ep) - 32:
                    raise ValueError('Incomplete probe action window')
                probes.append(dict(split=split, domain=ep.row['domain'], family_id=ep.row['family_id'],
                    source_path=str(ep.path), source_sha256=ep.sha256, first_left_close=close,
                    phase=phase, anchor=anchor, dataset_index=(ds.ends[i-1] if i else 0)+anchor))
    protocol = dict(schema='paired-firstgrasp-inference-probe-v21', checkpoints=checkpoints,
        original_episode_counts={s:len(d.episodes) for s,d in datasets.items()},
        probes=probes, noise='same batch order and fixed SHA-derived generator seed for every checkpoint',
        seed_prefix='firstgrasp-v21', integration_steps=10, batch_size=4,
        model_inputs=['current three-view frozen DINO', 'past measured states', 'past pooled features',
                      'causal clock', 'original TRAIN-only task condition'],
        no_future_or_target_inputs=True, heldout_unused=True, optimizer_updates=0,
        scope='all original TRAIN/VAL first left-closure windows; offline teacher-state probes, not full-task success or recovery rollout',
        execution_metrics='first8; raw native target plus original B-spline arms; no simulated physics/rate-limiter continuation',
        source_script_sha256=h.sha256_file(__file__), condition_sha256=array_digest(condition),
        statistics_sha256=h.digest(base['statistics']), cache_binding=base['hjepa']['cache_binding'])
    h.write_json(out/'plan.json', protocol, immutable=True)
    np.savez(out/'probe_labels_and_states.npz',
             labels=np.stack([datasets[p['split']].episodes[next(i for i,e in enumerate(datasets[p['split']].episodes)
                      if str(e.path)==p['source_path'])].action[p['anchor']:p['anchor']+32] for p in probes]))
    results = {}
    lo = wrapper.action_min.cpu().numpy()
    hi = wrapper.action_max.cpu().numpy()
    for cp in checkpoints:
        ck, receipt = check_checkpoint(cp['path'])
        wrapper.action_model.load_state_dict(ck['model_state_dict'], strict=True)
        del ck
        gc.collect()
        wrapper.eval()
        predictions, labels, states, clocks = [], [], [], []
        status('probing', checkpoint_step=cp['step'], completed_windows=0, total_windows=len(probes))
        for start in range(0, len(probes), 4):
            if gpu_lock.owner() != args.gpu_lock_owner:
                raise ValueError('Exclusive diagnostic GPU ownership lost')
            batch_probes = probes[start:start+4]
            batch = default_collate([datasets[p['split']][p['dataset_index']] for p in batch_probes])
            # Explicit whitelist: labels and fetched future32 features never enter inference.
            feat = batch['feats']
            layers = [feat[:, :, i] for i in range(feat.shape[2])]
            seed = int(hashlib.sha256(f'firstgrasp-v21:{start}'.encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device='cuda').manual_seed(seed)
            with torch.inference_mode():
                action, _ = wrapper.sample_actions(layers, batch['state'], batch['pool'], batch['task_cond'],
                    num_steps=10, generator=generator, event_state=batch['event_state'])
            predictions.extend(action.float().cpu().numpy())
            labels.extend(batch['action_sequence'].numpy())
            states.extend(batch['state'].numpy())
            clocks.extend(batch['event_state'].numpy())
            if start % 64 == 0 or start+4 >= len(probes):
                status('probing', checkpoint_step=cp['step'], completed_windows=min(start+4,len(probes)),
                       total_windows=len(probes), cache_stats=dict(cache.stats))
                print(json.dumps({'step':cp['step'],'windows':min(start+4,len(probes)),
                                  'total':len(probes),'elapsed':time.monotonic()-started}), flush=True)
        pred, target = np.asarray(predictions), np.asarray(labels)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite predictions')
        smoothed = np.stack([smooth_chunk(a,base['action_postprocessing']) for a in pred])
        np.savez(out/f'predictions_{cp["step"]:06d}.npz', model_raw=pred, spline_targets=smoothed,
                 expert_targets=target, measured_history=np.asarray(states), causal_clock=np.asarray(clocks))
        summaries = {}
        for split in ('train','val'):
            for domain in ('all','source','randomized'):
                for phase in ('all','reset','before_close_16','before_close_4','after_close_4'):
                    ix = [i for i,p in enumerate(probes) if p['split']==split and
                          (domain=='all' or p['domain']==domain) and (phase=='all' or p['phase']==phase)]
                    if ix:
                        summaries[f'{split}/{domain}/{phase}'] = {
                            'raw_first8':metrics(pred[ix,:8],target[ix,:8],lo,hi),
                            'spline_first8':metrics(smoothed[ix,:8],target[ix,:8],lo,hi),
                            'raw_full32':metrics(pred[ix],target[ix],lo,hi)}
        results[str(cp['step'])] = summaries
        h.write_json(out/f'metrics_{cp["step"]:06d}.json', summaries, immutable=True)
        log(f'首抓只读诊断{cp["step"]}完成{len(probes)}固定窗口，原160 TRAIN/20 VAL全回合、四阶段、同噪声/10 Euler；仅离线条件动作误差，未更新参数/评分/进行新物理测试。','firstgrasp_diagnostic')
    h.write_json(out/'summary.json', dict(protocol=protocol, results=results,
        cache_stats=dict(cache.stats), ram_cache_stats=dict(store.stats),
        completed=True, elapsed_seconds=time.monotonic()-started), immutable=True)
    status('completed', checkpoint_steps=[c['step'] for c in checkpoints], windows_per_checkpoint=len(probes))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu-lock-owner', required=True)
    args = parser.parse_args()
    gpu_lock.acquire(args.gpu_lock_owner, max_wait=0)
    try:
        return execute(args)
    except BaseException as exc:
        if args.out.exists():
            h.write_json(args.out/'status.json', dict(status='failed',error=f'{type(exc).__name__}: {exc}',
                optimizer_updates=0,evidence_preserved=True))
        raise
    finally:
        gpu_lock.release(args.gpu_lock_owner)


if __name__ == '__main__':
    raise SystemExit(main())
