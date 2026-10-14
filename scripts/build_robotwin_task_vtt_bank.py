#!/usr/bin/env python3
"""Task-scoped TRAIN-only endpoint VTTs; closure nominations remain advisory."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from harness.robotwin_policy_worker import atomic_json, sha256
from harness.robotwin_training_data import load_manifest, resolve_cache, episode_identity, validate_payload, episode_directory


def build(split_path, cache_path, task, out):
    split_path, cache_path, out = Path(split_path).resolve(), Path(cache_path).resolve(), Path(out).resolve()
    split = load_manifest(split_path)
    cache_root, cache_id = resolve_cache(split, cache_path)
    cache_config = json.loads((cache_root / 'cache_config.json').read_text())
    if cache_config['source']['task'] != task:
        raise ValueError('task does not match canonical feature cache')
    names = split['splits']['train']
    if len(names) < 5 or len(set(names)) != len(names):
        raise ValueError('at least five unique TRAIN episodes required')
    paths = {}
    for name in names:
        identity = episode_identity(Path(split['data_root']) / name, cache_id)
        validate_payload(cache_root, identity, load_features=False)
        paths[name] = episode_directory(cache_root, identity) / 'feat.npy'
    if set(paths) != set(names):
        raise ValueError('complete TRAIN feature cache required')
    transitions, starts, records = [], [], []
    for name in names:
        array = np.load(paths[name], mmap_mode='r', allow_pickle=False)
        if array.dtype != np.uint16 or array.shape[1:] != (3, 3, 201, 1024) or len(array) < 2:
            raise ValueError('invalid DINO feature geometry')
        indices = [round((len(array)-1)*fraction) for fraction in (0, .25, .5, .75, 1)]
        bits = np.array(array[indices, 0, -1, 0], copy=True).view(np.int16)
        cls = torch.from_numpy(bits).view(torch.bfloat16).float().numpy()
        if not np.isfinite(cls).all():
            raise ValueError('nonfinite TRAIN CLS')
        transitions.append(cls-cls[0]); starts.append(cls[0])
        records.append(dict(episode=name, frames=indices, length=len(array)))
    deltas = np.stack(transitions)
    fit_count = min(len(names)-1, max(1, int(.8*len(names))))
    prototype = deltas[:fit_count, -1].mean(0)
    norm = float(np.linalg.norm(prototype))
    if norm < 1e-3:
        raise ValueError('degenerate VTT prototype')
    unit = prototype / norm
    progress = deltas @ unit / norm
    alignment = (deltas @ unit) / np.maximum(np.linalg.norm(deltas, axis=-1), 1e-8)
    pt = float(np.quantile(progress[:fit_count, -1], .1))
    at = float(np.quantile(alignment[:fit_count, -1], .1))
    candidates = (progress >= pt) & (alignment >= at)
    for i, row in enumerate(records):
        row.update(partition='inner_fit' if i < fit_count else 'inner_check',
                   progress=progress[i].tolist(), alignment=alignment[i].tolist(), candidate=candidates[i].tolist())
    out.mkdir(parents=True, exist_ok=False)
    np.save(out/'prototypes.npy', prototype[None].astype(np.float32))
    np.save(out/'action_vtt_prototype.npy', deltas[:, -1].mean(0).astype(np.float32))
    np.save(out/'vtt_instances.npy', deltas[:, -1].astype(np.float32))
    np.save(out/'context_cls.npy', np.stack(starts).astype(np.float32))
    atomic_json(out/'meta.json', dict(format='task-vtt-bank-v2', classes={task: 0},
                representation='DINOv3 hidden_states[-4] head CLS endpoint minus frame0', source_split='train'))
    result = dict(format='task-vtt-bank-v2', task=task, completed=True, TRAIN_episodes=len(names),
                  inner_fit=fit_count, inner_check=len(names)-fit_count, prototype_norm=norm,
                  done_progress=pt, done_alignment=at, min_steps=30, heldout_used=False,
                  whole_task_monitor_admitted=False, physical_success_verifier=False,
                  split_sha256=sha256(split_path), records=records,
                  source_split='train', cache_id=cache_id,
                  cache_config_sha256=sha256(cache_root / 'cache_config.json'),
                  encoder_sha256=cache_config['encoder_sha256'],
                  preprocess_sha256=cache_config['preprocess_sha256'],
                  TRAIN_episode_names=names,
                  inner_check_candidate_counts_by_fraction=candidates[fit_count:].sum(0).tolist())
    atomic_json(out/'result.json', result)
    return {k: v for k, v in result.items() if k != 'records'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(build(args.split_manifest, args.cache, args.task, args.out), ensure_ascii=False))
