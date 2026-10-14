#!/usr/bin/env python3
"""CPU-only WAM design/data-budget audit; never fits statistics or launches training.

Reads formal success metadata and, optionally, TRAIN low-dimensional arrays only.
This is a dated diagnostic, NOT data admission or a replacement for full HDF hashes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/zhuhai2026_tasks12'
SOURCE = BASE / 'production_20261009_v1'
TASK2_RUN = BASE / 'production_task2_collection_only_20261009_v11'
PRIOR = {
    'zhuhai_task1': BASE / 'verification/tasks12_collection_repair/task1_prior_success_inventory.json',
    'zhuhai_task2': BASE / 'production_task2_final10_20k_20261009_v8/verification/expert_repair_v1/prior_success_inventory.json',
}
FRAME_CACHE_BYTES = 3 * 3 * 201 * 1024 * 2 + 3 * 2 * 1024 * 4
ARM = list(range(7)) + list(range(8, 15))
FIELDS = ('left_arm', 'left_gripper', 'right_arm', 'right_gripper')


def valid_endpoint_anchors(action_count, horizon):
    """N actions and N+1 real observations, including the saved terminal image."""
    if type(action_count) is not int or action_count < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError('nonnegative action count and positive integer horizon required')
    return max(0, action_count - horizon + 1)


def label_residuals(state, action, horizon=4):
    """u[t+k] - q[t], k=0..h-1. q is measured; u[t]=target_command[t+1]."""
    q, u = np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)
    if q.ndim != 2 or q.shape != u.shape or q.shape[1] != 16:
        raise ValueError('aligned native16 measured state / target action required')
    if not np.isfinite(q).all() or not np.isfinite(u).all():
        raise ValueError('nonfinite low-dimensional input')
    n = valid_endpoint_anchors(len(q), horizon)
    if not n:
        return np.empty((0, horizon, 16), dtype=np.float64)
    return np.stack([u[k:k+n] - q[:n] for k in range(horizon)], axis=1)


def family_split(rows, seed=20261008):
    families = sorted({str(r['family_id']) for r in rows if r['domain'] == 'source'})
    if len(families) != 100:
        raise ValueError('exactly 100 distinct source families are required')
    order = np.random.default_rng(seed).permutation(families).tolist()
    labels = {f: ('train' if i < 80 else 'val' if i < 90 else 'heldout') for i, f in enumerate(order)}
    seen, paths, hashes = set(), set(), set()
    out = {s: [] for s in ('train', 'val', 'heldout')}
    for row in rows:
        family, domain = str(row['family_id']), row['domain']
        if family not in labels or domain not in ('source', 'randomized'):
            raise ValueError('DR must belong to an admitted source family')
        key = family, domain
        if key in seen or row['path'] in paths or row['data_sha256'] in hashes:
            raise ValueError('duplicate family/domain, source path or payload hash')
        seen.add(key); paths.add(row['path']); hashes.add(row['data_sha256'])
        out[labels[family]].append(dict(row, split=labels[family]))
    return out


def coverage(task_windows, steps, batch=4):
    """Expected unique anchors for uniform task, then uniform within-task replacement."""
    if not task_windows or any(n <= 0 for n in task_windows.values()) or min(steps, batch) < 1:
        raise ValueError('positive task sizes, updates and batch required')
    k, draws = len(task_windows), steps * batch
    fractions = {t: -math.expm1(draws * math.log1p(-1.0/(k*n))) if k*n > 1 else 1.0
                 for t, n in task_windows.items()}
    total = sum(task_windows.values())
    return {'optimizer_updates': steps, 'batch_size': batch, 'sample_draws': draws,
            'equivalent_global_passes_not_epochs': draws / total,
            'expected_unique_anchor_fraction': sum(task_windows[t] * fractions[t] for t in task_windows) / total,
            'expected_unique_anchor_fraction_by_task': fractions}


def tail_eval_steps(total, interval=5000, tail=10000):
    """Checkpoints at entry to the last 10k, its midpoint and the end; no execution."""
    if any(type(v) is not int or v <= 0 for v in (total, interval, tail)) or total < tail or tail % interval:
        raise ValueError('valid fixed final-window schedule required')
    return [s for s in range(total-tail, total+1, interval) if s > 0]


def snapshot(path, receipts):
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    receipts.append({'path': str(path), 'sha256_at_read': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    return doc


def success_rows(doc, task, domain):
    contract = doc.get('contract', {})
    if contract.get('source_probe') or contract.get('allow_unvalidated_probe'):
        raise ValueError('probe data cannot enter the formal-budget audit')
    if contract and (contract.get('task') != task or contract.get('mode') != domain):
        raise ValueError('unexpected task/domain contract')
    rows = []
    for a in doc.get('attempts', []):
        if a.get('status') != 'success':
            continue
        for key in ('strict_predicate', 'plan_success', 'expert_receipt_valid', 'physics_safety_all_observations_passed'):
            if a.get(key) is not True:
                raise ValueError('success lacks strict evidence flag: ' + key)
        if a.get('domain') != domain or a.get('state_dim') != 16 or a.get('action_dim') != 16:
            raise ValueError('unexpected success schema')
        if a.get('state_semantics') != 'measured_joint_gripper' or a.get('action_semantics') != 'joint_position_target':
            raise ValueError('unexpected native16 semantics')
        n = a.get('observation_count')
        if type(n) is not int or n < 2:
            raise ValueError('real terminal observation count required')
        file = Path(a['path'])
        if file.is_symlink() or not file.is_file() or file.name.endswith('.partial'):
            raise ValueError('not a complete regular source HDF')
        rows.append({k: a[k] for k in ('family_id', 'domain', 'path', 'data_sha256')} | {'action_count': n-1})
    declared = doc.get('success_count', doc.get('successful_episodes'))
    if type(declared) is not int or declared != len(rows):
        raise ValueError('declared success count differs from formal successful records')
    return rows


def train_lowdim(rows):
    """Diagnostic only: never read VAL/heldout arrays or save reusable norm statistics."""
    import h5py
    count = stationary = intervals = 0
    sums, squares = np.zeros(16), np.zeros(16)
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('TRAIN-only diagnostic')
        with h5py.File(row['path'], 'r') as h:
            q = h['measured_state'][:-1].astype(np.float64)
            u = h['target_command'][1:].astype(np.float64)
            if q.shape != (row['action_count'], 16) or u.shape != q.shape:
                raise ValueError('manifest/native16 audit stream length mismatch')
            times = h['physics_time'][:]
            if len(times) != len(q)+1 or not np.allclose(np.diff(times), .1, rtol=0, atol=1e-7):
                raise ValueError('not actual uniform 10Hz data')
        y = label_residuals(q, u).reshape(-1, 16)
        count += len(y); sums += y.sum(0); squares += np.square(y).sum(0)
        delta = np.max(np.abs(np.diff(u[:, ARM], axis=0)), axis=1)
        stationary += int(np.sum(delta <= 1e-5)); intervals += len(delta)
    return {'scope': 'diagnostic only; not fitted policy normalization or q-only learned baseline',
            'train_episodes_opened': len(rows), 'val_or_heldout_arrays_opened': 0,
            'four_step_label_vectors': count,
            'raw_residual_mean_16d': (sums / count).tolist() if count else None,
            'zero_residual_baseline_raw_rmse_16d': np.sqrt(squares / count).tolist() if count else None,
            'unchanged_arm_target_interval_fraction': stationary / intervals if intervals else None,
            'unchanged_definition': 'max absolute change over 14 commanded arm joints <= 1e-5 rad; excludes grippers; not a physical hold award'}


def run_audit(include_lowdim=False):
    receipts, tasks, projected = [], {}, {}
    for task in ('zhuhai_task1', 'zhuhai_task2'):
        rows = success_rows(snapshot(SOURCE / task / 'source/manifest.json', receipts), task, 'source')
        prior = snapshot(PRIOR[task], receipts)
        for component in prior['components']:
            raw = Path(component['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != component['sha256']:
                raise ValueError('preserved prior component changed')
        rows += success_rows(prior, task, 'randomized')
        if task == 'zhuhai_task2':
            # Only this explicitly active formal run, never probes or guessed newer runs.
            for shard in sorted((TASK2_RUN / task / 'randomized/shards').glob('shard_*/manifest.json')):
                rows += success_rows(snapshot(shard, receipts), task, 'randomized')
        split = family_split(rows)
        current = Counter(r['domain'] for r in rows)
        if current['source'] != 100 or current['randomized'] > 100:
            raise ValueError('formal 100+100 quota violated')
        train = split['train']; dr_train = [r for r in train if r['domain'] == 'randomized']
        source_train = [r for r in train if r['domain'] == 'source']
        missing = 80 - len(dr_train)
        mean_dr = float(np.mean([r['action_count'] for r in dr_train or source_train]))
        observed = sum(r['action_count'] for r in train)
        projected[task] = observed + missing * mean_dr
        tasks[task] = {'formal_source': current['source'], 'formal_DR': current['randomized'],
            'remaining_DR': 100-current['randomized'], 'source_families': 100,
            'splits': {s: {'episodes': len(v), 'source': sum(r['domain']=='source' for r in v),
                           'DR': sum(r['domain']=='randomized' for r in v),
                           'action_anchors': sum(r['action_count'] for r in v)} for s,v in split.items()},
            'TRAIN_mean_episode_actions': observed/len(train),
            'TRAIN_DR_mean_episode_actions': mean_dr,
            'TRAIN_valid_future_anchors': {str(h): sum(valid_endpoint_anchors(r['action_count'],h) for r in train) for h in (4,8,32)},
            'projected_full_TRAIN_anchors': projected[task],
            'projection_method': 'observed TRAIN source+DR anchors + missing TRAIN families * observed TRAIN DR mean; estimate, not a guarantee',
            'TRAIN_episode_references': train}
        if include_lowdim:
            tasks[task]['lowdim_diagnostic'] = train_lowdim(train)
    total = sum(projected.values())
    return {'schema': 'wam_hjepa_cpu_design_budget_audit_v1',
        'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'scope': 'read-only formal metadata and optional TRAIN lowdim; no admission, optimizer, encoding, GPU, policy evaluation or video',
        'frozen_model_modified': False, 'training_started': False, 'current_run': str(TASK2_RUN),
        'source_snapshots': receipts, 'split_seed': 20261008, 'split_by_source_family': [80,10,10],
        'tasks': tasks, 'projected_full_TRAIN_anchors': total,
        'nominal_N_div_B4_updates_not_a_true_epoch': math.ceil(total/4),
        'coverage_if_full_data_uniform_task_then_anchor_with_replacement': [coverage(projected,s) for s in (20000,50000,100000,200000)],
        'cache': {'bytes_per_frame': FRAME_CACHE_BYTES, 'projected_TRAIN_dense_bytes_lower_bound': math.ceil(total*FRAME_CACHE_BYTES),
                  'hard_cap_bytes_unchanged': 10**12, 'excludes': 'terminal frames, VAL and metadata; frame caches shared by 4/8/32 horizons, never adapted trainable features'},
        'schedule_design_only': {'last_10000_every_5000': 'include T-10000, T-5000, T; same predeclared per-task dev seeds; heldout separate once',
                                 'example_T_100000': tail_eval_steps(100000), 'total_updates_not_selected_or_authorized': True}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--train-lowdim', action='store_true')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve previous audit; choose a fresh dated output')
    result = run_audit(args.train_lowdim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','source_snapshots')}, ensure_ascii=False, indent=2))
    print(json.dumps({t:{k:v for k,v in d.items() if k not in ('TRAIN_episode_references','lowdim_diagnostic')} for t,d in result['tasks'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
