"""Explicit expert-only DAgger continuation over the frozen Precision V2 base."""
from copy import deepcopy
import math
from pathlib import Path
from harness import zhuhai_precision_v2_training as h

SCHEMA = 'precision-v2-dagger-additional20k-v1'
START = 30000
END = 50000
SAMPLER = dict(batch_size=4, original_per_batch=3, expert_per_batch=1,
               within_group='uniform real supervised anchor with replacement',
               expert_policy_prefix='causal context only, never positive action labels')
SCHEDULE = dict(additional_updates=20000, warmup_updates=100,
                peak_learning_rate=3e-6, minimum_learning_rate=1e-6,
                kind='new explicit cosine finetuning stage, original 30k schedule preserved')


def rate(completed):
    if type(completed) is not int or not START <= completed < END:
        raise ValueError('Continuation optimizer boundary must be 30k..50k')
    n = completed - START
    warmup = SCHEDULE['warmup_updates']
    lo = SCHEDULE['minimum_learning_rate']; hi = SCHEDULE['peak_learning_rate']
    if n < warmup:
        return lo + (hi-lo)*(n+1)/warmup
    return lo+(hi-lo)*(1+math.cos(math.pi*(n-warmup)/(20000-warmup)))/2


def validate_rows(base, manifest):
    if (manifest.get('status') != 'complete' or manifest.get('success_count') != 50
            or manifest.get('record_video') is not False or manifest.get('heldout_used') is not False):
        raise ValueError('Fifty completed verified corrections required before training')
    rows = manifest.get('train_rows', [])
    if len(rows) != 50:
        raise ValueError('Exactly fifty admitted expert episodes required')
    train = {r['family_id'] for r in base['splits']['zhuhai_task2']['train']}
    forbidden = {r['family_id'] for k in ('val', 'heldout') for r in base['splits']['zhuhai_task2'][k]}
    seen = [set(), set(), set()]
    for r in rows:
        if (r.get('split') != 'train' or r.get('domain') != 'dagger'
                or r['family_id'] not in train or r['family_id'] in forbidden
                or r['takeover_step'] != r['policy_steps'] or r['policy_steps'] <= 0
                or r['expert_steps'] <= 0):
            raise ValueError('Correction split, authority, or step binding mismatch')
        for keys, value in zip(seen, (str(Path(r['path']).resolve()), r['sha256'], r['seed'])):
            if value in keys:
                raise ValueError('Duplicate correction seed, path, or content')
            keys.add(value)
    return deepcopy(rows)


def contract(base, rows, *, parent, collection, sources):
    result = deepcopy(base)
    result['config']['training'].update(max_steps=END, schedule_steps=END,
                                      learning_rate=SCHEDULE['peak_learning_rate'],
                                      lr_min=SCHEDULE['minimum_learning_rate'])
    result['source_files'].update(sources)
    result['hjepa']['code_hashes'] = deepcopy(result['source_files'])
    result['dagger_training'] = dict(schema=SCHEMA, start_step=START, target_step=END,
        parent_full_state=parent, collection=collection, train_rows=deepcopy(rows),
        sampler=deepcopy(SAMPLER), schedule=deepcopy(SCHEDULE),
        original_train_episodes=160, correction_episodes=50,
        normalization='frozen original TRAIN statistics, VTT and IDM scale; no refit on policy actions',
        original_splits_unchanged=True, prefix_used_for_action_labels=False,
        model_optimizer_rng_restored=True, intermediate_physical_evaluations=False,
        original_30k_training_contract_sha256=h.digest(base))
    result['initialization_load'] = dict(initialization='strict_30k_full_state_with_AdamW_and_RNG',
                                       projections_reset=False, parent=parent)
    h.validate_training_contract(result)
    return result


def mixed_samples(original, corrections, rng):
    """All original TRAIN anchors remain eligible; only expert anchors supplement."""
    if len(original) <= 0 or len(corrections) <= 0:
        raise ValueError('Both original successes and expert corrections required')
    items = [dict(original[int(rng.integers(len(original)))]) for _ in range(3)]
    item = dict(corrections[int(rng.integers(len(corrections)))])
    if item.pop('supervision_source', None) != 'expert_correction':
        raise ValueError('Expected an expert-only correction sample')
    if set(item) != set(items[0]) or any(set(s) != set(item) for s in items):
        raise ValueError('Original/correction batch schemas differ')
    items.append(item)
    return items
