"""Mixed immutable old continuations and newly verified intervention segments."""
from copy import deepcopy
from pathlib import Path
from harness import zhuhai_dagger_training as original
from harness.zhuhai_dagger_segmented import SegmentedEpisode

SCHEMA = 'precision-v2-segmented-dagger-additional20k-v5'
START, END, SCHEDULE, rate = original.START, original.END, original.SCHEDULE, original.rate
SAMPLER = dict(batch_size=4, original_per_batch=3, expert_per_batch=1,
    within_group='original uniform anchors; corrections uniform verified intervention span then uniform anchor',
    expert_policy_prefix='all real mixed-authority history is context only; only locally verified expert actions are labels')


def validate_rows(base, manifest):
    if (manifest.get('status') != 'complete' or manifest.get('success_count') != 50
            or manifest.get('record_video') is not False or manifest.get('heldout_used') is not False):
        raise ValueError('Exactly fifty audited correction episodes required')
    rows = manifest.get('train_rows', [])
    if len(rows) != 50:
        raise ValueError('Count episodes separately from intervention segments')
    train = {r['family_id'] for r in base['splits']['zhuhai_task2']['train']}
    forbidden = {r['family_id'] for split in ('val', 'heldout') for r in base['splits']['zhuhai_task2'][split]}
    seen = [set(), set(), set()]
    for row in rows:
        if (row.get('split') != 'train' or row.get('domain') != 'dagger'
                or row['family_id'] not in train or row['family_id'] in forbidden
                or row.get('policy_steps', 0) <= 0 or row.get('expert_steps', 0) <= 0):
            raise ValueError('Correction split or control counts invalid')
        for keys, value in zip(seen, (str(Path(row['path']).resolve()), row['sha256'], row['seed'])):
            if value in keys:
                raise ValueError('Duplicate correction path/hash/seed')
            keys.add(value)
        episode = SegmentedEpisode(row)
        if episode.supervised_count != row['expert_steps'] or episode.policy_count != row['policy_steps']:
            raise ValueError('Supervised and real policy counts differ from archive')
        if episode.variant == 'segmented_local_interventions':
            if row.get('handback_count', 0) < 1 or row.get('verified_segment_count') != len(episode.valid_segments):
                raise ValueError('Actual model handback and local goal counts required')
        elif row['takeover_step'] != row['policy_steps']:
            raise ValueError('Legacy continuation control boundary differs')
    return deepcopy(rows)


def contract(base, rows, *, parent, collection, sources):
    result = original.contract(base, rows, parent=parent, collection=collection, sources=sources)
    result['dagger_training'].update(schema=SCHEMA, sampler=deepcopy(SAMPLER),
        correction_protocol='immutable legacy full-task continuations plus locally verified intervention spans',
        whole_task_outcome_is_not_local_success=True, authority_boundary_masks_actions_future_and_idm=True)
    return result


def mixed_samples(original_dataset, corrections, rng):
    items = [dict(original_dataset[int(rng.integers(len(original_dataset)))]) for _ in range(3)]
    item = dict(corrections.sample(rng))
    if item.pop('supervision_source', None) != 'expert_correction':
        raise ValueError('Expert-only sampled correction required')
    if set(item) != set(items[0]) or any(set(s) != set(item) for s in items):
        raise ValueError('Batch schema differs')
    return items+[item]
