"""Fifty genuinely new DR corrections plus immutable seventy, strict 50k->80k."""
from copy import deepcopy
import math
from pathlib import Path
from harness import zhuhai_dagger_segmented_training_v7 as prior
from harness import zhuhai_precision_v2_training as h
SCHEMA='precision-v2-hundred-segmented-twenty-legacy-additional30k-v14'
LEGACY_EPISODES,NEW_EPISODES,TOTAL_EPISODES=20,100,120
NEWLY_ADDED_EPISODES=50
START,END=50000,80000
SAMPLER=deepcopy(prior.SAMPLER)
SCHEDULE=dict(additional_updates=30000,warmup_updates=100,peak_learning_rate=3e-6,
    minimum_learning_rate=1e-6,kind='explicit additional30k finetuning stage; prior schedules preserved')
correction_counts=prior.correction_counts
mixed_samples=prior.mixed_samples


def rate(completed):
    if type(completed) is not int or not START<=completed<END:
        raise ValueError('Continuation update boundary must be 50k..80k')
    n=completed-START;warm=SCHEDULE['warmup_updates'];lo=SCHEDULE['minimum_learning_rate'];hi=SCHEDULE['peak_learning_rate']
    if n<warm:return lo+(hi-lo)*(n+1)/warm
    return lo+(hi-lo)*(1+math.cos(math.pi*(n-warm)/(30000-warm)))/2


def validate_quota(manifest):
    counts=correction_counts(manifest.get('train_rows',[]))
    if (manifest.get('status')!='complete' or manifest.get('success_count')!=NEW_EPISODES
        or manifest.get('requested_success_count')!=NEW_EPISODES
        or counts!=dict(legacy=20,segmented=100,total=120)
        or manifest.get('legacy_corrections')!=20 or manifest.get('segmented_corrections')!=100
        or manifest.get('total_training_corrections')!=120
        or manifest.get('newly_added_segmented_episodes')!=50
        or manifest.get('quota_unit')!='accepted new segmented episodes'
        or manifest.get('record_video') is not False or manifest.get('heldout_used') is not False):
        raise ValueError('Exactly 70 immutable corrections plus 50 genuinely new DR episodes required')
    return counts


def validate_rows(base,manifest):
    validate_quota(manifest);binding=manifest['retained_collection']
    if h.sha256_file(binding['path'])!=binding['sha256']:raise ValueError('Retained seventy manifest changed')
    old=h.read_json(binding['path']);retained=prior.validate_rows(base,old);rows=manifest['train_rows']
    if rows[:70]!=retained:raise ValueError('All prior seventy correction rows must remain unchanged and ordered')
    extra=rows[70:]
    if len(extra)!=50 or any(r['correction_variant']!='segmented_local_interventions' for r in extra):
        raise ValueError('New fifty must be segmented recovery episodes')
    for key in ('seed','sha256','path'):
        values=[r[key] for r in rows]
        if len(set(values))!=120:raise ValueError('Duplicate correction '+key)
    for row in extra:
        raw=h.read_json(row['raw_result_path'])
        if (h.sha256_file(row['raw_result_path'])!=row['raw_result_sha256']
            or raw['scenario']['randomized'] is not True
            or raw['scene_manifest']['randomized'] is not True
            or raw['checkpoint_sha256']!=row['collection_policy_checkpoint_sha256']
            or raw['policy_ready']['runtime']['checkpoint_sha256']!=row['collection_policy_checkpoint_sha256']
            or raw['policy_ready']['runtime']['checkpoint_step']!=50000
            or raw['data_path']!=row['path'] or raw['data_sha256']!=row['sha256']):
            raise ValueError('New recovery must bind actual50k policy and actual DR scene evidence')
    legacy=[r for r in retained if r['correction_variant']=='legacy_full_continuation']
    subset=dict(old,train_rows=legacy+extra)
    prior.validate_rows(base,subset)
    return deepcopy(rows)


def contract(base,rows,*,parent,collection,sources):
    if parent.get('step')!=START or parent.get('model_only') is not False:
        raise ValueError('Exact50k full model/AdamW/RNG parent required')
    if correction_counts(rows)!=dict(legacy=20,segmented=100,total=120):raise ValueError('Correction quota differs')
    result=deepcopy(base);preceding=h.digest(base);original30=base['dagger_training']['original_30k_training_contract_sha256']
    result['config']['training'].update(max_steps=END,schedule_steps=END,learning_rate=3e-6,lr_min=1e-6)
    result['source_files'].update(sources);result['hjepa']['code_hashes']=deepcopy(result['source_files'])
    result['dagger_training']=dict(schema=SCHEMA,start_step=START,target_step=END,parent_full_state=parent,
        collection=collection,train_rows=deepcopy(rows),sampler=deepcopy(SAMPLER),schedule=deepcopy(SCHEDULE),
        original_train_episodes=160,correction_episodes=120,legacy_correction_episodes=20,new_segmented_episodes=100,
        newly_added_segmented_episodes=50,retained_correction_episodes=70,total_train_episodes=280,
        original_30k_training_contract_sha256=original30,preceding_50k_training_contract_sha256=preceding,
        normalization='frozen original TRAIN statistics, VTT and IDM scale',original_splits_unchanged=True,
        prefix_used_for_action_labels=False,model_optimizer_rng_restored=True,intermediate_physical_evaluations=False,
        authority_boundary_masks_actions_future_and_idm=True,whole_task_outcome_is_not_local_success=True)
    result['initialization_load']=dict(initialization='strict_50k_full_state_with_AdamW_and_RNG',projections_reset=False,parent=parent)
    h.validate_training_contract(result);return result
