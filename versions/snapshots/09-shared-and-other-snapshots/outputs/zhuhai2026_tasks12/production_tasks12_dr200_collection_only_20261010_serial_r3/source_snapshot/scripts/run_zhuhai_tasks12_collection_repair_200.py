#!/usr/bin/env python3
"""Independent collection-only task1 DR200 scheduler; task2 completed data imported.

Default is CPU planning. Explicit --execute requires the same exclusive GPU
owner and all four unchanged task1-v2 physical probes. No tiny/train/eval.
Original 48 DR, source pools, probe suites, score predicates and failed guards
remain immutable. Frozen scheduling code is reused in a private namespace.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_zhuhai_tasks12_collection_repair as frozen
from scripts import collect_isaac_zhuhai_tasks12_repair_200 as collector200

_p = collector200.isolate_protocol(frozen, __name__ + '._scheduling', __file__)
TASK1_SEED_START = 990000
DISABLED = dict(frozen.DISABLED)
for _key in ('GIB', 'RESOURCE_POLICY', 'WORKER_RECYCLING', 'RECYCLE_EXIT_CODE', 'MINIMUM_FREE_BYTES',
             'SECOND_START_DELAY_SECONDS', 'RECEIPT_FIELDS'):
    globals()[_key] = getattr(frozen, _key)


def parser():
    result = frozen.parser()
    result.description = __doc__
    for action in result._actions:
        if action.dest == 'randomized_count':
            action.choices, action.default = (200,), 200
        if action.dest == 'task1_collector':
            action.default = Path(collector200.__file__).resolve()
    return result


def validate_args(args):
    if (type(args.collection_workers) is not int or args.collection_workers not in (1, 2, 3, 4) or
            type(args.source_count) is not int or args.source_count != 100 or
            type(args.randomized_count) is not int or args.randomized_count != 200 or
            args.life_mapping not in ('body_text', 'scoring_table')):
        raise ValueError('Fixed task1 100-source/200-DR, task2 imported100/100, 1..4 collection workers')


def selected_collector(args):
    path = frozen.regular_path(args.task1_collector)
    if path != Path(collector200.__file__).resolve():
        raise ValueError('DR200 scheduler requires its independent hash-bound DR200 collector')
    for name, expected in (('PRODUCTION_QUOTAS', dict(source=100, randomized=200)),
            ('RESOURCE_POLICY', RESOURCE_POLICY), ('WORKER_RECYCLING', WORKER_RECYCLING),
            ('RECYCLE_EXIT_CODE', RECYCLE_EXIT_CODE), ('MINIMUM_FREE_BYTES', MINIMUM_FREE_BYTES)):
        if getattr(collector200, name) != expected:
            raise ValueError('Independent collector changed quota or original resource policy')
    return collector200


def accepted_task1_expert(path, collector=None):
    selected = collector or collector200
    value = frozen.read_json(path)
    bound = value.get('collector')
    if bound == frozen.reference(selected.__file__):
        proof_collector = selected
    elif bound == frozen.reference(collector200.frozen.__file__):
        proof_collector = collector200.frozen
    else:
        raise ValueError('Probe acceptance must bind frozen-v2 or independent-DR200 collector')
    result = frozen.accepted_task1_expert(path, proof_collector)
    if set(result['seeds']) != set(frozen.V2_PROBE_CASES) or result['probe_success_count'] != 4:
        raise ValueError('All four unchanged v2 full physical probe cases required')
    result.update(probe_collector=result['collector'], collector=frozen.reference(selected.__file__),
                  production_contract_version=collector200.CONTRACT_VERSION,
                  original_probe_acceptance_modified=False)
    return result


def family_integrity(source_rows, randomized_rows, *, complete=True, randomized_per_family=1):
    if randomized_per_family == 1:
        return frozen.family_integrity(source_rows, randomized_rows, complete=complete)
    if randomized_per_family != 2:
        raise ValueError('Only registered two-DR family protocol admitted')
    parents = {r['family_id']: r for r in source_rows}
    if len(source_rows) != 100 or len(parents) != 100:
        raise ValueError('100 distinct original source families required')
    frozen._unique_rows([*source_rows, *randomized_rows])
    counts = Counter()
    index = []
    for family, parent in sorted(parents.items()):
        label = parent.get('split', parent.get('metadata', {}).get('split'))
        if label not in (None, 'train', 'val', 'heldout') or parent.get('metadata', {}).get('split', label) != label:
            raise ValueError('Original source split label invalid')
        index.append(dict(family_id=family, source_seed=parent['seed'], split=label))
    split_counts = Counter(r['split'] for r in index)
    if None not in split_counts and split_counts != Counter(train=80, val=10, heldout=10):
        raise ValueError('Original materialized family split must remain80/10/10')
    labels = {r['family_id']: r['split'] for r in index}
    for row in randomized_rows:
        parent = parents.get(row.get('family_id')); md = row.get('metadata', {})
        if (parent is None or row.get('source_parent') != parent['seed'] or
                md.get('parent_source_seed') != parent['seed'] or
                row.get('split', md.get('split', labels[row['family_id']])) != labels[row['family_id']] or
                md.get('split', labels[row['family_id']]) != labels[row['family_id']]):
            raise ValueError('DR family/source/split provenance changed')
        counts[row['family_id']] += 1
    if any(n > 2 for n in counts.values()) or (complete and counts != Counter({f: 2 for f in parents})):
        raise ValueError('Exactly two DR per original source family required')
    return dict(family_index=index, family_index_sha256=frozen.json_digest(index),
        split_unit='source family; all relatives inherit the source split', randomized_per_family=2,
        source_split_counts={('unassigned' if key is None else key): value for key,value in split_counts.items()},
        dr_split_counts={('unassigned' if key is None else key): value for key,value in Counter(labels[r['family_id']] for r in randomized_rows).items()},
        planned_training_split=dict(source_families=dict(train=80,val=10,heldout=10),seed=20261008,
            method='harness.zhuhai_dataset.grouped_split NumPy default_rng permutation',
            successful_episodes_with_two_DR=dict(train=240,val=30,heldout=30)),
        training_split_materialized=False)


def task1_prior_import(args, collector=None):
    collector = collector or selected_collector(args)
    source = args.source_root.resolve()/'zhuhai_task1/source/manifest.json'
    original = collector.validate_manifest(source, task='zhuhai_task1', mode='source', verify_files=True)
    previous = collector.prior_rows(args.task1_prior_inventory, verify_files=True)
    if len(previous) != 48:
        raise ValueError('Exactly48 original physical DR retained')
    family = family_integrity(original, previous, complete=False, randomized_per_family=2)
    c = frozen.read_json(source)['contract']
    if c.get('life_mapping') != args.life_mapping:
        raise ValueError('Original source life mapping changed')
    binding = {key: c.get(key) for key in collector.SHARED_CONTRACT_FIELDS}
    binding.update(source_manifest=dict(path=str(source), sha256=frozen.sha(source), engine=c['engine']),
        prior_inventory=frozen.reference(args.task1_prior_inventory),
        collection_sharding=collector.sharding(args.collection_workers))
    remaining = [collector.residual_families(binding, i, args.collection_workers)
                 for i in range(args.collection_workers)]
    counts = Counter(r['family_id'] for r in previous)
    expected = [r for r in sorted(original, key=lambda r: r['seed']) for _ in range(2 - counts[r['family_id']])]
    flat = [r for rows in remaining for r in rows]
    if (len(flat) != 152 or Counter(r['family_id'] for r in flat) != Counter(r['family_id'] for r in expected)):
        raise ValueError('Preserved48 requires exactly152 remaining physical DR slots')
    for index, rows in enumerate(remaining):
        assigned = [r for j, r in enumerate(sorted(original, key=lambda r: r['seed']))
                    if j % args.collection_workers == index for _ in range(2 - counts[r['family_id']])]
        if rows != assigned:
            raise ValueError('Residual family shard assignment changed')
    return dict(**frozen.reference(args.task1_prior_inventory), source_manifest=binding['source_manifest'],
        successful_episodes=48, new_successful_episode_target=152, residual_shard_targets=list(map(len, remaining)),
        family_integrity=family, original_success_evidence_unchanged=True)


def collector_command(args, index):
    result = frozen.collector_command(args, index)
    result[result.index('--count') + 1] = '200'
    result[result.index('--seed-start') + 1] = str(TASK1_SEED_START)
    result.extend(['--task1-acceptance', str(args.task1_acceptance.resolve()), '--execute'])
    return result


def collection_barrier(state, mode, validators=None):
    if mode not in ('source', 'randomized') or validators is None or set(validators) != {'zhuhai_task1', 'zhuhai_task2'}:
        raise RuntimeError('Physical evidence validators for both tasks required')
    for task, validate in validators.items():
        stage, fresh = state['stages'].get(f'{mode}_task{task[-1]}', {}), validate()
        target = 200 if (mode, task) == ('randomized', 'zhuhai_task1') else 100
        if (stage.get('status') != 'completed' or fresh.get('successful_episodes') != target or
                fresh.get('source_families') != 100):
            raise RuntimeError('Strict source/DR200/task2 collection barrier incomplete')
        frozen._check_recorded_receipt(stage.get('validation'), fresh)


_original_build_plan = _p.build_plan

def build_plan(args):
    plan = _original_build_plan(args)
    source = frozen.read_json(plan['prior_DR_import']['zhuhai_task1']['source_manifest']['path'])
    inventory = frozen.read_json(args.task1_prior_inventory)
    attempted = {r['seed'] for r in source['attempts']} | set(plan['task1_expert_acceptance']['seeds'])
    for ref in inventory['components']:
        attempted.update(r['seed'] for r in frozen.read_json(ref['path'])['attempts'])
    if any(TASK1_SEED_START <= seed < TASK1_SEED_START + args.collection_workers * collector200.SEED_STRIDE
           for seed in attempted):
        raise ValueError('DR200 seed reservation overlaps prior/source/probe attempts')
    plan.update(schema='zhuhai-native16-task1-dr200-task2-import-collection-only-v1',
        source_count=100, randomized_count=200, total_all_tasks=500, total_per_task={'zhuhai_task1': 300, 'zhuhai_task2': 200},
        current_run_successful_episode_target=500, new_successful_episode_target=152,
        randomized_count_per_task={'zhuhai_task1': 200, 'zhuhai_task2': 100},
        inherited_scheduler=frozen.reference(frozen.__file__),
        inherited_collector=collector200.inherited_binding(),
        seed_reservation=dict(begin=TASK1_SEED_START, end_exclusive=TASK1_SEED_START + args.collection_workers * collector200.SEED_STRIDE,
                              per_shard_stride=collector200.SEED_STRIDE, probe_and_prior_disjoint=True),
        next_action='Only task1 DR152 new physical episodes; no tiny, training or policy evaluation')
    plan['family_split'].update(source_families=dict(train=80,val=10,heldout=10),seed=20261008,
        successful_randomized_per_family=2, task1_episodes=dict(train=240,val=30,heldout=30))
    for path in (frozen.__file__, collector200.frozen.__file__):
        plan['source_hashes'][str(Path(path).resolve())] = frozen.sha(path)
    _p.verify_plan_sources(plan)
    return plan


def execute_collection(args, plan, plan_path):
    state_path = args.out/'status.json'
    if state_path.exists() and not args.resume:
        raise RuntimeError('Existing collection requires explicit resume')
    state = _p.read_json(state_path) if state_path.exists() else dict(status='ready', active_stage=None,
        stages={}, plan_sha256=_p.sha(plan_path), **DISABLED)
    if state.get('plan_sha256') != _p.sha(plan_path) or any(state.get(k) is not False for k in DISABLED):
        raise ValueError('Frozen collection-only plan/status mismatch')
    if any(_p.re.fullmatch(r'(?:source_task[12]|randomized_task[12]|randomized_task1_shard[0-3])', n) is None for n in state['stages']):
        raise ValueError('No tiny, training or policy evaluation stage allowed')
    selected = selected_collector(args)
    borrowed = _p.completed_task2_import(args.task2_collection_run)
    if borrowed != plan['task2_collection_import']:
        raise ValueError('Imported completed task2 evidence changed')
    state['source_snapshot'] = _p.snapshot_sources(args.out, plan)
    specs = [dict(task='zhuhai_task1', source_manifest=str(args.source_root.resolve()/'zhuhai_task1/source/manifest.json'),
                  randomized_manifest=str(args.collection_root/'zhuhai_task1/randomized/manifest.json')),
             dict(task='zhuhai_task2', source_manifest=borrowed['source_manifest'], randomized_manifest=borrowed['randomized_manifest'])]
    _p.write_once(args.out/'tasks.json', dict(tasks=specs), identical_ok=True)
    source_validators = {s['task']: (lambda s=s: _p.collection_receipt(Path(s['source_manifest']), s['task'], 'source')) for s in specs}
    for spec in specs:
        name = 'source_task'+spec['task'][-1]; receipt = source_validators[spec['task']]()
        if state['stages'].get(name) and state['stages'][name].get('validation') != receipt:
            raise ValueError('Original source evidence changed')
        state['stages'][name] = dict(status='completed', imported=True, validation=receipt)
    state['stages']['randomized_task2'] = dict(status='completed', imported=True, validation=borrowed['randomized_validation'])
    _p.atomic(state_path, state); collection_barrier(state, 'source', source_validators)
    jobs = []
    for index in range(args.collection_workers):
        command = collector_command(args, index); target = Path(_p._command_value(command, '--out'))
        if (target/'manifest.json').exists():
            if not args.resume:
                raise RuntimeError('Existing task1 shard requires explicit resume')
            command.append('--resume')
        jobs.append(dict(name=f'randomized_task1_shard{index}', task='zhuhai_task1', shard_index=index,
            command=command, manifest=target/'manifest.json', validate=lambda target=target, index=index:
                _p.collection_receipt(target/'manifest.json', 'zhuhai_task1', 'randomized', collector=selected,
                                     shard_index=index, shard_count=args.collection_workers)))
    try:
        _p.run_collection_group(args.out, 'randomized', jobs, args, state, plan)
        new_rows = []
        for index, job in enumerate(jobs):
            new_rows.extend(selected.validate_manifest(job['manifest'], task='zhuhai_task1', mode='randomized',
                shard_index=index, shard_count=args.collection_workers))
        old_rows = selected.prior_rows(args.task1_prior_inventory)
        if len(old_rows) != 48 or len(new_rows) != 152:
            raise ValueError('Final quota exactly preserved48 + new152 required')
        source_rows = selected.validate_manifest(specs[0]['source_manifest'], task='zhuhai_task1', mode='source')
        family = family_integrity(source_rows, [*old_rows, *new_rows], randomized_per_family=2)
        _p._unique_rows([*source_rows, *old_rows, *new_rows, *plan['task1_expert_acceptance']['probe_episodes']])
        target = Path(specs[0]['randomized_manifest'])
        selected.aggregate_shards(target, [j['manifest'] for j in jobs], task='zhuhai_task1')
        dr_validators = {'zhuhai_task1': lambda: _p.collection_receipt(target, 'zhuhai_task1', 'randomized', collector=selected),
                         'zhuhai_task2': lambda: _p.completed_task2_import(args.task2_collection_run)['randomized_validation']}
        state['stages']['randomized_task1'] = dict(status='completed', validation=dr_validators['zhuhai_task1'](), family_integrity=family)
        collection_barrier(state, 'randomized', dr_validators); _p.verify_plan_sources(plan)
        state.update(status='completed_collection_only', active_stage=None, active_stages=[], completed_unix=_p.time.time(), **DISABLED)
        _p.atomic(state_path, state)
        _p.write_once(args.out/'completion.json', dict(status='completed_collection_only', tasks=specs,
            source_count_per_task=100, randomized_count_per_task={'zhuhai_task1': 200, 'zhuhai_task2': 100},
            total_successful_episodes=500, **DISABLED), identical_ok=True)
        _p.log('垃圾100源+200DR及餐具100源+100DR严格验收；无tiny、训练或评测', actor='tasks12_collection_only')
        return 0
    except BaseException as exc:
        state.update(status='halted_interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'halted_stage_failure',
                     error=f'{type(exc).__name__}: {exc}', active_stage=None, **DISABLED)
        _p.atomic(state_path, state)
        raise


_p.__dict__.update(parser=parser, validate_args=validate_args, selected_collector=selected_collector,
    accepted_task1_expert=accepted_task1_expert, task1_prior_import=task1_prior_import,
    collector_command=collector_command, collection_barrier=collection_barrier, build_plan=build_plan,
    execute_collection=execute_collection, TASK1_SEED_START=TASK1_SEED_START)
# Keep original family_integrity/default_collector for unchanged task2 import.
for _key in ('pipeline', 'main', 'run_collection_group', 'protect_output_paths', 'resource_ready',
             'collection_resource_admission', 'resume_target_guard', 'stage_needs_run', '_collection_jobs',
             'verify_plan_sources', 'collection_receipt', 'exclusive_lock', 'sha', 'reference', 'read_json'):
    globals()[_key] = getattr(_p, _key)

if __name__ == '__main__':
    raise SystemExit(main())
