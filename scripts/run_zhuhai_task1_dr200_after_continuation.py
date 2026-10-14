#!/usr/bin/env python3
"""CPU plan by default; explicit --execute queues one DR200 worker after continuation.

Only the private copy's tiny_review is redirected. Original serial execution,
GPU ownership, flock, resource, source/peak, foreign-job and failure guards stay
unchanged. This script never runs a controller, training or evaluation command.
The unit owner, not this wrapper, must restore the historical runtime environment.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/zhuhai2026_tasks12'
ORIGIN = BASE / 'wam_precision_v2_task2_30k_20261010_v1'
ORIGIN_PLAN_SHA256 = '940e22a7a2257c1d9db30cf9e2147f5e92d33a4520845c294ec24d082d5e55f9'
DEFAULT_RUN = BASE / 'wam_precision_v2_task2_30k_continuation_20261010_r2'
DEFAULT_OUT = BASE / 'production_tasks12_dr200_collection_only_20261010_serial_r2'
DEFAULT_OWNER = 'zhuhai-task1-dr200-serial-20261010-r2'
SCHEMA = 'task1_dr200_after_continuation_fixed_attempts_v2'
SEEDS = [932101, 932102, 932103, 932104, 932105]
FROZEN_SHA256 = {
    'scripts/run_zhuhai_task1_dr200_concurrent.py': '41db9b1627dcd0e1a10863d034884a6ae7df7f74388be96b7864f8fcf4d12289',
    'scripts/run_zhuhai_tasks12_collection_repair_200.py': 'd863ed6dd802cf07a1ac571880dc7e2e7e80356244030bb3e76eb97b3cac4029',
    'scripts/collect_isaac_zhuhai_tasks12_repair_200.py': '8e70fe190dac35c3167031c038375316161362c7934de6d53b3a2dd4165233f4',
    'scripts/gpu_lock.py': '0515d065cf758284cc52ed1c49c581d04455f38830cb3178edd777c0742224a6',
}

ATTEMPT_SHA256 = {
    'scripts/eval_zhuhai_precision_fixed_attempts.py': '4bb716ef38769a52fb368cb3a63ef10a1065181ad47653872dc60de0f7421d9f',
    'outputs/zhuhai2026_tasks12/verification/precision_v2_overnight_20261009/fixed_attempts_protocol.json': '33b0b31f7a76b31e1fedae420538865432ecec965990f05112fbe9e4156d27a0',
}


def fixed_api():
    # Frozen read-only API only; NEVER call its build_plan/execute/main.
    verify_frozen()
    from scripts import eval_zhuhai_precision_fixed_attempts
    return eval_zhuhai_precision_fixed_attempts


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    # Reject duplicate keys and nonfinite JSON instead of silently accepting drift.
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def canonical(path, parent):
    path = Path(path)
    require(path.is_absolute() and path.absolute() == path.resolve()
            and parent.resolve() in path.parents, 'Noncanonical/symlink/outside project path: ' + str(path))
    return path


def disjoint(a, b):
    require(a != b and a not in b.parents and b not in a.parents, 'Run paths must be disjoint')


def verify_frozen():
    for path, digest in {**FROZEN_SHA256, **ATTEMPT_SHA256}.items():
        require(sha(ROOT / path) == digest, 'Frozen queue/guard/source drift: ' + path)


def private_queue():
    verify_frozen()
    spec = importlib.util.spec_from_file_location('_dr200_continuation_private_queue',
                                                 ROOT / 'scripts/run_zhuhai_task1_dr200_concurrent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--owner', '--gpu-lock-owner', dest='gpu_lock_owner', default=DEFAULT_OWNER)
    p.add_argument('--workers', type=int, choices=(1,), default=1)
    p.add_argument('--schedule', choices=('after-training',), default='after-training')
    p.add_argument('--wait-seconds', type=float, default=172800)
    p.add_argument('--execute', action='store_true')
    return p


def validate_args(args, q):
    q.validate(args)
    require(args.schedule == 'after-training', 'Only serial after-training is allowed')
    canonical(args.training_run, BASE)
    canonical(args.out, BASE)
    origin = canonical(ORIGIN, BASE)
    disjoint(args.training_run, origin)
    disjoint(args.out, origin)
    control = args.out.with_name(args.out.name + '.queue')
    require(not args.out.exists() and not control.exists() and not control.is_symlink(),
            'Fresh output and queue required; never reset/reuse old guards')
    return control


def options(command, script, repeated=()):
    require(isinstance(command, list) and len(command) > 3 and all(isinstance(x, str) for x in command),
            'Invalid stage command')
    require(command[1:3] == ['-B', str(ROOT / 'scripts' / script)], 'Unexpected stage entrypoint')
    result = {}; key = None
    for token in command[3:]:
        if token.startswith('--'):
            require(token not in result or token in repeated, 'Duplicate stage option: ' + token)
            key = token; result.setdefault(key, [])
        else:
            require(key is not None, 'Unexpected positional stage argument')
            result[key].append(token)
    return result


def stage_options(stage, original, owner, adopt=False):
    require(stage['command'][0] == original['command'][0], 'Frozen stage interpreter changed')
    if stage['kind'] == 'train':
        script = 'train_zhuhai_wam_precision_v2.py'
        actual = options(stage['command'], script)
        expected = options(original['command'], script)
        expected['--out'] = [stage['out']]
        expected['--gpu-lock-owner'] = [owner]
    else:
        actual = options(stage['command'], 'eval_zhuhai_precision_fixed_attempts.py', ('--adopt-result',))
        expected = {'--checkpoint': [str(ORIGIN / 'train' / f"checkpoint_step_{stage['step']:06d}.pt")],
                    '--task': ['zhuhai_task2'], '--seeds': list(map(str, SEEDS)),
                    '--out': [stage['out']], '--gpu-lock-owner': [owner], '--execute': []}
        adopted = list(map(str, fixed_api().known_adoptions().values())) if adopt else []
        require(stage.get('adopted_results', []) == adopted, 'Adopted results must be the two original attempts at 20k only')
        if adopted:
            for path in adopted:
                canonical(path, ORIGIN)
            expected['--adopt-result'] = adopted
    require(actual == expected, 'Frozen task/full30k/seeds/stage command changed: ' + stage['name'])


def validate_binding(args, pinned=None):
    """Read-only binding checks. NEVER use origin/status.json as continuation status."""
    verify_frozen()
    origin = canonical(ORIGIN, BASE)
    run = canonical(args.training_run, BASE)
    disjoint(run, origin); disjoint(args.out, origin)
    path = canonical(run / 'plan.json', run)
    require(path.is_file(), 'Continuation plan is not published')
    digest = sha(path)
    require(pinned is None or digest == pinned, 'Continuation plan changed after first observation')
    plan = read(path)
    require(plan.get('training_origin') == str(origin), 'Wrong training_origin')
    origin_path = canonical(origin / 'plan.json', origin)
    require(plan.get('origin_plan_sha256') == ORIGIN_PLAN_SHA256 == sha(origin_path), 'Origin plan hash mismatch')
    original = read(origin_path)
    require(original.get('out') == str(origin) and plan.get('out') == str(run), 'Plan/run binding mismatch')
    expected = dict(schema='zhuhai-precision-wam-task2-30k-pipeline-v2', task='zhuhai_task2',
                    training_updates=30000, batch_size=4, development_seeds=SEEDS, final_seeds=[],
                    planned_physical_episodes=15, record_video=False, third_view=False, collection_parallel=False)
    for key, value in expected.items():
        require(type(plan.get(key)) is type(value) and plan.get(key) == value
                and original.get(key) == value, 'Frozen plan field mismatch: ' + key)
    # Continuation intentionally omits the redundant precision metadata; the
    # original plan SHA, all training source bindings and commands remain exact.
    for key in ('precision', 'checkpoint_selection'):
        if key != 'precision' or key in plan:
            require(plan.get(key) == original.get(key), 'Frozen protocol changed: ' + key)
    owner = plan.get('gpu_lock_owner')
    require(isinstance(owner, str) and owner and args.gpu_lock_owner not in (owner, original.get('gpu_lock_owner')),
            'Never share a training owner')
    hashes = plan.get('source_hashes')
    require(isinstance(hashes, dict) and bool(original.get('source_hashes')), 'Missing source bindings')
    require(all(hashes.get(p) == h for p, h in original['source_hashes'].items()), 'Original source binding changed/missing')
    require(all(hashes.get(str(ROOT / p)) == h for p, h in ATTEMPT_SHA256.items()),
            'Frozen fixed-attempt API/protocol binding missing or changed')
    for source, source_hash in hashes.items():
        canonical(source, ROOT)
        require(sha(source) == source_hash, 'Source hash drift: ' + source)
    stages = plan.get('stages')
    originals = original['stages']
    require(isinstance(stages, list) and len(stages) == 5, 'Exactly five continuation stages required')
    require([s.get('name') for s in stages] == [s['name'] for s in originals[2:]], 'Continuation stage order/inventory changed')
    eval_paths = []
    for stage, prior in zip(stages, originals[2:]):
        for key in ('name', 'kind', 'step'):
            require(stage.get(key) == prior[key], 'Stage identity changed: ' + key)
        out = canonical(stage['out'], BASE)
        disjoint(out, args.out)
        if stage['kind'] == 'train':
            require(stage.get('mode') == 'train' and out == origin / 'train', 'Training artifacts must remain in origin/train')
            require(not stage.get('adopt_existing', False), 'Remaining training cannot be adopted/skipped')
        else:
            require(stage.get('seeds') == SEEDS, 'Paired five seeds changed')
            # Recovery eval20 may be a sibling run, not a child of continuation.
            disjoint(out, origin)
            require(out != run and out not in run.parents, 'Eval cannot overwrite continuation root')
            for other in eval_paths:
                disjoint(out, other)
            eval_paths.append(out)
            require(stage.get('adopt_existing', False) is (stage['step'] == 20000),
                    'Only recovered eval20 must be adopted; eval25/30 must not be skipped')
            precompleted = stage.get('precompleted_summary', False)
            require(type(precompleted) is bool and (not precompleted or stage['step'] == 20000),
                    'Only 20k may reuse a precompleted whole summary')
            if precompleted:
                summary = canonical(out / 'summary.json', out)
                require(stage.get('completed_summary_path') == str(summary)
                        and stage.get('completed_summary_sha256') == sha(summary) == hashes.get(str(summary)),
                        'Precompleted summary path/hash/source binding mismatch')
        stage_options(stage, prior, owner, adopt=stage['step'] == 20000)
        require(stage['command'][stage['command'].index('--gpu-lock-owner') + 1] != args.gpu_lock_owner,
                'Never share an evaluation owner')
    require(sha(path) == digest and sha(origin_path) == ORIGIN_PLAN_SHA256, 'Plan changed during validation')
    return dict(plan=plan, origin=origin, plan_sha256=digest)


def verify_eval_evidence(binding, stage):
    out = Path(stage['out'])
    summary = canonical(out / 'summary.json', out)
    ck = canonical(binding['origin'] / 'train' / f"checkpoint_step_{stage['step']:06d}.pt", binding['origin'])
    receipt = read(canonical(ck.with_name(ck.name + '.sha256.json'), ck.parent))
    require(receipt.get('step') == stage['step'] and receipt.get('model_only') is True,
            'Training checkpoint receipt mismatch')
    value = fixed_api().verify_summary(summary, stage['step'], SEEDS, checkpoint_sha256=receipt['sha256'])
    require(value.get('schema') == 'precision-v2-fixed-attempt-summary-v1' and value.get('status') == 'completed',
            'Fixed-attempt summary not completed')
    if stage['step'] == 20000:
        require([row['raw_result_path'] for row in value['episodes'][:2]] == stage['adopted_results'],
                'Original attempted seeds were replaced or retried')
    # No success threshold: 0/5 is complete when the frozen verifier accepts it.
    return {key: value.get(key) for key in ('checkpoint_step', 'planned_episodes', 'normal_completed',
                                           'safety_aborted', 'full_successes', 'success_rate', 'status')}


def completed_evidence(binding):
    """Fixed 30k only, independent of optional later 50k; never require policy success."""
    train = canonical(binding['origin'] / 'train', binding['origin'])
    state = read(canonical(train / 'status.json', train))
    require(state.get('status') in ('completed', 'stopped_at_safe_boundary')
            and state.get('step') == state.get('target_steps') == 30000, 'Origin training has not completed full30k')
    for step in (20000, 25000, 30000):
        ck = canonical(train / f'checkpoint_step_{step:06d}.pt', train)
        receipt = read(canonical(ck.with_name(ck.name + '.sha256.json'), train))
        require(receipt.get('step') == step and receipt.get('model_only') is True
                and receipt.get('size') == ck.stat().st_size and receipt.get('sha256') == sha(ck),
                'Training checkpoint receipt mismatch')

    for stage in binding['plan']['stages']:
        if stage['kind'] == 'eval':
            verify_eval_evidence(binding, stage)


def command(args, execute=False):
    argv = [str(ROOT / '.venvs/wam-fixed/bin/python'), '-B', str(Path(__file__).resolve()),
            '--training-run', str(args.training_run), '--out', str(args.out), '--owner', args.gpu_lock_owner,
            '--workers', '1', '--schedule', 'after-training', '--wait-seconds', str(args.wait_seconds)]
    return argv + (['--execute'] if execute else [])


def review(args, q=None):
    q = q or private_queue()
    validate_args(args, q)
    plan_path = args.training_run / 'plan.json'
    binding = validate_binding(args) if plan_path.exists() or plan_path.is_symlink() else None
    adopted = [verify_eval_evidence(binding, stage) for stage in binding['plan']['stages']
               if stage.get('precompleted_summary')] if binding else []
    return dict(schema=SCHEMA, mode='plan', collection_started=False, scheduler_called=False,
                verified_precompleted_evaluations=adopted, policy_success_required=False,
                gate_training_updates=30000, optional_50k_is_not_a_dependency=True,
                training_run=str(args.training_run), training_origin=str(ORIGIN), origin_plan_sha256=ORIGIN_PLAN_SHA256,
                continuation_plan_present=binding is not None, continuation_plan_sha256=binding['plan_sha256'] if binding else None,
                status='validated_plan_not_execution_admission' if binding else 'awaiting_controller_plan',
                queue=str(args.out) + '.queue', owner=args.gpu_lock_owner, workers=1, wait_seconds=args.wait_seconds,
                preserved_dr=48, new_dr_target=152, source_target=100, randomized_target=200,
                record_video=False, extra_camera=False, onboard_sensor_archives_preserved=True,
                tiny_route='validated training_origin only; origin pipeline status ignored',
                requirements=['immutable source/plan bindings', 'all five planned stages completed',
                              'three fixed-seed eval summaries + full30k receipts', 'pipeline flock released',
                              'exclusive distinct owner', 'original tiny/peak/resource/foreign-job/probe guards'],
                argv=command(args, True), command=shlex.join(command(args, True)),
                environment='Restored by supervising unit, never modified here', frozen_sources=FROZEN_SHA256, fixed_attempt_sources=ATTEMPT_SHA256)


def execute(args, q=None):
    q = q or private_queue()
    control = validate_args(args, q)
    deadline = time.monotonic() + args.wait_seconds
    pinned = None
    # Preflight waits without importing/calling the scheduler, even if plan is absent.
    # Hold the SAME global queue mutex until handing off; no second waiting writer.
    with q.file_lock(q.QUEUE_LOCK):
        try:
            while True:
                require(time.monotonic() < deadline, 'Bounded continuation wait exhausted; no collection started')
                validate_args(args, q)
                status_path = args.training_run / 'status.json'
                if status_path.exists():
                    state = read(canonical(status_path, args.training_run))
                    require(state.get('status') in ('running', 'completed'), 'Continuation failed/halted/unknown status')
                path = args.training_run / 'plan.json'
                if path.exists() or path.is_symlink():
                    binding = validate_binding(args, pinned)
                    pinned = binding['plan_sha256']
                    if q.pipeline_finished(args.training_run):
                        break
                time.sleep(min(10, max(0, deadline - time.monotonic())))
        except BaseException as exc:
            # A new queue's refusal is durable, never clear/rewrite an older guard.
            if not control.exists() and not control.is_symlink() and not args.out.exists():
                canonical(control, BASE)
                control.mkdir(parents=True)
                state = dict(schema=SCHEMA, status='halted', training_run=str(args.training_run),
                             production_started=False, error=f'{type(exc).__name__}: {exc}')
                q.queue_record(control, state, 'failed_guard.json'); q.queue_record(control, state)
            raise
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'Bounded continuation wait exhausted before serial handoff')
    serial_args = copy(args); serial_args.wait_seconds = remaining
    original_tiny_review = q.tiny_review
    def routed_tiny(run):
        require(run == args.training_run, 'Unexpected tiny redirect target')
        require(time.monotonic() < deadline, 'Bounded wait exhausted before admission')
        binding = validate_binding(args, pinned)
        read(canonical(run / 'status.json', run))  # Strict JSON/path check at the handoff too.
        canonical(run / '.pipeline.lock', run)
        require(q.pipeline_finished(run), 'Continuation completion changed before admission')
        completed_evidence(binding)
        result = original_tiny_review(binding['origin'])
        require(time.monotonic() < deadline, 'Bounded wait exhausted verifying artifacts')
        return result
    q.tiny_review = routed_tiny  # ONLY overridden private queue global.
    try:
        return q.execute_serial(serial_args)
    finally:
        q.tiny_review = original_tiny_review


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.execute:
            return execute(args)
        print(json.dumps(review(args), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'No unsafe launch: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
