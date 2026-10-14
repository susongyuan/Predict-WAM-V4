#!/usr/bin/env python3
"""Exactly one new five-seed protocol, exclusive GPU, immutable original evidence."""
from pathlib import Path
import argparse, ast, fcntl, json, os, sys, time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as evidence
from scripts import eval_zhuhai_precision_official_braking as evaluator
from harness import zhuhai_official_braking as braking
from scripts import gpu_lock
from scripts.progress1_log import log
BASE = ROOT/'outputs/zhuhai2026_tasks12'
SCHEMA = 'precision-official-braking-fixed5-v1'
BASELINE = BASE/'wam_precision_v2_task2_final30k_trainingfirst_20261010/summary.json'
EXTENSION = [Path(braking.__file__), Path(evaluator.__file__), Path(__file__)]


def classify(path, metadata, frozen, source_hash):
    path = Path(path)
    raw = evidence.read_json(path)
    abort = raw.get('status') == 'infrastructure_error'
    if abort:
        prefix = f"InfrastructureError: wrong worker reply/ID, expected action/{evidence.TASK}_seed_{raw.get('seed')}/{raw.get('control_steps')}: "
        error = raw.get('error', '')
        evidence.require(error.startswith(prefix), 'Unknown infrastructure error; preserve and HALT')
        reply = ast.literal_eval(error[len(prefix):])
        evidence.require(reply == dict(version=1, kind='infrastructure_error', episode='startup', step=0,
            error='ValueError: '+braking.SIGNATURE), 'Unknown worker rejection; preserve and HALT')
    else:
        evidence.require(raw.get('status') == 'completed' and not raw.get('error'), 'Nonterminal or infrastructure result')
    seed = raw.get('seed')
    evidence.require(seed in evidence.SEEDS and 0 < raw.get('control_steps', 0) <= 2400, 'Invalid physical attempt')
    changed = evaluator.overlay(frozen)
    runtime = evidence.validate_binding(raw, metadata['step'], seed, metadata['sha256'], changed)
    evidence.require(runtime.get('evaluation_version') == braking.VERSION
        and runtime.get('evaluation_source_sha256') == source_hash
        and runtime.get('training_contract_sha256') == evidence.digest(frozen), 'Evaluation overlay identity mismatch')
    terminal = raw.get('last_rule_evidence' if abort else 'final_rule_evidence')
    score = evidence.validate_rule(terminal)
    if abort:
        evidence.require(raw.get('full_task_success') is None and raw.get('base_score') is None
            and score['full_task_success'] is False, 'Unsuccessful abort raw evidence changed')
    else:
        evidence.require(raw.get('full_task_success') == score['full_task_success']
            and raw.get('base_score') == score['base_score'], 'Raw score differs from physical evidence')
        evidence.require((score['full_task_success'] and raw.get('termination') == 'full_rule_success')
            or (not score['full_task_success'] and raw.get('termination') == 'fixed_step_limit'
                and raw['control_steps'] == 2400), 'Invalid termination/budget')
    files = evidence.archive_evidence(path.parent, raw, runtime, terminal)
    import numpy as np
    with np.load(path.parent/'onboard/observation_000000.npz', allow_pickle=False) as z:
        initial = z['commanded_joint_gripper_state'].copy()
    limiter = braking.BrakingRateLimiter(changed['action_postprocessing']['rate_limit']); limiter.reset(initial)
    brake_steps = 0; max_delta = 0.
    for line in (path.parent/'actions.jsonl').read_text().splitlines():
        action = evidence.loads(line)
        actual, stats = limiter.step(action['smoothed_action'])
        evidence.require(np.array_equal(actual, action['dispatched_action']), 'Discrete braking replay mismatch')
        evidence.require(stats == action['policy']['rate_limit_stats'], 'Limiter statistics replay mismatch')
        brake_steps += int(stats['braking_limited_dimensions'] > 0)
        max_delta = max(max_delta, stats['arm_limit_delta_max_rad'])
    if abort:
        lo, hi = limiter.feasible_interval()
        evidence.require(np.any(lo > hi+1e-10), 'Cannot prove terminal braking safety abort')
    return dict(task=evidence.TASK, seed=seed, status='control_safety_abort' if abort else 'completed',
        full_task_success=False if abort else raw['full_task_success'], base_score=raw['base_score'],
        observed_base_score=score['base_score'], control_steps=raw['control_steps'], attempt_count=1,
        raw_result_path=str(path.resolve()), raw_result_sha256=evidence.sha256_file(path),
        evidence_files=files, braking_steps=brake_steps, max_command_adjustment_rad=max_delta,
        deterministic_command_replay_verified=True)


def summary(plan, rows, status, error=None):
    successes = sum(r.get('full_task_success') is True for r in rows)
    completed = len(rows) == 5 and status == 'completed'
    if status == 'completed': evidence.require([r['seed'] for r in rows] == list(evidence.SEEDS), 'Incomplete fixed cohort')
    result = dict(schema=SCHEMA, status=status, checkpoint=plan['checkpoint'],
        evaluation_version=braking.VERSION, action_postprocessing_sha256=plan['action_postprocessing_sha256'],
        seeds=list(evidence.SEEDS), planned_episodes=5, episodes=rows,
        normal_completed=sum(r.get('status') == 'completed' for r in rows),
        safety_aborted=sum(r.get('status') == 'control_safety_abort' for r in rows),
        full_successes=successes, success_rate=successes/5 if completed else None,
        baseline=plan['baseline'], record_video=False, third_view=False, randomized=False,
        interpretation='New execution protocol, same30k/seeds/scoring; combined limits+braking effect, not isolated causality')
    if error: result['error'] = error
    return result


def build_plan(args):
    meta, frozen = evidence.checkpoint_metadata(args.checkpoint)
    evidence.require(meta['step'] == 30000, 'Exact30k model required')
    baseline = evidence.verify_summary(BASELINE, 30000, evidence.SEEDS, meta['sha256'])
    out = Path(args.out).absolute()
    evidence.require(out.resolve() == out and BASE in out.parents and not out.exists(), 'Fresh canonical output required')
    audit = evidence.read_json(args.cpu_audit)
    evidence.require(audit.get('status') == 'passed' and audit.get('gpu_used') is False
        and audit.get('checkpoint_sha256') == meta['sha256']
        and audit.get('action_postprocessing_sha256') == evidence.digest(braking.protocol()), 'Current CPU audit required')
    for name, sha in audit.get('extension_sources', {}).items():
        evidence.require(evidence.sha256_file(name) == sha, 'Audited source drift: '+name)
    child_env = evidence.read_json(evidence.ENV_LAUNCH)['plan']['child_env']
    settings = braking.protocol()
    hashes = {**audit['extension_sources'], **{str(p.resolve()): evidence.sha256_file(p) for p in EXTENSION}}
    commands = []
    for seed in evidence.SEEDS:
        target = out/evidence.TASK/f'seed_{seed}'
        commands.append(dict(seed=seed, out=str(target), command=[str(evidence.PYTHON), '-B', str(Path(evaluator.__file__).resolve()),
            '--checkpoint', meta['path'], '--out', str(target), '--task', evidence.TASK,
            '--gpu-lock-owner', args.gpu_lock_owner, '--episode-seed', str(seed), '--max-steps', '2400',
            '--wall-seconds', '3600', '--startup-timeout', '180', '--policy-timeout', '60']))
    return dict(schema=SCHEMA, out=str(out), checkpoint=meta, task=evidence.TASK, seeds=list(evidence.SEEDS),
        baseline=dict(path=str(BASELINE), sha256=evidence.sha256_file(BASELINE), success_rate=baseline['success_rate']),
        action_postprocessing=settings, action_postprocessing_sha256=evidence.digest(settings),
        frozen_sources=frozen['source_files'], extension_sources=hashes, commands=commands,
        gpu_lock_owner=args.gpu_lock_owner, child_env=child_env,
        cpu_audit=dict(path=str(Path(args.cpu_audit).resolve()), sha256=evidence.sha256_file(args.cpu_audit)),
        record_video=False, third_view=False, randomized=False, priority='evaluation_before_idle_collection')


def verify_sources(plan):
    for name, sha in {**plan['frozen_sources'], **plan['extension_sources']}.items():
        evidence.require(evidence.sha256_file(name) == sha, 'Frozen/evaluation source drift: '+name)
    for key in ('baseline', 'cpu_audit'):
        evidence.require(evidence.sha256_file(plan[key]['path']) == plan[key]['sha256'], key+' changed')


def execute(plan, wait_seconds):
    from scripts import run_zhuhai_task1_dr200_concurrent as admission
    for previous in BASE.glob('*/plan.json'):
        old = evidence.read_json(previous)
        if (old.get('schema') == SCHEMA and old.get('checkpoint', {}).get('sha256') == plan['checkpoint']['sha256']
                and old.get('action_postprocessing_sha256') == plan['action_postprocessing_sha256']):
            evidence.require(not list(previous.parent.glob('attempt_seed_*.json')),
                'Previous launch intent exists for this protocol; no retry: '+str(previous.parent))
    out = Path(plan['out']); out.mkdir(parents=True, exist_ok=False)
    evidence.write_json(out/'plan.json', plan)
    rows = []; held = False
    evidence.write_json(out/'summary.json', summary(plan, rows, 'waiting_for_exclusive_gpu'))
    try:
        verify_sources(plan)
        deadline = time.monotonic()+wait_seconds
        while True:
            evidence.require(time.monotonic() < deadline, 'GPU wait deadline exhausted')
            if gpu_lock.owner() is not None or admission.gpu_jobs(admission.processes()):
                time.sleep(10); continue
            try: gpu_lock.acquire(plan['gpu_lock_owner'], max_wait=0)
            except TimeoutError: time.sleep(10); continue
            held = True
            evidence.require(not admission.gpu_jobs(admission.processes()), 'Foreign GPU workload appeared')
            resource = admission.scheduler_module().collection_resource_admission()
            evidence.require(admission.scheduler_module().resource_ready(resource, expansion=True), 'RAM/GPU/swap admission denied')
            evidence.write_json(out/'resource_admission.json', resource)
            break
        meta, frozen = evidence.checkpoint_metadata(plan['checkpoint']['path'])
        evidence.require(meta == plan['checkpoint'], 'Checkpoint drift')
        with Path(meta['path']).open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            evidence.write_json(out/'summary.json', summary(plan, rows, 'evaluating'))
            log('官方Panda各轴限制+离散提前制动固定五测启动；同30k/932101–932105/原评分，无录像/DR；旧0/5保留，新协议独立计分。', actor='official_braking_eval')
            env = os.environ.copy(); env.update(plan['child_env'])
            for item in plan['commands']:
                verify_sources(plan); evidence.verify_external_owner(plan['gpu_lock_owner'])
                target = Path(item['out']); target.parent.mkdir(exist_ok=True)
                evidence.require(not target.exists(), 'No repeated physical attempt')
                evidence.write_json(out/f'attempt_seed_{item["seed"]}.json', item)
                code = evidence.run_episode(item, env)
                evidence.write_json(out/f'exit_seed_{item["seed"]}.json', dict(seed=item['seed'], returncode=code))
                row = classify(target/'result.json', meta, frozen, plan['extension_sources'][str(Path(evaluator.__file__).resolve())])
                evidence.require(code in (0, 2) if row['status'] == 'control_safety_abort' else code == 0, 'Case/exit mismatch')
                rows.append(row); evidence.write_json(out/'summary.json', summary(plan, rows, 'evaluating'))
                log(f'官方限制+制动 seed{item["seed"]}：{row["status"]}，{row["control_steps"]}步，完整成功{row["full_task_success"]}，原末分{row["observed_base_score"]}。', actor='official_braking_eval')
            verify_sources(plan)
            evidence.write_json(out/'summary.json', summary(plan, rows, 'completed'))
            log(f'官方Panda限制+提前制动固定五测完成：{sum(r["full_task_success"] for r in rows)}/5；旧成绩不替换。', actor='official_braking_eval')
        return 0
    except BaseException as exc:
        evidence.write_json(out/'summary.json', summary(plan, rows, 'halted', f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        if held:
            jobs = admission.gpu_jobs(admission.processes())
            if jobs:
                evidence.write_json(out/'owner_retained_guard.json', dict(jobs=jobs, owner=plan['gpu_lock_owner']))
            else: gpu_lock.release(plan['gpu_lock_owner'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    p.add_argument('--cpu-audit', required=True); p.add_argument('--execute', action='store_true')
    p.add_argument('--gpu-lock-owner', default='zhuhai-official-braking-fixed5-20261010-v1')
    p.add_argument('--wait-seconds', type=float, default=21600)
    args = p.parse_args()
    evidence.require(0 < args.wait_seconds <= 21600, 'Bounded6h maximum GPU wait')
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    return execute(plan, args.wait_seconds)


if __name__ == '__main__': raise SystemExit(main())
