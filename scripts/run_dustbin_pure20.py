#!/usr/bin/env python3
"""Frozen 25k pure-WAM, twenty fresh seeds; no training or source mutation.

The existing serial native queue owns each GPU child and archives invalid runs
separately. This CPU supervisor keeps a live report and verifies actual actions.
Without --execute, only read-only metadata/seed/checkpoint checks are performed.
"""
import argparse
import csv
import datetime
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import atomic_json, identity, sha256, training_contract
from scripts import run_robotwin_checkpoint_queue as queue
from scripts.eval_robotwin_wam import CONFIG_DIR, ROBOTWIN, dataset_seed_inventory
from scripts.gpu_lock import owner
from scripts.run_robotwin_posttrain_eval_pipeline import training_gate

TASK = 'put_bottles_dustbin'
TRAIN = ROOT / 'outputs/runs/put_bottles_dustbin/wam_30k_20261007/train'
CKPT = TRAIN / 'checkpoint_step_025000.pt'
TRAIN_UNIT = 'p2-dustbin-wam30k-20261007'
CONFIG = ROOT / 'configs/fixed_bimanual/robotwin_dustbin_50_wam.yaml'
BASE = ROOT / 'outputs/runs/put_bottles_dustbin/pure_25k_30k_5each_20261007'
SEEDS = list(range(921001, 921021))
ZERO_SHA = 'ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')


def bus_environment():
    runtime = f'/run/user/{os.getuid()}'
    if Path(runtime, 'bus').exists():
        os.environ.setdefault('XDG_RUNTIME_DIR', runtime)
        os.environ.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path={runtime}/bus')


def read(path):
    return queue.read_result(path)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def admission():
    contract, cache, _, receipt = training_contract(TRAIN, CKPT)
    if (contract['config']['dataset']['task_name'] != TASK or receipt['step'] != 25000
            or contract['config_sha256'] != sha256(CONFIG) or sha256(CKPT) != receipt['sha256']):
        raise ValueError('task/config/frozen 25k checkpoint mismatch')
    ready, gate = training_gate(TRAIN, TRAIN_UNIT)
    if not ready:
        raise ValueError('training/GPU admission not ready: ' + str(gate))
    dataset_seeds, seed_files = dataset_seed_inventory(cache)
    # Recorded smoke/development/evaluation seeds, never performance-selected.
    known = {910000, 920000, 920010, *range(920001, 920006)}
    evidence = []
    for directory in (ROOT / 'outputs/runs/put_bottles_dustbin',
                      ROOT / 'outputs/verification/dustbin_wam_20261007',
                      ROOT / 'outputs/verification/dustbin_middle25k_20261007'):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob('result.json')):
            result = read(path)
            if result.get('task') != TASK:
                continue
            found = [e['seed'] for e in result.get('episodes', []) if type(e.get('seed')) is int]
            if found:
                known.update(found)
                evidence.append(identity(path))
        for path in sorted(directory.rglob('manifest.json')):
            manifest = read(path)
            if manifest.get('task') == TASK:
                known.update(s for s in manifest.get('seeds', []) if type(s) is int)
                known.update(j['seed'] for j in manifest.get('jobs', []) if type(j.get('seed')) is int)
    rng = {contract['config']['training'][k] for k in ('seed', 'validation_seed')
           if k in contract['config']['training']}
    excluded = dataset_seeds | known | rng
    if len(SEEDS) != 20 or len(set(SEEDS)) != 20 or set(SEEDS) & excluded:
        raise ValueError('twenty distinct new seeds required; observed seed overlap')
    return dict(checked_at=now(), gate=gate, checkpoint_receipt=receipt,
                checkpoint=identity(CKPT, content=False), training_config=identity(CONFIG),
                excluded_seeds=sorted(excluded), seed_overlap=[], prior_result_evidence=evidence,
                seed_files=[identity(p) for p in sorted(seed_files)],
                provenance_limit='Audited local recorded runs and dataset seeds; not an attestation of unknown external tests.')


def build_manifest(out, admitted):
    original = read(BASE / 'manifest.json')
    reference = next(j for j in original['jobs'] if j['checkpoint_step'] == 25000)
    expected = dict(system_mode='pure', execution_mode='native8', task_config='demo_clean',
                    camera_readback='serial', max_steps=1700, settle_steps=0, record_onboard=True)
    if any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError('historical pure WAM protocol mismatch')
    jobs = [dict(reference, seed=seed, unit=f'p2-dustbin-pure20-25k-{seed}-20261007',
                 run_dir=str(out / f'seed_{seed}')) for seed in SEEDS]
    sources = [Path(__file__), ROOT / 'scripts/run_robotwin_checkpoint_queue.py',
               ROOT / 'scripts/eval_robotwin_wam.py', ROOT / 'scripts/gpu_lock.py',
               ROOT / 'scripts/run_robotwin_posttrain_eval_pipeline.py', CONFIG]
    sources += sorted((ROOT / 'harness').rglob('*.py'))
    sources += sorted((ROBOTWIN / 'envs').rglob('*.py')) + sorted(CONFIG_DIR.glob('*.yml'))
    return dict(protocol='dustbin_pure25k_fresh20_v1', task=TASK, created_at=now(),
                seeds=SEEDS, jobs=jobs, training_run_dir=str(TRAIN), training_unit=TRAIN_UNIT,
                require_video=True, max_invalid_retries=1, checkpoint_receipt=admitted['checkpoint_receipt'],
                baseline_manifest=identity(BASE / 'manifest.json'),
                checkpoint_selection='25k frozen before new tests; chosen after earlier 5-seed development results. Not a comparison selecting the best of two checkpoints on these 20 seeds.',
                source_sha256={str(p.resolve()): sha256(p) for p in sources},
                enabled=['frozen WAM25k', 'DINOv3', 'three RGB', 'native14 command state', 'native32/exec8'],
                disabled=['VLM', 'ASK', 'VTT', 'target heads', 'middle supervisor', 'recovery', 'external memory', 'retraining'],
                scoring='Native benchmark predicate only. No strict full-containment/release/stability claim. Infra/initialization failures remain unscored; at most one same-seed init retry.',
                comparison='Preserve old renderer(no denoiser), serial camera readback, native8, 1700 steps, settle0 and wall limits. New 20 are reported separately from old five.',
                estimated_wall_hours=[3, 5])


def verify_admission(manifest, admitted):
    ready, gate = training_gate(TRAIN, TRAIN_UNIT)
    if not ready:
        raise ValueError('training exit/GPU lock not ready before child: ' + str(gate))
    if read(CKPT.with_name(CKPT.name + '.sha256.json')) != admitted['checkpoint_receipt']:
        raise ValueError('checkpoint receipt changed')
    if CKPT.stat().st_size != admitted['checkpoint']['size']:
        raise ValueError('checkpoint size changed')
    for record in admitted['seed_files']:
        if sha256(record['path']) != record['sha256']:
            raise ValueError('dataset seed provenance changed')
    # Queue checks source SHA256 too; the actual worker rehashes checkpoint bytes.
    if owner() is not None:
        raise ValueError('GPU acquired by another job')


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z / total
    centre = (p + z*z/(2*total)) / d
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [centre-half, centre+half]


def verify_episode(job, attempt, manifest):
    directory = Path(attempt['run_dir'])
    result = read(directory / 'result.json')
    episode = result['episodes'][0]
    plan = read(directory / 'plan.json')
    ep_dir = directory / episode['episode']
    actions = [json.loads(line) for line in (ep_dir / 'actions.jsonl').read_text().splitlines()]
    submitted = [a for a in actions if a['status'] == 'submitted']
    returned = [a for a in actions if a['status'] == 'native_returned']
    video = read(ep_dir / 'third_view_video.json')
    checks = dict(task=result['task'] == TASK, seed=episode['seed'] == job['seed'],
                  checkpoint=plan['checkpoint_receipt']['sha256'] == manifest['checkpoint_receipt']['sha256'],
                  pure=plan['system']['mode'] == 'pure', native8=plan['system']['execution_mode'] == 'native8',
                  no_external_memory=plan['system']['memory_writes_enabled'] is False,
                  action_count=len(submitted) == len(returned) == episode['control_steps'] == episode['native_take_action_count']
                               and len(actions) == 2 * len(returned),
                  action_ids=all(a['step'] == i and a['episode'] == episode['episode']
                                 for records in (submitted, returned) for i, a in enumerate(records)),
                  dispatch_ack=all(a['step'] == b['step'] and a['bounded_action'] == b['bounded_action']
                                   for a, b in zip(submitted, returned)),
                  zero_conditions=all(a['policy']['task_condition'] == dict(source='zero_single_task', sha256=ZERO_SHA) for a in actions),
                  no_recovery=all(a['policy']['execution_prefix'] == 8 and a['policy']['execution_reason'] == 'native_exec8' for a in actions),
                  video=video.get('complete') is True and video.get('synthetic') is False
                        and video['frames'] == episode['control_steps'] + 1)
    return dict(seed=job['seed'], checks=checks, passed=all(checks.values()),
                result=identity(directory / 'result.json'), plan=identity(directory / 'plan.json'),
                actions=identity(ep_dir / 'actions.jsonl'), video_metadata=identity(ep_dir / 'third_view_video.json'))


def summarize(out, manifest):
    state = read(out / 'queue_status.json')
    rows, verification = [], []
    for job in manifest['jobs']:
        attempts = [a for a in state.get('jobs', []) if a['seed'] == job['seed']]
        valid = [a for a in attempts if a['status'] == 'valid_terminal']
        attempt = valid[-1] if valid else attempts[-1] if attempts else None
        result = read(Path(attempt['run_dir']) / 'result.json') if attempt else {}
        episode = (result.get('episodes') or [{}])[0]
        row = dict(seed=job['seed'], status=attempt['status'] if attempt else 'not_started',
                   native_success=episode.get('evidence', {}).get('native_benchmark_success') if valid else None,
                   control_steps=episode.get('control_steps'), wall_seconds=episode.get('wall_seconds'),
                   attempts=len(attempts), result_path=str(Path(attempt['run_dir']) / 'result.json') if attempt else '')
        if valid:
            try:
                verification.append(verify_episode(job, attempt, manifest))
            except Exception as exc:
                verification.append(dict(seed=job['seed'], passed=False, error=f'{type(exc).__name__}: {exc}'))
        rows.append(row)
    completed = [r for r in rows if r['status'] == 'valid_terminal']
    successes = sum(r['native_success'] is True for r in completed)
    count = len(completed)
    full = count == 20
    checked = full and len(verification) == 20 and all(r['passed'] for r in verification)
    status = state.get('status', 'preregistered')
    summary = dict(status=status, updated_at=now(), planned=20, scored=count, successes=successes,
                   failures=count-successes, native_success_rate=successes/20 if full else None,
                   partial_success_rate=successes/count if count else None,
                   wilson_95=wilson(successes, count) if full else None,
                   valid_completed_20=full, verified_complete=checked, episodes=rows,
                   error=state.get('error'), limitations=manifest['scoring'])
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'verification.json', dict(complete=checked, checked_episodes=len(verification), episodes=verification))
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(out / 'per_episode_results.csv', csv_buffer.getvalue())
    lines = ['# 25k纯WAM：20个全新种子', '', f'更新时间：{now()}；队列状态：{status}。', '',
             f'有效完成 **{count}/20**，原生成功 **{successes}**，有效失败 **{count-successes}**。']
    if full:
        interval = summary['wilson_95']
        lines += [f'成功率 **{successes/20:.0%}**；95% Wilson 区间 {interval[0]:.1%}–{interval[1]:.1%}。逐动作证据核对全部通过：{checked}。']
    else:
        lines += ['测试未全部完成，当前结果不是最终20次成功率。无效初始化、基础设施错误和中断不记作策略失败。']
    lines += ['', '固定权重25k（此前五测开发后选择，本轮不再选权重）、种子921001–921020；三个RGB、native14、native8、serial readback、1700步、settle0。无VLM/ASK/VTT/中层恢复/外部记忆/重训练。',
              '', '原生成功只检验三个刚体瓶原点在敞口垃圾桶阈值范围；没有开盖、柔性物体、完整容纳、释放或稳定验证。旧五测不合并计入本轮20测。',
              '', '本轮可估计纯WAM在当前任务/分布上的表现。系统优越性需要完整系统在同20种子、同权重和相同预算的配对结果，并报告恢复、误停和完成时间。',
              '', '|seed|状态|原生成功|控制动作|', '|---|---|---|---:|']
    for row in rows:
        value = '成功' if row['native_success'] is True else '失败' if row['native_success'] is False else '未评分'
        lines.append(f'|{row["seed"]}|{row["status"]}|{value}|{row["control_steps"] if row["control_steps"] is not None else ""}|')
    if state.get('error'):
        lines += ['', f'队列异常：{state["error"]}']
    lines += ['', '证据：同目录 manifest.json、training_admission.json、queue_status.json、summary.json、verification.json、per_episode_results.csv；每次测试目录保存动作、机载观测、独立评分和第三视角录像。', '']
    atomic_text(out / 'report.md', '\n'.join(lines))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', default='outputs/runs/put_bottles_dustbin/pure25k_fresh20_20261007')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    bus_environment()
    out = (ROOT / args.run_dir).resolve()
    if out.exists():
        raise ValueError('refuse existing archive; do not replay or overwrite tests')
    admitted = admission()
    manifest = build_manifest(out, admitted)
    queue.validate_manifest(manifest)
    if not args.execute:
        print(json.dumps(dict(protocol=manifest['protocol'], seeds=SEEDS, jobs=20,
                              checkpoint_sha256=admitted['checkpoint_receipt']['sha256'],
                              seed_overlap=[], estimated_wall_hours=[3, 5]), indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'manifest.json'
    atomic_json(path, manifest)
    atomic_json(out / 'training_admission.json', admitted)
    stop = threading.Event()
    reporting_errors = []
    def report_loop():
        while not stop.wait(30):
            try:
                summarize(out, manifest)
            except Exception as exc:
                reporting_errors.append(f'{type(exc).__name__}: {exc}')
    summarize(out, manifest)
    reporter = threading.Thread(target=report_loop, daemon=True)
    reporter.start()
    code = 1
    error = None
    try:
        code = queue.run_manifest(manifest, path, before_job=lambda: verify_admission(manifest, admitted))
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        stop.set()
        reporter.join(timeout=15)
        summary = summarize(out, manifest)
        if not summary['verified_complete']:
            code = 1
        atomic_json(out / 'completion.json', dict(finished_at=now(), exit_code=code, error=error,
                    verified_complete=summary['verified_complete'], reporting_errors=reporting_errors))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
