#!/usr/bin/env python3
"""CPU supervisor: audit r2's 2250-update boundary, resume once, evaluate once.

Never retries a GPU child or a physical episode. Existing frozen training and
evaluation sources are reused without modifications. Restarting this supervisor
requires a new operator decision because its immutable run directory persists.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / 'outputs/zhuhai2026_tasks12'
VERIFY = BASE / 'verification/precision_v4_20261012'
TRAINING = BASE / 'wam_precision_v4_task2_50k_20261012_r2/train'
EVALUATION = BASE / 'wam_precision_v4_task2_final_paired5_20261012_r2'
SHORT_UNIT = 'p2-precision-v4-short-20261012-r2.service'
TRAIN_OWNER = 'precision-v4-50k-20261012-r2'
EVAL_OWNER = 'precision-v4-final5-20261012-r2'
PYTHON = ROOT / '.venvs/wam-fixed/bin/python'
SCHEDULE = dict(steps=50000, predictor_steps=1000, gate_steps=2000,
    backbone_lr=1e-6, new_lr=1e-4, seed=20261012, batch_size=4,
    save_every=2500, resume_every=500, diagnostic_samples=8)
BOUNDARY = 2250


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: reject_constant(value))


def reject_constant(value):
    raise ValueError('Nonfinite JSON literal: ' + value)


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def write_once(path, data):
    """Never replace admission, attempts, exits or failure evidence."""
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def latest(run, phase, details=None):
    path = run / 'latest.json'
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(dict(phase=phase, unix_seconds=time.time(), details=details or {}),
                  stream, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def journal(message):
    from scripts.progress1_log_v41 import log
    log(message, 'v4_r2_supervisor')


def systemd_environment():
    environment = dict(os.environ)
    runtime = Path('/run/user') / str(os.getuid())
    require((runtime / 'bus').exists(), 'User systemd runtime bus is unavailable')
    environment.update(XDG_RUNTIME_DIR=str(runtime),
                       DBUS_SESSION_BUS_ADDRESS='unix:path=' + str(runtime / 'bus'))
    return environment


def short_service_state():
    result = subprocess.run(['systemctl', '--user', 'show', SHORT_UNIT,
        '--property=ActiveState,SubState,MainPID,ExecMainCode,ExecMainStatus,Result'],
        env=systemd_environment(), capture_output=True, text=True, timeout=10)
    require(result.returncode == 0, 'Cannot inspect short-run service: ' + result.stderr)
    row = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    require('MainPID' in row and 'ActiveState' in row, 'Incomplete short-run service identity')
    return row


def check_no_training_failure(training):
    require(not (training / 'failure_guard.json').exists(),
            'Training failure guard exists; halt without automatic recovery')
    status = read_json(training / 'status.json')
    require(status.get('status') not in ('failed', 'halted'), 'Training reported failure')
    return status


def validate_short_status(status):
    require(status.get('status') == 'stopped_at_safe_boundary'
        and status.get('updates') == BOUNDARY and status.get('step') == 80000 + BOUNDARY
        and status.get('target_updates') == SCHEDULE['steps']
        and status.get('target_step') == 80000 + SCHEDULE['steps'],
        'Short run must stop exactly at committed update 2250')


def wait_short(training, heartbeat=None):
    """No driver, GPU, feature-cache or model work while the short child is live."""
    last_report = -1800.
    while True:
        status = check_no_training_failure(training)
        state = short_service_state()
        if heartbeat:
            heartbeat(dict(status=status, short_service=state))
        require(state.get('ActiveState') != 'failed', 'Short-run systemd service failed')
        if status.get('status') == 'stopped_at_safe_boundary':
            validate_short_status(status)
            if state['MainPID'] == '0' and state['ActiveState'] == 'inactive':
                require(state.get('ExecMainStatus') == '0' and state.get('Result') == 'success',
                        'Short service did not exit successfully')
                return state
        elif state['MainPID'] == '0' and state['ActiveState'] == 'inactive':
            raise ValueError('Short service ended without its exact safe boundary')
        now = time.monotonic()
        if now - last_report >= 1800:
            journal(f"V4 r2继续队列CPU等待：{status.get('status')}，新增更新"
                    f"{status.get('updates')}/2250；未启动GPU子进程或物理测试。")
            last_report = now
        time.sleep(30)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_rows(rows, schedule=SCHEDULE, boundary=BOUNDARY):
    require(len(rows) == boundary, 'Short log must contain exactly every committed update')
    summaries = {}
    keys = ('loss', 'mse', 'future8', 'future32', 'idm', 'chunk_vtt_loss',
            'chunk_huber', 'chunk_cosine', 'alpha_after_update', 'gradient_norm',
            'predictor_grad_norm', 'projection_grad_norm', 'alpha_grad_norm',
            'backbone_grad_norm', 'seconds_per_step', 'cuda_allocated_bytes',
            'cuda_reserved_bytes', 'cuda_peak_allocated_bytes')
    for update, row in enumerate(rows, 1):
        require(row.get('updates') == update and row.get('step') == 80000 + update,
                'Training update gap, duplicate or checkpoint-step mismatch')
        expected = ('predictor' if update <= schedule['predictor_steps'] else
                    'gate' if update <= schedule['gate_steps'] else 'joint')
        require(row.get('phase') == expected, 'Staged training phase changed')
        for key in keys:
            require(finite(row.get(key)), 'Nonfinite/missing training metric: ' + key)
        require(row['seconds_per_step'] > 0, 'Training duration is invalid')
        require(type(row.get('chunk_vtt_valid_samples')) is int
            and row['chunk_vtt_valid_samples'] >= 0, 'Predictor validity mask missing')
        if row['chunk_vtt_valid_samples']:
            require(row['predictor_grad_norm'] > 0, 'Supervised predictor has no gradient')
        rates = row.get('learning_rates', {})
        require(set(rates) == {'backbone', 'predictor', 'projection', 'alpha'}
            and all(finite(value) and value >= 0 for value in rates.values()),
            'Optimizer group learning rates are incomplete')
        require(rates['predictor'] > 0, 'Predictor was disabled')
        if expected != 'joint':
            require(row['backbone_grad_norm'] == 0 and rates['backbone'] == 0,
                    'Backbone freeze was violated')
        if expected == 'predictor':
            require(row['alpha_after_update'] == 0 and row['projection_grad_norm'] == 0
                and row['alpha_grad_norm'] == 0 and rates['alpha'] == 0 and rates['projection'] == 0,
                'Predictor-only phase changed fusion')
        else:
            require(rates['alpha'] > 0 and rates['projection'] > 0, 'Fusion learning was disabled')
        if expected == 'joint':
            require(rates['backbone'] > 0, 'Joint training never enabled its original backbone')
        bucket = summaries.setdefault(expected, dict(updates=0, predictor_positive=0,
            backbone_positive=0, projection_positive=0, alpha_positive=0, alpha_nonzero=0))
        bucket['updates'] += 1
        for field in ('predictor', 'backbone', 'projection', 'alpha'):
            bucket[field + '_positive'] += int(row[field + '_grad_norm'] > 0)
        bucket['alpha_nonzero'] += int(row['alpha_after_update'] != 0)
    for phase in ('gate', 'joint'):
        bucket = summaries.get(phase, {})
        require(bucket.get('projection_positive', 0) > 0 and bucket.get('alpha_positive', 0) > 0
            and bucket.get('alpha_nonzero', 0) > 0, 'Fusion did not actually learn in ' + phase)
    require(summaries.get('joint', {}).get('backbone_positive', 0) > 0,
            'Joint training had no original-backbone gradient')
    return summaries


def compare_diagnostics(initial, latest, count=8):
    report = {}
    for split in ('train', 'val'):
        old, new = initial[split], latest[split]
        require(old.get('count') == count and new.get('count') == count
            and len(old['rows']) == count and len(new['rows']) == count,
            'Fixed diagnostic sample count differs')
        for before, after in zip(old['rows'], new['rows']):
            identity = ('index', 'episode', 'anchor')
            require(all(before.get(key) == after.get(key) for key in identity),
                    'Diagnostic train/validation anchors changed')
            for key in old['means']:
                require(finite(before.get(key)) and finite(after.get(key)),
                        'Nonfinite diagnostic row: ' + key)
        for data in (old, new):
            for key, value in data['means'].items():
                require(finite(value), 'Nonfinite diagnostic mean')
                actual = sum(row[key] for row in data['rows']) / count
                require(math.isclose(value, actual, rel_tol=1e-7, abs_tol=1e-10),
                        'Diagnostic mean does not match its fixed rows')
        require(old['means']['chunk_vtt_loss'] > 0 and old['means']['mse'] > 0,
                'Initial diagnostic denominator is invalid')
        report[split] = dict(initial=old['means'], short=new['means'],
            chunk_loss_ratio=new['means']['chunk_vtt_loss'] / old['means']['chunk_vtt_loss'],
            action_mse_ratio=new['means']['mse'] / old['means']['mse'],
            anchors=[{key: row[key] for key in ('index', 'episode', 'anchor')} for row in old['rows']])
    require(report['val']['chunk_loss_ratio'] < 1.,
            'Fixed validation Chunk predictor did not improve; do not continue automatically')
    require(report['val']['action_mse_ratio'] <= 1.25,
            'Fixed validation action MSE deteriorated by more than 25%; inspect negative transfer')
    return dict(splits=report, gates=dict(validation_chunk_below_initial=True,
        maximum_validation_action_mse_ratio=1.25),
        interpretation='Offline stability admission only; task success requires the final simulator trials.')


def verify_sources(sources):
    for path, sha in sources.items():
        require(digest_file(path) == sha, 'Frozen source changed: ' + path)


def receipt(path):
    saved = read_json(path.with_name(path.name + '.sha256.json'))
    require(saved.get('size') == path.stat().st_size and saved.get('sha256') == digest_file(path),
            'Checkpoint byte receipt mismatch: ' + str(path))
    return dict(path=str(path), **saved)


def audit_optimizer(optimizer, boundary=BOUNDARY, schedule=SCHEDULE):
    import torch
    groups = optimizer.get('param_groups', [])
    require([g.get('name') for g in groups] == ['backbone', 'predictor', 'projection', 'alpha'],
            'Resumable optimizer groups changed')
    states = optimizer.get('state', {})
    ids = [index for group in groups for index in group['params']]
    require(ids and len(ids) == len(set(ids)) and set(states).issubset(ids),
            'Optimizer parameter identities are invalid or duplicated')
    expected_steps = dict(predictor=boundary, projection=boundary - schedule['predictor_steps'],
        alpha=boundary - schedule['predictor_steps'], backbone=boundary - schedule['gate_steps'])
    report = {}
    for group in groups:
        name = group['name']
        rate = schedule['backbone_lr'] if name == 'backbone' else schedule['new_lr']
        require(group.get('base_lr') == rate, 'Optimizer base learning rate changed')
        learned = [states[index] for index in group['params'] if index in states]
        require(learned, 'Optimizer group has no learned state: ' + name)
        if name != 'backbone':
            require(len(learned) == len(group['params']), 'New module optimizer state missing: ' + name)
        counts = []
        for state in learned:
            require(all(not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
                        for value in state.values()), 'Nonfinite optimizer tensor')
            require('exp_avg' in state and 'exp_avg_sq' in state
                and state['exp_avg'].shape == state['exp_avg_sq'].shape, 'Incomplete AdamW state')
            count = float(state.get('step', 0))
            require(count.is_integer() and 0 < count <= expected_steps[name],
                    'Optimizer step count differs from enabled phase: ' + name)
            if name != 'backbone':
                require(count == expected_steps[name], 'New module skipped enabled updates: ' + name)
            counts.append(int(count))
        require(max(counts) == expected_steps[name], 'Optimizer group missed its actual update count: ' + name)
        report[name] = dict(expected_enabled_updates=expected_steps[name], learned_states=len(learned),
            parameters=len(group['params']), minimum_step=min(counts), maximum_step=max(counts))
    return report


def frozen_backbone_identity(baseline, gate_boundary):
    import torch
    original, gate = baseline['model_state_dict'], gate_boundary['model_state_dict']
    require(set(original) == set(gate), 'Frozen-stage model keys differ from baseline')
    checked = 0
    for key, tensor in original.items():
        if not key.startswith('chunk_vtt16.'):
            require(torch.equal(tensor, gate[key]), 'Original backbone changed while frozen: ' + key)
            checked += 1
    require(checked > 0 and float(original['chunk_vtt16.alpha']) == 0.,
            'Frozen baseline is missing its original backbone or zero gate')
    return dict(backbone_tensors_bitwise_equal=checked, baseline_alpha=0.)


def audit_boundary(training, launch, service):
    """CPU-only exact full-state audit, after the trainer and its GPU lock exit."""
    import numpy as np
    import torch
    from harness import zhuhai_precision_v4_training as v4
    from harness import zhuhai_precision_v2_training as v2
    torch.set_num_threads(4)
    from scripts import gpu_lock
    validate_short_status(check_no_training_failure(training))
    require(gpu_lock.owner() is None, 'GPU lock is still held after short training')
    verify_sources(launch['source_hashes'])
    manifest = read_json(training / 'manifest.json')
    contract = v4.validate_training_contract(manifest['contract'])
    require(manifest['contract_sha256'] == v2.digest(contract), 'Manifest digest mismatch')
    require(contract['v4_training']['schedule'] == SCHEDULE, 'Formal schedule must remain exactly unchanged')
    verify_sources(contract['source_files'])
    initializer = contract['v4_training']['initialization']
    require(digest_file(initializer['path']) == initializer['sha256'],
            'Original immutable 80k initializer changed')
    rows = [json.loads(line, parse_constant=reject_constant)
            for line in (training / 'log.jsonl').read_text().splitlines()]
    phases = audit_rows(rows)
    model_path = training / f'checkpoint_step_{80000 + BOUNDARY:06d}.pt'
    full_path = training / 'checkpoint_last.pt'
    checkpoints = [receipt(path) for path in (model_path, full_path)]
    model = torch.load(model_path, map_location='cpu', weights_only=False, mmap=True)
    full = torch.load(full_path, map_location='cpu', weights_only=False, mmap=True)
    for saved, record, model_only in zip((model, full), checkpoints, (True, False)):
        require(saved.get('step') == record.get('step') == 80000 + BOUNDARY
            and saved.get('updates') == record.get('updates') == BOUNDARY
            and saved.get('origin_step') == 80000 and saved.get('format') == v4.FORMAT
            and saved.get('model_only') is record.get('model_only') is model_only
            and saved.get('contract') == contract and saved.get('contract_sha256') == v2.digest(contract),
            'Checkpoint exact step/full-state/contract identity mismatch')
    states = full['model_state_dict']
    v4.validate_model_identity(states)
    require(set(states) == set(model['model_state_dict']), 'Model/full-state parameter keys differ')
    for key, tensor in states.items():
        require(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all())
            and torch.equal(tensor, model['model_state_dict'][key]),
            'Model/full-state tensor mismatch or nonfinite: ' + key)
    require(float(states['chunk_vtt16.alpha']) == rows[-1]['alpha_after_update']
        and float(states['chunk_vtt16.alpha']) != 0., 'Saved alpha differs from last actual update')
    optimizer = full.get('optimizer_state_dict', {})
    optimizer_audit = audit_optimizer(optimizer)
    previous_path, baseline_path = training / 'checkpoint_previous.pt', training / 'checkpoint_step_080000.pt'
    previous_receipt, baseline_receipt = receipt(previous_path), receipt(baseline_path)
    previous = torch.load(previous_path, map_location='cpu', weights_only=False, mmap=True)
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=False, mmap=True)
    for saved, rec, updates, model_only in ((previous, previous_receipt, 2000, False),
            (baseline, baseline_receipt, 0, True)):
        require(saved.get('updates') == rec.get('updates') == updates
            and saved.get('step') == rec.get('step') == 80000 + updates
            and saved.get('origin_step') == 80000 and saved.get('format') == v4.FORMAT
            and saved.get('contract') == contract and saved.get('contract_sha256') == v2.digest(contract)
            and saved.get('model_only') is rec.get('model_only') is model_only,
            'Previous/baseline optimizer boundary identity mismatch')
    frozen = frozen_backbone_identity(baseline, previous)
    frozen.update(baseline=baseline_receipt, gate_boundary=previous_receipt)
    rng = np.random.default_rng()
    rng.bit_generator.state = full['rng']
    generator = torch.Generator(device='cpu')
    generator.set_state(full['torch_rng'])
    cuda_states = full.get('cuda_rng')
    require(isinstance(cuda_states, list) and bool(cuda_states)
        and all(isinstance(t, torch.Tensor) and t.dtype == torch.uint8 and t.ndim == 1
                and t.numel() > 0 for t in cuda_states), 'Resumable CUDA RNG states missing')
    diagnostic = training / f'diagnostic_{80000 + BOUNDARY:06d}.json'
    comparison = compare_diagnostics(read_json(training / 'initial_diagnostic.json'), read_json(diagnostic))
    require(not torch.cuda.is_initialized(), 'CPU boundary audit unexpectedly initialized CUDA')
    return dict(status='passed', boundary_updates=BOUNDARY, checkpoint_step=80000 + BOUNDARY,
        service=service, checkpoints=checkpoints, optimizer_states=len(optimizer.get('state', {})),
        optimizer_update_counts=optimizer_audit, actual_first_2000_backbone_freeze=frozen,
        model_tensors=len(states), rng_checked=True, cuda_initialized=False,
        phases=phases, diagnostics=comparison, log_sha256=digest_file(training / 'log.jsonl'),
        manifest_sha256=digest_file(training / 'manifest.json'),
        diagnostic_sha256=digest_file(diagnostic), original_sources_verified=True)


def commands():
    train = ['/usr/bin/python3', str(ROOT / 'scripts/with_project_gpu.py'), '--',
        '/usr/bin/python3', str(ROOT / 'scripts/gpu_lock.py'), 'run', '--owner', TRAIN_OWNER,
        '--max-wait', '1', '--', str(PYTHON), '-B', str(ROOT / 'scripts/train_zhuhai_wam_precision_v4.py'),
        '--out', str(TRAINING), '--steps', '50000', '--predictor-steps', '1000', '--gate-steps', '2000',
        '--batch-size', '4', '--save-every', '2500', '--resume-every', '500',
        '--diagnostic-samples', '8', '--backbone-lr', '1e-6', '--new-lr', '1e-4',
        '--gpu-lock-owner', TRAIN_OWNER, '--resume', '--execute']
    evaluation = ['/usr/bin/python3', str(ROOT / 'scripts/with_project_gpu.py'), '--',
        str(PYTHON), '-B', str(ROOT / 'scripts/eval_precision_v4_final5.py'),
        '--training', str(TRAINING), '--out', str(EVALUATION), '--gpu-lock-owner', EVAL_OWNER, '--execute']
    return train, evaluation


def run_child(command, label, run, sources):
    verify_sources(sources)
    write_once(run / (label + '_attempt.json'), dict(command=command, automatic_retry=False))
    with (run / (label + '_stdout.log')).open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1'),
            start_new_session=True)
        write_once(run / (label + '_started.json'), dict(pid=child.pid))
        last_report = time.monotonic()
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    status_path = TRAINING / 'status.json' if label == 'training' else EVALUATION / 'summary.json'
                    status = read_json(status_path) if status_path.exists() else {}
                    latest(run, 'evaluating' if label == 'evaluation' else label,
                           dict(pid=child.pid, child_status=status))
                    if time.monotonic() - last_report >= 1800:
                        journal(f"V4 r2 {label}仍运行，pid={child.pid}，status={status.get('status')}，"
                                f"updates={status.get('updates')}；无自动重试。")
                        last_report = time.monotonic()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Trainer owns safe-boundary stopping; keep GPU participant/lock
                # intact rather than interrupting a checkpoint write with SIGKILL.
                write_once(run / (label + '_child_retained_guard.json'), dict(pid=child.pid,
                    reason='Child still stopping after 30s; no forced kill or retry'))
            raise
    write_once(run / (label + '_exit.json'), dict(returncode=code))
    require(code == 0, label + ' child failed; preserve guard and do not retry')
    verify_sources(sources)
    return code


def execute(run):
    from scripts import gpu_lock
    # Permanent unique directory makes service restart harmless even after a crash.
    run.mkdir(parents=False, exist_ok=False)
    with (run / '.writer.lock').open('x') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train, evaluation = commands()
        launch = read_json(VERIFY / 'r2_launch_and_admission.json')
        sources = dict(launch['source_hashes'])
        sources[str(Path(__file__).resolve())] = digest_file(Path(__file__).resolve())
        write_once(run / 'plan.json', dict(training=str(TRAINING), evaluation=str(EVALUATION),
            short_unit=SHORT_UNIT, boundary_updates=BOUNDARY, training_command=train,
            evaluation_command=evaluation, frozen_sources=sources, retries=0,
            scored_record_video=False, success_video_directory=str(ROOT / 'success video')))
        try:
            require(not EVALUATION.exists(), 'Evaluation output already exists; never repeat its physical attempts')
            verify_sources(sources)
            latest(run, 'waiting_short')
            service = wait_short(TRAINING, lambda status: latest(run, 'waiting_short', status))
            latest(run, 'audit')
            audit = audit_boundary(TRAINING, launch, service)
            write_once(run / 'short_admission_audit.json', audit)
            journal('V4 r2短跑2250更新CPU审查通过：精确模型/AdamW/RNG、冻结阶段和固定验证锚点通过；开始原计划50k续训。')
            latest(run, 'training')
            run_child(train, 'training', run, sources)
            status = check_no_training_failure(TRAINING)
            require(status.get('status') == 'completed' and status.get('updates') == 50000
                and status.get('step') == 130000 and status.get('target_updates') == 50000
                and status.get('target_step') == 130000, 'Trainer exited before formal completion')
            require(gpu_lock.owner() is None, 'Training GPU owner did not release; do not evaluate')
            require(not EVALUATION.exists(), 'Evaluation output appeared; never repeat physical attempts')
            journal('V4 r2正式50k续训已退出且GPU锁释放；启动末5000步内两个权重的同种子各5次真实仿真。')
            latest(run, 'evaluating')
            run_child(evaluation, 'evaluation', run, sources)
            summary = read_json(EVALUATION / 'summary.json')
            require(summary.get('status') == 'completed', 'Final physical evaluation incomplete')
            require(gpu_lock.owner() is None, 'Evaluation GPU owner was retained; inspect surviving child')
            write_once(run / 'completed.json', dict(status='completed', training_status=status,
                evaluation_summary=str(EVALUATION / 'summary.json'),
                evaluation_summary_sha256=digest_file(EVALUATION / 'summary.json'), physical_retries=0))
            journal('V4 r2训练与固定10次真实仿真完成；成功录像仅从原成功试次机载帧导出，未增加物理尝试。')
            latest(run, 'completed')
            return 0
        except BaseException as exc:
            write_once(run / 'failure_guard.json', dict(error=f'{type(exc).__name__}: {exc}',
                preserved=True, automatic_retry=False, physical_retries=0))
            latest(run, 'failed', dict(error=f'{type(exc).__name__}: {exc}'))
            journal(f'V4 r2继续队列停止：{type(exc).__name__}: {exc}；证据保留，无自动失败恢复/重试。')
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=VERIFY / 'r2_continuation_supervisor')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    run = args.out.absolute()
    require(run == run.resolve() and run.parent == VERIFY, 'Use a fresh direct verification child directory')
    if not args.execute:
        train, evaluation = commands()
        print(json.dumps(dict(status='plan', out=str(run), short_unit=SHORT_UNIT,
            training_command=train, evaluation_command=evaluation, boundary_updates=BOUNDARY), indent=2))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}; no automatic restart')
    signal.signal(signal.SIGTERM, interrupted)
    return execute(run)


if __name__ == '__main__':
    raise SystemExit(main())
