#!/usr/bin/env python3
"""CPU/read-only verifier for the frozen 20261010 Precision V2 run.

No training imports, model construction, policy, logger, locks or resume. Only
committed manifest/checkpoints/receipts and frozen sources are read. Exit codes:
0 verified, 1 invalid, 2 pending (retry after publication), 3 verifier error.
SHA scans checkpoint bytes; mmap avoids copying tensor storages into RAM.
Only a new, step-matched --out audit JSON in the authorized verification directory may be written; existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / 'outputs/zhuhai2026_tasks12/verification/precision_v2_overnight_20261009/checkpoint_verifier_review.json'
AUDIT_DIR = REVIEW_PATH.parent
FORMAT = 'zhuhai-precision-wam-native16-measured-10hz-v2'
STEPS = (5000, 20000, 25000, 30000)
# Original manifest bindings, not regenerated from possibly drifted source files.
SOURCE_DIGEST = 'e550ca8d303ac13174fa88b32fbd070bb4eb6d91022e63fdf9afb3917a24e4e9'
PRECISION_DIGEST = '96cffd3a367bddcd0e859e82c1106d5b8ce0f2e9c4d486035cd2b352b2e23fb1'


class Invalid(ValueError):
    pass


class Pending(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def digest(value):
    # EXACT frozen trainer encoding (not compact/canonical JSON).
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    try:
        s = path.lstat()
    except FileNotFoundError:
        return None
    require(stat.S_ISREG(s.st_mode), f'Not a regular nonsymlink file: {path}')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def read_json(path):
    before = fingerprint(path)
    if before is None:
        raise Pending(f'Not yet published: {path.name}')
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, FileNotFoundError):
        if fingerprint(path) != before:
            raise Pending(f'Changed while reading: {path.name}')
        raise Invalid(f'Invalid JSON: {path.name}')
    if fingerprint(path) != before:
        raise Pending(f'Changed while reading: {path.name}')
    require(isinstance(value, dict), f'JSON object required: {path.name}')
    return value, before


def file_sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def cpu_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This verifier process ONLY.
    import torch
    assert torch.cuda.is_initialized() is False, 'CUDA must remain uninitialized'
    return torch


def validate_contract(manifest):
    c = manifest.get('contract')
    require(isinstance(c, dict), 'Missing contract')
    require(manifest.get('contract_sha256') == digest(c), 'Manifest contract digest mismatch')
    require(c.get('format') == FORMAT and c.get('mode') == 'train', 'Wrong contract format/mode')
    cfg, training = c['config'], c['config']['training']
    for key, value in dict(max_steps=30000, schedule_steps=30000, batch_size=4,
                           grad_accum_steps=1, save_every=5000, warmup_steps=100,
                           learning_rate=1e-5, lr_min=1e-6, weight_decay=.01,
                           betas=[.9, .99]).items():
        require(training.get(key) == value, f'Frozen training contract differs: {key}')
    for key, value in dict(action_dim=16, state_dim=16, action_chunk_size=32,
                           action_execution_horizon=8).items():
        require(cfg['common'].get(key) == value, f'Not native16/chunk32/exec8: {key}')
    require(cfg['model'].get('proprio_mask') == 'native_all16', 'Wrong proprio mask')
    require(digest(c['precision']) == PRECISION_DIGEST, 'Wrong Precision V2 schema/settings')
    require(cfg['model'].get('precision') == c['precision'], 'Precision config alias differs')
    h = c['hjepa']
    require(h.get('variant') == 'combined' and h.get('task_scope') == ['zhuhai_task2'], 'Wrong variant/task')
    require(h['settings'].get('multi_horizon') is True and h['settings'].get('use_idm') is True,
            'Combined objectives disabled')
    require(training['auxiliary_loss'] == dict(future=.25, future8=.25, idm=.1,
                                               gate_kl=0., phase_ce=0., log_dist=0.), 'Wrong loss contract')
    for name in ('robot_contract', 'data_contract', 'statistics', 'idm_statistics', 'action_postprocessing'):
        require(c.get(name + '_sha256') == digest(c[name]), f'{name} digest mismatch')
    data = c['data_contract']
    for key, value in dict(schema='zhuhai-native16-measured-target-10hz-v1',
                           robot_dof=16, state_dim=16, action_dim=16, frequency_hz=10.,
                           gripper_indices=[7, 15], state_semantics='measured_joint_gripper',
                           action_semantics='joint_position_target').items():
        require(data.get(key) == value, f'Native16 data contract differs: {key}')
    for name in ('robot_contract', 'data_contract'):
        require(cfg.get(name) == c[name] and cfg.get(name + '_sha256') == c[name + '_sha256'],
                f'{name} config alias differs')
    require(c['protocol'].get('pure_wam') is True, 'Not pure-WAM')
    for key in ('record_video', 'third_view_camera_created', 'vlm', 'high_level_planner', 'ask', 'lifelong_memory_writes'):
        require(c['protocol'].get(key) is False, f'Unexpected online protocol: {key}')
    sources = c['source_files']
    require(isinstance(sources, dict) and len(sources) == 33 and digest(sources) == SOURCE_DIGEST,
            'Original 33-source hash binding changed')
    require(h.get('code_hashes') == sources, 'Source hash alias differs')
    return c


def verify_sources(sources):
    for name, expected in sources.items():
        path = Path(name)
        before = fingerprint(path)
        require(before is not None, f'Missing frozen source: {name}')
        with path.open('rb') as stream:
            actual = file_sha(stream)
        require(fingerprint(path) == before and actual == expected, f'Frozen source drift: {name}')
    return {'status': 'verified', 'count': len(sources), 'binding_sha256': digest(sources)}


def model_metadata(state, c, torch):
    require(isinstance(state, dict) and bool(state), 'Missing/empty model_state_dict')
    metadata = {}
    for name, tensor in state.items():
        require(isinstance(name, str) and bool(name) and isinstance(tensor, torch.Tensor), 'Invalid model key/tensor')
        require(tensor.device.type == 'cpu' and tensor.layout == torch.strided, f'Non-CPU/dense tensor: {name}')
        require(not name.startswith(('phase_net.', 'gate_head.', 'phase_head.', 'dist_head.')), 'Removed V1 head present')
        metadata[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'stride': list(tensor.stride())}
    version, config = state.get('precision.version'), state.get('precision.configuration')
    require(version is not None and version.dtype == torch.int64 and version.shape == () and version.item() == 2,
            'Wrong/missing V2 version buffer')
    require(config is not None and config.dtype == torch.int64 and tuple(config.shape) == (5,)
            and config.tolist() == [16, 8, 0, 1, 2], 'Wrong/missing V2 configuration buffer')
    hidden = c['config']['model']['action_expert']['hidden_size']
    for key, shape in {'precision.clock.weight': (hidden, 6), 'precision.residual.weight': (16, hidden),
                       'precision.patch_xy': (1, 1, 196, 2)}.items():
        require(key in state and tuple(state[key].shape) == shape, f'V2 shape mismatch: {key}')
    return metadata


def expected_lr(training, step):
    # Saved step is AFTER optimizer.step(); the last LR used update index step-1.
    i = step - 1
    warmup, total = training['warmup_steps'], training['max_steps']
    require(type(step) is int and 1 <= step <= total, 'Invalid optimizer boundary')
    factor = min(1., (i + 1) / max(1, warmup)) if i < warmup else .1 + .9 * (1 + math.cos(math.pi * (i - warmup) / max(1, total - warmup))) / 2
    return training['learning_rate'] * factor


def validate_full_state(saved, c, metadata, torch):
    opt = saved.get('optimizer_state_dict')
    require(isinstance(opt, dict) and isinstance(opt.get('state'), dict) and bool(opt['state']), 'Missing AdamW state')
    groups = opt.get('param_groups')
    require(isinstance(groups, list) and bool(groups), 'Missing AdamW parameter groups')
    lr = expected_lr(c['config']['training'], saved['step'])
    ids, lrs = [], []
    for group in groups:
        require(isinstance(group, dict) and isinstance(group.get('params'), list) and bool(group['params']), 'Invalid AdamW param group')
        ids.extend(group['params'])
        actual = group.get('lr')
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, lr, rel_tol=1e-10, abs_tol=1e-15),
                'Last parameter-group LR differs from absolute full-30k schedule')
        require(list(group.get('betas', [])) == [.9, .99] and group.get('weight_decay') == .01
                and group.get('eps') == 1e-8 and group.get('amsgrad') is False and group.get('maximize') is False,
                'Wrong AdamW group settings')
        lrs.append(actual)
    require(all(type(i) is int for i in ids) and len(set(ids)) == len(ids), 'Invalid/duplicate optimizer parameter IDs')
    require(set(opt['state']).issubset(ids), 'Orphan optimizer state')
    shapes = {(tuple(v['shape']), v['dtype']) for v in metadata.values()}
    for pid, state in opt['state'].items():
        require(type(pid) is int and isinstance(state, dict), 'Invalid optimizer state entry')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor = state.get(name)
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and tensor.is_floating_point(), f'Missing AdamW {name}')
            require((tuple(tensor.shape), str(tensor.dtype)) in shapes, f'AdamW moment has no matching model shape: {pid}')
        require(state['exp_avg'].shape == state['exp_avg_sq'].shape and state['exp_avg'].dtype == state['exp_avg_sq'].dtype,
                'AdamW moments disagree')
        t = state.get('step')
        require(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and t.numel() == 1 and t.item() == saved['step'],
                'AdamW update boundary differs')
    def byte_rng(value):
        return isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0
    require(byte_rng(saved.get('torch_rng')), 'Missing/invalid torch RNG bytes')
    cuda_rng = saved.get('cuda_rng')
    require(isinstance(cuda_rng, list) and bool(cuda_rng) and all(byte_rng(x) for x in cuda_rng), 'Missing/invalid CUDA RNG bytes')
    rng = saved.get('rng')
    require(isinstance(rng, dict) and rng.get('bit_generator') == 'PCG64' and isinstance(rng.get('state'), dict), 'Missing/invalid numpy RNG dict')
    require(all(type(rng['state'].get(k)) is int and 0 <= rng['state'][k] < 2**128 for k in ('state', 'inc'))
            and rng.get('has_uint32') in (0, 1) and type(rng.get('uinteger')) is int and 0 <= rng['uinteger'] < 2**32,
            'Invalid numpy PCG64 state')
    return {'optimizer_schema': 'AdamW (class inferred from frozen trainer; no class tag in state_dict)',
            'state_entries': len(opt['state']), 'parameter_ids': len(ids),
            'uninitialized_parameter_ids': len(ids) - len(opt['state']),
            'last_update_index': saved['step'] - 1, 'schedule_steps': 30000,
            'expected_last_lr': lr, 'parameter_group_lrs': lrs,
            'torch_rng_bytes': saved['torch_rng'].numel(), 'cuda_rng_bytes': [x.numel() for x in cuda_rng],
            'numpy_rng': 'PCG64', 'resume_executed': False}


def verify_snapshot(path, c, requested_step, model_only, torch):
    receipt_path = path.with_name(path.name + '.sha256.json')
    before = fingerprint(path)
    receipt, receipt_stamp = read_json(receipt_path)
    if before is None:
        raise Pending(f'Not yet published: {path.name}')
    def unchanged():
        return fingerprint(path) == before and fingerprint(receipt_path) == receipt_stamp
    try:
        require(type(receipt.get('step')) is int, 'Receipt step must be an integer')
        require(receipt.get('model_only') is model_only, 'Wrong receipt model_only')
        require(type(receipt.get('size')) is int and receipt['size'] > 0, 'Invalid receipt size')
        require(isinstance(receipt.get('sha256'), str) and len(receipt['sha256']) == 64, 'Invalid receipt SHA')
        with path.open('rb') as stream:
            s = os.fstat(stream.fileno())
            if (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) != before:
                raise Pending('Checkpoint replaced before hash')
            actual = file_sha(stream)
            if actual != receipt['sha256'] or before[2] != receipt['size']:
                # New weights can be fully renamed while the OLD receipt is still
                # visible during a multi-GB hash scan. Never call that corruption.
                if (not unchanged() or before[3] >= receipt_stamp[3]
                        or before[4] > receipt_stamp[4]):
                    raise Pending('Checkpoint/receipt publication gap; retry after receipt commit')
                raise Invalid('Checkpoint SHA/size differs from committed receipt')
            if not unchanged():
                raise Pending('Checkpoint/receipt changed during SHA scan')
            # Pin the exact hashed inode across a pathname replacement. mmap=True
            # requires a filename; Linux /proc/self/fd supplies the read-only FD.
            try:
                saved = torch.load(f'/proc/self/fd/{stream.fileno()}', map_location='cpu', mmap=True, weights_only=True)
            except (RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
                raise Invalid(f'Unreadable checkpoint schema: {type(exc).__name__}: {exc}') from exc
            assert torch.cuda.is_initialized() is False
            require(isinstance(saved, dict), 'Checkpoint object must be a dict')
            require(saved.get('format') == FORMAT and saved.get('model_only') is model_only, 'Wrong checkpoint format/model_only')
            step = saved.get('step')
            require(type(step) is int and 1 <= step <= 30000 and step == receipt['step'], 'Checkpoint/receipt step mismatch')
            if model_only:
                require(step == requested_step, 'Wrong milestone step')
                require(not any(k in saved for k in ('optimizer_state_dict', 'rng', 'torch_rng', 'cuda_rng')), 'Model-only contains full-state fields')
            require(saved.get('contract') == c and saved.get('contract_sha256') == digest(c), 'Checkpoint/manifest contract mismatch')
            metadata = model_metadata(saved.get('model_state_dict'), c, torch)
            report = dict(status='verified', path=str(path), step=step, model_only=model_only,
                          sha256=actual, size=before[2], format=FORMAT, contract_sha256=digest(c),
                          model_key_count=len(metadata), model_metadata_sha256=digest(metadata),
                          model_state_metadata=metadata, model_only_cannot_resume=model_only)
            if not model_only:
                report['full_state'] = validate_full_state(saved, c, metadata, torch)
            if not unchanged():
                raise Pending('Checkpoint/receipt changed during metadata verification')
            return report
    except Exception:
        if not unchanged():
            raise Pending('Checkpoint/receipt changed during verification') from None
        raise


def verify(run, step):
    torch = cpu_torch()
    report = {'schema': 'precision-v2-cpu-checkpoint-verification-v1', 'requested_step': step,
              'checked_at': datetime.now(timezone.utc).isoformat(), 'read_only_inputs': True,
              'limitations': ['No model construction, strict load, resume or policy execution.',
                              'Model keys/shapes/dtypes are compared between snapshots plus V2 identity checks; not an independent complete architecture oracle.',
                              'No all-weight/moment finite scan or RNG replay; no physical score or training-health conclusion.',
                              'Unmatched newer weights with an older receipt remain pending, including ambiguous post-publication edits.']}
    try:
        require(type(step) is int and step in STEPS, 'Unsupported requested milestone')
        run = Path(run).resolve()
        directory = run if (run / 'manifest.json').is_file() else run / 'train'
        report['run'] = str(run)
        report['checkpoint_directory'] = str(directory)
        manifest_path = directory / 'manifest.json'
        manifest, stamp = read_json(manifest_path)
        c = validate_contract(manifest)
        report['manifest'] = {'status': 'verified', 'contract_sha256': digest(c)}
        report['sources'] = verify_sources(c['source_files'])
        for key, path, only in [('model_only', directory / f'checkpoint_step_{step:06d}.pt', True),
                                ('full_state', directory / 'checkpoint_last.pt', False)]:
            try:
                report[key] = verify_snapshot(path, c, step, only, torch)
            except Pending as exc:
                report[key] = {'status': 'pending', 'path': str(path), 'reason': str(exc)}
                raise
        model, full = report['model_only'], report['full_state']
        require(model['model_state_metadata'] == full['model_state_metadata'], 'Model state keys/shapes/dtypes differ between snapshots')
        report['model_metadata_consistent'] = True
        report['full_state_step_relation'] = 'same' if full['step'] == step else ('newer' if full['step'] > step else 'older')
        report['full_state_is_requested_milestone'] = full['step'] == step
        if full['step'] < step:
            raise Pending('Full-state receipt has not caught up to the requested milestone')
        if fingerprint(manifest_path) != stamp:
            raise Pending('Manifest changed during verification')
        # Frozen source hash scans are small; repeat to detect drift during a long
        # checkpoint SHA scan without touching any other training files.
        verify_sources(c['source_files'])
        report['status'] = 'verified'
    except Pending as exc:
        report.update(status='pending', reason=str(exc))
    except (Invalid, KeyError, TypeError, ValueError) as exc:
        report.update(status='invalid', reason=f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        report.update(status='error', reason=f'{type(exc).__name__}: {exc}')
    finally:
        assert torch.cuda.is_initialized() is False, 'Verifier initialized CUDA'
        report['cuda_initialized'] = False
    return report


def validate_output_path(path, step):
    """Validate without creating anything; main calls this BEFORE reading weights."""
    require(type(step) is int and step in STEPS, 'An explicit supported audit step is required')
    path = Path(path).absolute()
    require('..' not in path.parts and path.parent == AUDIT_DIR,
            'Audit output must be directly inside the authorized verification directory')
    require(path.parent.is_dir() and path.parent.resolve() == AUDIT_DIR,
            'Audit directory must already exist and have no symlink components')
    pattern = rf'checkpoint_{step:06d}_audit(?:_attempt[1-9][0-9]*)?\.json'
    require(re.fullmatch(pattern, path.name) is not None,
            f'Output must match step {step}: checkpoint_{step:06d}_audit[_attemptN].json; review is protected')
    require(not os.path.lexists(path),
            'Output already exists (including links); preserve it and use a new explicit _attemptN suffix')
    return path


def write_report(path, report):
    path = validate_output_path(path, report.get('requested_step'))
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    directory_fd = os.open(AUDIT_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # O_EXCL is essential: a second writer/link appearing after validation
        # must fail, never truncate the existing evidence. No mkdir or lock.
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o644, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Report must not alias another file')
        with os.fdopen(fd, 'w', encoding='utf-8', closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Run root or its train directory')
    parser.add_argument('--step', type=int, choices=STEPS, required=True)
    parser.add_argument('--out', type=Path, required=True,
                        help=f'New {AUDIT_DIR}/checkpoint_STEP_audit[_attemptN].json; STEP is six digits matching --step')
    args = parser.parse_args(argv)
    try:
        validate_output_path(args.out, args.step)
    except (Invalid, OSError) as exc:
        parser.error(str(exc))
    report = verify(args.run, args.step)
    try:
        write_report(args.out, report)
    except (Invalid, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({'status': report['status'], 'out': str(args.out), 'reason': report.get('reason')}, ensure_ascii=False))
    return {'verified': 0, 'invalid': 1, 'pending': 2, 'error': 3}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
