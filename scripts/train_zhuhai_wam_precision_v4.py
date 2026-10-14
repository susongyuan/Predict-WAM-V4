#!/usr/bin/env python3
"""Staged V4 warm start on the existing fixed task; resumable optimizer boundaries."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import zhuhai_precision_v2_training as v2
from scripts import train_zhuhai_wam_precision_v2 as t2
from scripts.train_zhuhai_wam_precision_v3 import CK80K, CK80K_SHA
from scripts.progress1_log_v41 import log

ORIGIN_STEP = 80000


def phase_for(step, predictor_steps, gate_steps):
    return 'predictor' if step < predictor_steps else 'gate' if step < gate_steps else 'joint'


def configure_phase(optimizer, phase, update, schedule):
    # Preserve the optimizer and original frozen parameters across phase changes.
    joint_index = max(0, update - schedule['gate_steps'])
    joint_length = schedule['steps'] - schedule['gate_steps']
    decay = .1 + .9 * (1 + math.cos(math.pi * joint_index / max(1, joint_length))) / 2
    for group in optimizer.param_groups:
        name = group['name']
        enabled = name == 'predictor' or (name in ('projection', 'alpha') and phase != 'predictor') or phase == 'joint'
        for parameter in group['params']:
            parameter.requires_grad_(enabled)
        ramp = min(1., (joint_index + 1) / 200) if name == 'backbone' else 1.
        group['lr'] = group['base_lr'] * decay * ramp if enabled else 0.


def make_optimizer(model, training, schedule):
    import torch
    groups = []
    chunk = model.chunk_vtt16
    chunk_ids = {id(p) for p in chunk.parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in chunk_ids]
    for name, params, rate in (
            ('backbone', backbone, schedule['backbone_lr']),
            ('predictor', list(chunk.predictor.parameters()), schedule['new_lr']),
            ('projection', list(chunk.projection.parameters()), schedule['new_lr']),
            ('alpha', [chunk.alpha], schedule['new_lr'])):
        groups.append(dict(name=name, params=params, lr=rate, base_lr=rate,
                           weight_decay=0. if name == 'alpha' else training['weight_decay']))
    return torch.optim.AdamW(groups, betas=tuple(training['betas']))


def gradient_norm(parameters):
    import torch
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def save_checkpoint(out, wrapper, contract, updates, *, optimizer=None, rng=None):
    import torch
    from scripts.train_wam_v3 import cpu_tree
    from harness import zhuhai_precision_v4_training as v4
    step = ORIGIN_STEP + updates
    milestone = optimizer is None
    path = out / (f'checkpoint_step_{step:06d}.pt' if milestone else 'checkpoint_last.pt')
    if milestone and path.exists():
        receipt_path = path.with_name(path.name + '.sha256.json')
        receipt = v2.read_json(receipt_path) if receipt_path.exists() else None
        if receipt is not None and (receipt['sha256'] != v2.sha256_file(path) or receipt['step'] != step
                or receipt.get('size') != path.stat().st_size or receipt.get('model_only') is not True):
            raise ValueError('Existing immutable checkpoint changed')
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        current = wrapper.action_model.state_dict()
        if (saved.get('format') != v4.FORMAT or saved.get('model_only') is not True
                or saved.get('step') != step or saved.get('origin_step') != ORIGIN_STEP
                or saved.get('contract_sha256') != v2.digest(contract)
                or saved.get('contract') != contract or saved.get('updates') != updates
                or set(saved['model_state_dict']) != set(current)
                or any(not torch.equal(value.detach().cpu(), saved['model_state_dict'][key])
                       for key, value in current.items())):
            raise ValueError('Existing immutable checkpoint differs from current model boundary')
        if receipt is None:
            v2.write_json(receipt_path, dict(sha256=v2.sha256_file(path), size=path.stat().st_size,
                step=step, updates=updates, model_only=True, recovered_missing_receipt=True))
        return path
    value = dict(format=v4.FORMAT, step=step, updates=updates, origin_step=ORIGIN_STEP,
                 model_only=milestone, contract=contract, contract_sha256=v2.digest(contract),
                 model_state_dict=cpu_tree(wrapper.action_model.state_dict()))
    if not milestone:
        value.update(optimizer_state_dict=cpu_tree(optimizer.state_dict()), rng=rng,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    if not milestone and path.exists():
        # Keep the last committed pair across a crash between model and receipt writes.
        previous = out / 'checkpoint_previous.pt'
        _load_resume_path(path, contract)
        link = previous.with_name(previous.name + '.tmp')
        link.unlink(missing_ok=True)
        os.link(path, link)
        link.replace(previous)
        v2.write_json(previous.with_name(previous.name + '.sha256.json'),
                      v2.read_json(path.with_name(path.name + '.sha256.json')))
    temporary.replace(path)
    v2.write_json(path.with_name(path.name + '.sha256.json'),
                  dict(sha256=v2.sha256_file(path), size=path.stat().st_size,
                       step=step, updates=updates, model_only=milestone))
    return path


def _load_resume_path(path, contract):
    import torch
    receipt = v2.read_json(path.with_name(path.name + '.sha256.json'))
    if receipt['sha256'] != v2.sha256_file(path) or receipt['size'] != path.stat().st_size:
        raise ValueError('Resume checkpoint receipt mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    updates = saved.get('updates')
    if (type(updates) is not int or not 0 <= updates <= contract['v4_training']['schedule']['steps']
            or saved.get('contract') != contract or saved.get('contract_sha256') != v2.digest(contract)
            or saved.get('format') != contract['format'] or saved.get('model_only') is not False
            or receipt.get('model_only') is not False or saved.get('origin_step') != ORIGIN_STEP
            or saved.get('step') != ORIGIN_STEP + saved.get('updates', -1)
            or receipt.get('step') != saved['step'] or receipt.get('updates') != saved['updates']):
        raise ValueError('Exact V4 full-state optimizer boundary required')
    return saved


def load_resume(out, contract):
    path = out / 'checkpoint_last.pt'
    try:
        return _load_resume_path(path, contract)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        previous = out / 'checkpoint_previous.pt'
        saved = _load_resume_path(previous, contract)
        archive = out / f'checkpoint_interrupted_{time.time_ns()}.pt'
        preserved = []
        for source, target in ((path, archive),
                (path.with_name(path.name + '.sha256.json'), archive.with_name(archive.name + '.sha256.json'))):
            if source.exists():
                source.replace(target)
                preserved.append(str(target))
        saved['resume_recovery'] = dict(source=str(previous), reason=str(exc), preserved=preserved)
        return saved


def reconcile_resume_log(out, updates):
    """Keep interrupted output as evidence before replaying from its saved RNG boundary."""
    path = out / 'log.jsonl'
    raw = path.read_bytes() if path.exists() else b''
    committed, previous = 0, 0
    for line in raw.splitlines(keepends=True):
        if previous == updates:
            break
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError('Training log is incomplete before the saved boundary') from exc
        if (row.get('updates') != previous + 1
                or row.get('step') != ORIGIN_STEP + previous + 1):
            raise ValueError('Training log does not reach the saved optimizer boundary in order')
        previous += 1
        committed += len(line)
    if previous != updates:
        raise ValueError('Training log is missing committed optimizer updates')
    tail = raw[committed:]
    if not tail:
        return
    suffix = hashlib.sha256(tail).hexdigest()[:12]
    archive = out / f'log_uncommitted_after_{ORIGIN_STEP + updates:06d}_{suffix}.jsonl'
    if archive.exists():
        if archive.read_bytes() != tail:
            raise ValueError('Interrupted log archive changed')
    else:
        with archive.open('xb') as stream:
            stream.write(tail); stream.flush(); os.fsync(stream.fileno())
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(raw[:committed]); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def restore_state(optimizer, rng, saved):
    import torch
    expected = [(g['name'], len(g['params']), g['base_lr']) for g in optimizer.param_groups]
    incoming = [(g.get('name'), len(g['params']), g.get('base_lr'))
                for g in saved['optimizer_state_dict']['param_groups']]
    if incoming != expected:
        raise ValueError('Resume optimizer parameter groups or learning rates changed')
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and (not torch.isfinite(value).all()
                    or (key in ('exp_avg', 'exp_avg_sq') and value.shape != parameter.shape)):
                raise ValueError('Resume optimizer tensor is nonfinite or has the wrong shape')
    rng.bit_generator.state = saved['rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])


def diagnostic(wrapper, datasets, count):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from scripts.train_wam_v3 import isolated_rng
    result = {}
    mode = wrapper.training
    wrapper.eval()
    try:
        with torch.no_grad(), isolated_rng(20261012, wrapper.device):
            for split in ('train', 'val'):
                tasks = datasets.get(split, {})
                if not tasks:
                    continue
                ds = tasks[v2.TASK]
                rows = []
                for index in np.linspace(0, len(ds) - 33, count, dtype=int):
                    batch = default_collate([ds[int(index)]])
                    loss, info, out = wrapper(batch)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite diagnostic')
                    rows.append(dict(index=int(index), episode=batch['episode'][0], anchor=int(batch['anchor'][0]),
                                     loss=float(loss), **info))
                    del batch, loss, out
                keys = ('loss', 'mse', 'future32', 'future8', 'idm', 'chunk_vtt_loss', 'chunk_huber', 'chunk_cosine')
                means = {key: float(np.mean([r[key] for r in rows])) for key in keys if key in rows[0]}
                result[split] = dict(count=len(rows), means=means, rows=rows)
    finally:
        wrapper.train(mode)
    return result


def schedule_from(args):
    return dict(steps=args.steps, predictor_steps=args.predictor_steps, gate_steps=args.gate_steps,
                backbone_lr=args.backbone_lr, new_lr=args.new_lr, seed=args.seed,
                batch_size=args.batch_size, save_every=args.save_every,
                resume_every=args.resume_every, diagnostic_samples=args.diagnostic_samples)


def finish_run(out, wrapper, contract, step, datasets, count):
    """Idempotent publication after the complete optimizer boundary is committed."""
    total = contract['v4_training']['schedule']['steps']
    v2.verify_code_hashes(contract['source_files'])
    save_checkpoint(out, wrapper, contract, step)
    path = out / f'diagnostic_{ORIGIN_STEP + step:06d}.json'
    if not path.exists():
        v2.write_json(path, diagnostic(wrapper, datasets, count), immutable=True)
    else:
        v2.read_json(path)
    status = 'completed' if step == total else 'stopped_at_safe_boundary'
    v2.write_json(out / 'status.json', dict(status=status, updates=step, step=ORIGIN_STEP + step,
        target_updates=total, target_step=ORIGIN_STEP + total, physical_success_rate=None))
    log(f'V4 {status}，新增正式更新{step}/{total}；已存模型/AdamW/RNG，任务成功率尚待真实仿真。', 'v4_training')


def execute(args):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from harness import zhuhai_precision_v4_training as v4
    from harness import zhuhai_training as zt
    from harness.wam_behavior import load_upstream
    from harness.zhuhai_dataset import FrozenDinoProducer, FrameFeatureStore
    from harness.zhuhai_hybrid_cache import HybridFeatureCache

    t2.reserve(args, startup=True)
    if v2.sha256_file(CK80K) != CK80K_SHA:
        raise ValueError('Original 80k identity changed')
    initial = torch.load(CK80K, map_location='cpu', weights_only=False, mmap=True)
    if initial['format'] != v2.FORMAT or initial['step'] != ORIGIN_STEP:
        raise ValueError('Expected the original V2 80k initializer')
    base = initial['contract']
    v2.verify_code_hashes(base['source_files'])
    contract = v4.new_contract(base)
    contract['source_files'][str(Path(__file__).resolve())] = v2.sha256_file(Path(__file__).resolve())
    contract['hjepa']['code_hashes'] = copy.deepcopy(contract['source_files'])
    schedule = schedule_from(args)
    contract['v4_training'] = dict(schedule=schedule, origin_step=ORIGIN_STEP,
        initialization=dict(path=str(CK80K), sha256=CK80K_SHA, format=v2.FORMAT),
        optimizer='new staged AdamW; original moments audited but not resumed across architecture change',
        dataset='original successful TRAIN160, no recovery data or heldout; unchanged TRAIN statistics/Task-VTT',
        backbone_freeze='preserve all originally frozen parameters; only original trainable parameters unfreeze at joint',
        checkpoint_step='original 80000 plus V4 optimizer updates',
        evaluation_seeds=[932101, 932102, 932103, 932104, 932105],
        record_evaluation_video=False, success_video_directory=str(ROOT / 'success video'))
    v4.validate_training_contract(contract)
    saved = load_resume(args.out, contract) if args.resume else None
    manifest = dict(contract=contract, contract_sha256=v2.digest(contract))
    v2.write_json(args.out / 'manifest.json', manifest, immutable=True)
    v2.write_json(args.out / 'command.json', dict(argv=sys.argv, schedule=schedule), immutable=not args.resume)
    cfg = contract['config']
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(.90)
    step = int(saved['updates']) if saved is not None else 0
    v2.write_json(args.out / 'status.json', dict(status='building', updates=step, step=ORIGIN_STEP + step,
        target_updates=args.steps, target_step=ORIGIN_STEP + args.steps, physical_success_rate=None))
    enc_path = v2.resolve(cfg['model']['vision_encoder']['checkpoint_path'])
    enc_sha = base['cache']['encoder_sha256']
    if v2.sha256_file(enc_path / 'model.safetensors') != enc_sha:
        raise ValueError('Frozen DINO identity changed')
    _, runner = load_upstream()
    encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
        str(enc_path), dtype=torch.bfloat16, device='cuda')
    if (hidden, registers, patch) != (1024, 4, 16):
        raise ValueError('Unexpected DINO geometry')
    cache = HybridFeatureCache(args.cache_root, budget_bytes=args.cache_budget_bytes,
                               min_free_bytes=args.minimum_free_bytes)
    disk = FrameFeatureStore(v2.SourceBoundFeatureCache(cache, base['hjepa']['cache_binding']),
        FrozenDinoProducer(zt.FrozenFeatureEncoder(encoder, device='cuda'), encoder), encoder_sha256=enc_sha)
    store = zt.BoundedRamFeatures(disk, budget_bytes=args.cache_ram_budget_bytes)
    datasets = v4.build_datasets(base['splits'], store, cfg, variant=base['hjepa']['variant'])
    for tasks in datasets.values():
        for task, ds in tasks.items():
            if task != v2.TASK:
                raise ValueError('This run is restricted to the current fixed task')
            ds.condition = np.asarray(base['conditions'][task], dtype=np.float32)
    wrapper, notes = v4.build_wrapper(cfg, base['statistics'], variant=base['hjepa']['variant'],
        idm_statistics=base['idm_statistics'], device='cuda',
        state_dict=saved['model_state_dict'] if saved is not None else initial['model_state_dict'])
    del initial
    optimizer = make_optimizer(wrapper.action_model, cfg['training'], schedule)
    rng = np.random.default_rng(args.seed)
    if saved is not None:
        restore_state(optimizer, rng, saved)
        reconcile_resume_log(args.out, step)
        if step % args.save_every == 0 or (args.out / f'checkpoint_step_{ORIGIN_STEP + step:06d}.pt').exists():
            save_checkpoint(args.out, wrapper, contract, step)
        v2.write_json(args.out / f'resume_{ORIGIN_STEP + step:06d}.json', dict(updates=step,
            optimizer_states=len(optimizer.state), rng_restored=True, schedule_unchanged=True,
            recovery=saved.get('resume_recovery')), immutable=True)
        del saved
    else:
        v2.write_json(args.out / 'build_notes.json', notes, immutable=True)
        save_checkpoint(args.out, wrapper, contract, 0, optimizer=optimizer, rng=rng.bit_generator.state)
        save_checkpoint(args.out, wrapper, contract, 0)
        v2.write_json(args.out / 'initial_diagnostic.json',
                      diagnostic(wrapper, datasets, args.diagnostic_samples), immutable=True)

    stop = {'requested': False}
    signal.signal(signal.SIGTERM, lambda s, f: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda s, f: stop.update(requested=True))
    end = min(args.steps, args.max_updates) if args.max_updates is not None else args.steps
    if step == args.steps:
        finish_run(args.out, wrapper, contract, step, datasets, args.diagnostic_samples)
        return 0
    if not step < end:
        raise ValueError('No new optimizer updates requested after resume boundary')
    wrapper.train()
    started = time.monotonic()
    previous_phase = None
    last_report = started
    ds = datasets['train'][v2.TASK]
    saved_boundary = step
    try:
        while step < end and not stop['requested']:
            t2.reserve(args)
            tick = time.monotonic()
            phase = phase_for(step, args.predictor_steps, args.gate_steps)
            if phase != previous_phase:
                log(f'V4阶段{phase}，正式新增更新{step}/{args.steps}，原80k保留；不是任务成功证据。', 'v4_training')
                previous_phase = phase
            configure_phase(optimizer, phase, step, schedule)
            batch = default_collate([ds[int(rng.integers(len(ds)))] for _ in range(args.batch_size)])
            optimizer.zero_grad(set_to_none=True)
            loss, info, outputs = wrapper(batch, cfg['training']['auxiliary_loss'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            gradients = {g['name'] + '_grad_norm': gradient_norm(g['params']) for g in optimizer.param_groups}
            if any(not math.isfinite(v) for v in gradients.values()):
                raise ValueError('Nonfinite module gradient')
            if info['chunk_vtt_valid_samples'] and gradients['predictor_grad_norm'] <= 0:
                raise ValueError('Chunk predictor did not receive its supervision gradient')
            if phase != 'joint' and gradients['backbone_grad_norm'] != 0:
                raise ValueError('Frozen backbone received a gradient')
            if phase == 'predictor' and (float(wrapper.action_model.chunk_vtt16.alpha) != 0
                                         or gradients['projection_grad_norm'] != 0):
                raise ValueError('Predictor-only phase changed fusion')
            parameters = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['training']['grad_clip_norm'])
            if not torch.isfinite(norm) or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
                raise ValueError('Invalid clipped gradient or frozen DINO gradient')
            optimizer.step()
            step += 1
            row = dict(step=ORIGIN_STEP + step, updates=step, phase=phase, loss=float(loss.detach()),
                **info, **gradients, gradient_norm=float(norm),
                alpha_after_update=float(wrapper.action_model.chunk_vtt16.alpha.detach()),
                learning_rates={g['name']: g['lr'] for g in optimizer.param_groups},
                seconds_per_step=time.monotonic() - tick, elapsed_seconds=time.monotonic() - started,
                cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cache_stats=dict(cache.stats), ram_cache_stats=dict(store.stats))
            optimizer.zero_grad(set_to_none=True)
            del batch, loss, outputs
            with (args.out / 'log.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            v2.write_json(args.out / 'status.json', dict(status='training', step=ORIGIN_STEP + step, updates=step,
                target_updates=args.steps, target_step=ORIGIN_STEP + args.steps, phase=phase,
                last_loss=row['loss'], alpha=row['alpha_after_update'], physical_success_rate=None))
            if step == 1 or step % 25 == 0:
                print(json.dumps(row), flush=True)
            terminal = step == end or stop['requested']
            if step % args.resume_every == 0 or step % args.save_every == 0 or terminal:
                v2.verify_code_hashes(contract['source_files'])
                save_checkpoint(args.out, wrapper, contract, step, optimizer=optimizer, rng=rng.bit_generator.state)
                saved_boundary = step
            if step % args.save_every == 0 or terminal:
                save_checkpoint(args.out, wrapper, contract, step)
            if time.monotonic() - last_report >= 1800:
                log(f"V4新增更新{step}/{args.steps}，phase={phase}，loss={row['loss']:.6f}，"
                    f"chunk={row['chunk_vtt_loss']:.6f}，alpha={row['alpha_after_update']:.6f}；未做终局测试。", 'v4_training')
                last_report = time.monotonic()
        if step != saved_boundary:
            # A stop signal may arrive between iterations, after the last in-loop save check.
            v2.verify_code_hashes(contract['source_files'])
            save_checkpoint(args.out, wrapper, contract, step, optimizer=optimizer, rng=rng.bit_generator.state)
        finish_run(args.out, wrapper, contract, step, datasets, args.diagnostic_samples)
        return 0
    except BaseException as exc:
        v2.write_json(args.out / 'failure_guard.json', dict(updates=step, error=f'{type(exc).__name__}: {exc}',
            preserved=True, automatic_retry=False), immutable=True)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--max-updates', type=int)
    p.add_argument('--predictor-steps', type=int, default=1000)
    p.add_argument('--gate-steps', type=int, default=2000)
    p.add_argument('--backbone-lr', type=float, default=1e-6)
    p.add_argument('--new-lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--save-every', type=int, default=2500)
    p.add_argument('--resume-every', type=int, default=500)
    p.add_argument('--diagnostic-samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--gpu-lock-owner', required=True)
    p.add_argument('--cache-root', default='data/zhuhai_hjepa_hybrid_features_20261009')
    p.add_argument('--cache-budget-bytes', type=int, default=10 ** 12)
    p.add_argument('--cache-ram-budget-bytes', type=int, default=2 * 1024 ** 3)
    p.add_argument('--minimum-free-bytes', type=int, default=100 * 1024 ** 3)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not (0 < args.predictor_steps < args.gate_steps < args.steps
            and min(args.batch_size, args.save_every, args.resume_every, args.diagnostic_samples) > 0
            and all(math.isfinite(x) and x > 0 for x in (args.backbone_lr, args.new_lr))):
        raise ValueError('Invalid staged schedule')
    args.out = v2.resolve(args.out)
    if not args.execute:
        print(json.dumps(dict(status='plan', schedule=schedule_from(args), initializer=str(CK80K),
            initializer_sha256=CK80K_SHA, target_step=ORIGIN_STEP + args.steps, out=str(args.out))))
        return 0
    with v2.output_lock(args.out):
        occupied = any(p.name != '.hjepa_training.lock' for p in args.out.iterdir())
        if occupied and not args.resume:
            raise FileExistsError('Preserve nonempty run; choose a new output or exact --resume')
        if args.resume and ((args.out / 'failure_guard.json').exists() or not (args.out / 'manifest.json').exists()):
            raise ValueError('Resume requires a committed run without a failure guard')
        try:
            return execute(args)
        except BaseException as exc:
            old = v2.read_json(args.out / 'status.json') if (args.out / 'status.json').exists() else {}
            v2.write_json(args.out / 'status.json', dict(old, status='failed', error=f'{type(exc).__name__}: {exc}',
                                                       evidence_preserved=True))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
