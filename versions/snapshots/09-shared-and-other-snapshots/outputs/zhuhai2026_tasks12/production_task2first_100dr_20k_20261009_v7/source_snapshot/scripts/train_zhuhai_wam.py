#!/usr/bin/env python3
"""Train native16 measured-feedback WAM from the immutable native14 50k backbone.

Default plan is CPU-only. Tiny runs explicitly relax data quota, use a separate
immutable contract, and cannot establish a competition success rate. The
coordinator alone owns the GPU lock. Encoder weights remain frozen, training
conditions use TRAIN endpoints, and every derived cache byte is bounded.
"""
from __future__ import annotations

import argparse
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

from scripts.progress1_log import log
from harness import zhuhai_training as zt


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', required=True, help='JSON tasks with source_manifest/randomized_manifest')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--mode', choices=['plan', 'tiny', 'train'], default='plan')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--save-every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261009)
    ap.add_argument('--cache-root', default='data/zhuhai_native16_hybrid_features_20261009')
    ap.add_argument('--cache-budget-bytes', type=int, default=10**12)
    ap.add_argument('--cache-ram-budget-bytes', type=int, default=2 * 1024**3)
    ap.add_argument('--minimum-free-bytes', type=int, default=100 * 1024**3)
    ap.add_argument('--source-count', type=int, default=100)
    ap.add_argument('--randomized-count', type=int, default=500)
    ap.add_argument('--allow-small-data', action='store_true')
    ap.add_argument('--overfit-anchors', type=int, default=0,
                    help='Tiny mode only: repeat this many fixed TRAIN anchors per task')
    ap.add_argument('--initialization', default=str(zt.INITIALIZATION))
    ap.add_argument('--initialization-sha256', default=zt.INITIALIZATION_SHA)
    ap.add_argument('--gpu-lock-owner')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-updates', type=int, help='Safe absolute stopping boundary; fixed schedule unchanged')
    return ap


def verify_owner(owner):
    from scripts.gpu_lock import owner as current
    if not owner or current() != owner:
        raise RuntimeError('matching externally owned GPU lock required')


def save_checkpoint(out, wrapper, contract, step, *, optimizer=None, rng=None):
    import torch
    from scripts.train_wam_v3 import cpu_tree
    zt.validate_training_contract(contract)
    milestone = optimizer is None
    name = f'checkpoint_step_{step:06d}.pt' if milestone else 'checkpoint_last.pt'
    path = out / name
    if milestone and path.exists():
        raise FileExistsError('immutable milestone already exists')
    ck = dict(format=zt.FORMAT, step=step, model_only=milestone,
              model_state_dict=cpu_tree(wrapper.action_model.state_dict()),
              contract=contract, contract_sha256=zt.digest(contract))
    if not milestone:
        ck.update(optimizer_state_dict=cpu_tree(optimizer.state_dict()), rng=rng,
                  torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    tmp = path.with_name(name + '.tmp')
    torch.save(ck, tmp)
    tmp.replace(path)
    zt.atomic_json(path.with_name(name + '.sha256.json'), dict(sha256=zt.sha256_file(path),
        size=path.stat().st_size, step=step, model_only=milestone))
    return path


def diagnostic(wrapper, datasets, rng_seed=20261008):
    import torch
    from torch.utils.data import default_collate
    from scripts.train_wam_v3 import isolated_rng
    result = {}
    wrapper.eval()
    with torch.no_grad(), isolated_rng(rng_seed):
        for task, ds in datasets.items():
            batch = default_collate([ds[min(len(ds) - 1, len(ds) // 2)]])
            total, info, _ = wrapper(batch, {'future': 0.5, 'gate_kl': 0, 'phase_ce': 0, 'log_dist': 0})
            result[task] = dict(loss=float(total), flow_mse=info['mse'],
                                scope='fixed offline diagnostic, no physical-success claim')
    wrapper.train()
    return result


def train(args):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from harness.wam_behavior import load_upstream
    from harness.zhuhai_dataset import FrozenDinoProducer, FrameFeatureStore, PREPROCESS_SHA256
    from harness.zhuhai_hybrid_cache import HybridFeatureCache
    from harness.zhuhai_action_smoothing import protocol as smoothing_protocol, protocol_hash

    if min(args.steps, args.batch_size, args.save_every) < 1:
        raise ValueError('positive steps/batch/save-every required')
    if args.mode == 'train' and (args.allow_small_data or args.overfit_anchors):
        raise ValueError('production train cannot relax quota or repeat tiny anchors')
    if not 0 <= args.cache_budget_bytes <= 10**12 or not 0 <= args.cache_ram_budget_bytes <= 2 * 1024**3:
        raise ValueError('feature cache exceeds user disk/RAM budget')
    if args.minimum_free_bytes < 100 * 1024**3:
        raise ValueError('100GiB disk reserve cannot be weakened')
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or not (args.out / 'manifest.json').is_file():
            raise FileExistsError('nonempty output preserved; select a NEW native16 output or explicit --resume')
        previous = json.loads((args.out / 'manifest.json').read_text())
        zt.validate_training_contract(previous['contract'])
        if previous['contract_sha256'] != zt.digest(previous['contract']):
            raise ValueError('resume manifest digest mismatch')
    elif args.resume:
        raise FileNotFoundError('--resume requires an existing native16 manifest')
    log(f'{args.mode}接入前核对native16机器人/实测state/目标action合同、TRAIN family与10Hz时序', actor='train_zhuhai_wam')
    splits, sources = zt.task_sources(args.tasks, allow_small=args.allow_small_data,
        source_count=args.source_count, randomized_count=args.randomized_count)
    ck, initialization = zt.checkpoint_header(args.initialization, expected_sha=args.initialization_sha256)
    cfg = zt.config_for(ck, steps=args.steps, seed=args.seed, batch=args.batch_size)
    cfg['training'].update(save_every=args.save_every, init_from=str(zt.resolve(args.initialization)),
                            init_from_sha256=args.initialization_sha256)
    smoothing = zt.validate_postprocessing(smoothing_protocol())
    cfg.setdefault('inference', {})['action_postprocessing'] = smoothing
    encoder_path = zt.resolve(cfg['model']['vision_encoder']['checkpoint_path'])
    provenance = json.loads((encoder_path / 'DOWNLOAD_PROVENANCE.json').read_text())
    encoder_sha = provenance['files']['model.safetensors']['sha256']
    base = dict(format=zt.FORMAT, mode=args.mode, config=cfg, sources=sources, splits=splits,
        initialization=initialization, time=zt.time_contract(cfg), **zt.protocol_binding(),
        numerical_normalization='new successful TRAIN-only native16 measured-state/target-action statistics; native14 normalization forbidden',
        cache=dict(root=str(zt.resolve(args.cache_root)), disk_budget_bytes=args.cache_budget_bytes,
                   ram_budget_bytes=args.cache_ram_budget_bytes, minimum_free_bytes=args.minimum_free_bytes,
                   encoder_sha256=encoder_sha, preprocess_sha256=PREPROCESS_SHA256,
                   raw_data_evicted=False, policy='bounded derived-feature RAM LRU + indexed disk LRU + online fallback; no decoded RGB cache',
                   ram_scope='resident derived-feature arrays and index; model/optimizer/minibatch transient memory reported separately'),
        protocol=dict(record_video=False, third_view_camera_created=False,
                      pure_wam=True, vlm=False, high_level_planner=False, ask=False,
                      lifelong_memory_writes=False, capacity_is_not_task_success=True),
        action_postprocessing=smoothing, action_postprocessing_sha256=protocol_hash(smoothing),
        overfit_anchors=args.overfit_anchors, source_files={str(p): zt.sha256_file(p) for p in (
            Path(__file__).resolve(), ROOT / 'harness/zhuhai_training.py', ROOT / 'harness/zhuhai_dataset.py',
            ROOT / 'harness/zhuhai_action_smoothing.py', ROOT / 'third_party/LiLa-WAM/robotwin_infer.py',
            ROOT / 'harness/wam_v3.py', ROOT / 'harness/wam_behavior.py',
            ROOT / 'harness/zhuhai_robot_spec.py', ROOT / 'harness/zhuhai_demo_recording.py',
            ROOT / 'scripts/eval_zhuhai_wam.py')})
    args.out.mkdir(parents=True, exist_ok=True)
    zt.immutable_json(args.out / 'plan.json', dict(contract=base, cuda_initialized=False,
        frame_counts_unknown_until_hdf_validation=True, statistics='pending_new_TRAIN_only_fit_before_model_creation',
        status='cpu_preflight_completed'))
    if args.mode == 'plan':
        print(json.dumps({'status': 'cpu_preflight_completed', 'out': str(args.out)}, ensure_ascii=False), flush=True)
        return 0
    verify_owner(args.gpu_lock_owner)
    if (args.out / 'manifest.json').exists() and not args.resume:
        raise FileExistsError('existing training run; use --resume or new output')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(0.90)
    log('先加载共享冻结DINO建立新TRAIN统计/VTT；禁止使用14D norm，之后仅重置三投影迁移backbone', actor='train_zhuhai_wam')
    if zt.sha256_file(encoder_path / 'model.safetensors') != encoder_sha:
        raise ValueError('frozen encoder digest changed')
    _, runner = load_upstream()
    encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(str(encoder_path),
        dtype=torch.bfloat16, device='cuda')
    if (hidden, registers, patch) != (1024, 4, 16):
        raise ValueError('unexpected frozen encoder geometry')
    cache = HybridFeatureCache(args.cache_root, budget_bytes=args.cache_budget_bytes,
                               min_free_bytes=args.minimum_free_bytes)
    feature_encoder = zt.FrozenFeatureEncoder(encoder, device='cuda')
    disk_store = FrameFeatureStore(cache, FrozenDinoProducer(feature_encoder, encoder), encoder_sha256=encoder_sha)
    store = zt.BoundedRamFeatures(disk_store, budget_bytes=args.cache_ram_budget_bytes)
    datasets = zt.build_datasets(splits, store, cfg)
    # SourceEpisode checks SHA and successful_expert before any model sample.
    feature_identity = dict(encoder_sha256=encoder_sha, preprocess_sha256=PREPROCESS_SHA256)
    statistics = zt.fit_train_statistics(datasets['train'], cfg, feature_identity=feature_identity,
                                         check=lambda: verify_owner(args.gpu_lock_owner))
    condition_bank = zt.install_conditions(datasets, cfg, statistics)
    wrapper, notes = zt.build_wrapper(cfg, statistics, device='cuda', initialization=ck)
    del ck
    contract = dict(base, statistics=statistics, statistics_sha256=zt.digest(statistics),
        conditions=condition_bank['conditions'], condition_bank=condition_bank, initialization_load=notes,
        train_frame_counts={t: len(ds) for t, ds in datasets['train'].items()},
        condition_source='successful native16 TRAIN-only head CLS terminal-minus-start means; robot/statistics/features bound')
    zt.validate_training_contract(contract)
    zt.immutable_json(args.out / 'statistics.json', statistics)
    zt.immutable_json(args.out / 'condition_bank.json', condition_bank)
    rng = np.random.default_rng(args.seed)
    parameters = [p for p in wrapper.action_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['training']['learning_rate'],
        betas=tuple(cfg['training']['betas']), weight_decay=cfg['training']['weight_decay'])
    start = 0
    if args.resume:
        old = json.loads((args.out / 'manifest.json').read_text())
        if old['contract'] != contract or old['contract_sha256'] != zt.digest(contract):
            raise ValueError('immutable training source/config/scene contract changed; new run required')
        last = args.out / 'checkpoint_last.pt'
        receipt = json.loads(last.with_name(last.name + '.sha256.json').read_text())
        if receipt['sha256'] != zt.sha256_file(last):
            raise ValueError('resume checkpoint digest changed')
        saved = torch.load(last, map_location='cpu', weights_only=False, mmap=True)
        if (saved.get('format') != zt.FORMAT or saved['contract'] != contract
                or saved.get('contract_sha256') != zt.digest(contract) or saved['model_only']):
            raise ValueError('resume must use exact full-state last checkpoint')
        wrapper.action_model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start = saved['step']
        del saved
    else:
        zt.immutable_json(args.out / 'manifest.json', dict(contract=contract, contract_sha256=zt.digest(contract)))
        save_checkpoint(args.out, wrapper, contract, 0)
    stopped = {'requested': False}
    def stop(signum, frame):
        stopped['requested'] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tasks = list(datasets['train'])
    fixed = {task: np.linspace(0, len(ds) - 1, args.overfit_anchors, dtype=int).tolist()
             for task, ds in datasets['train'].items()} if args.overfit_anchors else None
    end = min(args.steps, args.max_updates) if args.max_updates is not None else args.steps
    wrapper.train()
    started = time.monotonic()
    step = start
    if not args.resume:
        initial_diagnostic = diagnostic(wrapper, datasets['train'])
        zt.immutable_json(args.out / 'initial_diagnostic.json', initial_diagnostic)
    try:
        while step < end and not stopped['requested']:
            verify_owner(args.gpu_lock_owner)
            from shutil import disk_usage
            if disk_usage(args.out).free < args.minimum_free_bytes:
                raise RuntimeError('100GiB disk reserve breached')
            items = []
            for _ in range(args.batch_size):
                task = tasks[int(rng.integers(len(tasks)))]
                ds = datasets['train'][task]
                anchor = int(rng.choice(fixed[task])) if fixed else int(rng.integers(len(ds)))
                items.append(ds[anchor])
            batch = default_collate(items)
            warmup = cfg['training']['warmup_steps']
            factor = min(1., (step + 1) / max(1, warmup)) if step < warmup else (
                0.1 + 0.9 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))) / 2)
            for group in optimizer.param_groups:
                group['lr'] = cfg['training']['learning_rate'] * factor
            optimizer.zero_grad(set_to_none=True)
            loss, info, _ = wrapper(batch, cfg['training']['auxiliary_loss'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, cfg['training']['grad_clip_norm'])
            projection = wrapper.action_model.task_cond_proj
            evidence = dict(vtt_projection_grad_norm=float(projection.residual.weight.grad.float().norm()),
                shared_action_grad_norm=float(wrapper.action_model.action_proj.weight.grad.float().norm()),
                frozen_base_has_grad=any(p.grad is not None for p in projection.base.parameters()),
                dino_encoder_loaded=True, dino_trainable_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad))
            if evidence['frozen_base_has_grad'] or evidence['dino_trainable_parameters']:
                raise ValueError('frozen condition-base/encoder received optimization gradients')
            if float(grad) <= 0 or not torch.isfinite(grad):
                raise ValueError('no finite optimization gradient')
            optimizer.step()
            step += 1
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad), **info,
                **evidence, elapsed_seconds=time.monotonic() - started, cache_stats=dict(cache.stats),
                ram_cache_stats=dict(store.stats),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            with (args.out / 'log.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            zt.atomic_json(args.out / 'status.json', dict(status='training', step=step,
                target_steps=args.steps, last_loss=row['loss'], elapsed_seconds=row['elapsed_seconds']))
            if step == 1 or step % 25 == 0:
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == end or stopped['requested']:
                log(f'保存训练安全边界step={step}及固定诊断；不按验证分数选择更好权重', actor='train_zhuhai_wam')
                save_checkpoint(args.out, wrapper, contract, step)
                save_checkpoint(args.out, wrapper, contract, step, optimizer=optimizer,
                                rng=rng.bit_generator.state)
                zt.atomic_json(args.out / f'diagnostic_{step:06d}.json', {
                    'train': diagnostic(wrapper, datasets['train']),
                    'val': diagnostic(wrapper, datasets['val']) if datasets['val'] else {}})
        if step == start or (step % args.save_every and step != end):
            save_checkpoint(args.out, wrapper, contract, step, optimizer=optimizer, rng=rng.bit_generator.state)
        state = 'completed' if step == args.steps else 'stopped_at_safe_boundary'
        zt.atomic_json(args.out / 'status.json', dict(status=state, step=step, target_steps=args.steps,
            physical_success_rate=None, data_quota_relaxed=args.allow_small_data,
            tiny_run_cannot_establish_competition_rate=args.mode == 'tiny'))
        return 0
    except BaseException as exc:
        zt.atomic_json(args.out / 'status.json', dict(status='failed', step=step,
            error=f'{type(exc).__name__}: {exc}', preserved_last_checkpoint=True))
        raise


if __name__ == '__main__':
    raise SystemExit(train(parser().parse_args()))
