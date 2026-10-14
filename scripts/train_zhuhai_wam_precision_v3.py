#!/usr/bin/env python3
"""Fine-tune Precision-WAM v3 (full-chunk camera-ray refinement) from the 80k v2 weights on the
ORIGINAL dessert data (frozen DINOv3 feature cache). No control run; fork-80k warm-start chosen for
highest success. The new refinement head trains at a higher lr; the warm-started trunk/adapter use the
small 80k lr so the already-learned behaviour is protected. Plan unless --execute (needs the GPU lock).
"""
from __future__ import annotations
import argparse, copy, json, math, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v2_training as v2
from harness import zhuhai_precision_v3_training as v3
from scripts.progress1_log import log
import scripts.train_zhuhai_wam_precision_v2 as t2

CK80K = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger30k_diagnostic_20261011_v23/train/checkpoint_step_080000.pt'
CK80K_SHA = '1de0c8b337bd3a01eec102e3b44df121dfa999bb9324d1c51b2f2a26f264f93c'
EXT_V3 = ('harness/wam_precision_v3.py', 'harness/zhuhai_precision_v3_training.py', 'scripts/train_zhuhai_wam_precision_v3.py')
PRECISION_HEAD_LR = 1e-4   # new refinement head learns fast; trunk stays at the 80k lr (cfg.training.learning_rate)


def v3_contract(base):
    c = copy.deepcopy(base)
    c['format'] = v3.FORMAT
    c['precision_v3'] = copy.deepcopy(v3.PRECISION_V3_SPEC)
    c['config']['model']['precision_v3'] = copy.deepcopy(v3.PRECISION_V3_SPEC)
    for rel in EXT_V3:
        c['source_files'][str(ROOT / rel)] = v2.sha256_file(ROOT / rel)
    c['hjepa']['code_hashes'] = copy.deepcopy(c['source_files'])
    c['warm_start'] = dict(fork='80k v2 Precision', checkpoint_sha256=CK80K_SHA,
                           control_run='none (user directive: fork-80k only, highest success)',
                           precision_head_lr=PRECISION_HEAD_LR)
    return c


def save_checkpoint(out, wrapper, contract, step, *, optimizer=None, rng=None):
    import torch
    from scripts.train_wam_v3 import cpu_tree
    milestone = optimizer is None
    path = out / (f'checkpoint_step_{step:06d}.pt' if milestone else 'checkpoint_last.pt')
    if milestone and path.exists():
        raise FileExistsError('Preserve existing immutable milestone')
    value = dict(format=v3.FORMAT, step=step, model_only=milestone, contract=contract, contract_sha256=v2.digest(contract),
                 model_state_dict=cpu_tree(wrapper.action_model.state_dict()))
    if not milestone:
        value.update(optimizer_state_dict=cpu_tree(optimizer.state_dict()), rng=rng,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temp = path.with_name(path.name + '.tmp'); torch.save(value, temp); temp.replace(path)
    v2.write_json(path.with_name(path.name + '.sha256.json'),
                  dict(sha256=v2.sha256_file(path), size=path.stat().st_size, step=step, model_only=milestone))
    return path


def lr_factor(step, warmup, total):
    if step < warmup:
        return min(1.0, (step + 1) / max(1, warmup))
    return 0.1 + 0.9 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))) / 2


def execute(args):
    import numpy as np, torch
    from harness import zhuhai_training as zt
    from harness.wam_behavior import load_upstream
    from harness.zhuhai_dataset import FrozenDinoProducer, FrameFeatureStore
    from harness.zhuhai_hybrid_cache import HybridFeatureCache
    t2.reserve(args, startup=True)
    ck = torch.load(CK80K, map_location='cpu', weights_only=False, mmap=True)
    if ck.get('format') != v2.FORMAT or v2.sha256_file(CK80K) != CK80K_SHA:
        raise ValueError('80k warm-start checkpoint identity mismatch')
    base = ck['contract']; cfg = base['config']; stats = base['statistics']; idm = base['idm_statistics']
    variant = base['hjepa']['variant']; binding = base['hjepa']['cache_binding']
    v2.verify_code_hashes(base['source_files'])
    contract = v3_contract(base)
    total = args.steps; warmup = min(200, max(1, total // 50)); base_lr = cfg['training']['learning_rate']
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(.90)
    v2.write_json(args.out / 'plan.json', dict(status='v3_fork80k', contract_sha256=v2.digest(contract), steps=total,
        warmup=warmup, base_lr=base_lr, precision_head_lr=PRECISION_HEAD_LR, mode=args.mode), immutable=True)
    v2.write_json(args.out / 'status.json', dict(status='building', step=0, target_steps=total, physical_success_rate=None))
    enc_path = v2.resolve(cfg['model']['vision_encoder']['checkpoint_path']); enc_sha = base['cache']['encoder_sha256']
    if v2.sha256_file(enc_path / 'model.safetensors') != enc_sha: raise ValueError('DINO changed')
    _, runner = load_upstream()
    encoder, hid, reg, patch = runner.ModelFactory.create_vision_encoder(str(enc_path), dtype=torch.bfloat16, device='cuda')
    if (hid, reg, patch) != (1024, 4, 16): raise ValueError('Unexpected DINO geometry')
    cache = HybridFeatureCache(args.cache_root, budget_bytes=args.cache_budget_bytes, min_free_bytes=args.minimum_free_bytes)
    disk = FrameFeatureStore(v2.SourceBoundFeatureCache(cache, binding),
        FrozenDinoProducer(zt.FrozenFeatureEncoder(encoder, device='cuda'), encoder), encoder_sha256=enc_sha)
    store = zt.BoundedRamFeatures(disk, budget_bytes=args.cache_ram_budget_bytes)
    datasets = v2.build_datasets(base['splits'], store, cfg, variant=variant)
    for split in datasets.values():
        for task, ds in split.items():
            ds.condition = np.asarray(base['conditions'][task], dtype=np.float32)
    wrapper, notes = v3.build_wrapper(cfg, stats, variant=variant, idm_statistics=idm, device='cuda', state_dict=ck['model_state_dict'])
    del ck
    v2.write_json(args.out / 'build_notes.json', notes, immutable=True)
    prec_ids = {id(p) for p in wrapper.action_model.precision.parameters()}
    rest = [p for p in wrapper.action_model.parameters() if p.requires_grad and id(p) not in prec_ids]
    prec = [p for p in wrapper.action_model.precision.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=rest, lr=base_lr), dict(params=prec, lr=PRECISION_HEAD_LR)],
        betas=tuple(cfg['training']['betas']), weight_decay=cfg['training']['weight_decay'])
    group_bases = [base_lr, PRECISION_HEAD_LR]
    from torch.utils.data import default_collate
    stopped = {'v': False}
    signal.signal(signal.SIGTERM, lambda s, f: stopped.update(v=True))
    signal.signal(signal.SIGINT, lambda s, f: stopped.update(v=True))
    tasks = list(datasets['train']); rng = np.random.default_rng(args.seed)
    v2.write_json(args.out / 'manifest.json', dict(contract=contract, contract_sha256=v2.digest(contract)), immutable=True)
    save_checkpoint(args.out, wrapper, contract, 0)
    wrapper.train(); started = time.monotonic(); step = 0
    v2.write_json(args.out / 'initial_diagnostic.json', t2.diagnostic(wrapper, datasets['train']), immutable=True)
    try:
        while step < total and not stopped['v']:
            t2.reserve(args); items = []
            for _ in range(args.batch_size):
                task = tasks[int(rng.integers(len(tasks)))]; ds = datasets['train'][task]
                items.append(ds[int(rng.integers(len(ds)))])
            batch = default_collate(items); del items
            fac = lr_factor(step, warmup, total)
            for group, gb in zip(optimizer.param_groups, group_bases): group['lr'] = gb * fac
            optimizer.zero_grad(set_to_none=True)
            loss, info, _ = wrapper(batch, cfg['training']['auxiliary_loss'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], cfg['training']['grad_clip_norm'])
            if not torch.isfinite(grad) or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
                raise ValueError('Invalid gradient or frozen DINO received a gradient')
            optimizer.step(); step += 1; optimizer.zero_grad(set_to_none=True)
            row = dict(step=step, loss=float(loss.detach()), grad=float(grad), lr_trunk=optimizer.param_groups[0]['lr'],
                lr_prec=optimizer.param_groups[1]['lr'], **info, elapsed=time.monotonic() - started)
            del batch, loss
            with (args.out / 'log.jsonl').open('a') as s: s.write(json.dumps(row, allow_nan=False) + '\n')
            v2.write_json(args.out / 'status.json', dict(status='training', step=step, target_steps=total, last_loss=row['loss'], elapsed_seconds=row['elapsed']))
            if step == 1 or step % 25 == 0: print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == total:
                v2.verify_code_hashes(contract['source_files'])
                save_checkpoint(args.out, wrapper, contract, step)
                save_checkpoint(args.out, wrapper, contract, step, optimizer=optimizer, rng=rng.bit_generator.state)
                v2.write_json(args.out / f'diagnostic_{step:06d}.json', dict(train=t2.diagnostic(wrapper, datasets['train']), val=t2.diagnostic(wrapper, datasets['val'])), immutable=True)
        v2.write_json(args.out / 'status.json', dict(status='completed' if step >= total else 'stopped', step=step, target_steps=total, physical_success_rate=None))
        return 0
    except BaseException as exc:
        v2.write_json(args.out / 'status.json', dict(status='failed', step=step, error=f'{type(exc).__name__}: {exc}', evidence_preserved=True)); raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--mode', choices=('plan', 'tiny', 'train'), default='plan')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--variant', default='combined')
    p.add_argument('--cache-root', default='data/zhuhai_hjepa_hybrid_features_20261009')
    p.add_argument('--cache-budget-bytes', type=int, default=10 ** 12)
    p.add_argument('--cache-ram-budget-bytes', type=int, default=2 * 1024 ** 3)
    p.add_argument('--minimum-free-bytes', type=int, default=100 * 1024 ** 3)
    p.add_argument('--gpu-lock-owner')
    args = p.parse_args(argv)
    if args.mode == 'tiny':
        args.steps = 2; args.save_every = 2
    if not args.execute:
        ok = CK80K.exists() and v2.sha256_file(CK80K) == CK80K_SHA and all((ROOT / r).exists() for r in EXT_V3)
        print(json.dumps(dict(status='cpu_preflight_completed' if ok else 'incomplete',
                              checkpoint=str(CK80K), steps=args.steps, mode=args.mode)))
        return 0 if ok else 2
    args.out = v2.resolve(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        return execute(args)
    except BaseException as exc:
        state = v2.read_json(args.out / 'status.json') if (args.out / 'status.json').exists() else {}
        v2.write_json(args.out / 'status.json', dict(state, status='failed', error=f'{type(exc).__name__}: {exc}', evidence_preserved=True))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
