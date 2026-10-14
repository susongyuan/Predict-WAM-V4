#!/usr/bin/env python3
"""Paired, cached-only V2 es-best vs V3 on an EXISTING pilot validation_plan.json.

CPU preflight (safe during training):
  .conda-envs/lilawam/bin/python scripts/eval_v2_on_v3_plan.py --plan <pilot>/validation_plan.json --check-plan-only
After training ONLY, under scripts/gpu_lock.py run --owner <unique-owner> -- ...:
  .../python scripts/eval_v2_on_v3_plan.py --plan <pilot>/validation_plan.json \
      --v3-checkpoint <pilot>/checkpoint_best.pt --out <NEW-directory> --device cuda --training-finished
Prefer --v3-result <same-pilot>/validation_<checkpoint-step>.json to reuse the trainer's V3 fine rows
and run ONLY V2 inference. Step, full plan, exact fine identities and Euler10 are checked.
Defaults: pinned original V2 es-best, exactly 256 fine validation anchors, K=1, trainer.noise_seed,
Euler10, active-hand action[31] FK and hold baseline. No DINO model is ever created. No training or step-0
baseline is substituted for V2. Never overwrites an existing output directory. No new dependencies.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('P2_TASK_ID', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
from torch.utils.data import default_collate
from omegaconf import OmegaConf
import train_wam_v3 as trainer
from harness.wam_behavior import BehaviorVLAWrapper, create_action_model, hold_baseline, load_upstream, model_proprio_lags
from harness.wam_v3 import WamV3Dataset

VERSION = 'paired-v2-v3-fine-plan-v1'
EULER_STEPS = 10


def identity(path):
    p = trainer.resolve(path)
    return {'path': str(p), 'sha256': trainer.sha256(p), 'stat': trainer.stat_identity(p)}


def validate_plan(plan):
    if (plan.get('version') != 1 or plan.get('split') != 'validation'
            or plan.get('samples_per_anchor') != 1 or plan.get('terminal_step') != 31):
        raise ValueError('require registered validation plan v1, K=1, terminal_step=31 (not tiny/train)')
    records = plan.get('fine', [])
    if len(records) != 256 or len({(r['episode'], r['anchor']) for r in records}) != 256:
        raise ValueError('require exactly 256 unique registered fine anchors; never resample a plan')
    for r in records:
        if not r.get('fine') or r['phase'] not in (1, 2) or r['active_hand'] not in (0, 1):
            raise ValueError('fine plan must contain only labelled near/contact anchors with known active hand')
    int(plan['seed'])
    return records


def validate_dataset(dataset, plan):
    records, _ = trainer.anchor_records(dataset)
    if trainer.digest_json(records) != plan['full_census_sha256']:
        raise ValueError('validation census/labels/indices changed since pilot plan registration')
    for r in plan['fine']:
        if records[r['index']] != r:
            raise ValueError('fine anchor metadata differs from registered plan')


def check_configs(v2, v3):
    if 'v3' in v2.model or 'v3' not in v3.model or model_proprio_lags(v2) != [0]:
        raise ValueError('require original current-state V2 and a V3 checkpoint')
    for path in ('model.views', 'model.vision_encoder.feat_layers', 'dataset.image_size',
                 'dataset.contract_targets_sha256', 'model.proprio_mask', 'dataset.cond_mode'):
        if OmegaConf.select(v2, path) != OmegaConf.select(v3, path):
            raise ValueError(f'checkpoint input contract mismatch: {path}')
    if v2.dataset.cond_mode != 'contract_proto':
        raise ValueError('comparison requires the same frozen training contract vectors')
    if (trainer.resolve(v2.model.vision_encoder.checkpoint_path) != trainer.resolve(v3.model.vision_encoder.checkpoint_path)
            or trainer.sha256(trainer.resolve(v2.dataset.stats_path)) != trainer.sha256(trainer.resolve(v3.dataset.stats_path))):
        raise ValueError('encoder identity or action/current-state normalization differs')
    if not v2.model.vision_encoder.get('include_cls_register', True):
        raise ValueError('cached features require CLS/register inclusion')


@contextmanager
def forbid_encoder():
    """Fail closed if any reused builder accidentally attempts to load DINO weights."""
    from unittest.mock import patch
    _, runner = load_upstream()
    with patch.object(runner.ModelFactory, 'create_vision_encoder', side_effect=RuntimeError('DINO loading forbidden')):
        yield runner


def build_v2(cfg, checkpoint, device):
    ec = json.loads((trainer.resolve(cfg.model.vision_encoder.checkpoint_path) / 'config.json').read_text())
    with forbid_encoder() as runner:
        action = create_action_model(cfg, dino_hidden_size=ec['hidden_size'],
                                     num_dino_layers=len(cfg.model.vision_encoder.feat_layers),
                                     task_cond_dim=ec['hidden_size'], patch_size=ec['patch_size'])
        action.load_state_dict(checkpoint['model_state_dict'], strict=True)
        w = BehaviorVLAWrapper(runner.VLAWrapper, vision_encoder=None, action_model=action,
            time_sampler=cfg.training.get('time_sampler', 'logit_normal'),
            feat_layers=list(cfg.model.vision_encoder.feat_layers), include_cls_register=True,
            num_register_tokens=ec['num_register_tokens'], device=torch.device(device), dtype=torch.float32,
            norm_stats_path=str(trainer.resolve(cfg.dataset.stats_path)), norm_stats_key='behavior1k',
            encoder_dtype=torch.bfloat16, proprio_mask=cfg.model.get('proprio_mask'),
            action_repr=cfg.model.get('action_repr'), views=list(cfg.model.views))
    # BehaviorVLAWrapper.train() calls vision_encoder.eval(); cached-only has no encoder.
    # The base implementation still sets every action-model child to eval mode.
    return torch.nn.Module.train(w.to(device), False)


@torch.no_grad()
def sample_v2(wrapper, feats, state, condition, generator):
    """Reuse the live V2 Euler sampler + real wrapper normalization, with cached features and ONLY current state."""
    from harness.policy_server import WamPolicy
    current = state[:, 0] if state.ndim == 3 else state
    if current.shape != (1, 61):
        raise ValueError('V2 requires one raw CURRENT (61,) state, not normalized V3 differences/history')
    sampler = SimpleNamespace(wrapper=wrapper, model=SimpleNamespace(is_v3=False, proprio_lags=[0], name='v2'),
                              num_steps=EULER_STEPS, gen=generator)
    result = WamPolicy._sample_once(sampler, feats, current[0].cpu().numpy(), condition)
    return torch.from_numpy(result)[None]


@torch.no_grad()
def evaluate_fine(wrapper, dataset, plan, kin, architecture):
    """Same trainer noise_seed/action_eef/hold recipe. No V3-only gate or projection diagnostics fabricated for V2."""
    device = torch.device(wrapper.device)
    rows = []
    for rec in sorted(plan['fine'], key=lambda r: (r['episode'], r['anchor'])):
        b = default_collate([dataset[rec['index']]])
        if (int(b['episode'][0]), int(b['anchor'][0]), int(b['phase'][0]), int(b['active_hand'][0])) != (
                rec['episode'], rec['anchor'], rec['phase'], rec['active_hand']):
            raise ValueError('cached sample identity/labels do not match fixed plan')
        features = b['feats'].to(device).view(torch.bfloat16)
        feats = [features[:, :, i] for i in range(features.shape[2])]
        seed = trainer.noise_seed(plan['seed'], rec['episode'], rec['anchor'])
        gen = torch.Generator(device=device).manual_seed(seed)
        if architecture == 'v2':
            pred = sample_v2(wrapper, feats, b['state'], b['task_cond'], gen)
        elif architecture == 'v3':
            pred, _ = wrapper.sample_actions(feats, b['state'], b['pool'], b['task_cond'],
                                            num_steps=EULER_STEPS, generator=gen)
        else:
            raise ValueError('unknown architecture')
        pred = pred.float().cpu()
        target = b['action_sequence'].float()
        held = hold_baseline(b['state'].float(), chunk_len=32).cpu()
        if pred.shape != (1, 32, 23):
            raise ValueError('K1 requires one (1,32,23) action chunk')
        for name, x in (('predicted action', pred), ('target', target), ('hold', held)):
            trainer.finite(x, name)
        hand = rec['active_hand']
        gt = trainer.action_eef(kin, target[0, 31].numpy(), hand)
        err = lambda x: float(np.linalg.norm(trainer.action_eef(kin, x[0, 31].numpy(), hand) - gt))
        rows.append({**rec, 'noise_seed': seed, 'fine_eef_m': err(pred), 'fine_hold_eef_m': err(held),
                     'action_mae': float((pred - target).abs().mean())})
    return rows


def registered_plan_digest(plan):
    # Legacy trainer hashed in-memory label_report with integer episode keys; JSON converts
    # them to strings and changes sort ordering. Reconstruct only this known old schema.
    old = dict(plan)
    old["label_report"] = {int(k): v for k, v in plan.get("label_report", {}).items()}
    return trainer.digest_json(old)


def reuse_v3_rows(path, plan, checkpoint):
    """Reuse trainer validation_<step>.json fine rows; no V3 model build or inference."""
    payload = json.loads(Path(path).read_text())
    report = payload["report"]
    if (payload.get("mode") != "pilot" or int(payload["step"]) != int(checkpoint["step"])
            or report.get("plan_sha256") not in (trainer.digest_json(plan), registered_plan_digest(plan))
            or report.get("samples_per_anchor") != 1 or report.get("terminal_step") != 31
            or report.get("split") != "validation"
            or int(checkpoint["config"]["common"]["num_inference_steps"]) != EULER_STEPS):
        raise ValueError("V3 result must match pilot checkpoint step, exact validation plan, K1 and Euler10")
    selected = [r for r in payload["rows"] if r.get("is_fine")]
    by_key = {(r["episode"], r["anchor"]): r for r in selected}
    want = {(r["episode"], r["anchor"]): r for r in plan["fine"]}
    if len(selected) != len(want) or set(by_key) != set(want):
        raise ValueError("V3 result fine rows differ from the fixed plan")
    rows = []
    for key, rec in sorted(want.items()):
        row = by_key[key]
        if any(row[k] != rec[k] for k in ("index", "skill", "phase", "active_hand", "contract_start")):
            raise ValueError("V3 result fine metadata drift")
        for k in ("fine_eef_m", "fine_hold_eef_m"):
            trainer.finite(row[k], k)
        rows.append({**row, "noise_seed": trainer.noise_seed(plan["seed"], *key)})
    return rows


def compare(v2_rows, v3_rows):
    a = {(r['episode'], r['anchor']): r for r in v2_rows}
    b = {(r['episode'], r['anchor']): r for r in v3_rows}
    if not a or set(a) != set(b) or len(a) != len(v2_rows) or len(b) != len(v3_rows):
        raise ValueError('paired comparison requires identical unique anchors')
    paired = []
    for key, r in a.items():
        q = b[key]
        for field in ('skill', 'phase', 'active_hand', 'noise_seed', 'fine_hold_eef_m'):
            if r[field] != q[field]:
                raise ValueError(f'paired input/baseline mismatch: {field}')
        paired.append({'episode': key[0], 'anchor': key[1], 'skill': r['skill'],
                       'v2_eef_m': r['fine_eef_m'], 'v3_eef_m': q['fine_eef_m'],
                       'hold_eef_m': r['fine_hold_eef_m'], 'v3_minus_v2_m': q['fine_eef_m'] - r['fine_eef_m']})
    def summary(rs):
        mean = lambda k: float(np.mean([r[k] for r in rs]))
        return {'samples': len(rs), **{k: mean(k) for k in ('v2_eef_m', 'v3_eef_m', 'hold_eef_m', 'v3_minus_v2_m')},
                'v2_minus_hold_m': mean('v2_eef_m') - mean('hold_eef_m'),
                'v3_minus_hold_m': mean('v3_eef_m') - mean('hold_eef_m')}
    return {'overall': summary(paired), 'by_skill': {s: summary([r for r in paired if r['skill'] == s])
             for s in sorted({r['skill'] for r in paired})}, 'paired': paired,
            'interpretation': 'Offline paired FK proxy only, not physical/BDDL success or statistical significance.'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--v2-checkpoint', default=trainer.DEFAULT_INIT)
    ap.add_argument('--v3-checkpoint')
    ap.add_argument('--v3-result', help='reuse matching trainer validation_<step>.json fine rows; skip V3 inference')
    ap.add_argument('--out', help='NEW directory only; never overwrite pilot/old comparison outputs')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--training-finished', action='store_true', help='explicit acknowledgement; GPU lock still required externally')
    ap.add_argument('--check-plan-only', action='store_true', help='CPU JSON preflight; no weights/data/model loaded')
    args = ap.parse_args(argv)
    plan = json.loads(trainer.resolve(args.plan).read_text())
    validate_plan(plan)
    if args.check_plan_only:
        print(json.dumps({'plan': identity(args.plan), 'fine_samples': 256, 'samples': 1, 'euler_steps': 10,
                          'terminal_step': 31, 'plan_seed': plan['seed'], 'status': 'plan structure valid'}, indent=2))
        return 0
    if not args.v3_checkpoint or not args.out:
        ap.error('--v3-checkpoint and --out required for evaluation')
    if args.device == 'cuda' and not args.training_finished:
        ap.error('GPU comparison is for AFTER training; require --training-finished and external GPU lock')
    out = trainer.resolve(args.out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.70)
    if os.environ['P2_TASK_ID'] != '0':
        raise ValueError('this pinned V2 comparison is Task 0 only')
    ids = {'v2': identity(args.v2_checkpoint), 'v3': identity(args.v3_checkpoint), 'plan': identity(args.plan)}
    if ids['v2']['sha256'] != trainer.V2_SHA256:
        raise ValueError('V2 source is not the pinned original es-best checkpoint')
    v2ck = torch.load(trainer.resolve(args.v2_checkpoint), map_location='cpu', mmap=True, weights_only=False)
    v3ck = torch.load(trainer.resolve(args.v3_checkpoint), map_location='cpu', mmap=True, weights_only=False)
    c2, c3 = OmegaConf.create(v2ck['config']), OmegaConf.create(v3ck['config'])
    check_configs(c2, c3)
    manifest_path = trainer.resolve(args.v3_checkpoint).parent / 'manifest.json'
    trainer.require_hash(manifest_path, v3ck['manifest_sha256'])
    manifest = json.loads(manifest_path.read_text())
    accepted_plan_hashes = (trainer.digest_json(plan), registered_plan_digest(plan))
    if int(v2ck['step']) != 500 or manifest.get('validation_plan_sha256') not in accepted_plan_hashes:
        raise ValueError('V2 step or V3 source manifest registered validation plan mismatch')
    stats_path = trainer.resolve(c3.model.v3.stats_path)
    stats = json.loads(stats_path.read_text())
    if v3ck.get('v3_stats') != stats:
        raise ValueError('V3 training stats snapshot differs from current stats file')
    reused_rows = None
    if args.v3_result:
        result_path = trainer.resolve(args.v3_result)
        if result_path.parent != trainer.resolve(args.v3_checkpoint).parent:
            raise ValueError("reused V3 result must be from the checkpoint's own run directory")
        reused_rows = reuse_v3_rows(result_path, plan, v3ck)
        ids['v3_result'] = identity(result_path)
    d, v = c3.dataset, c3.model.v3
    dataset = WamV3Dataset('validation', feature_cache=trainer.resolve(d.feature_cache), views=list(c3.model.views),
        proprio_lags=list(v.proprio_lags), memory_lags=list(v.memory_lags),
        contract_targets=str(trainer.resolve(d.contract_targets)), contract_targets_sha256=d.contract_targets_sha256,
        seed=int(c3.training.seed))
    validate_dataset(dataset, plan)
    from harness.r1pro_kinematics import R1ProKinematics, URDF, OFFSETS
    kin = R1ProKinematics()
    sources = [Path(__file__), ROOT / 'scripts/train_wam_v3.py', ROOT / 'harness/wam_behavior.py',
               ROOT / 'harness/wam_v3.py', ROOT / 'harness/policy_server.py', ROOT / 'harness/r1pro_kinematics.py',
               stats_path, trainer.resolve(c2.dataset.stats_path), trainer.resolve(d.contract_targets), URDF, OFFSETS,
               trainer.resolve(c3.model.vision_encoder.checkpoint_path) / 'config.json',
               trainer.resolve(d.feature_cache) / 'cache_config.json']
    from harness import task_profile as tp
    sources += [tp.SPLIT, tp.SEGMENTS]
    versions = [identity(p) for p in sources]
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'version': VERSION, 'v3_result_reused': reused_rows is not None, 'checkpoints': ids, 'v2_step': int(v2ck['step']), 'v3_step': int(v3ck['step']),
                'sources': versions, 'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
                'samples': 1, 'euler_steps': 10, 'terminal_step': 31, 'plan_seed': plan['seed'],
                'noise_rule': 'train_wam_v3.noise_seed(plan.seed, episode, anchor)',
                'checkpoint_loading': 'strict; one action model resident at a time; no DINO weights',
                'validation_plan_sha256': trainer.digest_json(plan), 'device': args.device}
    trainer.atomic_json(out / 'metadata.json', metadata)
    w = build_v2(c2, v2ck, args.device)
    rows2 = evaluate_fine(w, dataset, plan, kin, 'v2')
    trainer.atomic_json(out / 'v2_rows.json', rows2)
    del w, v2ck
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    if reused_rows is not None:
        rows3 = reused_rows
    else:
        with forbid_encoder():
            w, _ = trainer.build_cached_wrapper(OmegaConf.to_container(c3), stats, None, None,
                                                 args.device, resume_checkpoint=v3ck)
        w = w.to(args.device).eval()
        rows3 = evaluate_fine(w, dataset, plan, kin, 'v3')
    trainer.atomic_json(out / 'v3_rows.json', rows3)
    # Refuse a completed comparison if checkpoints/stats/code changed while evaluating.
    for item in list(ids.values()) + versions:
        if trainer.stat_identity(item['path']) != item['stat']:
            raise RuntimeError(f'source changed during evaluation: {item["path"]}')
    report = compare(rows2, rows3)
    trainer.atomic_json(out / 'comparison.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'paired'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
