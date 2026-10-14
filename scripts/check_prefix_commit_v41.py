#!/usr/bin/env python3
"""Synthetic mechanism tests and real frozen-model preflight for hard prefix commitment (v41).

cpu(): exact algebraic properties of replacement inpainting plus the toy
"postponed closure" contrast against the v37 soft overlap.
gpu(): replay archived onboard observations of original-protocol seeds through
the unchanged 80k model with the committed sampler. Open-loop counterfactual on
a frozen observation stream; no physics, no score claim.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from harness import zhuhai_prefix_commit_v41 as c
from harness import zhuhai_overlap_guidance_v37 as g

ARCHIVE = ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_80k_extra20_20261011_v24'


def cpu():
    torch.manual_seed(0)
    M = torch.randn(16, 16)*.05
    b = torch.randn(16)*.01
    x0 = torch.randn(1, 32, 16)
    field = lambda x, t: x@M.T+b
    plain, _ = c.integrate(field, x0)
    expected = x0.clone()
    for i in range(10):
        expected = expected+(expected@M.T+b)/10
    assert torch.allclose(plain, expected, rtol=0, atol=1e-6), 'target=None must be the plain Euler sampler'
    target = torch.randn(1, 32, 16)
    seen = []
    def spy(x, t):
        seen.append((t, x[:, :24].clone()))
        return x@M.T+b
    out, traces = c.integrate(spy, x0, target)
    assert len(seen) == 10 and all(torch.allclose(xp, t*target[:, :24]+(1-t)*x0[:, :24], rtol=0, atol=1e-6)
                                   for t, xp in seen), 'pinned rows must follow the interpolant at every step'
    assert torch.equal(out[:, :24], target[:, :24]), 'committed prefix must be exact at tau=1'
    assert all(tr['suffix_flow_rms'] is not None for tr in traces)
    coupled = lambda x, t: x@M.T+b+.5*x.mean(dim=1, keepdim=True)
    out_a, _ = c.integrate(coupled, x0, target)
    shifted = target.clone(); shifted[:, :24] += 1.
    out_b, _ = c.integrate(coupled, x0, shifted)
    assert not torch.allclose(out_a[:, 24:], out_b[:, 24:], atol=1e-3), 'suffix must be conditioned on the pinned prefix'
    # Toy: the model always proposes closure (gripper 0) from chunk index 20 on; exec8.
    proposed = torch.ones(1, 32, 16); proposed[:, 20:, 7] = 0.
    zero = lambda x, t: torch.zeros_like(x)
    old = None; hard = []; hard_index = []
    for query in range(5):
        if old is None:
            chunk, _ = c.integrate(zero, proposed.clone())
        else:
            y = torch.zeros_like(proposed); y[:, :24] = old[:, 8:]
            chunk, _ = c.integrate(zero, proposed.clone(), y)
        hard.append(bool((chunk[:, :8, 7] <= .2).any()))
        closed = (chunk[0, :, 7] <= .2).nonzero()
        hard_index.append(int(closed[0]) if len(closed) else None)
        old = chunk
    # Closure persists once committed: after it reaches index 4 the whole pinned
    # region is closed, so the next queries report first-close index 0.
    assert hard == [False, False, True, True, True] and hard_index == [20, 12, 4, 0, 0], hard_index
    soft_old = proposed.clone(); soft = []
    weights = torch.from_numpy(g.mask())[None, :, None]
    for query in range(5):
        y = torch.zeros_like(soft_old); y[:, :24] = soft_old[:, 8:]
        soft_old, _ = g.integrate(zero, proposed.clone(), y, weights)
        soft.append(bool((soft_old[:, :8, 7] <= .2).any()))
    assert not any(soft), 'v37 soft overlap toy must stay unexecuted (documented in check_overlap_guidance_v37)'
    return dict(status='passed', plain_sampler_exact=True, interpolant_pinning_exact=True,
        suffix_conditioned_on_prefix=True, toy_postponed_closure_hard_commit_executed_prefix=hard,
        toy_hard_commit_first_close_index_per_query=hard_index, toy_soft_overlap_v37_executed_prefix=soft,
        limitations='synthetic mechanism verification, not a robot success result; '
                    'hard commitment executes stale plans up to 2.4s old by design')


def gpu(seeds, max_steps, out_dir):
    from scripts import gpu_lock
    from scripts import eval_zhuhai_wam_precision_v2 as original
    from harness import zhuhai_precision_v2_training as p
    from harness.robotwin_training_data import preprocess_rgb
    from harness.wam_behavior import load_upstream
    from harness.wam_v3 import V3Context
    from harness.wam_event_clock import EventClock
    plan = json.loads((ARCHIVE/'plan.json').read_text())
    owner = 'zhuhai-prefix-commit-preflight-v41'
    assert gpu_lock.owner() is None, 'GPU owner must be free'
    gpu_lock.acquire(owner, max_wait=0)
    report = dict(status='running', checkpoint_sha256=plan['checkpoint']['sha256'], seeds={}, max_steps=max_steps,
        scope='archived original-protocol observations replayed open-loop; counterfactual after the first executed chunk; no physics rollout or score claim')
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        wrapper, condition, contract = p.load_policy(plan['checkpoint']['path'], 'zhuhai_task2', 'cuda')
        _, runner = load_upstream(); cfg = contract['config']
        encoder, *_ = runner.ModelFactory.create_vision_encoder(
            str(ROOT/cfg['model']['vision_encoder']['checkpoint_path']), dtype=torch.bfloat16, device='cuda')
        wrapper.base.vision_encoder = encoder.eval().requires_grad_(False)
        v3 = cfg['model']['v3']
        for seed in seeds:
            directory = ARCHIVE/f'zhuhai_task2/seed_{seed}'
            recorded = [json.loads(l) for l in (directory/'actions.jsonl').read_text().splitlines()]
            sampler = c.PrefixCommitSampler(wrapper, condition, lambda: original.verify_owner(owner))
            context = V3Context(v3['proprio_lags'], v3['memory_lags'], original.base.MODEL_VIEWS, state_dim=16)
            context.reset(); clock = EventClock(**p.CLOCK_CONFIG)
            rows = []; started = time.monotonic()
            for step in range(min(max_steps, len(recorded))):
                with np.load(directory/f'onboard/observation_{step:06d}.npz', allow_pickle=False) as z:
                    measured = z['measured_joint_gripper_state'].copy(); command = z['commanded_joint_gripper_state'].copy()
                    pixels = np.stack([preprocess_rgb(z[v]) for v in original.base.VIEWS])
                event = clock.update(measured, command)
                with torch.inference_mode():
                    feats, pool = wrapper.encode(torch.from_numpy(pixels)[None])
                context.push(measured, pool[0].float().cpu().numpy())
                if step % 8:
                    continue
                value = sampler(feats, context.inputs(), seed, step, event)
                rec = np.asarray(recorded[step]['policy']['raw_denormalized_chunk'], dtype=np.float32)
                assert recorded[step]['policy']['replanned'] is True
                if step == 0:
                    assert np.array_equal(value, rec), 'first chunk changed'
                left, oleft = value[:, 7], rec[:, 7]
                def first(v):
                    idx = np.nonzero(v <= .2)[0]
                    return int(idx[0]) if len(idx) else None
                rows.append(dict(step=step, committed_first_close=first(left), original_first_close=first(oleft),
                    committed_close_in_executed=bool((left[:8] <= .2).any()), original_close_in_executed=bool((oleft[:8] <= .2).any()),
                    fresh_suffix_close=bool((left[24:] <= .2).any()) if step else None,
                    original_tail_close=bool((oleft[8:] <= .2).any()),
                    right_committed_close_in_executed=bool((value[:8, 15] <= .2).any()),
                    suffix_boundary=sampler.last_diagnostic['suffix_boundary'],
                    inference_seconds=sampler.last_diagnostic['inference_seconds']))
            exec_c = sum(r['committed_close_in_executed'] for r in rows)
            exec_o = sum(r['original_close_in_executed'] for r in rows)
            any_o = sum(r['original_first_close'] is not None for r in rows)
            fresh_c = sum(bool(r['fresh_suffix_close']) for r in rows)
            jumps = [r['suffix_boundary']['position_jump_rad'] for r in rows if r['suffix_boundary']]
            within = [r['suffix_boundary']['within_prefix_step_rad'] for r in rows if r['suffix_boundary']]
            report['seeds'][str(seed)] = dict(queries=len(rows), first_chunk_exact_original=True,
                original_queries_with_close_anywhere=any_o, original_queries_with_close_in_executed=exec_o,
                committed_queries_with_close_in_executed=exec_c, committed_queries_fresh_suffix_close=fresh_c,
                first_committed_executed_close_step=next((r['step'] for r in rows if r['committed_close_in_executed']), None),
                first_original_executed_close_step=next((r['step'] for r in rows if r['original_close_in_executed']), None),
                suffix_boundary_jump_rad=dict(mean=float(np.mean(jumps)) if jumps else None, p95=float(np.percentile(jumps, 95)) if jumps else None, max=float(np.max(jumps)) if jumps else None),
                within_prefix_step_rad=dict(mean=float(np.mean(within)) if within else None, p95=float(np.percentile(within, 95)) if within else None),
                mean_inference_seconds=float(np.mean([r['inference_seconds'] for r in rows])),
                wall_seconds=time.monotonic()-started, rows=rows)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir)/f'prefix_commit_preflight_v41_seed{seed}.json').write_text(json.dumps(report['seeds'][str(seed)], indent=1)+'\n')
        assert all(p_.grad is None for p_ in wrapper.parameters())
        report.update(status='passed', parameter_gradients_absent=True)
        return report
    finally:
        gpu_lock.release(owner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[932106, 932120])
    parser.add_argument('--max-steps', type=int, default=2400)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out); assert not path.exists(), 'fresh output path required'
    result = dict(cpu=cpu(), gpu=gpu(args.seeds, args.max_steps, path.parent/'prefix_commit_preflight_v41_rows') if args.gpu else None,
                  source_sha256=c.implementation_sha256(), check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'gpu'} | dict(gpu={k: v for k, v in (result['gpu'] or {}).items() if k != 'seeds'} | dict(seeds={s: {k: v for k, v in r.items() if k != 'rows'} for s, r in (result['gpu'] or {}).get('seeds', {}).items()})), ensure_ascii=False, indent=1))
