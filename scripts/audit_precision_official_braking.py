#!/usr/bin/env python3
"""CPU-only command audit over TRAIN experts and retained failed policy outputs."""
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import h5py, numpy as np
from scripts import eval_zhuhai_precision_fixed_attempts as evidence
from scripts import eval_zhuhai_precision_official_braking as evaluator
from harness import zhuhai_official_braking as braking
from harness import zhuhai_action_smoothing as old


def replay(initial, targets, settings):
    limiter = braking.BrakingRateLimiter(settings['rate_limit']); limiter.reset(initial)
    changed = events = count = 0; max_delta = 0.
    for target in targets:
        actual, stats = limiter.step(target)
        count += 1; changed += int(stats['arm_limit_delta_max_rad'] > 1e-5)
        events += int(stats['braking_limited_dimensions'] > 0)
        max_delta = max(max_delta, stats['arm_limit_delta_max_rad'])
        lo, hi = limiter.feasible_interval()
        evidence.require(np.all(lo <= hi+1e-10), 'No legal future stop after accepted target')
    # Check a finite legal stopping tail after the last retained target.
    for _ in range(12):
        target = np.asarray(initial, dtype=np.float64).copy(); target[old.ARM] = limiter.previous
        limiter.step(target)
    evidence.require(np.all(np.abs(limiter.velocity) < 1e-9), 'Stopping tail did not settle')
    return dict(control_steps=count, changed_steps=changed, braking_steps=events,
                max_adjustment_rad=max_delta, feasible_stopping_tail=True, aborted=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); out = Path(args.out).resolve()
    evidence.require(not out.exists(), 'Keep previous audit immutable')
    meta, frozen = evidence.checkpoint_metadata(args.checkpoint); settings = braking.protocol()
    evaluator.configure()
    changed, receipt = evaluator.read_contract(args.checkpoint)
    evidence.require(changed['action_postprocessing'] == settings and receipt['sha256'] == meta['sha256'], 'Evaluator overlay failed')
    rows = []
    for e in frozen['splits']['zhuhai_task2']['train']:
        with h5py.File(e['path']) as h: q = np.asarray(h['target_command'], dtype=np.float64)
        raw = q[1:]
        result = replay(q[0], raw, settings)
        smoothed = []
        for t in range(0, len(raw), 8):
            ids = np.minimum(np.arange(t, t+32), len(raw)-1)
            smoothed.extend(old.smooth_chunk(raw[ids], settings)[:min(8, len(raw)-t)])
        spline = replay(q[0], smoothed, settings)
        rows.append(dict(seed=e['seed'], domain=e['domain'], source_sha256=e['sha256'],
                         direct=result, original_spline_exec8=spline))
    failures = []
    for summary_path in [evidence.RUN.parent/'wam_precision_v2_task2_30k_continuation_20261010_r1/development_20000/summary.json',
                         evidence.RUN.parent/'wam_precision_v2_task2_final30k_trainingfirst_20261010/summary.json']:
        summary = evidence.read_json(summary_path)
        for e in summary['episodes']:
            directory = Path(e['raw_result_path']).parent
            with np.load(directory/'onboard/observation_000000.npz', allow_pickle=False) as z:
                initial = z['commanded_joint_gripper_state'].copy()
            targets = [json.loads(line)['smoothed_action'] for line in (directory/'actions.jsonl').read_text().splitlines()]
            failures.append(dict(step=summary['checkpoint_step'], seed=e['seed'], original_status=e['status'],
                retained_actions_sha256=evidence.sha256_file(directory/'actions.jsonl'),
                replay=replay(initial, targets, settings),
                scope='Counterfactual command replay only; original observations/model outputs, not new closed-loop success'))
    result = dict(status='passed', gpu_used=False, checkpoint_sha256=meta['sha256'],
        checkpoint_step=meta['step'], action_postprocessing=settings,
        action_postprocessing_sha256=evidence.digest(settings), train_episodes=len(rows), train_audits=rows,
        retained_failure_episodes=len(failures), retained_failure_audits=failures,
        original_training_sources_verified=True, model_and_checkpoint_modified=False,
        physical_evaluation_performed=False, heldout_used=False,
        original_source_hashes=frozen['source_files'],
        extension_sources={str(Path(x).resolve()):evidence.sha256_file(x)
            for x in [braking.__file__, evaluator.__file__, __file__]})
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_json(out, result)
    print(json.dumps(dict(status='passed', train_episodes=len(rows), retained_failure_episodes=len(failures),
        aborted_expert_replays=0, aborted_retained_command_replays=0, GPU=False, out=str(out)), ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
