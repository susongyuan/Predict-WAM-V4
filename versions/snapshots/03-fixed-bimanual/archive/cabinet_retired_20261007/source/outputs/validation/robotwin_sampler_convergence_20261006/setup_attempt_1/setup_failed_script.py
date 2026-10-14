#!/usr/bin/env python3
"""CPU-only paired Euler10/20/40 diagnostic on the first eight sorted VAL starts.

No CLI knobs for checkpoint, subset, seeds, or output: these are fixed before
sampling. Reuses check_robotwin_start_states.read_anchor/build_wrapper and the
unmodified production sample_actions. Does not import an encoder or simulator,
train, use physical outcomes, or read other episode payloads. --self-test runs
synthetic checks only; --verify rechecks the saved numerical artifacts without
loading a model. Only the assigned output directory is writable at runtime.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import gc
import json
import math
import signal
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# First numerical import. This helper owns the one-time torch inter-op setup;
# never call set_num_interop_threads again or dynamically reload the helper.
from scripts import check_robotwin_start_states as base
np, torch, psutil = base.np, base.torch, base.psutil
require = base.require
OUT = ROOT / 'outputs/validation/robotwin_sampler_convergence_20261006'
CKPT = base.RUN / 'checkpoint_step_030000.pt'
STEPS = (10, 20, 40)
COUNT = 8
SEED = 20261004
WINDOWS = {'exec8': 8, 'chunk32': 32}
GROUPS = {'arms_rad': [*range(6), *range(7, 13)],
          'left_arm_rad': list(range(6)), 'right_arm_rad': list(range(7, 13)),
          'left_gripper_units': [6], 'right_gripper_units': [13],
          'grippers_units': [6, 13]}
THRESHOLDS = (0., 1e-4, 1e-3)
RSS_LIMIT = 6 * 1024**3
MIN_AVAILABLE = 4 * 1024**3


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))))) if np.size(x) else None


def trajectory_metrics(prediction, target, valid, commands):
    """Pool scalar differences within valid prefixes, never across episodes.

    Frame-based finite differences (NOT velocities in seconds). A reversal is
    two adjacent increments with opposite signs and BOTH abs(delta)>epsilon.
    The denominator counts pairs with both increments above epsilon.
    """
    result = {}
    for group, cols in GROUPS.items():
        diffs, derr = {d: [] for d in (1, 2, 3)}, {d: [] for d in (1, 2)}
        jumps, tv, reversals = [], [], {str(e): [0, 0] for e in THRESHOLDS}
        for p, t, mask, command in zip(prediction, target, valid, commands):
            n = int(mask.sum())
            require(n > 0 and np.array_equal(mask, np.arange(len(mask)) < n), 'valid mask must be a nonempty prefix')
            p = p[:n, cols].astype(np.float64)
            t = t[:n, cols].astype(np.float64)
            jumps.append(p[0] - command[cols])
            delta = np.diff(p, axis=0)
            tv.append(np.abs(delta).sum(axis=0))
            for d in diffs:
                diffs[d].append(np.diff(p, n=d, axis=0).reshape(-1))
                if d in derr:
                    derr[d].append((np.diff(p, n=d, axis=0) - np.diff(t, n=d, axis=0)).reshape(-1))
            for e in THRESHOLDS:
                eligible = (np.abs(delta[:-1]) > e) & (np.abs(delta[1:]) > e)
                count = eligible & (delta[:-1] * delta[1:] < 0)
                reversals[str(e)][0] += int(count.sum())
                reversals[str(e)][1] += int(eligible.sum())
        result[group] = {
            'first_command_jump_rms': rms(jumps),
            'first_command_jump_max_abs': float(np.max(np.abs(jumps))),
            'total_variation_mean_per_episode_channel': float(np.mean(tv)),
            **{f'difference{d}_rms': rms(np.concatenate(x)) for d, x in diffs.items()},
            **{f'difference{d}_scalar_count': int(sum(len(a) for a in x)) for d, x in diffs.items()},
            **{f'difference{d}_error_rms_vs_target': rms(np.concatenate(x)) for d, x in derr.items()},
            'direction_reversals': {e: {'count': c, 'eligible_pairs': n,
                'fraction': c / n if n else None} for e, (c, n) in reversals.items()}}
    return result


def predictions_distance(a, b, valid, scale):
    """Numerical sensitivity, not error to ground truth: Euler40 is not truth."""
    diff = (a - b)[valid].astype(np.float64)
    return {'normalized_rmse': rms(diff / scale),
            **{group + '_rmse': rms(diff[:, cols]) for group, cols in GROUPS.items()}}


def scalar_delta(before, after):
    return {'before': before, 'after': after, 'delta': after - before,
            'relative_reduction': 1 - after / before if before else None}


def self_tests():
    """Synthetic checks of masks, pooling, reversals, bounds, and REAL sampler."""
    from harness.wam_v3 import WamV3Wrapper
    passed = []
    command = np.zeros((2, 14), dtype=np.float32)
    p = np.zeros((2, 4, 14), dtype=np.float32)
    p[0, :, 0] = [0, 1, 0, 1]
    p[1, :, 0] = [0, 2, 999, 999]
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    m = trajectory_metrics(p, p, mask, command)['left_arm_rad']
    require(m['direction_reversals']['0.001'] == {'count': 2, 'eligible_pairs': 2, 'fraction': 1.}, 'reversal or prefix masking test')
    require(math.isclose(m['difference1_rms'], math.sqrt(7/24)), 'scalar-weighted difference pooling test')
    require(m['difference1_error_rms_vs_target'] == 0 and m['first_command_jump_rms'] == 0, 'derivative error/start jump test')
    require(math.isclose(m['total_variation_mean_per_episode_channel'], 5/12), 'total variation test')
    passed.append('finite differences, ragged prefix masks, reversals, start jump, scalar pooling')
    tiny = np.zeros((1, 4, 14), dtype=np.float32)
    tiny[0, :, 0] = [0, 1e-5, 0, 1e-5]
    mt = trajectory_metrics(tiny, tiny, mask[:1], command[:1])['left_arm_rad']
    require(mt['direction_reversals']['0.0']['count'] == 2 and mt['direction_reversals']['0.0001']['count'] == 0, 'deadband test')
    short = trajectory_metrics(tiny[:, :1], tiny[:, :1], mask[:1, :1], command[:1])['arms_rad']
    require(short['difference1_rms'] is None and short['direction_reversals']['0.0']['fraction'] is None, 'short prefix test')
    bad_mask = mask.copy(); bad_mask[0] = [1, 0, 1, 0]
    try:
        trajectory_metrics(p, p, bad_mask, command)
    except RuntimeError:
        pass
    else:
        raise AssertionError('non-prefix mask accepted')
    passed.append('deadbands, one-frame edge case, non-prefix rejection')
    bound_input = np.zeros((2, 14), dtype=np.float32)
    bound_input[:, 6] = [-.1, 1.2]
    require(base.bounds(bound_input)['both']['out_of_range_count'] == 2, 'gripper bounds test')
    require(predictions_distance(p, p, mask, np.ones(14))['normalized_rmse'] == 0, 'distance identity test')
    passed.append('unclipped gripper bounds and pairwise distance')

    class Field(torch.nn.Module):
        action_len = 32
        def __init__(self):
            super().__init__(); self.x0s = []; self.calls = 0
        def forward(self, t, **kw):
            if float(t[0]) == 0:
                self.x0s.append(kw['noisy_actions'].clone())
            self.calls += 1
            x = kw['noisy_actions']
            return {'final_pred': x, 'gate': torch.zeros(1), 'phase_logits': torch.zeros(1), 'log_dist': torch.zeros(1)}
    class Stub:
        device = torch.device('cpu')
        action_min = torch.zeros(14)
        action_model = Field()
        normalize_state = staticmethod(lambda x: x)
        normalize_memory = staticmethod(lambda x: x)
        denormalize_action = staticmethod(lambda x: x)
    stub = Stub()
    g = torch.Generator(device='cpu').manual_seed(SEED)
    expected = torch.randn((1, 32, 14), generator=g)
    errors = []
    for steps in STEPS:
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        pred, _ = WamV3Wrapper.sample_actions(stub, [torch.zeros(1)], torch.zeros(1, 7, 14),
            torch.zeros(1), torch.zeros(1, 1024), num_steps=steps, generator=generator)
        require(torch.equal(stub.action_model.x0s[-1], expected), 'sampler x0 pairing test')
        require(torch.equal(generator.get_state(), g.get_state()), 'sampler RNG consumption test')
        require(torch.allclose(pred, expected * (1 + 1/steps)**steps, rtol=2e-6, atol=2e-6), 'production Euler recurrence test')
        errors.append(float((pred - math.e * expected).square().mean()))
    require(errors[0] > errors[1] > errors[2] and stub.action_model.calls == sum(STEPS), 'known ODE convergence test')
    passed.append('production sampler: paired x0/RNG, 70 forwards, known Euler ODE convergence')
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2 and not torch.cuda.is_initialized(), 'CPU thread test')
    passed.append('single helper thread initialization, two CPU threads, no CUDA')
    return {'status': 'passed', 'checks': passed, 'known_ode_squared_errors': errors}


def summarize_arrays(pred, target, valid, commands, scale, wrapper):
    """Metrics recomputable from stored raw arrays; no policy/label mixing."""
    out = {'errors': {}, 'smoothness': {}, 'raw_gripper_bounds': {}, 'pairwise': {}}
    hold = np.repeat(commands[:, None, :], 32, axis=1)
    for window, h in WINDOWS.items():
        mask, truth = valid[:, :h], target[:, :h]
        metrics, smooth, bounds = {}, {}, {}
        for key, values in [('target', target), ('hold', hold), *[(str(s), pred[:, i]) for i, s in enumerate(STEPS)]]:
            p = values[:, :h]
            # Reuse exact trainer float32 metrics and helper row weighting.
            rows = [base.action_metrics(torch.from_numpy(a[None]), torch.from_numpy(t[None]),
                        torch.from_numpy(v[None]), wrapper) for a, t, v in zip(p, truth, mask)]
            metrics[key] = base.summarize(rows)
            smooth[key] = trajectory_metrics(p, truth, mask, commands)
            bounds[key] = base.bounds(p[mask])
        out['errors'][window], out['smoothness'][window], out['raw_gripper_bounds'][window] = metrics, smooth, bounds
        out['pairwise'][window] = {f'{a}_vs_{b}': predictions_distance(pred[:, STEPS.index(a), :h],
            pred[:, STEPS.index(b), :h], mask, scale) for a, b in ((10, 20), (20, 40), (10, 40))}
    return out


def paired_report(summary, rows):
    result = {}
    for window in WINDOWS:
        result[window] = {}
        for a, b in ((10, 20), (20, 40), (10, 40)):
            errors = summary['errors'][window]
            smooth = summary['smoothness'][window]
            deltas = [r['metrics']['errors'][window][str(b)]['normalized_mse'] -
                      r['metrics']['errors'][window][str(a)]['normalized_mse'] for r in rows]
            pair = {'normalized_mse': scalar_delta(errors[str(a)]['normalized_mse'], errors[str(b)]['normalized_mse']),
                    'per_episode_normalized_mse_deltas': deltas,
                    'episodes_improved': sum(d < 0 for d in deltas), 'episodes_worsened': sum(d > 0 for d in deltas),
                    'episodes_tied': sum(d == 0 for d in deltas), 'groups': {}}
            for group in GROUPS:
                ma, mb = smooth[str(a)][group], smooth[str(b)][group]
                pair['groups'][group] = {key: scalar_delta(ma[key], mb[key]) for key in
                    ('difference1_rms', 'difference2_rms', 'difference1_error_rms_vs_target',
                     'difference2_error_rms_vs_target', 'first_command_jump_rms') if ma[key] is not None and mb[key] is not None}
                pair['groups'][group]['reversals_epsilon_0.001'] = scalar_delta(
                    ma['direction_reversals']['0.001']['count'], mb['direction_reversals']['0.001']['count'])
            result[window][f'{a}_to_{b}'] = pair
        distances = summary['pairwise'][window]
        den = distances['10_vs_20']['normalized_rmse']
        result[window]['normalized_distance_20_40_over_10_20'] = distances['20_vs_40']['normalized_rmse'] / den if den else None
    return result


def write_summary(summary, paired, runtime):
    lines = ['# Fixed 30k sampler convergence — offline VAL start diagnostic', '',
             'Eight predetermined lexicographically first VAL episodes, frame 0; one seed per episode, identical x0 for Euler10/20/40.',
             'No physical results were read or used. No retraining, checkpoint/subset/seed selection, clipping, simulator, or GPU.', '',
             '| Window | Steps | Norm MSE | Arm Δ1 RMS | Arm Δ2 RMS | Arm reversals (>0.001 rad/frame) | Gripper outside [0,1] |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for window in WINDOWS:
        for step in ('target', 'hold', '10', '20', '40'):
            err = summary['errors'][window][step]['normalized_mse']
            sm = summary['smoothness'][window][step]['arms_rad']
            gr = summary['raw_gripper_bounds'][window][step]['both']
            lines.append(f"| {window} | {step} | {err:.7f} | {sm['difference1_rms']:.6f} | {sm['difference2_rms']:.6f} | {sm['direction_reversals']['0.001']['count']} | {gr['out_of_range_count']}/{gr['scalar_count']} |")
    lines += ['', '## Paired findings']
    for window in WINDOWS:
        for pair in ('10_to_20', '10_to_40'):
            item = paired[window][pair]
            lines.append(f"- {window} {pair}: normalized MSE reduction {item['normalized_mse']['relative_reduction']:.2%}; improved {item['episodes_improved']}/8 starts; arm Δ2 RMS reduction {item['groups']['arms_rad']['difference2_rms']['relative_reduction']:.2%}.")
        lines.append(f"- {window}: ||Euler40−Euler20|| / ||Euler20−Euler10|| normalized RMS = {paired[window]['normalized_distance_20_40_over_10_20']:.3f}; Euler40 is not a reference solution.")
    e = paired['exec8']['10_to_40']
    supported = (e['normalized_mse']['relative_reduction'] > 0 and
                 e['groups']['arms_rad']['difference2_rms']['relative_reduction'] > 0 and
                 e['episodes_improved'] >= 6)
    lines += ['', '## Interpretation',
              ('The paired offline directions support a bounded higher-step real-inference test, not adoption or a success claim.' if supported else
               'These offline results do not consistently support higher steps as a remedy for noisy first chunks; any real-inference test remains exploratory.'),
              'Coarse Euler sampling can contribute only to the measured step-dependent component. Residual error/jitter can come from the fixed model, single-sample uncertainty, reset history, or data mismatch; this diagnostic cannot identify those causes.',
              'Small fixed subset (8/50), one CPU float32 seed per start, no confidence interval or population claim. Commanded state is not measured qpos. No simulated/physical success or visual causality claim.',
              'Differences are frame-based, not rad/s; smoothness alone is not correctness. Target/hold baselines and derivative errors are included. Reversal counts include explicit 0, 1e-4, 1e-3 deadbands. Gripper ranges are raw diagnostics, not proof of failure.',
              'All history is clamped to frame 0. Only expert action[0:32] is read for scoring; it never enters the sampler. No future visual features, dev/heldout episode payloads, or encoder weights are read.',
              '', '## Resources', f"CPU threads/inter-op: 2/2. CUDA initialized: {runtime['cuda_initialized']}. Elapsed: {runtime['elapsed_seconds']:.1f}s. Peak RSS: {runtime['sampled_resource_peaks']['process_rss_bytes']/1024**3:.2f} GiB. Peak system RAM: {runtime['sampled_resource_peaks']['system_ram_percent']:.1f}%.", '']
    (OUT / 'summary.md').write_text('\n'.join(lines))
    return supported


def run():
    require(not OUT.exists(), f'refuse overwrite/retry: {OUT}; use --verify to inspect completed output')
    OUT.mkdir(parents=True, exist_ok=False)
    base.OUT, base.CKPT = OUT, CKPT
    base.START = time.monotonic()
    base.READS.clear(); base.SOURCES.clear()
    rows, telemetry, opened = [], [], {}
    allowed_hdfs = set()
    # Supplement the precise native HDF/cache array ledger with Python open
    # paths. Audit hooks do not see all native-library/mmap reads; say so.
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            require(path.is_relative_to(OUT) or str(path) == '/dev/null', f'write outside output denied: {path}')
        elif path.is_file():
            if path.suffix in ('.hdf5', '.h5'):
                require(str(path.resolve()) in allowed_hdfs, f'nonselected HDF read denied: {path}')
            entry = opened.setdefault(str(path), {'read_open_count': 0})
            entry['read_open_count'] += 1
    sys.addaudithook(audit)
    def check(*unused):
        base.check()
        require(base.PROCESS.memory_info().rss < RSS_LIMIT, '6 GiB process RSS safety cap')
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'thread settings changed')
        if not telemetry or time.monotonic() - telemetry[-1]['monotonic'] >= .9:
            mem = psutil.virtual_memory()
            telemetry.append({'monotonic': time.monotonic(), 'elapsed_seconds': time.monotonic()-base.START,
                              'ram_percent': mem.percent, 'available_bytes': mem.available,
                              'rss_bytes': base.PROCESS.memory_info().rss})
    signal.signal(signal.SIGALRM, check)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda n, f: (_ for _ in ()).throw(RuntimeError(f'signal {n}')))
    signal.setitimer(signal.ITIMER_REAL, 1, 1)
    try:
        check()
        vm = psutil.virtual_memory()
        require(vm.available >= MIN_AVAILABLE and .85 * vm.total - vm.used >= MIN_AVAILABLE, 'insufficient 4 GiB model/load headroom below RAM85')
        base.write_json('self_tests.json', self_tests())
        base.register(__file__); base.register(base.__file__)
        base.register(base.RUN / 'manifest.json')
        manifest = json.loads((base.RUN / 'manifest.json').read_text())
        contract = manifest['contract']; cfg = contract['config']
        require(base.digest(contract) == manifest['contract_sha256'], 'contract digest changed')
        require(base.register(base.CONFIG)['sha256'] == contract['config_sha256'], 'frozen config changed')
        split_path = ROOT / cfg['dataset']['split_manifest']
        require(base.register(split_path)['sha256'] == cfg['dataset']['split_sha256'] == contract['split_sha256'], 'split mismatch')
        split = json.loads(split_path.read_text())
        val_names = sorted(split['splits']['val']); names = val_names[:COUNT]
        require(len(val_names) == len(set(val_names)) == 50 and len(names) == COUNT, 'expected fixed 50 VAL starts')
        require(not set(val_names) & set().union(*(set(v) for k, v in split['splits'].items() if k != 'val')), 'VAL overlap with other splits')
        allowed_hdfs.update(str((Path(split['data_root']) / n).resolve()) for n in names)
        require(cfg['common']['action_chunk_size'] == 32 and cfg['common']['action_execution_horizon'] == 8
                and cfg['common']['num_inference_steps'] == 10 and cfg['common']['proprio_len'] == 7
                and cfg['dataset']['cond_mode'] == 'zero_single_task'
                and cfg['model']['v3']['proprio_lags'] == [2, 4, 8, 16, 32]
                and cfg['model']['v3']['memory_lags'] == [4, 16], 'unexpected frozen recipe')
        protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint': str(CKPT), 'checkpoint_step': 30000, 'split': 'val', 'available_val_starts': 50,
            'selection': 'first 8 lexicographically sorted VAL episode names; frame 0; predetermined, never physical-test selection',
            'episode_plan': [{'episode': n, 'ordinal': i, 'anchor': 0, 'seed': SEED+i} for i, n in enumerate(names)],
            'Euler_steps_in_execution_order': list(STEPS), 'K': 1, 'windows': WINDOWS,
            'pairing': 'fresh CPU Generator(seed) for each step count; actual first model input x0 bitwise-checked against independent seeded torch.randn; post-sampler RNG state checked',
            'policy_inputs': ['frame0 cached features', 'frame0 commanded state repeated 7', 'frame0 pool repeated 4', 'zero task condition'],
            'scoring_only': 'action[0:min(32,length)] with prefix masks, never passed to sampler',
            'smoothness': 'per-group finite differences 1/2/3 pooled within valid prefixes only; native units/frame^order; no episode boundary differences; target and hold baselines',
            'reversal_deadbands_native_units_per_frame': list(THRESHOLDS), 'raw_unclipped_grippers': [6, 13],
            'precision': 'CPU float32; cached BF16 bits decoded by reused read_anchor',
            'cpu_threads': 2, 'interop_threads': 2, 'CUDA_VISIBLE_DEVICES': '',
            'stop_limits': {'system_ram_percent': 85, 'system_cpu_percent': 85, 'process_rss_bytes': RSS_LIMIT,
                'minimum_preload_headroom_bytes': MIN_AVAILABLE, 'total_seconds': 1800,
                'per_sampling_call_seconds': 120, 'first_anchor_triplet_seconds': 180},
            'interpretation_rule': 'descriptive support for a bounded real-inference test only if exec8 10->40 reduces both normalized MSE and arm second-difference RMS and improves MSE on >=6/8 starts; no physical-success inference',
            'no_training_or_selection': True, 'boundary': 'offline prediction, not physical or closed-loop success'}
        base.write_json('protocol.json', protocol)
        for path, expected in contract['source_hashes'].items():
            require(base.register(path)['sha256'] == expected, f'frozen source changed: {path}')
        paths = {k: str(base.RUN / f'{k}.json') for k in ('normalization', 'temporal_stats')}
        stats = {}
        for k, p in paths.items():
            require(base.register(p)['sha256'] == contract['stats'][k]['sha256'], 'statistics changed')
            stats[k] = json.loads(Path(p).read_text()); pr = stats[k]['provenance']
            require(pr['fit_split'] == 'train' and pr['episodes'] == contract['episodes']['train']
                and pr['episode_count'] == len(split['splits']['train'])
                and pr['split_manifest_sha256'] == base.data_digest(split)
                and pr['cache_config_id'] == contract['cache_config_id'], 'TRAIN normalization provenance changed')
        cache_root, config_id = base.resolve_cache(split, ROOT / cfg['dataset']['feature_cache'])
        require(config_id == contract['cache_config_id'], 'cache identity mismatch')
        require(base.register(cache_root / 'cache_config.json') == contract['cache_config'], 'cache config changed')
        frozen_cache = {x['path']: x for x in contract['cache_files']}
        identities = {x['episode']: x for x in contract['episodes']['val']}
        require(set(identities) == set(val_names), 'frozen VAL membership mismatch')
        receipt_path = CKPT.with_name(CKPT.name + '.sha256.json'); base.register(receipt_path)
        receipt = json.loads(receipt_path.read_text()); ck_identity = base.register(CKPT)
        require(ck_identity['sha256'] == receipt['sha256'] and ck_identity['size'] == receipt['size']
                and receipt['step'] == 30000 and receipt['model_only'] is True, 'checkpoint receipt mismatch')
        check()
        checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
        require(checkpoint['step'] == 30000 and checkpoint['model_only'] is True
                and checkpoint['contract'] == contract and checkpoint['contract_sha256'] == manifest['contract_sha256'], 'wrong checkpoint/contract')
        wrapper, load_notes = base.build_wrapper(cfg, paths, stats, 'cpu', checkpoint=checkpoint)
        wrapper.eval(); wrapper.requires_grad_(False)
        del checkpoint; gc.collect()
        require(load_notes == {'strict_resume': True, 'step': 30000} and wrapper.vision_encoder is None, 'unexpected load/encoder')
        require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in wrapper.parameters()), 'model not CPU float32')
        require(not any(m.training for m in wrapper.modules()), 'model must be eval throughout')
        scale = (wrapper.normalize_action(torch.ones((1, 14))) - wrapper.normalize_action(torch.zeros((1, 14)))).numpy()[0]
        require(np.all(np.isfinite(scale)) and np.all(scale > 0), 'invalid normalization slope')
        # Distance routine divides by scale; save the physical span per unit of normalized action.
        scale = 1. / scale
        calls = {'action_model_forward': 0, 'future_decoder_forward': 0}
        current = {}
        def model_check(module, args, kwargs):
            check()
            require(time.monotonic() - current['started'] < 120, 'sampling call exceeded 120 seconds')
            require(set(kwargs) == {'noisy_actions', 'qpos_history', 'dino_features_list', 'task_cond', 'memory'}, 'unexpected sampler conditioning')
            require(bool((kwargs['task_cond'] == 0).all()), 'nonzero task condition')
            index, steps = current['calls'], current['steps']
            require(torch.equal(args[0], torch.full((1,), index / steps)), 'wrong Euler time grid')
            if index == 0:
                require(torch.equal(kwargs['noisy_actions'], current['x0']), 'actual x0 differs across step counts')
                current['observed_x0_sha256'] = base.array_sha(kwargs['noisy_actions'].numpy())
            current['calls'] += 1; calls['action_model_forward'] += 1
        def forbid_future(*unused):
            calls['future_decoder_forward'] += 1
            raise RuntimeError('future feature decoder must not execute')
        wrapper.action_model.register_forward_pre_hook(model_check, with_kwargs=True)
        wrapper.action_model.future_feat_decoder.register_forward_pre_hook(forbid_future)
        # Hash project modules used by the imported numerical routines, not environments.
        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and any(Path(path).resolve().is_relative_to(ROOT / sub) for sub in ('harness', 'scripts')) and Path(path).is_file():
                base.register(path)
        def provenance():
            return {'source_files': base.SOURCES, 'run_contract_sha256': manifest['contract_sha256'],
                'cache_config_id': config_id, 'load_notes': load_notes, 'TRAIN_stats_provenance': stats['normalization']['provenance'],
                'verification': 'fresh checkpoint SHA vs receipt; source/config/stat/meta hashes; whole HDF hash inherited from frozen run, NOT freshly rehashed; exact used arrays hashed; cache payload stat/header checked, NOT whole cache hash',
                'import_initialization': 'helper imported before numerical work; no duplicate inter-op setter and no monkeypatch',
                'read_audit_limits': 'Python open ledger is supplemental, not a native syscall audit. HDF slices/cache mmap pages are documented separately in reused read_anchor ledger.'}
        base.write_json('provenance.json', provenance())
        preds, targets, masks, commands, noise = [], [], [], [], []
        for i, name in enumerate(names):
            check(); tick = time.monotonic()
            feats, state, pool, target, valid = base.read_anchor(identities[name], split, cache_root, frozen_cache)
            require(not wrapper.normalize_state(state)[:, 1:].any() and not wrapper.normalize_memory(pool).any(), 'nonzero reset history')
            expected_gen = torch.Generator(device='cpu').manual_seed(SEED+i)
            x0 = torch.randn((1, 32, 14), generator=expected_gen)
            row = {'episode': name, 'ordinal': i, 'anchor': 0, 'seed': SEED+i,
                   'expected_x0_sha256': base.array_sha(x0.numpy()), 'samplers': {}}
            anchor_preds = []
            for steps in STEPS:
                generator = torch.Generator(device='cpu').manual_seed(SEED+i)
                current.clear(); current.update(steps=steps, calls=0, x0=x0, started=time.monotonic())
                with torch.inference_mode():
                    pred, info = wrapper.sample_actions([feats[:, :, layer] for layer in range(3)], state, pool,
                        torch.zeros((1, 1024), dtype=torch.float32), num_steps=steps, generator=generator)
                seconds = time.monotonic() - current['started']
                require(seconds < 120 and current['calls'] == steps, 'sampling time/count bound')
                require(pred.shape == (1, 32, 14) and pred.dtype == torch.float32 and torch.isfinite(pred).all(), 'invalid prediction')
                require(torch.equal(generator.get_state(), expected_gen.get_state()), 'extra sampler RNG consumption')
                anchor_preds.append(pred.numpy()[0].copy())
                row['samplers'][str(steps)] = {'seconds': seconds, 'forward_count': current['calls'],
                    'actual_x0_sha256': current['observed_x0_sha256'], 'rng_state_sha256': base.array_sha(generator.get_state().numpy()),
                    'same_x0_bitwise_verified': True, 'same_rng_consumption_verified': True, 'gate': info['gate'].tolist()}
            elapsed = time.monotonic() - tick
            if i == 0:
                require(elapsed < 180, 'first anchor triplet exceeded 180 seconds')
                base.write_json('first_anchor.json', {'passed': True, 'episode': name, 'triplet_seconds': elapsed,
                    'prediction_shape_per_sampler': list(pred.shape), 'feature_shape': list(feats.shape),
                    'state_shape': list(state.shape), 'pool_shape': list(pool.shape), 'runtime': base.runtime()})
            preds.append(np.stack(anchor_preds)); targets.append(target.numpy()[0]); masks.append(valid.numpy()[0])
            commands.append(state.numpy()[0, 0]); noise.append(x0.numpy()[0])
            row['metrics'] = summarize_arrays(np.stack(anchor_preds)[None], target.numpy(), valid.numpy(), state.numpy()[:, 0], scale, wrapper)
            row['seconds_including_metrics'] = time.monotonic()-tick; rows.append(row)
            base.write_json('per_episode.json', rows)
            temp = OUT / 'predictions.tmp.npz'
            np.savez(temp, episode=np.array(names[:i+1]), steps=np.array(STEPS), prediction=np.stack(preds),
                target=np.stack(targets), valid=np.stack(masks), current_command=np.stack(commands),
                x0=np.stack(noise), seeds=np.arange(SEED, SEED+i+1, dtype=np.int64), normalized_action_scale=scale,
                action_min=wrapper.action_min.numpy(), action_max=wrapper.action_max.numpy())
            temp.replace(OUT / 'predictions.npz')
            base.write_json('resource_samples.json', telemetry)
            print(json.dumps({'event': 'anchor_complete', 'episodes': i+1, 'episode': name,
                  'seconds': row['seconds_including_metrics'], 'exec8_normalized_mse':
                  {str(s): row['metrics']['errors']['exec8'][str(s)]['normalized_mse'] for s in STEPS}}), flush=True)
        summary = summarize_arrays(np.stack(preds), np.stack(targets), np.stack(masks), np.stack(commands), scale, wrapper)
        paired = paired_report(summary, rows)
        require(calls == {'action_model_forward': COUNT*sum(STEPS), 'future_decoder_forward': 0}, 'unexpected model forward count')
        require([r['episode'] for r in base.READS] == names, 'read set differs from predetermined anchors')
        # Register additional small provenance reads by resolve_cache, excluding
        # outputs, dependencies and payloads already covered by the slice ledger.
        for path in list(opened):
            p = Path(path)
            if p.is_relative_to(ROOT) and not p.is_relative_to(OUT) and not any(part.startswith('.') for part in p.relative_to(ROOT).parts) and p.suffix in ('.py', '.json', '.yaml'):
                if str(p.resolve()) not in base.SOURCES:
                    base.register(p)
        for path, before in list(base.SOURCES.items()):
            require(base.file_identity(path) == {k: v for k, v in before.items() if k != 'sha256'}, f'input stat changed: {path}')
            if 'sha256' in before and path != str(CKPT):
                require(base.sha(path) == before['sha256'], f'input content changed: {path}')
        check()
        base.write_json('provenance.json', provenance())
        base.write_json('python_read_opens.json', {'paths': opened, 'limitations': provenance()['read_audit_limits']})
        base.write_json('resource_samples.json', telemetry)
        base.write_json('paired_comparisons.json', paired)
        runtime = base.runtime()
        supported = write_summary(summary, paired, runtime)
        result = {'status': 'complete', 'checkpoint_step': 30000, 'episodes': COUNT, 'Euler_steps': list(STEPS),
            'split': 'val', 'frame': 0, 'K': 1, 'metrics': summary, 'paired_comparisons': paired,
            'supports_bounded_higher_step_real_inference_test_descriptive_only': supported,
            'runtime': runtime, 'forward_counts': calls, 'inputs_unchanged_verified': True,
            'reads': {'val_hdf_payloads': COUNT, 'train_hdf_payloads': 0, 'dev_hdf_payloads': 0,
                'heldout_hdf_payloads': 0, 'future_visual_rows': 0, 'image_payloads': 0, 'encoder_weight_loads': 0,
                'physical_test_results': 0, 'frozen_train_statistics_metadata_only': True},
            'latency_seconds_by_steps': {str(s): {'median': float(np.median([r['samplers'][str(s)]['seconds'] for r in rows])),
                'min': min(r['samplers'][str(s)]['seconds'] for r in rows), 'max': max(r['samplers'][str(s)]['seconds'] for r in rows)} for s in STEPS},
            'interpretation_limit': 'offline paired CPU float32 sensitivity; Euler40 is not exact solution; no physical success, population, visual causality or retraining conclusion',
            'artifacts': {p.name: {'sha256': base.sha(p), 'bytes': p.stat().st_size} for p in OUT.iterdir() if p.is_file()}}
        base.write_json('result.json', result)
        print(json.dumps({'event': 'complete', 'result': str(OUT/'result.json'), 'runtime': runtime,
                          'supports_bounded_real_inference_test_only': supported}), flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        base.write_json('failure.json', {'status': 'stopped', 'reason': str(exc), 'traceback': traceback.format_exc(),
            'completed_anchors': len(rows), 'runtime': base.runtime(), 'source_files': base.SOURCES,
            'read_ledger': base.READS, 'python_read_opens': opened, 'resource_samples': telemetry})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def verify():
    result = json.loads((OUT / 'result.json').read_text())
    require(result['status'] == 'complete', 'run incomplete')
    for name, entry in result['artifacts'].items():
        require(base.sha(OUT / name) == entry['sha256'], f'artifact changed: {name}')
    rows = json.loads((OUT / 'per_episode.json').read_text())
    plan = json.loads((OUT / 'protocol.json').read_text())['episode_plan']
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as a:
        require(a['prediction'].shape == (COUNT, 3, 32, 14), 'wrong artifact shape')
        require(a['episode'].tolist() == [r['episode'] for r in plan], 'plan mismatch')
        require(a['steps'].tolist() == list(STEPS) and a['seeds'].tolist() == [SEED+i for i in range(COUNT)], 'steps/seeds mismatch')
        from harness.wam_v3 import WamV3Wrapper
        class Normalizer:
            action_min = torch.from_numpy(a['action_min'])
            action_max = torch.from_numpy(a['action_max'])
            normalize_action = WamV3Wrapper.normalize_action
        recalculated = summarize_arrays(a['prediction'], a['target'], a['valid'], a['current_command'], a['normalized_action_scale'], Normalizer())
        require(recalculated == result['metrics'], 'recomputed metrics differ')
        require(paired_report(recalculated, rows) == result['paired_comparisons'], 'paired comparisons differ')
        for i, row in enumerate(rows):
            x0 = torch.randn((1, 32, 14), generator=torch.Generator(device='cpu').manual_seed(SEED+i)).numpy()
            require(np.array_equal(x0[0], a['x0'][i]), 'stored x0/seed mismatch')
            require(base.array_sha(x0) == row['expected_x0_sha256'], 'stored x0 hash mismatch')
            require(all(v['actual_x0_sha256'] == row['expected_x0_sha256'] for v in row['samplers'].values()), 'unpaired sampler')
    message = {'status': 'passed', 'checks': ['all artifact SHA256s', 'prediction shapes and fixed plan',
        'exact aggregate error/smoothness/gripper/distance recomputation', 'paired summaries', 'seed/x0 hashes'],
        'model_loaded': False, 'output': str(OUT)}
    print(json.dumps(message, indent=2))
    # Verification receipt deliberately not in the immutable run artifact manifest.
    base.OUT = OUT; base.write_json('verification.json', message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true', help='synthetic tests; no model/checkpoint/data reads or output writes')
    mode.add_argument('--verify', action='store_true', help='verify completed run artifacts without model load')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(), indent=2))
    elif args.verify:
        verify()
    else:
        run()


if __name__ == '__main__':
    main()
