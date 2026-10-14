#!/usr/bin/env python3
"""Fixed K4/Euler10 CPU candidate medoid diagnostic, not a deployment integration.

Only first eight frozen VAL starts at 30k. Candidate 0 exactly reproduces the
previous K1 seed; candidates 1..3 use preregistered SHA256-derived seeds. Select
one actual sample using only pairwise TRAIN-normalized exec8 prediction MSE.
No labels, oracle, weights, physical outcomes, or future features in selection.
--self-test is synthetic only; --verify recomputes stored outputs without model.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import gc
import hashlib
import inspect
import json
import signal
import tempfile
import time
import traceback
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# This is the FIRST numerical import; the existing helper initializes inter-op
# threads once. No reload, duplicate thread setters, or monkeypatches.
from scripts import diagnose_robotwin_sampler_convergence as previous
base = previous.base
np, torch, psutil = base.np, base.torch, base.psutil
require = base.require
OUT = ROOT / 'outputs/validation/robotwin_candidate_consensus_20261006'
OLD = previous.OUT
CKPT = previous.CKPT
COUNT, K, EULER, SEED = 8, 4, 10, 20261004
WINDOWS = previous.WINDOWS
RSS_LIMIT, MIN_AVAILABLE = previous.RSS_LIMIT, previous.MIN_AVAILABLE
SEED_NAMESPACE = 'robotwin-candidate-consensus-v1'


def seed_entry(episode, ordinal, candidate):
    baseline = SEED + ordinal
    if candidate == 0:
        return {'candidate': 0, 'seed': baseline, 'hash_input': None, 'sha256': None,
                'rule': 'unchanged baseline seed'}
    text = f'{SEED_NAMESPACE}|baseline_seed={baseline}|episode={episode}|candidate={candidate}'
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest)[:8], 'big') & ((1 << 63) - 1)
    return {'candidate': candidate, 'seed': seed, 'hash_input': text, 'sha256': digest,
            'rule': 'SHA256 UTF-8; first 8 bytes unsigned big-endian, clear top bit'}


def select_medoid(candidates, normalizer):
    """Label-free selector. No target, valid mask, state, or expert inputs.

    D[i,j] = mean over fixed 8x14 coordinates of (norm_i - norm_j)^2.
    Score = sum to the other three candidates / 3. Diagonal is zero.
    Exact float64 ties select the lowest index (np.argmin); NO tolerance tuning.
    Normalization is exact production float32 TRAIN minmax, then distances are
    accumulated in float64. All 14 channels get equal weight, no clipping.
    """
    require(candidates.shape == (K, 32, 14) and np.isfinite(candidates).all(), 'bad candidate array')
    normalized = normalizer.normalize_action(torch.from_numpy(candidates[:, :8].copy())).numpy().astype(np.float64)
    distances = np.square(normalized[:, None] - normalized[None, :]).mean(axis=(2, 3))
    scores = distances.sum(axis=1) / (K - 1)
    index = int(np.argmin(scores))
    tied = np.flatnonzero(scores == scores[index]).tolist()
    ordered = np.sort(scores)
    selected = candidates[index].copy()  # full chunk from ONE sample, never mean
    return index, selected, {'distance_matrix_normalized_exec8_mse': distances.tolist(),
        'mean_distance_to_other_candidates': scores.tolist(), 'selected_index': index,
        'exact_minimum_tied_indices': tied, 'second_smallest_minus_smallest_score': float(ordered[1]-ordered[0]),
        'reason': 'minimum mean pairwise normalized exec8 MSE; exact tie -> lowest candidate index',
        'selection_input': 'only 4 candidate exec8 predictions and frozen TRAIN normalization',
        'selector_used_target_or_roughness_or_bounds': False, 'output_is_actual_candidate': True,
        'selected_chunk_sha256': base.array_sha(selected),
        'candidate_chunk_sha256': [base.array_sha(p) for p in candidates]}


def metrics_for(values, target, valid, commands, normalizer):
    out = {}
    for label, h in WINDOWS.items():
        p, t, v = values[:, :h], target[:, :h], valid[:, :h]
        errors = [base.action_metrics(torch.from_numpy(a[None]), torch.from_numpy(b[None]),
                     torch.from_numpy(c[None]), normalizer) for a, b, c in zip(p, t, v)]
        out[label] = {'errors': base.summarize(errors),
            'roughness': previous.trajectory_metrics(p, t, v, commands),
            'raw_gripper_bounds': base.bounds(p[v])}
    return out


def comparison_metrics(baseline, medoid, target, valid, commands, normalizer):
    hold = np.repeat(commands[:, None], 32, axis=1)
    return {name: metrics_for(array, target, valid, commands, normalizer)
            for name, array in [('baseline_k1', baseline), ('medoid_k4', medoid), ('target', target), ('hold', hold)]}


def compare(summary, rows):
    result = {}
    for window in WINDOWS:
        a, b = summary['baseline_k1'][window], summary['medoid_k4'][window]
        deltas = [r['metrics']['medoid_k4'][window]['errors']['normalized_mse'] -
                  r['metrics']['baseline_k1'][window]['errors']['normalized_mse'] for r in rows]
        groups = {}
        for group in previous.GROUPS:
            groups[group] = {key: previous.scalar_delta(a['roughness'][group][key], b['roughness'][group][key])
                for key in ('difference1_rms', 'difference2_rms', 'difference3_rms',
                    'difference1_error_rms_vs_target', 'difference2_error_rms_vs_target', 'first_command_jump_rms')}
            groups[group]['reversals_epsilon_0.001'] = previous.scalar_delta(
                a['roughness'][group]['direction_reversals']['0.001']['count'],
                b['roughness'][group]['direction_reversals']['0.001']['count'])
        result[window] = {'normalized_mse': previous.scalar_delta(a['errors']['normalized_mse'], b['errors']['normalized_mse']),
            'per_episode_normalized_mse_deltas': deltas, 'episodes_improved': sum(d < 0 for d in deltas),
            'episodes_worsened': sum(d > 0 for d in deltas), 'episodes_tied': sum(d == 0 for d in deltas),
            'groups': groups, 'raw_gripper_out_of_range_count': previous.scalar_delta(
                a['raw_gripper_bounds']['both']['out_of_range_count'], b['raw_gripper_bounds']['both']['out_of_range_count'])}
    return result


def self_tests():
    tests = previous.self_tests()
    normalizer = previous.saved_normalizer(np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32))
    candidates = np.broadcast_to(np.array([0, 1, 2, 20], dtype=np.float32)[:, None, None], (4, 32, 14)).copy()
    index, selected, reason = select_medoid(candidates, normalizer)
    require(index == 2 and np.array_equal(selected, candidates[index]), 'must select actual medoid, not mean')
    expected = np.square(2 * (np.array([0, 1, 2, 20])[:, None] - np.array([0, 1, 2, 20])[None, :]))
    require(np.array_equal(expected, reason['distance_matrix_normalized_exec8_mse']), 'distance or scaling mismatch')
    changed = candidates.copy(); changed[:, 8:] = -10000
    require(select_medoid(changed, normalizer)[2]['mean_distance_to_other_candidates'] == reason['mean_distance_to_other_candidates'], 'selector accessed chunk tail')
    tie = np.broadcast_to(np.array([0, 0, 1, 1], dtype=np.float32)[:, None, None], (4, 32, 14)).copy()
    require(select_medoid(tie, normalizer)[0] == 0 and select_medoid(tie, normalizer)[2]['exact_minimum_tied_indices'] == [0, 1, 2, 3], 'lowest index tie break failed')
    require(list(inspect.signature(select_medoid).parameters) == ['candidates', 'normalizer'], 'selector accepts labels')
    # Equal contribution from any of 12 arm / 2 gripper normalized coordinates.
    one_channel = np.zeros((4, 32, 14), dtype=np.float32)
    one_channel[:, :8, 0] = np.array([0, 1, 2, 20])[:, None]
    gripper = np.zeros_like(one_channel); gripper[:, :, 13] = one_channel[:, :, 0]
    require(select_medoid(one_channel, normalizer)[2]['mean_distance_to_other_candidates'] == select_medoid(gripper, normalizer)[2]['mean_distance_to_other_candidates'], 'arm/gripper weighting changed')
    seeds = [seed_entry('episode_0000003.hdf5', 0, k) for k in range(K)]
    require(seeds[0]['seed'] == SEED and len({s['seed'] for s in seeds}) == K, 'seed rule mismatch')
    require(seeds == [seed_entry('episode_0000003.hdf5', 0, k) for k in range(K)], 'seed rule nondeterministic')
    tests['checks'] += ['label-free actual-sample medoid and known distances', 'exec8 only, equal channel weights, exact ties select lowest index', 'deterministic SHA256 candidate seed derivation and unchanged candidate0 seed']
    return tests


def old_inventory():
    return {str(p.relative_to(OLD)): {'sha256': base.sha(p), 'stat': base.file_identity(p)}
            for p in sorted(OLD.rglob('*')) if p.is_file()}


def summary_text(metrics, paired, cost, selected_counts, runtime):
    lines = ['# Candidate consensus — fixed K4 / Euler10, offline only', '',
        '30k; same fixed first 8 sorted VAL starts; candidate0 bitwise identical to prior K1 baseline.',
        'Selection: minimum mean pairwise TRAIN-normalized exec8 MSE across all 14 channels; exact tie -> lowest index. Return one full candidate chunk, not an average.',
        'No labels, roughness, bounds, or physical results enter selection. No weight tuning, oracle, retraining, or deployment integration.', '',
        '| Window | Policy | Normalized MSE | Arm Δ2 RMS | Arm reversals (>0.001 rad/frame) | Gripper outside [0,1] |',
        '|---|---|---:|---:|---:|---:|']
    for window in WINDOWS:
        for name in ('baseline_k1', 'medoid_k4', 'target', 'hold'):
            m = metrics[name][window]; g = m['raw_gripper_bounds']['both']; a = m['roughness']['arms_rad']
            lines.append(f"| {window} | {name} | {m['errors']['normalized_mse']:.8f} | {a['difference2_rms']:.6f} | {a['direction_reversals']['0.001']['count']} | {g['out_of_range_count']}/{g['scalar_count']} |")
    lines += ['', '## Paired findings']
    for window in WINDOWS:
        p = paired[window]
        lines.append(f"- {window}: MSE reduction {p['normalized_mse']['relative_reduction']:.2%}; improved/worsened/tied = {p['episodes_improved']}/{p['episodes_worsened']}/{p['episodes_tied']}; arm Δ2 RMS reduction {p['groups']['arms_rad']['difference2_rms']['relative_reduction']:.2%}.")
    lines += ['', '## Costs and boundaries', f'- Selected candidate indices: {selected_counts}.',
        f"- CPU median baseline sampling {cost['baseline_k1_median_seconds']:.3f}s; K4 sampling plus selection {cost['medoid_k4_median_seconds']:.3f}s; median ratio {cost['median_k4_over_k1']:.3f}x. 320 total action-model forwards, no additional baseline pass.",
        f"- Runtime {runtime['elapsed_seconds']:.1f}s; peak RSS {runtime['sampled_resource_peaks']['process_rss_bytes']/1024**3:.2f} GiB; peak system RAM {runtime['sampled_resource_peaks']['system_ram_percent']:.1f}%; torch/inter-op 2/2; CUDA never initialized.",
        '- Finite differences use frames, not seconds; raw gripper excursions are diagnostics, not physical failure. Both windows, per-arm/gripper errors, derivative errors, and multiple reversal deadbands are saved.',
        '- Only 8 fixed starts and one preregistered four-seed set each. Medoid is centrality, not task correctness; candidate modes can disagree. No generalization, closed-loop success, real-time deployability, or connected system-module claim.', '']
    (OUT / 'summary.md').write_text('\n'.join(lines))


def run():
    require(not OUT.exists(), f'refuse overwrite/retry: {OUT}')
    OUT.mkdir(parents=True, exist_ok=False)
    scratch = OUT / 'runtime_tmp'; scratch.mkdir()
    os.environ['TMPDIR'] = str(scratch); tempfile.tempdir = str(scratch)
    base.OUT, base.CKPT, base.START = OUT, CKPT, time.monotonic()
    base.READS.clear(); base.SOURCES.clear()
    process = {'pid': os.getpid(), 'output': str(OUT), 'status': 'running',
               'started_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
               'cpu_threads': 2, 'interop_threads': 2, 'CUDA_VISIBLE_DEVICES': ''}
    base.write_json('process.json', process)
    print(json.dumps(process), flush=True)
    rows, selections, telemetry, opened = [], [], [], {}
    allowed_hdfs = set()
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
                require(str(path.resolve()) in allowed_hdfs, 'nonselected HDF read denied')
            entry = opened.setdefault(str(path), {'read_open_count': 0})
            entry['read_open_count'] += 1
    sys.addaudithook(audit)
    def check(*unused):
        base.check()
        require(time.monotonic()-base.START < 600, '600 second total time limit')
        require(base.PROCESS.memory_info().rss < RSS_LIMIT, '6 GiB process RSS cap')
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'thread count changed')
        if not telemetry or time.monotonic()-telemetry[-1]['monotonic'] >= .9:
            m = psutil.virtual_memory()
            telemetry.append({'monotonic': time.monotonic(), 'elapsed_seconds': time.monotonic()-base.START,
                              'ram_percent': m.percent, 'available_bytes': m.available, 'rss_bytes': base.PROCESS.memory_info().rss})
    signal.signal(signal.SIGALRM, check)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda n, f: (_ for _ in ()).throw(RuntimeError(f'signal {n}')))
    signal.setitimer(signal.ITIMER_REAL, 1, 1)
    try:
        check(); vm = psutil.virtual_memory()
        require(vm.available >= MIN_AVAILABLE and .85*vm.total-vm.used >= MIN_AVAILABLE, 'insufficient 4 GiB headroom below RAM85')
        base.write_json('self_tests.json', self_tests())
        base.register(__file__); base.register(previous.__file__); base.register(base.__file__)
        old_before = old_inventory()
        base.write_json('prior_results_inventory.json', old_before)
        base.register(base.RUN / 'manifest.json')
        manifest = json.loads((base.RUN / 'manifest.json').read_text())
        contract = manifest['contract']; cfg = contract['config']
        require(base.digest(contract) == manifest['contract_sha256'], 'contract digest changed')
        require(base.register(base.CONFIG)['sha256'] == contract['config_sha256'], 'frozen config changed')
        split_path = ROOT / cfg['dataset']['split_manifest']
        require(base.register(split_path)['sha256'] == cfg['dataset']['split_sha256'] == contract['split_sha256'], 'split mismatch')
        split = json.loads(split_path.read_text())
        val_names = sorted(split['splits']['val']); names = val_names[:COUNT]
        require(len(val_names) == len(set(val_names)) == 50 and len(names) == COUNT, 'fixed 50 VAL membership required')
        require(not set(val_names) & set().union(*(set(v) for k, v in split['splits'].items() if k != 'val')), 'split overlap')
        allowed_hdfs.update(str((Path(split['data_root']) / n).resolve()) for n in names)
        old_protocol = json.loads((OLD / 'protocol.json').read_text())
        require([x['episode'] for x in old_protocol['episode_plan']] == names and old_protocol['checkpoint'] == str(CKPT), 'prior baseline plan mismatch')
        plan = [{'episode': n, 'ordinal': i, 'anchor': 0, 'candidates': [seed_entry(n, i, k) for k in range(K)]} for i, n in enumerate(names)]
        require(len({v['seed'] for p in plan for v in p['candidates']}) == COUNT*K, 'seed collision; stop without substitution')
        protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(CKPT),
            'checkpoint_step': 30000, 'split': 'val', 'episode_plan': plan, 'K': K, 'Euler_steps_per_candidate': EULER,
            'anchor_selection': 'same first 8 lexicographically sorted frozen VAL episodes, frame0', 'windows': WINDOWS,
            'seed_namespace': SEED_NAMESPACE, 'seed_rule': 'candidate0=20261004+VAL ordinal; others SHA256(first 8 bytes big-endian)&(2^63-1), exact hash inputs recorded in plan',
            'medoid': {'space': 'production TRAIN minmax-normalized absolute actions, all native 14 dimensions',
                'coordinate_count': 8*14, 'channels': '12 arm + 2 gripper, equal weight per coordinate',
                'distance': 'mean of squared differences over fixed exec8 x14; float32 normalization then float64 distances',
                'score': 'sum distances to other 3 candidates /3', 'tie': 'exact minimum, lowest index',
                'output': 'entire original chunk32 from the winning candidate, not a mean',
                'forbidden': ['labels', 'target errors', 'roughness scoring', 'bounds scoring', 'oracle', 'weights tuning', 'physical tests']},
            'policy_inputs': ['cached frame0 DINO features', 'frame0 commanded state repeated7', 'frame0 pool repeated4', 'zero task condition'],
            'target_only': 'stored action[:32] and masks for scoring AFTER selection ledger is persisted; never passed to selector',
            'precision': 'CPU float32; no output clipping', 'cpu_threads': 2, 'interop_threads': 2, 'CUDA_VISIBLE_DEVICES': '',
            'resource_limits': {'ram_percent': 85, 'system_cpu_percent': 85, 'rss_bytes': RSS_LIMIT, 'time_seconds': 600,
                'per_candidate_seconds': 60, 'first_anchor_seconds': 120},
            'cost': '4 independent batch1 sampler calls; no extra baseline call; report summed sampling+selection vs candidate0 only; I/O and scoring excluded from inference latency',
            'prior_results': str(OLD), 'boundary': 'offline transferable sampling-robustness evidence only; no connected system-module or closed-loop success claim'}
        base.write_json('protocol.json', protocol)  # before model load or any VAL payload
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
        require(cfg['common']['action_chunk_size'] == 32 and cfg['common']['action_execution_horizon'] == 8
                and cfg['common']['num_inference_steps'] == 10 and cfg['common']['proprio_len'] == 7
                and cfg['dataset']['cond_mode'] == 'zero_single_task'
                and cfg['model']['v3']['proprio_lags'] == [2, 4, 8, 16, 32]
                and cfg['model']['v3']['memory_lags'] == [4, 16], 'frozen recipe mismatch')
        with np.load(OLD / 'predictions.npz', allow_pickle=False) as old_arrays:
            require(old_arrays['episode'].tolist() == names and old_arrays['steps'].tolist()[0] == 10, 'prior array order mismatch')
            old_pred = old_arrays['prediction'][:, 0].copy()
            old_x0 = old_arrays['x0'].copy()
            require(np.array_equal(old_arrays['seeds'], [SEED+i for i in range(COUNT)]), 'prior noise seed mismatch')
            require(np.array_equal(old_arrays['action_min'], wrapper.action_min.numpy()) and np.array_equal(old_arrays['action_max'], wrapper.action_max.numpy()), 'prior normalization mismatch')
        old_reads = json.loads((OLD / 'read_ledger.json').read_text())
        calls = {'action_model_forward': 0, 'future_decoder_forward': 0}
        current = {}
        def model_check(module, args, kwargs):
            check()
            require(time.monotonic()-current['started'] < 60, '60 second candidate limit')
            require(set(kwargs) == {'noisy_actions', 'qpos_history', 'dino_features_list', 'task_cond', 'memory'}, 'unexpected conditioning')
            require(bool((kwargs['task_cond'] == 0).all()), 'nonzero task condition')
            require(torch.equal(args[0], torch.full((1,), current['calls']/EULER)), 'wrong Euler time grid')
            if current['calls'] == 0:
                require(torch.equal(kwargs['noisy_actions'], current['x0']), 'seeded x0 mismatch')
                current['actual_x0_sha256'] = base.array_sha(kwargs['noisy_actions'].numpy())
            current['calls'] += 1; calls['action_model_forward'] += 1
        def forbid_future(*unused):
            calls['future_decoder_forward'] += 1
            raise RuntimeError('future decoder forbidden')
        wrapper.action_model.register_forward_pre_hook(model_check, with_kwargs=True)
        wrapper.action_model.future_feat_decoder.register_forward_pre_hook(forbid_future)
        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and any(Path(path).resolve().is_relative_to(ROOT / sub) for sub in ('harness', 'scripts', 'third_party/LiLa-WAM')) and Path(path).is_file():
                base.register(path)
        def provenance():
            return {'source_files': base.SOURCES, 'run_contract_sha256': manifest['contract_sha256'],
                'cache_config_id': config_id, 'load_notes': load_notes, 'TRAIN_stats_provenance': stats['normalization']['provenance'],
                'reused': ['check_robotwin_start_states.read_anchor', 'check_robotwin_start_states.build_wrapper',
                    'production WamV3Wrapper.sample_actions', 'trainer.action_metrics', 'prior diagnostic trajectory_metrics'],
                'verification': 'fresh checkpoint SHA vs receipt; fresh source/config/stats/meta hashes; full HDF hash inherited from frozen run, not rehashed; exact read arrays hashed; cache payload stat/header checked, not whole cache SHA',
                'read_audit_limits': 'Python open paths supplement native HDF slice/cache mmap ledger; not a native syscall audit',
                'pairing': 'candidate0 seed, actual x0, prediction, normalization, and input-array hashes must match prior baseline bitwise',
                'selector_source_sha256': hashlib.sha256(inspect.getsource(select_medoid).encode()).hexdigest()}
        base.write_json('provenance.json', provenance())
        predictions, choices, targets, masks, commands, noises, seeds = [], [], [], [], [], [], []
        for i, name in enumerate(names):
            check(); anchor_tick = time.monotonic()
            feats, state, pool, target, valid = base.read_anchor(identities[name], split, cache_root, frozen_cache)
            require(valid[0, :8].all(), 'exec8 requires all 8 valid rows; do not shorten selection horizon')
            require(not wrapper.normalize_state(state)[:, 1:].any() and not wrapper.normalize_memory(pool).any(), 'reset history mismatch')
            require(base.READS[-1]['fresh_read_array_sha256'] == old_reads[i]['fresh_read_array_sha256'], 'anchor inputs/targets differ from prior baseline')
            sample_predictions, sample_noises, sampler_records = [], [], []
            for candidate, spec in enumerate(plan[i]['candidates']):
                seed = spec['seed']
                expected_generator = torch.Generator(device='cpu').manual_seed(seed)
                x0 = torch.randn((1, 32, 14), generator=expected_generator)
                generator = torch.Generator(device='cpu').manual_seed(seed)
                current.clear(); current.update(started=time.monotonic(), calls=0, x0=x0)
                with torch.inference_mode():
                    prediction, _ = wrapper.sample_actions([feats[:, :, layer] for layer in range(3)], state, pool,
                        torch.zeros((1, 1024), dtype=torch.float32), num_steps=EULER, generator=generator)
                seconds = time.monotonic()-current['started']
                require(seconds < 60 and current['calls'] == EULER, 'candidate time/forward bound')
                require(prediction.shape == (1, 32, 14) and prediction.dtype == torch.float32 and torch.isfinite(prediction).all(), 'invalid prediction')
                require(torch.equal(generator.get_state(), expected_generator.get_state()), 'unexpected RNG consumption')
                p = prediction.numpy()[0].copy()
                if candidate == 0:
                    require(np.array_equal(x0.numpy()[0], old_x0[i]) and np.array_equal(p, old_pred[i]), 'candidate0 not BITWISE identical to old K1; stop, no retry')
                sample_predictions.append(p); sample_noises.append(x0.numpy()[0])
                sampler_records.append({'candidate': candidate, 'seed': seed, 'seconds': seconds,
                    'actual_x0_sha256': current['actual_x0_sha256'], 'rng_state_sha256': base.array_sha(generator.get_state().numpy()),
                    'prediction_sha256': base.array_sha(p), 'forward_count': current['calls']})
            candidates = np.stack(sample_predictions)
            selection_tick = time.monotonic()
            index, selected, reason = select_medoid(candidates, wrapper)
            selection_seconds = time.monotonic()-selection_tick
            reason.update(episode=name, ordinal=i, selection_seconds=selection_seconds)
            selections.append(reason)
            # Commit the target-free choice BEFORE computing any current metrics.
            base.write_json('selection_ledger.json', selections)
            predictions.append(candidates); choices.append(index); noises.append(np.stack(sample_noises))
            targets.append(target.numpy()[0]); masks.append(valid.numpy()[0]); commands.append(state.numpy()[0, 0])
            seeds.append([s['seed'] for s in plan[i]['candidates']])
            row = {'episode': name, 'ordinal': i, 'anchor': 0, 'selected_index': index, 'samplers': sampler_records,
                'selection_seconds': selection_seconds, 'baseline_pairing_bitwise': True,
                'baseline_sampling_seconds': sampler_records[0]['seconds'],
                'k4_sampling_plus_selection_seconds': sum(s['seconds'] for s in sampler_records)+selection_seconds,
                'metrics': comparison_metrics(candidates[0:1], selected[None], target.numpy(), valid.numpy(), state.numpy()[:, 0], wrapper)}
            row['elapsed_including_scoring_seconds'] = time.monotonic()-anchor_tick
            rows.append(row)
            if i == 0:
                require(row['elapsed_including_scoring_seconds'] < 120, 'first anchor exceeds120 seconds')
                base.write_json('first_anchor.json', {'passed': True, 'episode': name, 'candidate_shape': list(candidates.shape),
                    'seconds': row['elapsed_including_scoring_seconds'], 'baseline_pairing_bitwise': True, 'runtime': base.runtime()})
            temp = OUT / 'predictions.tmp.npz'
            np.savez(temp, episode=np.array(names[:i+1]), candidates=np.stack(predictions), selected_indices=np.array(choices),
                selected_prediction=np.stack([p[k] for p, k in zip(predictions, choices)]),
                target=np.stack(targets), valid=np.stack(masks), current_command=np.stack(commands),
                x0=np.stack(noises), seeds=np.array(seeds, dtype=np.int64),
                action_min=wrapper.action_min.numpy(), action_max=wrapper.action_max.numpy())
            temp.replace(OUT / 'predictions.npz')
            base.write_json('per_episode.json', rows); base.write_json('resource_samples.json', telemetry)
            print(json.dumps({'event': 'anchor_complete', 'pid': os.getpid(), 'episodes': i+1, 'episode': name,
                'selected_index': index, 'seconds': row['elapsed_including_scoring_seconds'],
                'baseline_exec8_mse': row['metrics']['baseline_k1']['exec8']['errors']['normalized_mse'],
                'medoid_exec8_mse': row['metrics']['medoid_k4']['exec8']['errors']['normalized_mse']}), flush=True)
        p = np.stack(predictions); selected = np.stack([a[k] for a, k in zip(predictions, choices)])
        metrics = comparison_metrics(p[:, 0], selected, np.stack(targets), np.stack(masks), np.stack(commands), wrapper)
        paired = compare(metrics, rows)
        old_result = json.loads((OLD / 'result.json').read_text())
        for window in WINDOWS:
            require(metrics['baseline_k1'][window]['errors'] == old_result['metrics']['errors'][window]['10'], 'baseline aggregate metrics differ')
        require(calls == {'action_model_forward': COUNT*K*EULER, 'future_decoder_forward': 0}, 'forward count mismatch')
        require([r['episode'] for r in base.READS] == names, 'unexpected payload read set')
        for path in list(opened):
            q = Path(path)
            if q.is_relative_to(ROOT) and not q.is_relative_to(OUT) and not q.is_relative_to(OLD) and not any(part.startswith('.') for part in q.relative_to(ROOT).parts) and q.suffix in ('.py', '.json', '.yaml') and str(q.resolve()) not in base.SOURCES:
                base.register(q)
        for path, before in list(base.SOURCES.items()):
            require(base.file_identity(path) == {k: v for k, v in before.items() if k != 'sha256'}, f'input stat changed: {path}')
            if 'sha256' in before and path != str(CKPT):
                require(base.sha(path) == before['sha256'], f'input bytes changed: {path}')
        require(old_inventory() == old_before, 'old output files changed')
        check()
        base.write_json('provenance.json', provenance())
        base.write_json('python_read_opens.json', {'paths': opened, 'limitations': provenance()['read_audit_limits']})
        base.write_json('resource_samples.json', telemetry)
        base.write_json('paired_comparisons.json', paired)
        k1 = [r['baseline_sampling_seconds'] for r in rows]
        k4 = [r['k4_sampling_plus_selection_seconds'] for r in rows]
        cost = {'baseline_k1_median_seconds': float(np.median(k1)), 'medoid_k4_median_seconds': float(np.median(k4)),
            'median_k4_over_k1': float(np.median(np.array(k4)/np.array(k1))),
            'selection_median_seconds': float(np.median([r['selection_seconds'] for r in rows])),
            'total_32_sampler_seconds': float(sum(sum(s['seconds'] for s in r['samplers']) for r in rows)),
            'baseline_forwards_per_anchor': 10, 'k4_forwards_per_anchor': 40,
            'cpu_sequential_batch1_not_real_time_deployment_latency': True}
        counts = {str(k): choices.count(k) for k in range(K)}
        runtime = base.runtime()
        summary_text(metrics, paired, cost, counts, runtime)
        process.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        base.write_json('process.json', process)
        result = {'status': 'complete', 'checkpoint_step': 30000, 'episodes': COUNT, 'K': K, 'Euler_steps_per_candidate': EULER,
            'split': 'val', 'anchor': 0, 'metrics': metrics, 'paired_comparisons': paired, 'selected_candidate_counts': counts,
            'cost': cost, 'runtime': runtime, 'pid': os.getpid(), 'forward_counts': calls,
            'all_baseline_pairing_bitwise': True, 'prior_results_unchanged': True, 'source_hashes_unchanged': True,
            'selection_without_labels_or_tuned_weights': True, 'actual_candidate_not_mean': True,
            'reads': {'val_hdf_payloads': COUNT, 'train_hdf_payloads': 0, 'dev_hdf_payloads': 0, 'heldout_hdf_payloads': 0,
                'future_visual_rows': 0, 'image_payloads': 0, 'encoder_weight_loads': 0, 'physical_results': 0,
                'frozen_TRAIN_statistics_metadata_only': True},
            'boundary': 'offline candidate consensus evidence; not integrated deployment module, physical success, population or latency guarantee',
            'artifacts': {q.name: {'sha256': base.sha(q), 'bytes': q.stat().st_size} for q in OUT.iterdir() if q.is_file()}}
        base.write_json('result.json', result)
        print(json.dumps({'event': 'complete', 'pid': os.getpid(), 'output': str(OUT), 'paired_comparisons':
            {w: {'normalized_mse': paired[w]['normalized_mse'], 'improved': paired[w]['episodes_improved'],
                'worsened': paired[w]['episodes_worsened'], 'tied': paired[w]['episodes_tied']} for w in WINDOWS},
            'cost': cost, 'runtime': runtime}), flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        process.update(status='stopped', reason=str(exc)); base.write_json('process.json', process)
        base.write_json('failure.json', {'status': 'stopped', 'reason': str(exc), 'traceback': traceback.format_exc(),
            'completed_anchors': len(rows), 'runtime': base.runtime(), 'source_files': base.SOURCES,
            'read_ledger': base.READS, 'resource_samples': telemetry, 'python_read_opens': opened})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def verify():
    base.OUT = OUT
    result = json.loads((OUT / 'result.json').read_text())
    require(result['status'] == 'complete', 'run incomplete')
    for name, receipt in result['artifacts'].items():
        require(base.sha(OUT / name) == receipt['sha256'], f'artifact changed: {name}')
    require(old_inventory() == json.loads((OUT / 'prior_results_inventory.json').read_text()), 'old results changed')
    plan = json.loads((OUT / 'protocol.json').read_text())['episode_plan']
    rows = json.loads((OUT / 'per_episode.json').read_text())
    selections = json.loads((OUT / 'selection_ledger.json').read_text())
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as a, np.load(OLD / 'predictions.npz', allow_pickle=False) as old:
        require(a['candidates'].shape == (COUNT, K, 32, 14) and a['x0'].shape == (COUNT, K, 32, 14), 'wrong saved shape')
        require(a['episode'].tolist() == [r['episode'] for r in plan] == old['episode'].tolist(), 'episode order mismatch')
        require(np.array_equal(a['candidates'][:, 0], old['prediction'][:, 0]) and np.array_equal(a['x0'][:, 0], old['x0']), 'baseline prediction/x0 mismatch')
        normalizer = previous.saved_normalizer(a['action_min'], a['action_max'])
        for i, entry in enumerate(plan):
            require(entry['candidates'] == [seed_entry(entry['episode'], i, k) for k in range(K)], 'preregistered seed rule mismatch')
            for k, spec in enumerate(entry['candidates']):
                require(int(a['seeds'][i, k]) == spec['seed'], 'stored seed mismatch')
                x0 = torch.randn((1, 32, 14), generator=torch.Generator(device='cpu').manual_seed(spec['seed'])).numpy()
                require(np.array_equal(x0[0], a['x0'][i, k]) and base.array_sha(x0) == rows[i]['samplers'][k]['actual_x0_sha256'], 'actual x0 pairing mismatch')
                require(base.array_sha(a['candidates'][i, k]) == rows[i]['samplers'][k]['prediction_sha256'], 'candidate hash mismatch')
            index, selected, reason = select_medoid(a['candidates'][i], normalizer)
            require(index == int(a['selected_indices'][i]) == rows[i]['selected_index'], 'medoid index mismatch')
            require(np.array_equal(selected, a['selected_prediction'][i]), 'output not actual medoid sample')
            require(all(selections[i][key] == value for key, value in reason.items()), 'selection rationale mismatch')
            m = comparison_metrics(a['candidates'][i:i+1, 0], a['selected_prediction'][i:i+1], a['target'][i:i+1], a['valid'][i:i+1], a['current_command'][i:i+1], normalizer)
            require(m == rows[i]['metrics'], 'per-anchor metrics mismatch')
        recomputed = comparison_metrics(a['candidates'][:, 0], a['selected_prediction'], a['target'], a['valid'], a['current_command'], normalizer)
        require(recomputed == result['metrics'] and compare(recomputed, rows) == result['paired_comparisons'], 'aggregate recomputation mismatch')
    for path, before in json.loads((OUT / 'provenance.json').read_text())['source_files'].items():
        require(base.file_identity(path) == {k:v for k,v in before.items() if k != 'sha256'}, f'input stat changed: {path}')
        if 'sha256' in before:
            require(base.sha(path) == before['sha256'], f'input content changed: {path}')
    receipt = {'status': 'passed', 'model_loaded': False, 'checks': ['all output SHA256s', 'all 32 seeds and actual x0 hashes',
        'candidate0 bitwise equals previous K1', 'target-free medoid indices and distance/score matrices exactly recomputed',
        'chosen full chunk is original candidate, not mean', 'all per-anchor and aggregate metrics exactly recomputed',
        'previous output inventory unchanged', 'all source/input hashes including full checkpoint unchanged']}
    base.write_json('verification.json', receipt)
    print(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(), indent=2))
    elif args.verify:
        verify()
    else:
        run()


if __name__ == '__main__':
    main()
