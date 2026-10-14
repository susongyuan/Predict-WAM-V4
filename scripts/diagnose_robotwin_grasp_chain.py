"""Bounded CPU diagnostic of frozen cabinet30k; no training or policy changes.

Four first sorted TRAIN and four VAL episodes, first commanded closure of each
hand and eight frames before it. Expert histories are supplied: this measures
teacher-context prediction, NOT autonomous grasp success. GT is read separately
for offline execution diagnosis only. Heldout is never opened.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_robotwin_wam import build_wrapper, action_metrics, sha256
from harness.robotwin_dataset import _lowdim
from harness.robotwin_training_data import validate_payload, episode_directory
from harness.wam_v3 import history_frames, bf16_from_bits
from harness.robotwin_eval_evidence import ResourceGuard

RUN = ROOT / 'outputs/runs/put_object_cabinet/wam_30k_20261005'
OUT = ROOT / 'outputs/validation/wam_grasp_chain_20261007'


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(('median', 'p95', 'max'), map(float, np.quantile(a, [.5, .95, 1])))) if a.size else None


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    guard = ResourceGuard(); guard.check(); start = time.monotonic()
    contract = json.loads((RUN / 'manifest.json').read_text())['contract']
    cfg = contract['config']; root = ROOT / Path(cfg['dataset']['split_manifest']).parent / 'data'
    paths = {k: RUN / (k + '.json') for k in ('normalization', 'temporal_stats')}
    stats = {k: json.loads(p.read_text()) for k, p in paths.items()}
    for k, p in paths.items():
        if sha256(p) != contract['stats'][k]['sha256']: raise ValueError('changed statistics')
    checkpoint = RUN / 'checkpoint_step_030000.pt'
    receipt = json.loads(checkpoint.with_name(checkpoint.name + '.sha256.json').read_text())
    if sha256(checkpoint) != receipt['sha256']: raise ValueError('changed weights')
    selected = {k: sorted(contract['episodes'][k], key=lambda i: i['episode'])[:4] for k in ('train', 'val')}
    write('protocol.json', {'checkpoint_sha256': receipt['sha256'], 'selected': selected,
        'anchors': 'first state gripper <0.2, and 8 frames before, each arm',
        'window': 'exec8', 'inference': 'production sampler K1 Euler10 CPU float32 zero-condition',
        'seed': 'SHA256(20261007:episode:frame) first15 hex digits',
        'teacher_context': True, 'heldout_read': False, 'training': False,
        'GT_use': 'offline execution diagnosis only; not a policy input or calibration'})
    # Label alignment and motion distribution, low-dimensional TRAIN data only.
    frames, alignment_max, still = 0, 0., {str(e): 0 for e in (1e-6, .001, .01)}
    lengths, changes = [], []
    for identity in contract['episodes']['train']:
        guard.check()
        with h5py.File(root / identity['episode'], 'r') as f:
            state, action = _lowdim(f, 'state'), _lowdim(f, 'action')
        frames += len(state); lengths.append(len(state))
        alignment_max = max(alignment_max, float(np.max(np.abs(action[:-1] - state[1:]))))
        delta = np.max(np.abs(action - state), axis=1)
        for e in still: still[e] += int((delta < float(e)).sum())
        changes.extend(np.linalg.norm((action - state)[:, [*range(6), *range(7, 13)]], axis=1).tolist())
    write('training_distribution.json', {'episodes': 400, 'frames': frames,
        'anchor_exposure_equivalent': 30000 * 16 * 2 / frames,
        'action_t_equals_state_t_plus1_max_abs_error': alignment_max,
        'all14_command_delta_below_diagnostic_threshold_counts': still,
        'all14_command_delta_below_diagnostic_threshold_fractions': {e: n / frames for e, n in still.items()},
        'note': 'Thresholds are diagnostic, mixed rad/aperture units, not physical stationary/grasp criteria.',
        'arm12_command_delta_l2_rad': quantiles(changes), 'episode_length': quantiles(lengths)})
    # The original transparent scores include read-only real arm tracking.
    executions = []
    for p in sorted((ROOT / 'outputs/validation/system_comparison_10_20261006').glob('*_receipt.json')):
        r = json.loads(p.read_text())
        if r.get('status') != 'valid_terminal': continue
        run = Path(r['run_dir']); result = json.loads((run / 'result.json').read_text()); episode = result['episodes'][0]
        source = run / episode['episode'] / 'evaluation_gt.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        control = [v for v in rows if v.get('settle_step', 0) == 0]
        arm = control[0]['native_diagnostics']['native_predicate_evidence']['arm_tag']
        col = 0 if arm == 'left' else 1
        close = next((v['control_step'] for v in control if v['commanded_grippers'][col] < .2), None)
        errors, closure_errors = [], []
        for v in control:
            tracking = v['native_diagnostics']['arm_joint_tracking'][arm]
            values = list(map(abs, tracking['actual_minus_drive_target']))
            errors.extend(values)
            if close is not None and abs(v['control_step'] - close) <= 16: closure_errors.extend(values)
        closed_row = next((v for v in control if v['control_step'] == close), None)
        executions.append({'seed': r['seed'], 'run': str(run), 'object': control[0]['object_model'],
            'native_success': r.get('native_successes'), 'selected_arm_offline_GT': arm,
            'first_command_below_0_2_step': close,
            'lift_at_first_closure_m': closed_row['native_diagnostics']['native_predicate_evidence']['height_delta_m'] if closed_row else None,
            'robot_contact_at_first_closure': closed_row['contact_evidence'].get('target_robot_contact_count') if closed_row else None,
            'max_lift_m': max(v['native_diagnostics']['native_predicate_evidence']['height_delta_m'] for v in control),
            'joint_tracking_abs_error_rad': quantiles(errors),
            'closure_window_joint_tracking_abs_error_rad': quantiles(closure_errors)})
    write('execution_tracking.json', executions)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if ck['contract'] != contract: raise ValueError('checkpoint/manifest mismatch')
    wrapper, _ = build_wrapper(cfg, paths, stats, torch.device('cpu'), checkpoint=ck)
    del ck; wrapper.eval(); records, saved = [], {}
    for split, identities in selected.items():
        for identity in identities:
            guard.check()
            with h5py.File(root / identity['episode'], 'r') as f:
                state, action = _lowdim(f, 'state'), _lowdim(f, 'action')
            cache = Path(contract['cache_config']['path']).parent
            arrays = validate_payload(cache, identity)
            events = []
            for side, col in (('left', 6), ('right', 13)):
                ix = np.flatnonzero(state[:, col] < .2)
                if len(ix):
                    events.extend((side, max(0, int(ix[0]) - d), int(ix[0])) for d in (8, 0))
            for side, frame, closure in events:
                guard.check()
                if time.monotonic() - start > 600: raise TimeoutError('600s CPU diagnostic budget')
                pframes = [frame] + history_frames(frame, 0, cfg['model']['v3']['proprio_lags'])
                mframes = [frame] + history_frames(frame, 0, cfg['model']['v3']['memory_lags'])
                feats = bf16_from_bits(torch.from_numpy(arrays['feat'][frame].copy())[None]).float()
                proprio = torch.from_numpy(state[pframes].copy())[None]
                pool = torch.from_numpy(arrays['pool'][mframes].transpose(1, 0, 2).copy())[None]
                seed = int(hashlib.sha256(f'20261007:{identity["episode"]}:{frame}'.encode()).hexdigest()[:15], 16)
                with torch.inference_mode():
                    pred, extra = wrapper.sample_actions([feats[:, :, j] for j in range(3)], proprio, pool,
                        torch.zeros(1, 1024), num_steps=10, generator=torch.Generator(device='cpu').manual_seed(seed))
                n = min(8, len(action) - frame); target = torch.from_numpy(action[frame:frame+n])[None]
                metrics = action_metrics(pred[:, :n], target, torch.ones(1, n, dtype=torch.bool), wrapper)
                prediction = pred[0, :n].numpy(); gt = target[0].numpy(); cols = range(6) if side == 'left' else range(7, 13)
                k = 6 if side == 'left' else 13
                row = {'split': split, 'episode': identity['episode'], 'side': side, 'frame': frame,
                    'first_close_frame': closure, 'metrics': metrics,
                    'selected_arm_mae_rad': float(np.abs(prediction[:, list(cols)] - gt[:, list(cols)]).mean()),
                    'selected_gripper_mae': float(np.abs(prediction[:, k] - gt[:, k]).mean()),
                    'selected_gripper_first_error': float(prediction[0, k] - gt[0, k]),
                    'gate_head_left_right': extra['gate'][0].tolist(),
                    'first_prediction': prediction[0].tolist(), 'first_expert_target': gt[0].tolist()}
                records.append(row); saved[f'prediction_{len(records)-1}'] = prediction
                saved[f'target_{len(records)-1}'] = gt
                with (OUT / 'closure_predictions.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
                print('ANCHOR', len(records), split, identity['episode'], side, frame,
                      'arm_mae', round(row['selected_arm_mae_rad'], 4),
                      'grip_mae', round(row['selected_gripper_mae'], 4), flush=True)
    np.savez_compressed(OUT / 'predictions.npz', **saved)
    result = {'status': 'completed', 'elapsed_seconds': time.monotonic() - start,
        'anchors': len(records), 'CUDA_initialized': torch.cuda.is_initialized(), 'physical_evaluation': False,
        'teacher_context': True, 'heldout_read': False, 'resources': guard.peaks,
        'groups': {split: {'anchors': len(items := [r for r in records if r['split'] == split]),
            'selected_arm_mae_rad': quantiles([r['selected_arm_mae_rad'] for r in items]),
            'selected_gripper_mae': quantiles([r['selected_gripper_mae'] for r in items])} for split in selected}}
    write('result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
