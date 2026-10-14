"""CPU visual branch diagnosis; no deployed policy changes or heldout reads.

TRAIN expert first-closing arm supplies labels for fixed ridge probes. VAL and
five already-observed native starts are diagnostic checks, not admission or new
success-rate samples. Native GT labels are used only after predictions for
offline scoring. Float32 CPU DINO/flow need not match BF16 CUDA bitwise.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import json
import sys
import time
import hashlib
import numpy as np
import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.robotwin_dataset import _lowdim, decode_rgb
from harness.robotwin_training_data import preprocess_rgb, validate_payload
from harness.wam_v3 import bf16_from_bits, pool_from_hidden
from harness.wam_behavior import load_upstream
from harness.robotwin_eval_evidence import ResourceGuard
from scripts.train_robotwin_wam import build_wrapper, sha256

RUN = ROOT / 'outputs/runs/put_object_cabinet/wam_30k_20261005'
OUT = ROOT / 'outputs/validation/wam_visual_branch_v2_20261007'
VIEWS = ('head_camera', 'left_camera', 'right_camera')
CASES = (10010, 10011, 10012, 10014, 10017)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def moving_arm(prediction, state):
    diff = prediction[:8] - state
    left = float(np.linalg.norm(diff[:, :6])); right = float(np.linalg.norm(diff[:, 7:13]))
    return {'arm': 'left' if left > right else 'right', 'left_excursion_rad': left, 'right_excursion_rad': right}


def main():
    OUT.mkdir(exist_ok=False); start = time.monotonic()
    torch.set_num_threads(2); torch.set_num_interop_threads(2); guard = ResourceGuard()
    c = json.loads((RUN / 'manifest.json').read_text())['contract']; cfg = c['config']
    cache = Path(c['cache_config']['path']).parent
    data = ROOT / Path(cfg['dataset']['split_manifest']).parent / 'data'
    selections = {'head_cls': np.arange(1024), 'head_cls_patchmean': np.arange(2048), 'all_view_cls_patchmean': np.arange(6144)}
    bank = {}
    for split in ('train', 'val'):
        names, xs, ys = [], [], []
        for identity in c['episodes'][split]:
            guard.check()
            with h5py.File(data / identity['episode'], 'r') as f:
                state = _lowdim(f, 'state')
            closings = []
            for side, col in (('left', 6), ('right', 13)):
                ix = np.flatnonzero(state[:, col] < .2)
                if not len(ix): raise ValueError('expert branch unavailable')
                closings.append((int(ix[0]), side))
            if closings[0][0] == closings[1][0]: raise ValueError('ambiguous expert branch')
            arrays = validate_payload(cache, identity)
            names.append(identity['episode']); xs.append(arrays['pool'][0].reshape(-1).copy())
            ys.append(1. if min(closings)[1] == 'right' else -1.)
        bank[split] = {'names': names, 'x': np.asarray(xs, dtype=np.float64), 'y': np.asarray(ys)}
    write('protocol.json', {'TRAIN': 400, 'VAL': 50, 'heldout_read': False, 'cases': CASES,
        'labels': 'expert first commanded closure; not a physical grasp verifier',
        'ridge': 'TRAIN standardization; alpha=.1*trace(TRAIN kernel)/n; no tuning',
        'CPU_model': 'frozen original30k, K1 Euler10, 2 predetermined noise seeds per native start',
        'numerics': 'CPU float32 DINO -> BF16 feature quantization; no bitwise CUDA equivalence claim',
        'native_GT': 'offline reference arm only; never enters model or ridge fit',
        'physical_evaluation': False, 'initialization_sha256': c['initialization']['sha256'],
        'note': 'deployed30k hash recorded in result'})
    probes, summary = {}, {}
    for key, cols in selections.items():
        x = bank['train']['x'][:, cols]; mean = x.mean(0); scale = np.maximum(x.std(0), 1e-3)
        z = (x - mean) / scale; kernel = z @ z.T
        alpha = .1 * float(np.trace(kernel)) / len(kernel)
        dual = np.linalg.solve(kernel + alpha * np.eye(len(kernel)), bank['train']['y'])
        weight = z.T @ dual
        def scores(values): return ((values[:, cols] - mean) / scale) @ weight
        train_scores, val_scores = scores(bank['train']['x']), scores(bank['val']['x'])
        probes[key] = (cols, mean, scale, weight)
        summary[key] = {'train_correct': int(((train_scores > 0) == (bank['train']['y'] > 0)).sum()),
                        'val_correct': int(((val_scores > 0) == (bank['val']['y'] > 0)).sum()),
                        'val_left_correct': int(((val_scores <= 0) & (bank['val']['y'] < 0)).sum()),
                        'val_left_count': int((bank['val']['y'] < 0).sum()), 'alpha': alpha}
    write('ridge_diagnostic.json', summary)
    print('RIDGE', json.dumps(summary), flush=True)
    _, runner = load_upstream()
    encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
        str(ROOT / cfg['model']['vision_encoder']['checkpoint_path']), torch.float32, torch.device('cpu'))
    if (hidden, registers, patch) != (1024, 4, 16): raise ValueError('encoder differs')
    def encode(rgb):
        guard.check()
        pixels = torch.from_numpy(np.stack([preprocess_rgb(rgb[v]) for v in VIEWS]))
        with torch.inference_mode():
            hs = encoder(pixel_values=pixels, output_hidden_states=True, return_dict=True).hidden_states
            feats = [hs[l].to(torch.bfloat16).float()[None] for l in (-12, -8, -4)]
            pool = pool_from_hidden(feats[-1], 4)[0].float().numpy().reshape(3, 2048)
        return feats, pool
    identity = c['episodes']['train'][0]
    with h5py.File(data / identity['episode'], 'r') as f:
        rgb = {v: decode_rgb(f['vision/' + old + '/colors'][0]) for v, old in
               zip(VIEWS, ('cam_head', 'cam_left_wrist', 'cam_right_wrist'))}
    _, encoded = encode(rgb); recorded = bank['train']['x'][0].reshape(3, 2048)
    cosine = (encoded * recorded).sum(1) / (np.linalg.norm(encoded, axis=1) * np.linalg.norm(recorded, axis=1))
    write('CPU_encoder_check.json', {'TRAIN_case': identity['episode'], 'pool_cosine_by_view': cosine.tolist(),
                                    'relative_l2_by_view': (np.linalg.norm(encoded-recorded, axis=1)/np.linalg.norm(recorded, axis=1)).tolist()})
    cached = []
    for seed in CASES:
        guard.check()
        if time.monotonic() - start > 600: raise TimeoutError('600s visual diagnostic budget')
        run = ROOT / f'outputs/runs/put_object_cabinet/system10_transparent_seed{seed}_20261006'
        episode = json.loads((run / 'result.json').read_text())['episodes'][0]['episode']; directory = run / episode
        meta = json.loads((directory / 'onboard/frames.jsonl').read_text().splitlines()[0])
        with np.load(directory / 'onboard' / meta['file']) as a:
            rgb = {v: a[v].copy() for v in VIEWS}; state = a['commanded_joint_gripper_state'].copy()
        feats, pool = encode(rgb)
        scores = {k: float(((pool.reshape(-1)[cols]-mean)/scale) @ weight)
                  for k, (cols, mean, scale, weight) in probes.items()}
        cached.append((seed, directory, feats, pool, state, scores))
        print('ENCODED', seed, scores, flush=True)
    del encoder
    checkpoint = RUN / 'checkpoint_step_030000.pt'
    receipt = json.loads(checkpoint.with_name(checkpoint.name + '.sha256.json').read_text())
    if sha256(checkpoint) != receipt['sha256']: raise ValueError('weight hash changed')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    stats_paths = {k: RUN / (k + '.json') for k in ('normalization', 'temporal_stats')}
    stats = {k: json.loads(p.read_text()) for k, p in stats_paths.items()}
    wrapper, _ = build_wrapper(cfg, stats_paths, stats, torch.device('cpu'), checkpoint=ck); del ck
    wrapper.eval(); native_rows = []
    for seed, directory, feats, pool, state, scores in cached:
        guard.check()
        q = torch.from_numpy(np.repeat(state[None], 7, axis=0))[None].float()
        memory = torch.from_numpy(np.repeat(pool[:, None], 4, axis=1))[None].float()
        predictions = []
        for index in range(2):
            noise_seed = int(hashlib.sha256(f'visual-branch:{seed}:{index}'.encode()).hexdigest()[:15], 16)
            with torch.inference_mode():
                pred, extra = wrapper.sample_actions(feats, q, memory, torch.zeros(1, 1024), num_steps=10,
                    generator=torch.Generator(device='cpu').manual_seed(noise_seed))
            predictions.append({'noise_seed': noise_seed, **moving_arm(pred[0].numpy(), state),
                                'gate': extra['gate'][0].tolist()})
        # Native reference is accessed only after all predictions, outside inference.
        reference = json.loads((directory / 'evaluation_gt.jsonl').read_text().splitlines()[0])
        row = {'seed': seed, 'reference_arm_offline_GT': reference['native_diagnostics']['native_predicate_evidence']['arm_tag'],
            'ridge_scores_positive_right': scores, 'CPU_flow_start_predictions': predictions}
        native_rows.append(row); print('NATIVE', json.dumps(row), flush=True)
    write('native_start_diagnostic.json', native_rows)
    write('result.json', {'status': 'completed', 'elapsed_seconds': time.monotonic()-start,
        'physical_evaluation': False, 'CUDA_initialized': torch.cuda.is_initialized(),
        'deployed_checkpoint_sha256': receipt['sha256'], 'CPU_encoder_check_cosine': cosine.tolist(),
        'ridge': summary, 'native': native_rows, 'resources': guard.peaks,
        'limitations': '5 already-observed native starts; CPU numerics differ from live BF16 CUDA. Diagnostic probes are not admitted controllers or task verifiers.'})


if __name__ == '__main__': main()
