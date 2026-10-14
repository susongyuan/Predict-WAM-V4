"""Joint native14 WAM V3+VTT worker; import/metadata preflight is CPU-only.

Reuses bounded JSON/RGB IPC and ChunkPolicy, not the frozen single-task loader.
DINO stays frozen. The sole condition is the selected task's TRAIN prototype;
shared TRAIN normalization and residual projection come from the SAME checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import (Channel, ChunkPolicy, InfrastructureError,
    MODEL_VIEWS, VIEWS, digest, envelope, identity, serve, sha256,
    verify_gpu_owner as _verify_gpu_owner)

FORMAT = 'robotwin-multitask-native14-vtt-v1'
BANK_FORMAT = 'robotwin-multitask-train-vtt-v1'


def data_digest(value):
    # robotwin_training_data uses pretty JSON+newline, unlike checkpoint digest.
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def resolve(path):
    return (ROOT / path).resolve()


def verify_gpu_owner(expected):
    """Read-only ownership admission; never acquire or release the pipeline lock."""
    _verify_gpu_owner(expected)
    # The legacy guard predates this trainer. Never admit overlapping training
    # merely because both children inherited the same pipeline owner name.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(arg).name == 'train_robotwin_multitask_wam.py' for arg in argv if arg):
            raise InfrastructureError(f'multitask trainer {entry.name} still present')


def task_specs(value):
    rows = value.get('tasks') if isinstance(value, dict) else value
    if not isinstance(rows, list) or not 2 <= len(rows) <= 50:
        raise InfrastructureError('joint contract needs 2..50 tasks')
    result = []
    for row in rows:
        name = row.get('task')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name):
            raise InfrastructureError('invalid task name')
        result.append(dict(task=name, **{k: str(resolve(row[k])) for k in ('split_manifest', 'feature_cache')}))
    if len({r['task'] for r in result}) != len(result):
        raise InfrastructureError('duplicate tasks')
    return result


def validate_condition_statistics(contract):
    """No torch needed; reject zero fallback and mismatched/shared provenance."""
    import numpy as np
    names = [r['task'] for r in task_specs(contract)]
    bank = contract['condition_bank']
    stats = contract['statistics']
    train = contract['episodes']['train']
    if (bank.get('format') != BANK_FORMAT or bank.get('source_split') != 'train'
            or set(bank['conditions']) != set(names) or set(train) != set(names)):
        raise InfrastructureError('TRAIN condition task membership mismatch')
    for value in [bank, stats['normalization'], stats['temporal_stats']]:
        provenance = value['provenance']
        if (provenance.get('fit_split') != 'train' or provenance.get('heldout_used') is not False
                or set(provenance['tasks']) != set(names)):
            raise InfrastructureError('shared statistics/condition provenance is not TRAIN-only')
        for task in names:
            if provenance['tasks'][task]['episodes'] != train[task]:
                raise InfrastructureError('TRAIN episode identity mismatch: ' + task)
    normalized = []
    for task in names:
        x = np.asarray(bank['conditions'][task], dtype=np.float32)
        if (x.shape != (1024,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-6
                or np.linalg.norm(x - x.mean()) < 1e-6):
            raise InfrastructureError('invalid/LayerNorm-degenerate task prototype: ' + task)
        unit = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        if any(np.allclose(unit, old, rtol=1e-5, atol=1e-6) for old in normalized):
            raise InfrastructureError('task prototypes indistinguishable after LayerNorm')
        normalized.append(unit)
    normal = stats['normalization']
    if normal.get('action_repr') != 'absolute':
        raise InfrastructureError('absolute actions required')
    for field in ('action', 'state'):
        lo, hi = [np.asarray(normal['robotwin'][field][k], dtype=np.float32) for k in ('min', 'max')]
        if lo.shape != (14,) or hi.shape != (14,) or not np.isfinite([lo, hi]).all() or (lo > hi).any():
            raise InfrastructureError('invalid shared ' + field + ' min/max')
    cfg = contract['config']
    v3, temporal = cfg['model']['v3'], stats['temporal_stats']
    if (tuple(temporal['views']) != MODEL_VIEWS or temporal['proprio_lags'] != v3['proprio_lags']
            or temporal['memory_lags'] != v3['memory_lags']):
        raise InfrastructureError('history/statistics lags or views differ')
    for name, shape in [('proprio_diff_std', (len(v3['proprio_lags']) + 1, 14)),
                        ('memory_diff_std', (3, len(v3['memory_lags']) + 1, 2048))]:
        x = np.asarray(temporal[name], dtype=np.float32)
        if x.shape != shape or not np.isfinite(x).all() or (x <= 0).any():
            raise InfrastructureError('invalid shared ' + name)


def load_contract(checkpoint, multitask_manifest=None):
    """Validate JSON/provenance/receipt without unpickling weights or using CUDA.

    --multitask-manifest accepts the trainer's input tasks JSON or its output
    manifest.json. The checkpoint's sibling manifest.json is authoritative.
    """
    checkpoint = Path(checkpoint).resolve()
    path = checkpoint.parent / 'manifest.json'
    manifest = json.loads(path.read_text())
    contract = manifest['contract']
    if contract.get('format') != FORMAT or manifest['contract_sha256'] != digest(contract):
        raise InfrastructureError('joint training manifest format/digest mismatch')
    specs = task_specs(contract)
    if multitask_manifest is not None:
        supplied = json.loads(Path(multitask_manifest).read_text())
        if isinstance(supplied, dict) and 'contract' in supplied:
            if supplied != manifest:
                raise InfrastructureError('supplied training manifest differs from checkpoint run')
        elif task_specs(supplied) != specs:
            raise InfrastructureError('supplied task manifest differs from joint checkpoint tasks')
    cfg = contract['config']
    expected = {'action_dim': 14, 'state_dim': 14, 'action_chunk_size': 32, 'num_inference_steps': 10}
    if (any(cfg['common'].get(k) != v for k, v in expected.items())
            or cfg['common'].get('action_execution_horizon', 8) != 8
            or cfg['dataset'].get('cond_mode') != 'train_task_vtt'
            or tuple(cfg['model']['views']) != MODEL_VIEWS or not cfg['model']['use_task_cond']
            or contract.get('heldout_used') is not False or contract.get('encoder_loaded_for_training') is not False):
        raise InfrastructureError('joint native14/train_task_vtt/frozen-DINO contract mismatch')
    validate_condition_statistics(contract)
    for source, expected_hash in contract['source_hashes'].items():
        if sha256(source) != expected_hash:
            raise InfrastructureError('frozen joint training source changed: ' + source)
    caches, splits, files = {}, {}, {}
    for item in contract['cache_files']:
        if Path(item['path']).name == 'cache_config.json':
            files[str(Path(item['path']).resolve())] = item
    for spec in specs:
        task = spec['task']
        provenance = contract['condition_bank']['provenance']['tasks'][task]
        split = json.loads(Path(spec['split_manifest']).read_text())
        if data_digest(split) != provenance['split_manifest_sha256']:
            raise InfrastructureError('task split changed: ' + task)
        cache_path = Path(spec['feature_cache']) / 'cache_config.json'
        if not cache_path.is_file():
            cache_path = Path(spec['feature_cache']) / provenance['cache_config_id'] / 'cache_config.json'
        bound = files.get(str(cache_path.resolve()))
        if not bound or sha256(cache_path) != bound['sha256']:
            raise InfrastructureError('task cache config differs: ' + task)
        cache = json.loads(cache_path.read_text())
        if cache['source']['task'] != task or data_digest(cache) != provenance['cache_config_id']:
            raise InfrastructureError('task/cache identity mismatch: ' + task)
        for value in contract['statistics'].values():
            if value['provenance']['tasks'][task] != provenance:
                raise InfrastructureError('condition/statistics provenance differ: ' + task)
        splits[task], caches[task] = split, {'path': str(cache_path), 'value': cache}
    receipt = json.loads(checkpoint.with_name(checkpoint.name + '.sha256.json').read_text())
    if (receipt.get('model_only') is not True or receipt.get('size') != checkpoint.stat().st_size
            or receipt.get('step') != cfg['training']['max_steps']
            or not re.fullmatch(r'[0-9a-f]{64}', receipt.get('sha256', ''))):
        raise InfrastructureError('requires immutable final-budget model-only checkpoint receipt')
    return {'contract': contract, 'receipt': receipt, 'tasks': specs, 'caches': caches, 'splits': splits,
            'manifest': str(path), 'checkpoint': str(checkpoint)}


def select_condition(contract, task):
    import numpy as np
    try:
        value = contract['condition_bank']['conditions'][task]
    except KeyError as exc:
        raise InfrastructureError('unknown checkpoint task: ' + task) from exc
    return np.asarray(value, np.float32).copy()


def die_with_parent():
    """Worker owns a new session; still die on evaluator hard timeout (Linux)."""
    import ctypes
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if parent == 1 or libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        raise InfrastructureError('could not bind worker lifetime to evaluator parent')


class MultitaskPolicy(ChunkPolicy):
    def __init__(self, checkpoint, manifest, task, gpu_owner):
        from harness.robotwin_eval_evidence import ResourceGuard
        self.guard = ResourceGuard()
        self.guard.check()
        verify_gpu_owner(gpu_owner)
        bundle = load_contract(checkpoint, manifest)
        contract, receipt = bundle['contract'], bundle['receipt']
        self.task_condition = select_condition(contract, task)
        self.condition_hash = hashlib.sha256(self.task_condition.tobytes()).hexdigest()
        cache = bundle['caches'][task]['value']
        import cv2
        import numpy as np
        import torch
        from harness.robotwin_training_data import preprocess_rgb, cache_identity
        from harness.robotwin_multitask_training import load_multitask_policy
        from harness.wam_behavior import load_upstream
        from harness.wam_v3 import V3Context
        if cache_identity(bundle['splits'][task]) != cache:
            raise InfrastructureError('inference RGB/cache ABI differs from training')
        encoder_path = resolve(contract['config']['model']['vision_encoder']['checkpoint_path'])
        if sha256(encoder_path / 'model.safetensors') != cache['encoder_sha256']:
            raise InfrastructureError('frozen DINO content differs')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        cv2.setNumThreads(2)
        wrapper, condition, loaded_contract = load_multitask_policy(checkpoint, task, device='cuda')
        if loaded_contract != contract or not np.array_equal(condition, self.task_condition):
            raise InfrastructureError('loaded model/condition differs from verified task contract')
        verify_gpu_owner(gpu_owner)
        _, runner = load_upstream()
        encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
            str(encoder_path), torch.bfloat16, torch.device('cuda'))
        if (hidden, registers, patch) != (1024, 4, 16):
            raise InfrastructureError('unexpected frozen DINO geometry')
        encoder.requires_grad_(False).eval()
        wrapper.vision_encoder = encoder
        wrapper.eval()
        self.wrapper = wrapper
        self.runtime = {'task': task, 'task_condition_source': 'embedded_TRAIN_task_prototype',
            'task_condition_sha256': self.condition_hash, 'normalization': 'shared_all_TRAIN_frames',
            'statistics_sha256': digest(contract['statistics']), 'condition_bank_sha256': digest(contract['condition_bank']),
            'checkpoint_sha256': receipt['sha256'], 'checkpoint_step': receipt['step'],
            'contract_sha256': digest(contract), 'encoder_sha256': cache['encoder_sha256'],
            'preprocess_sha256': cache['preprocess_sha256'], 'strict_load': True, 'device': 'cuda',
            'numpy': np.__version__, 'opencv': cv2.__version__, 'torch': str(torch.__version__),
            'dino_frozen': not any(p.requires_grad for p in encoder.parameters()),
            'projection': 'task_cond_proj.residual.weight_joint_checkpoint',
            'middle_enabled': False, 'vlm_enabled': False, 'ask_enabled': False,
            'memory_updates_enabled': False, 'condition_switch_enabled': False}

        def encode(rgb):
            verify_gpu_owner(gpu_owner)
            self.guard.check()
            with torch.inference_mode():
                pixels = np.stack([preprocess_rgb(rgb[view]) for view in VIEWS])
                features, pool = wrapper.encode(torch.from_numpy(pixels)[None])
                return features, pool[0].float().cpu().numpy()

        def sample(features, inputs, seed, step):
            state, pool = inputs
            sample_seed = int(hashlib.sha256(f'{seed}:{step}'.encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device='cuda').manual_seed(sample_seed)
            with torch.inference_mode():
                action, _ = wrapper.sample_actions(features, torch.from_numpy(state)[None],
                    torch.from_numpy(pool)[None], torch.from_numpy(self.task_condition.copy())[None],
                    num_steps=10, generator=generator)
                return action[0].float().cpu().numpy()

        v3 = contract['config']['model']['v3']
        super().__init__(V3Context(v3['proprio_lags'], v3['memory_lags'], MODEL_VIEWS, state_dim=14),
                         encode, sample, execution_mode='native8')

    def observe(self, episode, step, observation):
        result = super().observe(episode, step, observation)
        import resource
        import torch
        result['task_condition'] = {'task': self.runtime['task'], 'source': 'embedded_TRAIN_task_prototype',
                                    'sha256': self.condition_hash}
        result['resources'] = {**self.guard.check(), 'policy_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cuda_allocated_bytes': torch.cuda.memory_allocated(), 'cuda_reserved_bytes': torch.cuda.memory_reserved(),
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ipc-fd', required=True, type=int)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--multitask-manifest', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu-lock-owner', required=True)
    args = parser.parse_args(argv)
    channel = Channel(socket.socket(fileno=args.ipc_fd))
    try:
        die_with_parent()
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '2'
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        serve(channel, MultitaskPolicy(args.checkpoint, args.multitask_manifest, args.task, args.gpu_lock_owner))
        return 0
    except BaseException as exc:
        try:
            channel.send(envelope('infrastructure_error', 'startup', 0, error=f'{type(exc).__name__}: {exc}'), 1)
        except BaseException:
            pass
        print(f'infrastructure_error: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    finally:
        channel.sock.close()


if __name__ == '__main__':
    raise SystemExit(main())
