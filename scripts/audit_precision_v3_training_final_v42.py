#!/usr/bin/env python3
"""Read-only CPU gate for the *completed* 30k V3 fork. No model construction/GPU.

Exit 0 / evaluation_ready=True requires final weights, 30000 logged updates,
exact full-state/initialization/data contracts and the departed trainer/free GPU
owner. Exit 3 means not ready (including a live trainer); exit 2 means an audit
invariant failed. Neither result changes training, its source or its status file.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TOTAL = 30000
UNIT = 'p2-zhuhai-v3fork80k-train-20261012.service'
OWNER = 'zhuhai-v3fork80k-20261012'
FORMAT = 'zhuhai-precision-wam-native16-measured-10hz-v3'
TASK = 'zhuhai_task2'
INITIALIZER = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger30k_diagnostic_20261011_v23/train/checkpoint_step_080000.pt'
INIT_SHA = '1de0c8b337bd3a01eec102e3b44df121dfa999bb9324d1c51b2f2a26f264f93c'
TRUSTED_EXTENSION = {
    'scripts/train_zhuhai_wam_precision_v3.py': 'a02503309d77fff6d051ddd30e94ae82b3cc20b6683e37afeb461f87f550a530',
    'harness/wam_precision_v3.py': '134a75da4af18dd905a77ac83bb543a63b9cc67688ab4934ecf3aca0ae1f74fc',
    'harness/zhuhai_precision_v3_training.py': 'db854407dba339fb3b180a13eabc2b9ab215265cafb98984b0e397db8364ef89',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def idm_digest(value):
    # IDM producer intentionally uses compact JSON, unlike the outer contract.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inspect_runtime(train_root):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', XDG_RUNTIME_DIR='/run/user/1000',
               DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    result = subprocess.run(['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState',
                             '-p', 'SubState', '-p', 'MainPID', '-p', 'Result',
                             '-p', 'ExecMainStatus', '-p', 'LoadState'],
                            capture_output=True, text=True, timeout=15, env=env)
    unit = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode:
        unit['query_error'] = result.stderr.strip()
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [a for a in (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0') if a]
            scripts = [a for a in argv if a.endswith('.py')]
            # The lock parent also mentions the trainer later in argv; it is not
            # an additional trainer and must not double the writer count.
            if not scripts or Path(scripts[0]).name != 'train_zhuhai_wam_precision_v3.py':
                continue
            if str(train_root.resolve()) not in argv:
                continue
            selected_env = dict(x.split('=', 1) for x in (entry / 'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
            processes.append(dict(pid=int(entry.name), argv=argv,
                                  pinned_library_path=selected_env.get('LD_LIBRARY_PATH')))
        except (OSError, ValueError):
            continue
    lock = Path('/tmp/project2_gpu.lock')
    try:
        owner = lock.read_text().strip()
    except FileNotFoundError:
        owner = None
    return dict(unit=unit, trainer_processes=processes, gpu_lock_owner=owner,
                gpu_queried=False)


def runtime_ready(runtime):
    unit = runtime['unit']
    return (not runtime['trainer_processes'] and runtime['gpu_lock_owner'] is None
            and not unit.get('query_error') and unit.get('LoadState') == 'loaded'
            and unit.get('ActiveState') == 'inactive' and unit.get('MainPID') == '0'
            and unit.get('ExecMainStatus') == '0' and unit.get('Result') == 'success')


def lr_factor(step_zero_based, total=TOTAL, warmup=200):
    if step_zero_based < warmup:
        return min(1., (step_zero_based + 1) / max(1, warmup))
    return .1 + .9 * (1 + math.cos(math.pi * (step_zero_based - warmup) / max(1, total - warmup))) / 2


def validate_log(path, *, total=TOTAL, base_lr=3e-6, precision_lr=1e-4, final=False):
    last = None
    losses = []
    idm_updates = 0
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            # In the live run a trailing write can be in flight. Complete lines
            # remain strict; no full row is silently dropped.
            if not line.endswith('\n') and not final:
                break
            row = json.loads(line)
            require(type(row.get('step')) is int and row['step'] == number, 'optimizer log skipped/repeated/reordered a step')
            require(number <= total, 'optimizer log exceeded target updates')
            for key in ('loss', 'grad', 'mse', 'future', 'future32', 'future8', 'idm', 'elapsed'):
                require(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]), 'nonfinite/missing log value: ' + key)
            require(row['loss'] >= 0 and row['grad'] >= 0, 'negative loss/gradient norm')
            require(last is None or row['elapsed'] > last['elapsed'], 'non-monotonic log elapsed time')
            factor = lr_factor(number - 1, total, min(200, max(1, total // 50)))
            for key, base in [('lr_trunk', base_lr), ('lr_prec', precision_lr)]:
                require(isinstance(row.get(key), (int, float)) and
                        math.isclose(row[key], base * factor, rel_tol=2e-12, abs_tol=1e-16), 'wrong LR schedule: ' + key)
            require(row.get('phase_ce') == 0. and row.get('gate_kl') == 0. and row.get('log_dist') == 0., 'unexpected removed phase/gate/dist losses')
            require(row.get('hjepa_variant') == 'combined', 'wrong actual HJEPA variant')
            require(all(type(row.get(key)) is int and 0 <= row[key] <= 4
                        for key in ('future8_valid_samples', 'idm_valid_samples')), 'invalid auxiliary validity counts')
            # Near terminal observations legitimately mask auxiliary horizons.
            # Invalid samples are not filtered from the action sampler.
            idm_updates += row['idm_valid_samples'] > 0
            require([row.get(k) for k in ['future32_weight', 'future8_weight', 'idm_weight']] == [.25, .25, .1], 'auxiliary training weights changed')
            last = row
            losses.append(row['loss'])
    count = last['step'] if last else 0
    if final:
        require(count == total, 'final log lacks exactly all target optimizer updates')
    return dict(optimizer_updates=count, last=last,
                idm_parameter_updates=idm_updates,
                mean_loss=sum(losses) / len(losses) if losses else None,
                recent_200_mean_loss=sum(losses[-200:]) / len(losses[-200:]) if losses else None,
                scope='training/auxiliary losses; no inference or task-success evidence')


def validate_manifest(train_root):
    manifest = read(train_root / 'manifest.json')
    contract = manifest['contract']
    require(manifest.get('contract_sha256') == digest(contract), 'manifest contract digest mismatch')
    require(contract.get('format') == FORMAT, 'not V3 FORMAT')
    require(contract.get('warm_start', {}).get('checkpoint_sha256') == INIT_SHA, 'wrong 80k warm-start identity')
    require(contract.get('warm_start', {}).get('precision_head_lr') == 1e-4, 'wrong refinement LR declaration')
    require(contract.get('hjepa', {}).get('variant') == 'combined', 'wrong HJEPA contract')
    require(contract.get('precision_v3', {}).get('refine_steps') == 32, 'V3 full chunk contract absent')
    require(contract['config']['model'].get('precision_v3') == contract['precision_v3'], 'two V3 specifications disagree')
    plan = read(train_root / 'plan.json')
    require(plan == dict(status='v3_fork80k', contract_sha256=digest(contract), steps=TOTAL,
                         warmup=200, base_lr=3e-6, precision_head_lr=1e-4, mode='train'), 'actual train plan differs from approved 30k fork')
    sources = contract['source_files']
    for rel, expected in TRUSTED_EXTENSION.items():
        require(sources.get(str(ROOT / rel)) == expected, 'frozen V3 extension identity changed: ' + rel)
    require(contract['hjepa']['code_hashes'] == sources, 'source code hash declarations disagree')
    for path, expected in sources.items():
        require(Path(path).is_file() and sha(path) == expected, 'frozen source drift: ' + path)
    require(set(contract['splits']) == {TASK}, 'unexpected training task')
    groups = contract['splits'][TASK]
    require(set(groups) == {'train', 'val', 'heldout'}, 'unexpected split names')
    seen_paths, seen_sha, families = set(), set(), {}
    for split, expected_n in [('train', 160), ('val', 20), ('heldout', 20)]:
        rows = groups[split]
        require(len(rows) == expected_n, 'original dataset count changed: ' + split)
        require(sum(r.get('domain') == 'source' for r in rows) == expected_n // 2 and
                sum(r.get('domain') == 'randomized' for r in rows) == expected_n // 2, 'source/DR count changed')
        for row in rows:
            require(row.get('split') == split and row.get('domain') in ('source', 'randomized')
                    and 'correction.hdf5' not in row['path'] and 'dagger' not in row['path'], 'DAgger/non-original row used in actual splits')
            require(row['path'] not in seen_paths and row['sha256'] not in seen_sha, 'episode duplicated between splits')
            seen_paths.add(row['path']); seen_sha.add(row['sha256'])
            family = row['family_id']
            require(family not in families or families[family] == split, 'family leak between splits')
            families[family] = split
    require(contract.get('train_frame_counts') == {TASK: 154871}, 'original TRAIN anchor count changed')
    for value, key in [(contract['statistics'], 'statistics_sha256'), (contract['idm_statistics'], 'idm_statistics_sha256')]:
        require(digest(value) == contract[key], 'TRAIN statistics digest mismatch: ' + key)
    provenance = contract['statistics']['normalization']['provenance']
    require(provenance.get('fit_split') == 'train' and provenance.get('heldout_used') is False
            and provenance.get('episode_count') == 160 and provenance.get('frame_count') == 154871, 'normalization used another split/data')
    require(contract['statistics']['temporal_stats']['provenance'] == provenance, 'temporal TRAIN provenance differs')
    require(provenance.get('train_data_sha256') == digest(provenance['records']), 'statistics source list digest mismatch')
    require({r['path']: r['sha256'] for r in provenance['records']} ==
            {r['path']: r['sha256'] for r in groups['train']}, 'statistics not fitted on exactly actual TRAIN')
    idm = contract['idm_statistics']
    require(idm.get('fit_split') == 'train' and idm.get('heldout_used') is False
            and idm.get('train_sources_sha256') == idm_digest(idm['train_sources'])
            and idm.get('statistics_sha256') == idm_digest({key: value for key, value in idm.items() if key != 'statistics_sha256'}),
            'IDM TRAIN provenance mismatch')
    require({r['path']: r['sha256'] for r in idm['train_sources']} ==
            {r['path']: r['sha256'] for r in groups['train']}, 'IDM source set differs from actual TRAIN')
    bank = contract['condition_bank']
    require(bank.get('source_split') == 'train' and bank['provenance'].get('heldout_used') is False
            and bank['provenance']['records'] == provenance['records']
            and bank['conditions'] == contract['conditions'], 'task condition used another split/data')
    inherited = dict(dagger_rows_retained_as_history=len(contract.get('dagger_training', {}).get('train_rows', [])),
                     actual_dagger_rows_used=0, stale_precision_steps=contract['precision_v3']['model'].get('precision_steps'),
                     stale_refined_flow_prefix=contract['precision_v3'].get('refined_flow_prefix'),
                     stale_checkpoint_model_version=contract['precision_v3'].get('checkpoint_model_version'),
                     stale_config_max_steps=contract['config']['training'].get('max_steps'),
                     actual_target_optimizer_updates=TOTAL,
                     explanation='inherited V2/80k metadata is not the actual V3 head or the executed sampler; actual source/plan/tensors are checked independently')
    return contract, dict(source_files_exact=len(sources), actual_train_episodes=160, val_episodes=20,
                          heldout_episodes_declared_but_not_constructed=20, train_action_frames=154871,
                          actual_training_data='original source80 + DR80; no DAgger recovery episodes',
                          frozen_dino=True, inherited_metadata=inherited)


def validate_head(state):
    import torch
    required = ['precision.version', 'precision.configuration', 'precision.patch_proj.weight',
                'precision.patch_proj.bias', 'precision.patch_xy', 'precision.patch_ray']
    require(all(k in state for k in required), 'missing V3 head tensor')
    require(state['precision.version'].numel() == 1 and int(state['precision.version'].item()) == 3, 'V3 tensor version mismatch')
    require(state['precision.configuration'].tolist() == [16, 32, 0, 1, 2], 'V3 tensor refinement/config mismatch')
    require(tuple(state['precision.patch_proj.weight'].shape) == (768, 1029)
            and tuple(state['precision.patch_proj.bias'].shape) == (768,), 'V3 full patch/ray projection shape mismatch')
    xy, ray = state['precision.patch_xy'], state['precision.patch_ray']
    require(tuple(xy.shape) == (1, 1, 196, 2) and tuple(ray.shape) == (1, 1, 196, 3), 'V3 patch geometry mismatch')
    t = math.tan(math.radians(65.) / 2)
    target = torch.cat((xy[..., 0:1] * t, xy[..., 1:2] * t, -torch.ones_like(xy[..., 0:1])), dim=-1)
    target = target / target.norm(dim=-1, keepdim=True)
    require(torch.equal(ray, target), 'V3 ray tensor differs from frozen implementation')
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu', 'model tensor not CPU')
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all()), 'nonfinite model tensor: ' + name)
    return dict(version=3, configuration=[16, 32, 0, 1, 2], refinement_steps=32,
                dense_kv_patch_tokens=588, patch_projection_shape=[768, 1029],
                ray_shape=[1, 1, 196, 3], ray_kind='fixed camera-local direction derived from xy; no depth/extrinsics',
                all_model_tensors_finite=True)


def validate_rng(saved):
    import numpy as np
    import torch
    require(isinstance(saved.get('rng'), dict) and saved['rng'].get('bit_generator') == 'PCG64', 'missing/invalid numpy sampler RNG')
    rng = np.random.default_rng(0)
    rng.bit_generator.state = saved['rng']  # real CPU parser validates full state
    require(isinstance(saved.get('torch_rng'), torch.Tensor) and saved['torch_rng'].dtype == torch.uint8
            and saved['torch_rng'].ndim == 1 and saved['torch_rng'].numel() > 0, 'missing/invalid torch RNG')
    require(isinstance(saved.get('cuda_rng'), list) and len(saved['cuda_rng']) == 1
            and all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and t.dtype == torch.uint8
                    and t.ndim == 1 and t.numel() > 0 for t in saved['cuda_rng']), 'missing/invalid single-GPU RNG archive')
    generator = torch.Generator(device='cpu')
    generator.set_state(saved['torch_rng'])  # does not touch global RNG or CUDA


def validate_optimizer(optimizer, final_lr, *, total=TOTAL, parameter_names=None, idm_updates=None, parameter_shapes=None):
    import torch
    require(isinstance(optimizer, dict) and set(optimizer) == {'state', 'param_groups'}, 'missing AdamW state')
    groups = optimizer['param_groups']
    require(len(groups) == 2, 'wrong optimizer parameter groups')
    ids = [i for group in groups for i in group['params']]
    require(ids and len(ids) == len(set(ids)), 'duplicate/empty optimizer parameter IDs')
    require(optimizer['state'] and set(optimizer['state']) == set(ids), 'missing/invalid optimizer parameter state IDs')
    require(ids == list(range(len(ids))), 'optimizer parameter registration order changed')
    by_id = {}
    if parameter_names is not None:
        require(len(parameter_names) == 2 and all(len(group['params']) == len(names)
                                                 for group, names in zip(groups, parameter_names)), 'optimizer/model parameter counts differ')
        by_id = {parameter_id: name for group, names in zip(groups, parameter_names)
                 for parameter_id, name in zip(group['params'], names)}
    for group, lr in zip(groups, final_lr):
        require(math.isclose(group['lr'], lr, rel_tol=2e-12, abs_tol=1e-16), 'final optimizer LR mismatch')
        require(tuple(group['betas']) == (.9, .99) and group['weight_decay'] == .01, 'AdamW hyperparameters changed')
    for parameter_id, state in optimizer['state'].items():
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'incomplete AdamW moments')
        step = state['step'].item() if isinstance(state['step'], torch.Tensor) else state['step']
        expected_step = (idm_updates if by_id.get(parameter_id, '').startswith(('hjepa_aux.idm_head.', 'hjepa_aux.step_embedding.'))
                         and idm_updates is not None else total)
        require(step == expected_step, 'AdamW parameter update count differs from actual valid training updates')
        require(state['exp_avg'].shape == state['exp_avg_sq'].shape, 'AdamW moment shapes disagree')
        if parameter_shapes is not None:
            require(tuple(state['exp_avg'].shape) == tuple(parameter_shapes[by_id[parameter_id]]), 'AdamW moment shape differs from its model parameter')
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = state[key]
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
                    and bool(torch.isfinite(tensor).all()), 'nonfinite/invalid AdamW moment')
        require(bool((state['exp_avg_sq'] >= 0).all()), 'negative AdamW second moment')
    return dict(parameter_groups=2, parameters_in_groups=len(ids), parameters_with_full_moments=len(optimizer['state']),
                main_optimizer_updates=total, idm_updates_excluding_all_invalid_batches=idm_updates,
                note='all declared trainable parameters have full AdamW state; IDM head skips all-invalid batches; other stored moments span all 30000 updates')


def verified_checkpoint(path, contract, *, step, model_only):
    import torch
    receipt = read(path.with_name(path.name + '.sha256.json'))
    require(receipt.get('size') == path.stat().st_size and receipt.get('sha256') == sha(path)
            and receipt.get('step') == step and receipt.get('model_only') is model_only, 'checkpoint receipt mismatch: ' + path.name)
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    require(saved.get('format') == FORMAT and saved.get('step') == step and saved.get('model_only') is model_only,
            'checkpoint FORMAT/step/model_only mismatch: ' + path.name)
    require(saved.get('contract') == contract and saved.get('contract_sha256') == digest(contract), 'checkpoint contract mismatch: ' + path.name)
    return saved, dict(path=str(path), **receipt)


def exact_state(first, second, message):
    import torch
    require(set(first) == set(second), message + ': keys')
    for name in first:
        require(first[name].dtype == second[name].dtype and first[name].shape == second[name].shape
                and torch.equal(first[name], second[name]), message + ': ' + name)


def validate_final_artifacts(train_root, contract, log):
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'audit must never initialize CUDA')
    require(sha(INITIALIZER) == INIT_SHA, 'original 80k checkpoint no longer preserved')
    original = torch.load(INITIALIZER, map_location='cpu', weights_only=False, mmap=True)
    require(original.get('step') == 80000, 'original 80k step mismatch')
    expected = copy.deepcopy(original['contract'])
    expected['format'] = FORMAT
    spec = copy.deepcopy(original['contract']['precision'])
    spec.update(schema='full_chunk_camera_ray_precision_v3', refine_steps=32)
    expected['precision_v3'] = spec
    expected['config']['model']['precision_v3'] = copy.deepcopy(spec)
    for rel, hash_value in TRUSTED_EXTENSION.items():
        expected['source_files'][str(ROOT / rel)] = hash_value
    expected['hjepa']['code_hashes'] = copy.deepcopy(expected['source_files'])
    expected['warm_start'] = dict(fork='80k v2 Precision', checkpoint_sha256=INIT_SHA,
                                  control_run='none (user directive: fork-80k only, highest success)', precision_head_lr=1e-4)
    require(contract == expected, 'V3 contract is not exactly the approved fork of original 80k')
    initial, initial_receipt = verified_checkpoint(train_root / 'checkpoint_step_000000.pt', contract, step=0, model_only=True)
    head = validate_head(initial['model_state_dict'])
    old_state, zero_state = original['model_state_dict'], initial['model_state_dict']
    require(set(zero_state) == set(old_state) | {'precision.patch_ray'}, 'V3 initialization unexpectedly changed model structure')
    exceptions = {'precision.version', 'precision.configuration', 'precision.patch_proj.weight'}
    for name in old_state:
        if name not in exceptions:
            exact_state({name: old_state[name]}, {name: zero_state[name]}, '80k inherited tensor changed at warm start')
    projection = zero_state['precision.patch_proj.weight']
    require(torch.equal(projection[:, :1026], old_state['precision.patch_proj.weight'])
            and bool((projection[:, 1026:] == 0).all()), 'V3 feature+xy/ray warm start differs')
    final, final_receipt = verified_checkpoint(train_root / 'checkpoint_step_030000.pt', contract, step=TOTAL, model_only=True)
    final_head = validate_head(final['model_state_dict'])
    require(set(final['model_state_dict']) == set(zero_state), 'final model structure changed')
    for name in zero_state:
        require(final['model_state_dict'][name].dtype == zero_state[name].dtype
                and final['model_state_dict'][name].shape == zero_state[name].shape, 'final model tensor schema changed: ' + name)
    for name in ('precision.version', 'precision.configuration', 'precision.patch_xy', 'precision.patch_ray'):
        exact_state({name: final['model_state_dict'][name]}, {name: zero_state[name]}, 'frozen V3 architecture buffer changed')
    for name in zero_state:
        if name.startswith('task_cond_proj.base.'):
            exact_state({name: final['model_state_dict'][name]}, {name: zero_state[name]}, 'frozen task-condition base changed')
    full, full_receipt = verified_checkpoint(train_root / 'checkpoint_last.pt', contract, step=TOTAL, model_only=False)
    exact_state(full['model_state_dict'], final['model_state_dict'], 'full-state and final milestone weights differ')
    buffers = {'action_pos_emb', 'proprio_pos_emb', 'hjepa_aux.idm_scale', 'precision.version',
               'precision.configuration', 'precision.patch_xy', 'precision.patch_ray'}
    names = [name for name in zero_state if name not in buffers and not name.startswith('task_cond_proj.base.')]
    require(len(names) == 323, 'actual trainable parameter inventory changed')
    changed_names = [name for name in names if not torch.equal(final['model_state_dict'][name], zero_state[name])]
    require(changed_names and any(name.startswith('precision.') for name in changed_names), 'final training did not change the new precision head')
    parameter_names = [[name for name in names if not name.startswith('precision.')],
                       [name for name in names if name.startswith('precision.')]]
    optimizer = validate_optimizer(full.get('optimizer_state_dict'), [log['last']['lr_trunk'], log['last']['lr_prec']],
                                   parameter_names=parameter_names, idm_updates=log['idm_parameter_updates'],
                                   parameter_shapes={name: zero_state[name].shape for name in names})
    validate_rng(full)
    milestones = []
    for step in range(5000, TOTAL + 1, 5000):
        path = train_root / f'checkpoint_step_{step:06d}.pt'
        receipt = read(path.with_name(path.name + '.sha256.json'))
        require(receipt.get('step') == step and receipt.get('model_only') is True
                and receipt.get('size') == path.stat().st_size and receipt.get('sha256') == sha(path), 'immutable milestone bytes/receipt missing or changed')
        diagnostic = read(train_root / f'diagnostic_{step:06d}.json')
        require(set(diagnostic) == {'train', 'val'} and
                all(set(diagnostic[split]) == {TASK} and math.isfinite(diagnostic[split][TASK]['loss'])
                    for split in ('train', 'val')), 'train/VAL milestone diagnostic missing or nonfinite')
        milestones.append(dict(receipt, path=str(path)))
    # Raw training/validation archives are hashed, never encoded or sampled by
    # this audit. Heldout files remain unopened: frozen source confirms they are
    # declared for split validation and never constructed for training.
    rows = [row for split in ('train', 'val') for row in contract['splits'][TASK][split]]
    archive_bytes = 0
    for row in rows:
        archive = Path(row['path'])
        require(archive.is_file() and sha(archive) == row['sha256'], 'actual TRAIN/VAL archive changed: ' + row['path'])
        archive_bytes += archive.stat().st_size
    encoder = ROOT / contract['config']['model']['vision_encoder']['checkpoint_path'] / 'model.safetensors'
    require(sha(encoder) == contract['cache']['encoder_sha256'], 'frozen DINO encoder bytes changed')
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return dict(initializer_preserved_sha256=INIT_SHA, warm_start_all_inherited_tensors_exact=True,
                checkpoint0=initial_receipt, checkpoint_final=final_receipt, checkpoint_full_state=full_receipt,
                preserved_milestones=milestones, diagnostic_scope='one fixed TRAIN and one fixed VAL anchor per milestone, not a full VAL evaluation',
                head=final_head, optimizer=optimizer, numpy_torch_and_single_gpu_rng_restorable=True,
                trainable_tensors_changed_from_checkpoint0=len(changed_names),
                precision_tensors_changed_from_checkpoint0=sum(name.startswith('precision.') for name in changed_names),
                learned_ray_projection_max_abs=float(final['model_state_dict']['precision.patch_proj.weight'][:, 1026:].abs().max()),
                original_train_val_archives_verified=180, original_train_val_archive_bytes=archive_bytes,
                heldout_archives_opened=0, frozen_dino_bytes_preserved=True,
                model_instances_constructed=0, cuda_initialized=False,
                full_state_equals_final_milestone=True)


def audit(train_root, *, runtime=None):
    train_root = Path(train_root).resolve()
    report = dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                  auditor='precision_v3_final_training_v42', train_root=str(train_root),
                  evaluation_ready=False, status='waiting_for_training_completion',
                  mutates_training=False, gpu_used=False, target_optimizer_updates=TOTAL)
    try:
        status = read(train_root / 'status.json')
        report['training_status_observed'] = status
        require(status.get('target_steps') == TOTAL and type(status.get('step')) is int
                and 0 <= status['step'] <= TOTAL, 'unexpected training target/step')
        report['runtime'] = runtime if runtime is not None else inspect_runtime(train_root)
        require(len(report['runtime']['trainer_processes']) <= 1, 'duplicate trainer writers detected')
        if not (train_root / 'manifest.json').exists() and status.get('status') == 'building' and status['step'] == 0:
            report['reason'] = 'Trainer is constructing the warm-start model before the manifest/optimizer.'
            return report
        contract, report['contract_and_sources'] = validate_manifest(train_root)
        path = train_root / 'log.jsonl'
        report['optimizer_log'] = validate_log(path, final=False) if path.exists() else dict(optimizer_updates=0, last=None)
        # Atomic status may lag a just-appended log by one during live updates.
        if report['runtime']['trainer_processes']:
            status_after = read(train_root / 'status.json')
            report['training_status_after_log_snapshot'] = status_after
            require(status['step'] <= report['optimizer_log']['optimizer_updates'] <= status_after['step'] + 1,
                    'live status/log optimizer count mismatch')
        if status.get('status') in ('failed', 'stopped'):
            report.update(status='training_terminal_without_completed_target', reason=status.get('error') or status['status'])
            return report
        if status.get('status') != 'completed' or status['step'] != TOTAL:
            report['reason'] = 'The final 30000-update status is not present; step0/milestones do not satisfy final completion.'
            return report
        if not runtime_ready(report['runtime']):
            report['reason'] = 'Final status exists but trainer/unit exit or GPU-owner release is not yet independently proven.'
            return report
        report['optimizer_log'] = validate_log(path, final=True)
        report['artifacts'] = validate_final_artifacts(train_root, contract, report['optimizer_log'])
        # Recheck source and terminal state after potentially long archive hashes;
        # release at audit start does not prove the GPU remains available.
        _, report['contract_and_sources'] = validate_manifest(train_root)
        after = runtime if runtime is not None else inspect_runtime(train_root)
        report['runtime_after_artifact_hashes'] = after
        require(read(train_root / 'status.json') == status, 'terminal training status changed during audit')
        if not runtime_ready(after):
            report['reason'] = 'GPU/trainer state changed during final audit; training artifacts verified but evaluation must wait.'
            return report
        report.update(status='passed_final_training_and_gpu_released', evaluation_ready=True,
                      scope='training admission only; no robot success-rate or architecture-causality claim')
        return report
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report.update(status='failed_audit_invariant', error=f'{type(exc).__name__}: {exc}',
                      explanation='Audit gate rejected evidence; the trainer and its status were not changed.')
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    report = audit(args.train_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(json.dumps(dict(status=report['status'], evaluation_ready=report['evaluation_ready'],
                          actual_updates=report.get('optimizer_log', {}).get('optimizer_updates'),
                          reason=report.get('reason') or report.get('error'), out=str(args.out)), ensure_ascii=False))
    return 0 if report['evaluation_ready'] else (2 if report['status'] in ('failed_audit_invariant', 'training_terminal_without_completed_target') else 3)


if __name__ == '__main__':
    raise SystemExit(main())
