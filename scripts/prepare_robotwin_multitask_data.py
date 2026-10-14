#!/usr/bin/env python3
"""Prepare a CPU-only five-task demo_clean pilot; never edit existing sources/splits.

Run with .venvs/wam-fixed/bin/python scripts/prepare_robotwin_multitask_data.py
--download-missing. Downloads are restricted to the two selected official clean50
archives. No simulator, encoder, torch, feature cache, training, or video is run.
Each task's split is compatible with harness.robotwin_dataset.validate_split.
The top-level manifest.json exposes tasks[*].split_manifest for trainer/runner use.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = 'TianxingChen/RoboTwin2.0'
REVISION = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
TASKS = ('put_bottles_dustbin', 'place_cans_plasticbox', 'open_laptop',
         'click_bell', 'press_stapler')
NEW_TASKS = ('click_bell', 'press_stapler')
SEED = 20261004  # Same split algorithm/seed as frozen bottle experiment.
GIB = 1024 ** 3
BYTES_PER_CACHED_FRAME = 3 * 3 * 201 * 1024 * 2 + 3 * 2048 * 4
SPLITS = ('train', 'val', 'heldout')
VIEWS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
JOINT_FIELDS = (('left_arm_joint_states', 6), ('left_ee_joint_states', 1),
                ('right_arm_joint_states', 6), ('right_ee_joint_states', 1))
RIGID_ASSETS = {
    'put_bottles_dustbin': {'114_bottle': [1, 2, 3], '011_dustbin': [0]},
    'place_cans_plasticbox': {'062_plasticbox': [3, 5], '071_can': [0, 1, 2, 3, 5, 6]},
    'open_laptop': {}, 'click_bell': {'050_bell': [0, 1]},
    'press_stapler': {'048_stapler': list(range(7))},
}
REASONS = {
    'put_bottles_dustbin': 'existing clean50, frozen split/cache reuse; long multi-object transport',
    'place_cans_plasticbox': 'existing clean50; bilateral two-object placement',
    'open_laptop': 'existing clean50; articulated-object manipulation',
    'click_bell': 'small official clean50 archive; short contact task; local assets',
    'press_stapler': 'small official clean50 archive; short contact task; local assets',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()


def write_once(path, value):
    """Do not silently alter a preregistered split or completed artifact on rerun."""
    path = Path(path)
    content = json_bytes(value)
    if path.exists():
        require(path.read_bytes() == content, f'Refusing to change existing artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(content)


def status(out, value):
    path = Path(out) / 'preparation_status.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_bytes(json_bytes(value))
    tmp.replace(path)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def file_digest(path):
    h, crc = hashlib.sha256(), 0
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024 ** 2):
            h.update(block)
            crc = zlib.crc32(block, crc)
    return h.hexdigest(), f'{crc:08x}'


def identity(path):
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size > 0, f'Missing/empty file: {path}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_digest(path)[0]}


def budget_guard(free_bytes, additional_bytes, archive_bytes, extraction_ceiling=8 * GIB,
                 max_additional_bytes=50 * GIB, min_free_bytes=300 * GIB):
    """Budget includes compressed transport plus worst-case extracted bytes."""
    require(0 < archive_bytes <= 4 * GIB, 'Archive outside clean50 size limit')
    require(0 <= additional_bytes, 'Invalid cumulative download accounting')
    peak = archive_bytes + extraction_ceiling
    require(additional_bytes + peak <= max_additional_bytes,
            'Additional source-data budget would be exceeded')
    require(free_bytes - peak >= min_free_bytes, '300GiB free-space reserve would be breached')


def fetch_metadata(task):
    require(task in NEW_TASKS, 'Only the two selected official tasks can be downloaded')
    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/dataset/{task}'
    request = urllib.request.Request(url, headers={'User-Agent': 'robotwin-pilot-preparer/1'})
    with urllib.request.urlopen(request, timeout=120) as f:
        rows = json.load(f)
    row = next(x for x in rows if x.get('path') == f'dataset/{task}/demo_clean.zip')
    digest = row.get('lfs', {}).get('oid') or row.get('lfs', {}).get('sha256')
    require(isinstance(digest, str) and len(digest) == 64, 'Missing official archive SHA256')
    return row


def source_location(base, task):
    if task == 'put_bottles_dustbin':
        folder = base / 'put_bottles_dustbin_50_v1'
        return folder, folder / 'data'
    folder = base / task
    return folder, folder / 'source' / task / 'aloha_agilex' / 'data'


def read_source(folder, task):
    receipt_path = folder / 'SOURCE.json'
    source = json.loads(receipt_path.read_text())
    require(source.get('status') == 'complete', f'Incomplete source: {receipt_path}')
    require(source.get('repo') == REPO and source.get('revision') == REVISION,
            f'Unexpected source revision: {receipt_path}')
    require(source.get('task') == task, 'Source task mismatch')
    archive_receipt = receipt_path
    if 'raw_source_receipt' in source:
        archive_receipt = Path(source['raw_source_receipt'])
        require(file_digest(archive_receipt)[0] == source['raw_source_sha256'],
                'Preserved raw source receipt changed')
    archive = json.loads(archive_receipt.read_text())
    require(archive.get('repo') == REPO and archive.get('revision') == REVISION
            and archive.get('task') == task and archive.get('status') == 'complete'
            and archive.get('archive') == 'demo_clean.zip', 'Wrong/unfinished source archive')
    require(archive.get('expected_sha256') == archive.get('sha256')
            and len(archive.get('sha256', '')) == 64 and archive.get('zip_crc_verified'),
            'Missing successful upstream archive verification')
    if 'raw_archive_sha256' in source:
        require(source['raw_archive_sha256'] == archive['sha256'], 'Raw archive identity changed')
    expected_prefix = f'source/{task}/aloha_agilex/data/'
    members = {Path(row['path']).name: row for row in archive['members']
               if row['path'].startswith(expected_prefix) and row['path'].endswith('.hdf5')}
    require(len(members) == 50, f'Expected exactly 50 native aloha_agilex HDF5 members: {task}')
    return {'receipt': identity(receipt_path), 'archive_receipt': identity(archive_receipt),
            'archive_sha256': archive['sha256'], 'archive_url': archive['url'],
            'repo': REPO, 'revision': REVISION, 'archive': 'demo_clean.zip',
            'archive_bytes': archive['received_bytes'], 'extracted_bytes': archive['extracted_bytes']}, members


def check_urdf(path):
    """Check each URDF mesh reference without importing SAPIEN or touching GPU."""
    path = Path(path).resolve()
    tree = ET.parse(path)
    refs = []
    for elem in tree.iter('mesh'):
        name = elem.attrib.get('filename', '')
        require(name and '://' not in name, f'Unsupported mesh URI in {path}: {name}')
        mesh = (path.parent / name).resolve()
        require(mesh.is_file() and mesh.stat().st_size > 0, f'Missing URDF mesh: {mesh}')
        refs.append(str(mesh))
    require(refs, f'URDF has no mesh references: {path}')
    return {'urdf': identity(path), 'mesh_references_checked': sorted(set(refs))}


def rigid_asset_check(objects, model, model_id):
    root = objects / model
    result = {'model': model, 'model_id': model_id,
              'metadata': identity(root / f'model_data{model_id}.json')}
    for kind in ('collision', 'visual'):
        candidates = [parent / name for parent in (root / kind, root)
                      for name in (f'base{model_id}.glb', f'textured{model_id}.obj')]
        mesh = next((x for x in candidates if x.is_file()), None)
        require(mesh is not None and mesh.stat().st_size > 0, f'Missing {kind}: {model}/{model_id}')
        result[kind] = identity(mesh)
    return result


def audit_native(robotwin, task):
    import yaml
    robotwin = Path(robotwin)
    config_path = robotwin / 'env_cfg/task_config/demo_clean.yml'
    cfg = yaml.safe_load(config_path.read_text())
    require(cfg['embodiment'] == ['aloha-agilex'], 'demo_clean robot mismatch')
    require(cfg['episode_num'] == 50 and cfg['save_freq'] == 15, 'demo_clean collection contract changed')
    require(cfg['data_type']['rgb'] and not cfg['data_type']['third_view'], 'Unexpected camera contract')
    camera_cfg = cfg['camera']
    require(camera_cfg['collect_head_camera'] and camera_cfg['collect_wrist_camera']
            and camera_cfg['head_camera_type'] == camera_cfg['wrist_camera_type'] == 'D435',
            'Unexpected onboard camera contract')
    dr = cfg['domain_randomization']
    require(not any(dr[k] for k in ('random_background', 'cluttered_table', 'random_light')),
            'Selected config is not demo_clean')
    embodiment = robotwin / 'assets/embodiments/aloha-agilex'
    robot_cfg_path = embodiment / 'config.yml'
    robot_cfg = yaml.safe_load(robot_cfg_path.read_text())
    require([len(x) for x in robot_cfg['arm_joints_name']] == [6, 6]
            and len(robot_cfg['gripper_name']) == 2 and robot_cfg['dual_arm'], 'Robot is not native dual-arm14')
    robot = {'config': identity(robot_cfg_path),
             'urdf_check': check_urdf(embodiment / robot_cfg['urdf_path']),
             'srdf': identity(embodiment / robot_cfg['srdf_path']),
             'planner_configs': [identity(embodiment / f'curobo_{side}.yml') for side in ('left', 'right')]}
    task_path = robotwin / 'envs' / f'{task}.py'
    syntax = ast.parse(task_path.read_text())
    cls = next((node for node in syntax.body if isinstance(node, ast.ClassDef) and node.name == task), None)
    require(cls is not None, f'Missing task implementation: {task}')
    methods = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
    require({'load_actors', 'setup_demo', 'play_once', 'check_success'} <= methods,
            f'Incomplete native task entry points: {task}')
    objects = robotwin / 'assets/objects'
    assets = [rigid_asset_check(objects, model, mid)
              for model, ids in RIGID_ASSETS[task].items() for mid in ids]
    if task == 'open_laptop':
        variants = sorted((p for p in (objects / '015_laptop').iterdir()
                           if p.is_dir() and p.name != 'visual'), key=lambda p: int(p.name))
        require(len(variants) >= 11, 'Missing laptop variants 0..10')
        for i, variant in enumerate(variants[:11]):
            assets.append({'model': '015_laptop', 'model_id': i,
                           'native_directory': variant.name,
                           'metadata': identity(variant / 'model_data.json'),
                           'urdf_check': check_urdf(variant / 'mobility.urdf')})
    limit_path = robotwin / 'env_cfg/task_config/_eval_step_limit.yml'
    limits = yaml.safe_load(limit_path.read_text())
    require(isinstance(limits.get(task), int) and limits[task] > 0, 'Missing native evaluation step limit')
    instruction = robotwin / 'description/task_instruction' / f'{task}.json'
    require(instruction.is_file(), f'Missing task instructions: {instruction}')
    return {'status': 'passed_static_only', 'task': task, 'embodiment': 'aloha-agilex',
            'dataset_embodiment_directory': 'aloha_agilex', 'state_dim': 14, 'action_dim': 14,
            'joint_field_order': [name for name, _ in JOINT_FIELDS], 'policy_views': list(VIEWS),
            'task_source': identity(task_path), 'task_config': identity(config_path),
            'task_instructions': identity(instruction), 'robot': robot, 'object_assets': assets,
            'native_eval_step_limit': limits[task], 'eval_step_config': identity(limit_path),
            'simulation_run': False, 'gpu_used': False,
            'runtime_preflight_required': True,
            'scope': 'CPU file/URDF-mesh/config/HDF contract checks, NOT simulator rollout validation',
            'evaluation_overrides_required': {'record_video': False, 'eval_video_log': False,
                                              'third_view': False},
            'state_semantics': 'controller_joint_targets_and_gripper_command_not_measured_qpos',
            'action_semantics': 'native_absolute_joint_targets_and_gripper_command_no_shift'}


def make_task_split(data_root, inherited=None):
    from harness.robotwin_dataset import make_split, validate_split
    expected = make_split(data_root, seed=SEED, counts=(40, 5, 5))
    if inherited is not None:
        original = json.loads(Path(inherited).read_text())
        validate_split(original)
        require(original == expected, 'Existing frozen split must be preserved exactly')
    return expected


def audit_episode(path, split, archive_member):
    import h5py
    import numpy as np
    from harness.robotwin_dataset import RoboTwinEpisode
    require(split in SPLITS, 'Unknown split')
    path = Path(path)
    require(path.stat().st_size == archive_member['bytes'], f'Official member size mismatch: {path}')
    digest, crc = file_digest(path)
    require(crc == archive_member['zip_crc32'], f'Official ZIP CRC32 mismatch: {path}')
    with h5py.File(path, 'r') as f:
        require(f['data_format_version'].asstr()[()] == 'v1.0', 'Expected native dataset v1.0')
        require(f.attrs.get('source_format') == 'RoboTwin', 'Unexpected source format')
        require(float(f['additional_info/frequency'][()]) == 15, 'Unexpected nominal frequency')
        frames = f['state/left_arm_joint_states'].shape[0]
        require(frames > 0, 'Empty episode')
        for group in ('state', 'action'):
            for field, width in JOINT_FIELDS:
                arr = f[f'{group}/{field}']
                require(arr.shape == (frames, width) and arr.dtype.kind in 'fiu',
                        f'Invalid native14 layout: {path}:{group}/{field}')
        for view in VIEWS:
            cam = f[f'vision/{view}']
            arr = cam['colors']
            require(arr.shape[0] == frames, 'Camera/frame alignment mismatch')
            require(np.array_equal(cam['shape'][:], [240, 320, 3]), 'Camera shape mismatch')
        extra_views = sorted(set(f['vision']) - set(VIEWS))
    # Heldout is sealed: only integrity hashing and structural metadata, no
    # lowdim payload/instructions/image decoding and no model-derived features.
    if split != 'heldout':
        ep = RoboTwinEpisode(path)
        require(len(ep) == frames and ep.state.shape == ep.actions.shape == (frames, 14),
                'Training reader contract mismatch')
    return {'episode': path.name, 'split': split, 'bytes': path.stat().st_size,
            'sha256': digest, 'official_zip_crc32': crc, 'frames': frames,
            'native14_metadata_validated': True, 'ignored_nonpolicy_views': extra_views,
            'lowdim_finite_and_first_onboard_rgb_validated': split != 'heldout',
            'heldout_integrity_metadata_only': split == 'heldout'}


def ensure_source(base, task, download, additional_bytes):
    folder, data_root = source_location(base, task)
    if (folder / 'SOURCE.json').exists():
        source, members = read_source(folder, task)
        require(data_root.is_dir(), f'Missing source HDF5 directory: {data_root}')
        return source, members, additional_bytes
    require(not folder.exists(), f'Refusing to alter unrecognized/partial source directory: {folder}')
    require(download and task in NEW_TASKS, f'Missing {task}; pass --download-missing')
    metadata = fetch_metadata(task)
    budget_guard(shutil.disk_usage(base).free, additional_bytes, metadata['size'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/download_robotwin_subset.py'),
                    '--task', task, '--archive', 'demo_clean.zip', '--out', str(folder)],
                   check=True, env=env, stdout=subprocess.PIPE, text=True)
    source, members = read_source(folder, task)
    additional_bytes += source['archive_bytes'] + source['extracted_bytes']
    require(additional_bytes <= 50 * GIB and shutil.disk_usage(base).free >= 300 * GIB,
            'Post-download resource guard failed')
    return source, members, additional_bytes


def prepare(args):
    base, robotwin = args.data_base.resolve(), args.robotwin_root.resolve()
    out = args.out.resolve() if args.out else base / 'multitask_pilot5_20261008'
    require(out.parent == base and out.name.startswith('multitask_pilot'),
            'Output must be a new multitask_pilot* child of data-base')
    require(base.is_dir(), 'Dataset base directory missing')
    out.mkdir(exist_ok=True)
    selection = {'schema_version': 1, 'task_order': list(TASKS), 'task_config': 'demo_clean',
                 'official_repo': REPO, 'revision': REVISION, 'split_seed': SEED,
                 'counts_per_task': dict(zip(SPLITS, (40, 5, 5))),
                 'selection_reasons': REASONS, 'selected_without_policy_results': True,
                 'max_additional_source_bytes': 50 * GIB, 'min_free_bytes': 300 * GIB,
                 'gpu_jobs': 0, 'feature_cache_created': False, 'record_video': False}
    write_once(out / 'selection.json', selection)
    additional = 0
    entries = []
    try:
        for task_id, task in enumerate(TASKS):
            require(shutil.disk_usage(base).free >= 300 * GIB, 'Free space below required reserve')
            status(out, {'status': 'preparing', 'task': task, 'completed_tasks': [x['task'] for x in entries]})
            # Fail before downloading if local evaluation assets are missing.
            native = audit_native(robotwin, task)
            source, members, additional = ensure_source(base, task, args.download_missing, additional)
            folder, data_root = source_location(base, task)
            inherited = folder / 'split_manifest.json' if task == TASKS[0] else None
            split = make_task_split(data_root, inherited)
            task_out = out / task
            rows = [audit_episode(data_root / name, group, members[name])
                    for group in SPLITS for name in split['splits'][group]]
            # No aliases or byte-identical episodes across split/task boundaries.
            require(len({x['sha256'] for x in rows}) == 50, f'Duplicate episode bytes: {task}')
            frames = {group: sum(x['frames'] for x in rows if x['split'] == group) for group in SPLITS}
            provenance = {'schema_version': 1, 'task': task, **source,
                          'data_root': str(data_root), 'native_embodiment': 'aloha-agilex',
                          'original_hdf5_unchanged': True, 'copied_or_converted': False,
                          'inherited_split': identity(inherited) if inherited else None,
                          'heldout_policy': 'integrity/structural metadata only; no image decode/features/fitting'}
            write_once(task_out / 'split_manifest.json', split)
            write_once(task_out / 'SOURCE.json', provenance)
            write_once(task_out / 'native_contract.json', native)
            write_once(task_out / 'episodes.json', {'task': task, 'episodes': rows, 'frames_by_split': frames})
            entry = {'task_id': task_id, 'task': task, 'task_name': task,
                     'task_config': 'demo_clean', 'embodiment': 'aloha-agilex',
                     'split_manifest': str(task_out / 'split_manifest.json'),
                     'data_root': str(data_root), 'source_receipt': str(task_out / 'SOURCE.json'),
                     'original_source_receipt': source['receipt']['path'],
                     'native_contract': str(task_out / 'native_contract.json'),
                     'episode_inventory': str(task_out / 'episodes.json'),
                     'counts': split['counts'], 'frames_by_split': frames,
                     'native_eval_step_limit': native['native_eval_step_limit'],
                     'uncached_train_val_feature_bytes_estimate':
                         (frames['train'] + frames['val']) * BYTES_PER_CACHED_FRAME,
                     'cache_identity_preserved_by_original_data_root': True,
                     'source_bytes': sum(x['bytes'] for x in rows)}
            entries.append(entry)
            print(json.dumps({'task_prepared': task, 'frames': frames,
                              'split_manifest': entry['split_manifest']}, ensure_ascii=False), flush=True)
        all_hashes = [row['sha256'] for task in TASKS
                      for row in json.loads((out / task / 'episodes.json').read_text())['episodes']]
        require(len(all_hashes) == len(set(all_hashes)), 'Cross-task duplicate episode bytes')
        manifest = {'schema_version': 1, 'kind': 'robotwin_multitask_pilot',
                    'experiment_id': out.name, 'selection': str(out / 'selection.json'),
                    'task_order': list(TASKS), 'task_config': 'demo_clean',
                    'tasks': entries, 'split_manifests': {x['task']: x['split_manifest'] for x in entries},
                    'counts_total': {g: sum(x['counts'][g] for x in entries) for g in SPLITS},
                    'frames_total': {g: sum(x['frames_by_split'][g] for x in entries) for g in SPLITS},
                    'feature_cache_bytes_per_frame': BYTES_PER_CACHED_FRAME,
                    'uncached_train_val_feature_bytes_estimate': sum(
                        x['uncached_train_val_feature_bytes_estimate'] for x in entries),
                    'source_hdf5_bytes_total': sum(x['source_bytes'] for x in entries),
                    'training_requirements': {'normalization_fit_split': 'train', 'vtt_source_split': 'train',
                        'heldout_allowed_for_training_or_features': False,
                        'task_sampling': 'task_uniform_recommended', 'episode_history_crossing_allowed': False,
                        'action_repr': 'absolute', 'action_dim': 14, 'state_dim': 14,
                        'dino_frozen': True, 'vtt_projection_trainable': True,
                        'shared_action_network_trainable': True},
                    'evaluation_requirements': {'episodes_per_task': 10, 'task_config': 'demo_clean',
                        'use_native_task_success': True, 'record_video': False, 'third_view': False,
                        'onboard_observations_and_actions_retained': True,
                        'no_selection_by_policy_results': True,
                        'runtime_preflight_required': True},
                    'validation_scope': 'source integrity + stable splits + CPU native contract/asset audit',
                    'runtime_simulation_validated': False, 'gpu_used': False}
        write_once(out / 'manifest.json', manifest)
        status(out, {'status': 'complete', 'manifest': str(out / 'manifest.json'),
                     'completed_tasks': list(TASKS), 'free_bytes': shutil.disk_usage(base).free,
                     'downloaded_this_invocation_bytes_including_transport': additional,
                     'gpu_used': False, 'simulation_run': False})
        return manifest
    except Exception as exc:
        status(out, {'status': 'failed', 'completed_tasks': [x['task'] for x in entries],
                     'error': f'{type(exc).__name__}: {exc}', 'gpu_used': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-base', type=Path, default=ROOT / 'data/benchmarks/robotwin2')
    parser.add_argument('--robotwin-root', type=Path, default=ROOT / 'third_party/RoboTwin')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--download-missing', action='store_true')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
