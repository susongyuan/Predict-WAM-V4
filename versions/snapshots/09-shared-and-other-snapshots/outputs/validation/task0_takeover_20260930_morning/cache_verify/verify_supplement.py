#!/usr/bin/env python3
"""Independent bounded source/migration verification; CPU only, no production model/data loading."""
import ast
from contextlib import ExitStack
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path('/home/user/Robotic/project2')
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()

reads = []
def digest(path):
    path = Path(path).resolve()
    before = path.stat()
    assert before.st_size <= 16 * 1024 * 1024, f'bounded verification refuses {path}'
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_dev) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev)
    reads.append({'path': str(path), 'bytes': before.st_size, 'sha256': value})
    return value

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

plan_path = ROOT / 'outputs/validation/task0_continuation_20k_20260929/continuation_plan.json'
backup_root = ROOT / 'outputs/validation/cache_ledger_fix_20260930/backups'
plan = json.loads(plan_path.read_text())
old_plan = json.loads((backup_root / 'continuation_plan.json').read_text())
tracked = [ROOT / 'scripts/train_wam_v3.py', ROOT / 'scripts/test_train_wam_v3.py', plan_path,
           ROOT / 'data/dino_cache_task000/sha256_ledger_micro16_30k_v1.json']
source_before = {str(p): digest(p) for p in tracked}
old_ast = ast.parse((backup_root / 'train_wam_v3.py').read_text())
new_ast = ast.parse((ROOT / 'scripts/train_wam_v3.py').read_text())
assert len(old_ast.body) == len(new_ast.body)
changed = [getattr(b, 'name', 'module_docstring') for a,b in zip(old_ast.body,new_ast.body) if ast.dump(a) != ast.dump(b)]
assert changed == ['module_docstring', 'CacheLedger'], changed
unchanged_fields = [k for k in old_plan if k != 'source_migrations']
assert {k:old_plan[k] for k in unchanged_fields} == {k:plan[k] for k in unchanged_fields}
assert set(old_plan) == set(plan)
old_migrations = {r['path']:r for r in old_plan['source_migrations']}
migrations = []
for row in plan['source_migrations']:
    assert digest(ROOT / row['path']) == row['to_sha256']
    assert digest(ROOT / row['backup']) == row['from_sha256']
    migrations.append({'path': row['path'], 'backup_and_live_sha_match': True, 'from': row['from_sha256'], 'to': row['to_sha256']})
assert next(r for r in plan['source_migrations'] if r['path']=='harness/orchestrator.py') == old_migrations['harness/orchestrator.py']
assert digest(ROOT / plan['parent']['manifest']) == plan['parent']['manifest_sha256']
assert digest(ROOT / plan['parent']['config']) == plan['parent']['config_sha256']

continuation = load('independent_continuation', plan_path.parent / 'test_continuation.py')
integration = load('independent_integration', plan_path.parent / 'completion_20260930/test_integration.py')
T = sys.modules['train_wam_v3']
method_names = (
    'test_real_plan_reconstructs_parent_contract_and_new_contract',
    'test_no_other_source_inventory_data_model_stats_workers_or_budget_changes',
    'test_plan_rejects_bad_hash_parent_source_schedule_and_config',
    'test_new_approval_required_pending_refused_no_auto_gpu',
    'test_lr_history_unchanged_and_new_stage_separate_from_original_50k',
    'test_cli_legacy_defaults_and_continuation_require_explicit_resume_prefix_target',
)
integration_names = (
    'test_runtime_training_dependency_and_live_inventory',
    'test_cpu_four_stage_train_eval_select_resume',
    'test_cpu_restart_after_eval_before_receipt',
    'test_cpu_restart_after_selector',
)
suite = unittest.TestSuite([continuation.MigrationTests(n) for n in method_names] + [integration.Integration(n) for n in integration_names])
start = time.monotonic()
with ExitStack() as stack:
    stack.enter_context(patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')))
    stack.enter_context(patch.object(torch, 'load', side_effect=AssertionError('checkpoint loading excluded')))
    with (OUT / 'supplementary_tests.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
elapsed = time.monotonic() - start

receipt = json.loads(tracked[-1].read_text())
bindings = receipt['bindings']
cfg = json.loads((ROOT / plan['parent']['checkpoint']).parent.joinpath('config.json').read_text())
assert T.digest_json(cfg) == bindings['resolved_training_config']
binding_rows = []
for filename, expected in bindings.items():
    if filename == 'resolved_training_config':
        continue
    path = Path(filename)
    assert path.is_relative_to(ROOT) and path.suffix == '.json'
    assert path.stat().st_size <= 1024 * 1024
    value = digest(path)
    assert value == expected, filename
    binding_rows.append({'path': filename, 'bytes': path.stat().st_size, 'sha256': value})
cache = ROOT / cfg['dataset']['feature_cache']
feature_meta = [r for r in binding_rows if Path(r['path']).parent.parent == cache]
views = cfg['model']['views']
meta_episode_views = []
for record in feature_meta:
    path = Path(record['path'])
    meta = json.loads(path.read_text())
    source_path = ROOT / 'data/frame_cache' / meta['view'] / path.name
    assert meta['complete'] is True
    assert meta['view'] in views
    assert meta['source_frame_meta_sha256'] == bindings[str(source_path)]
    assert meta['feat_layers'] == cfg['model']['vision_encoder']['feat_layers']
    assert meta['stride'] == 2 and meta['feat_shape'] == [meta['rows'],3,201,1024]
    assert path.name == f"episode_{meta['episode']}.meta.json"
    meta_episode_views.append((meta['episode'],meta['view']))
assert len(feature_meta) == 600
assert len(set(meta_episode_views)) == 600
assert {ep for ep,_ in meta_episode_views} == set(range(200))
assert set(meta_episode_views) == {(ep, view) for ep in range(200) for view in views}

source_after = {str(p): digest(p) for p in tracked}
assert source_before == source_after
assert not torch.cuda.is_initialized()
assert not any(k.startswith(('isaacsim','omnigibson')) for k in sys.modules)
output = {
    'verified_at': datetime.datetime.now().astimezone().isoformat(),
    'passed': result.wasSuccessful(),
    'source_audit': {'changed_trainer_ast_nodes': changed, 'unchanged_plan_fields': unchanged_fields,
                     'migrations': migrations, 'files_before': source_before, 'files_after': source_after,
                     'parent_manifest_and_config_sha_match': True,
                     'real_parent_checkpoint_rehashed_or_loaded': False},
    'supplementary_tests': {'run': result.testsRun, 'passed': result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                            'failures': [(str(t),e) for t,e in result.failures], 'errors': [(str(t),e) for t,e in result.errors],
                            'skipped': [(str(t),r) for t,r in result.skipped], 'elapsed_seconds': elapsed,
                            'migration_methods': method_names, 'integration_methods': integration_names,
                            'log': str(OUT / 'supplementary_tests.log')},
    'fresh_bounded_binding_check': {'passed': True, 'files_hashed': len(binding_rows),
        'bytes_hashed': sum(r['bytes'] for r in binding_rows),
        'resolved_parent_training_config_digest_matches': True,
        'feature_metadata_rows': len(feature_meta), 'episode_ids': '0..199, three configured views each',
        'feature_meta_links_and_shape_fields_checked': True,
        'production_payload_contents_read_in_this_supplement': False,
        'boundary': 'Freshly hashes historical receipt-bound JSON config/metadata and checks feature->RGB metadata links. Does NOT verify underlying RGB pixels, full feature/pool bytes, regenerate manipulation_frames, audit lowdim/parquet/statistics contents or prove model inference/physical success.',
        'files': binding_rows},
    'resources': {'cuda_initialized': False, 'sim_imported': False, 'checkpoint_loaded': False,
                  'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'cpu_affinity': sorted(os.sched_getaffinity(0))},
    'prohibited_actions': {'production_files_modified': False, 'progress_modified': False, 'training_started': False,
                           'gpu_work_started': False, 'artifact_published': False},
    'command': "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 /home/user/Robotic/project2/.conda-envs/lilawam/bin/python -B " + str(Path(__file__).resolve()),
}
with (OUT / 'supplementary_verification.json').open('x') as file:
    json.dump(output,file,indent=2,ensure_ascii=False); file.write('\n')
print(json.dumps({k:v for k,v in output.items() if k not in ('fresh_bounded_binding_check',)},ensure_ascii=False,indent=2))
print('BINDING_CHECK',len(binding_rows),sum(r['bytes'] for r in binding_rows))
raise SystemExit(0 if result.wasSuccessful() else 1)
