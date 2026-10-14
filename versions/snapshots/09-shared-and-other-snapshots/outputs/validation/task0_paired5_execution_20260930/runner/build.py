#!/usr/bin/env python3
"""One-time CPU-only assembly; no modification of historical files or large payload reads."""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path('/home/user/Robotic/project2')
BASE = PROJECT / 'outputs/validation/task0_paired5_execution_20260930'
OLD = PROJECT / 'outputs/validation/task0_v3_25k_repeat_20260929'
PLAN = PROJECT / 'outputs/validation/task0_paired5_protocol_20260930'
AUTH = PROJECT / 'outputs/validation/task0_gpu_peak_authorization_20260930/authorization.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False); f.write('\n')

def stat(p):
    s = p.stat()
    return {k: getattr(s, k) for k in ('st_size','st_mtime_ns','st_ctime_ns','st_ino','st_dev')}

def main():
    assert not (BASE / 'assembly.json').exists(), 'Already assembled; refusing overwrite'
    plan = json.loads((PLAN / 'protocol.json').read_text())
    inv = json.loads((OLD / 'runner/freeze_inventory.json').read_text())
    snapshot = {}
    for name in ('protocol.json', 'preflight.json'):
        src = PLAN / name
        write(BASE / 'plan_snapshot' / name, json.loads(src.read_text()))
        snapshot[str(src)] = sha(src)
    bound = {str(AUTH): sha(AUTH), **snapshot,
             str(OLD / 'runner/repeat.py'): sha(OLD / 'runner/repeat.py'),
             str(OLD / 'runner/bootstrap.py'): sha(OLD / 'runner/bootstrap.py'),
             str(OLD / 'runner/freeze_inventory.json'): sha(OLD / 'runner/freeze_inventory.json')}
    write(BASE / 'authorization_supplement.json', {
        'schema':'paired5_execution_supplement_v1', 'bound_files':bound,
        'actual_user_text':json.loads(AUTH.read_text())['user_text'],
        'old_blocked_history_modified':False, 'gpu_utilization_allowed_percent':100,
        'global_vram_abort_mib':15300, 'cpu_ram_abort_percent':85,
        'monitor_is_polling_not_hard_quota':True, 'gpu_concurrency':'exclusive_serial',
        'first_requested_run':'original_seed1', 'run_order_change':'single explicit dispatch; no auto chain',
        'original_seed0':'reuse only', 'improved_candidate':'HOLD; no verifier or safe release admitted',
        'full_task_success_inferred_from_benchmark':False})
    image_info = {}
    for variant in ('original','improved'):
        image = BASE / 'images' / variant
        root = image / 'frozen_runtime'
        rows = []
        deltas = {r['relative']:r for r in plan['groups']['improved']['planned_source_deltas']} if variant == 'improved' else {}
        for old in inv['files']:
            row = dict(old)
            relative = row['relative']
            source = Path(row['runtime'])
            if relative in deltas:
                source = Path(deltas[relative]['path'])
                assert sha(source) == deltas[relative]['sha256'], f'CPU accepted source drift: {source}'
                row['sha256'] = deltas[relative]['sha256']
                row['accepted_delta'] = True
            if row['copied']:
                assert sha(source) == row['sha256'], f'Frozen source drift: {source}'
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists()
                # Independent read-only byte-identical copy; never writable hardlinks to historical source.
                shutil.copyfile(source, target)
                target.chmod(0o444)
                row['runtime'] = str(target)
                row['source'] = str(source)
            else:
                if source.stat().st_size < 16 * 1024**2:
                    assert sha(source) == row['sha256'], f'Shared source drift: {source}'
                row['assembly_stat'] = stat(source)
            rows.append(row)
        for relative, delta in deltas.items():
            if any(r['relative'] == relative for r in rows): continue
            source, target = Path(delta['path']), root / relative
            assert sha(source) == delta['sha256']
            shutil.copyfile(source, target); target.chmod(0o444)
            rows.append({**delta, 'runtime':str(target), 'source':str(source), 'original':str(source),
                         'copied':True, 'baseline_bound':False, 'accepted_delta':True})
        for relative, target in inv['shared_links'].items():
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.symlink_to(target, target_is_directory=True)
        image_runner = image / 'runner'
        image_runner.mkdir(parents=True, exist_ok=True)
        for file in ('bootstrap.py',):
            shutil.copyfile(BASE / 'runner' / file, image_runner / file)
        shutil.copyfile(OLD / 'runner/check_closed_loop_video.py', image_runner / 'check_closed_loop_video.py')
        for name in ('lilawam','behavior6'):
            env = root / '.conda-envs' / name
            (env / 'bin').mkdir(parents=True)
            for child in (PROJECT / '.conda-envs' / name).iterdir():
                if child.name != 'bin': (env / child.name).symlink_to(child, target_is_directory=child.is_dir())
            wrapper = ('#!/bin/sh\nset -eu\n'
                'unset PYTHONPATH PYTHONHOME PYTHONSAFEPATH\n'
                'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1\n'
                f'exec {PROJECT}/.conda-envs/{name}/bin/python -B -E -s {image_runner}/bootstrap.py {name} "$@"\n')
            (env / 'bin/python').write_text(wrapper); (env / 'bin/python').chmod(0o555)
        new_inv = {**inv, 'frozen_root':str(root), 'files':rows, 'variant':variant,
                   'limitation':'Small baseline files copied read-only; shared data/models stat receipt at CPU assembly, large files content-verified once per executing launcher. No dataset/cache sweep.'}
        write(image_runner / 'freeze_inventory.json', new_inv)
        old_protocol = json.loads((OLD / 'protocol.json').read_text())
        orchestrate = old_protocol['orchestrate'] if variant == 'original' else json.dumps(plan['groups'][variant]['orchestrate'])
        from admission import resource_policy
        protocol = {**old_protocol, 'orchestrate':orchestrate,
                    'historical_resource_policy_not_authorization':resource_policy(json.loads((root / 'configs/behavior_environment.json').read_text())['resource_policy']),
                    'execution_supplement':str(BASE / 'authorization_supplement.json'),
                    'reused_seed0':{'run_dir':str(root / 'outputs/behavior/closed_loop' / f'task0_paired5_{variant}_20260930_s025000_dev10_seed0_k1')}}
        write(image_runner / 'runtime_protocol.json', protocol)
        image_info[variant] = {'root':str(root), 'inventory':str(image_runner / 'freeze_inventory.json'),
                               'runtime_protocol':str(image_runner / 'runtime_protocol.json'), 'files':len(rows)}
    write(BASE / 'assembly.json', {'schema':1, 'images':image_info, 'plan_snapshot_sha256':snapshot,
          'large_payloads_content_hashed':False, 'old_directories_modified':False})
    print(json.dumps({'assembled':list(image_info), 'gpu_started':False}))

if __name__ == '__main__': main()
