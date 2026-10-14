#!/usr/bin/env python3
"""25k development-repeat controller. Default is CPU dry-run; --execute is never implicit.
No polling of training/checkpoints. New seeds are serial and a valid physical failure is a terminal result.
An infrastructure error keeps pending/evidence/GPU lock and stops the entire chain for human inspection.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
FROZEN = BASE / 'frozen_runtime'
RUNNER = BASE / 'runner'
PROTOCOL = BASE / 'protocol.json'
INVENTORY = RUNNER / 'freeze_inventory.json'
PROJECT = BASE.parents[2]
STATE = RUNNER / 'state'
GPU_LOCK = Path('/tmp/project2_gpu.lock')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def emit(event, **kwargs):
    print(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **kwargs}), flush=True)


def publish_once(path, payload):
    """Complete JSON or no final name; no existing receipt can ever be replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.writing-{os.getpid()}')
    with temporary.open('x') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)  # atomic and refuses an existing final path
    finally:
        temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_bundle():
    rows = read(RUNNER / 'runner_integrity.json')
    for path, digest in rows.items():
        require(sha(path) == digest, f'Controller/protocol changed: {path}')
    return sha(RUNNER / 'runner_integrity.json')


def verify_image():
    inventory = read(INVENTORY)
    require(sha(inventory['baseline_manifest']) == inventory['baseline_manifest_sha256'], 'Original manifest changed')
    baseline = read(inventory['baseline_manifest'])
    verified = 0
    for row in inventory['files']:
        path = Path(row['runtime'])
        if row['copied']:
            require(path.resolve() == path and path.is_relative_to(FROZEN), f'Frozen source symlink/escape: {path}')
        require(sha(path) == row['sha256'], f'Frozen/shared SHA mismatch: {path}')
        if row['baseline_bound'] is True:
            require(baseline['artifact_sha256'].get(row['original']) == row['sha256'], 'Unbound baseline source')
            verified += 1
    require(verified == len(baseline['artifact_sha256']) == 52, 'Incomplete original source inventory')
    for relative, expected in inventory['shared_links'].items():
        require((FROZEN / relative).is_symlink() and (FROZEN / relative).resolve() == Path(expected),
                f'Shared mount path changed: {relative}')
    derived = read(FROZEN / 'data/skill_vtt_task000/contract_targets.json')['source_sha256']
    for relative, digest in derived.items():
        require(sha(FROZEN / relative) == digest, f'Baseline target-source dependency changed: {relative}')
    # The active trainer and mutable root policy are deliberately neither loaded nor consulted.
    return {'baseline_files_verified': verified, 'supplemental_files': len(inventory['files']) - verified,
            'target_source_dependencies_verified': len(derived),
            'checkpoint_sha256': baseline['checkpoint_sha256'], 'all_match': True}


def validate_seeds(seeds):
    require(seeds and len(set(seeds)) == len(seeds), 'Duplicate/empty seed selection refused')
    require(all(type(s) is int and 1 <= s <= 9 for s in seeds), 'Only new seeds 1..9 are allowed; seed0 is reused')
    require(seeds == sorted(seeds), 'Seeds must be ascending')


def run_name(seed):
    validate_seeds([seed])
    return f'task0_v3_25krepeat_v1_20260929_s025000_dev10_seed{seed}_k1'


def run_path(seed):
    name = run_name(seed)
    require(re.fullmatch(r'[A-Za-z0-9_-]+', name) is not None, 'Unsafe run name')
    path = FROZEN / 'outputs/behavior/closed_loop' / name
    require(path.resolve() == path, 'Run output path contains symlink/escape')
    return path


def command(seed):
    protocol = read(PROTOCOL)
    return [str(FROZEN / '.conda-envs/lilawam/bin/python'), '-B', str(FROZEN / 'scripts/behavior_closed_loop.py'),
            '--profile', 'development', '--run-name', run_name(seed), '--checkpoint', protocol['checkpoint']['path'],
            '--orchestrate', protocol['orchestrate'], '--instances', '10', '--exec-horizon', '8',
            '--samples', '1', '--num-steps', '10', '--seed', str(seed), '--timeout', '3600',
            '--policy-device', 'cuda', '--policy-cuda-fraction', '0.3', '--port', '8000',
            '--write-video', '--video-view', 'viewer', '--video-width', '1280', '--video-height', '720',
            '--video-framing', 'auto', '--log-object-poses', '--env-wrapper', 'harness.head_rgbd_wrapper.HeadRGBDWrapper']


def cpu_env():
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'P2_TASK_ID': '0', 'P2_HEAD_RES': '720', 'P2_WRIST_RES': '224',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1'}


def cpu_probe():
    """Actual policy imports and checkpoint metadata; CPU only, no model construction and no nvidia-smi."""
    code = r'''
import hashlib, inspect, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
# Fail before any attempted CUDA initialization, not merely after a successful allocation.
torch.cuda._lazy_init = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('CPU probe forbids CUDA initialization'))
import harness.policy_server as policy
import harness.task0_predicates as predicates
import harness.wam_v3 as wam
import harness.radio_perception as perception
from harness.wam_behavior import load_upstream
upstream = load_upstream()  # source import only; never construct encoder/action models
import behavior_closed_loop as closed
root = closed.ROOT
checkpoint = torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)
assert checkpoint['step'] == 25000
cfg = checkpoint['config']
assert len(cfg['model']['views']) == 3
assert checkpoint['v3_stats'] == json.loads((root / cfg['model']['v3']['stats_path']).read_text())
assert cfg['dataset']['contract_targets_sha256'] == hashlib.sha256((root / 'data/skill_vtt_task000/contract_targets.json').read_bytes()).hexdigest()
assert 'z_band' in inspect.signature(perception.obstacle_points).parameters
args = SimpleNamespace(checkpoint=sys.argv[1], orchestrate=sys.argv[2], program=None, memory_vtt=None,
                       env_wrapper='harness.head_rgbd_wrapper.HeadRGBDWrapper')
paths = closed.runtime_artifact_paths(args, closed.eval_command('development', '127.0.0.1', 8000), cfg)
assert all(p.is_file() for p in paths)
assert not torch.cuda.is_initialized()
assert not any(k == 'omnigibson' or k.startswith('isaacsim') for k in sys.modules)
assert not any(k.startswith('scripts.train') or k.startswith('train_wam') for k in sys.modules)
print(json.dumps({'checkpoint_step':checkpoint['step'], 'cuda_initialized':False, 'simulator_imported':False,
 'root':str(root), 'modules':{m.__name__:m.__file__ for m in (policy,predicates,wam,perception,closed)},
 'runtime_artifacts':[str(p) for p in paths]}))
'''
    protocol = read(PROTOCOL)
    result = subprocess.run([str(FROZEN / '.conda-envs/lilawam/bin/python'), '-B', '-c', code,
                             protocol['checkpoint']['path'], protocol['orchestrate']], cwd=FROZEN, env=cpu_env(),
                            capture_output=True, text=True, timeout=120, check=True)
    payload = json.loads(result.stdout)
    expected = {str(Path(r['runtime']).resolve()) for r in read(INVENTORY)['files'] if r['baseline_bound'] is True}
    require(set(payload['runtime_artifacts']) == expected, 'Actual launcher runtime inventory differs from original baseline')
    behavior_code = "import importlib.util,json,harness;print(json.dumps({'harness':harness.__file__, 'omnigibson_origin':importlib.util.find_spec('omnigibson').origin, 'simulator_imported':'omnigibson' in __import__('sys').modules}))"
    other = subprocess.run([str(FROZEN / '.conda-envs/behavior6/bin/python'), '-I', '-c', behavior_code],
                           cwd=PROJECT, env=cpu_env(), capture_output=True, text=True, timeout=30, check=True)
    payload['evaluator_import_probe'] = json.loads(other.stdout)
    require(not payload['evaluator_import_probe']['simulator_imported'], 'CPU probe imported simulator')
    return payload


def expected_artifacts(seed):
    rows = read(INVENTORY)['files']
    return {r['original'] if seed == 0 else str(Path(r['runtime']).resolve()): r['sha256']
            for r in rows if r['baseline_bound'] is True}


def validate_result(path, seed, *, decode=True):
    """Task success comes only from the evaluator; no proxy or program completion counts."""
    path = Path(path)
    protocol = read(PROTOCOL)
    if seed == 0:
        require(path == Path(protocol['reused_seed0']['run_dir']), 'Wrong seed0 path')
    else:
        require(path == run_path(seed), 'Wrong new-run path')
    manifest = read(path / 'manifest.json')
    required = {'task_id': 0, 'task_name': 'turning_on_radio', 'profile': 'development', 'seed': seed,
                'effective_policy_seed': seed, 'effective_evaluator_seed': seed, 'samples_averaged': 1,
                'exec_horizon': 8, 'num_inference_steps': 10, 'max_steps_override': None,
                'instances_override': [10], 'evaluation_complete': True, 'evaluator_returncode': 0,
                'launcher_returncode': 0, 'expected_episodes': 1, 'artifact_changes': [],
                'policy_device': 'cuda', 'policy_vram_reserve_mib': 2100, 'official_score_valid': False,
                'checkpoint': protocol['checkpoint']['path'], 'checkpoint_sha256': protocol['checkpoint']['sha256'],
                'orchestrate': protocol['orchestrate']}
    for key, value in required.items():
        require(manifest.get(key) == value and (value is not True or manifest.get(key) is True), f'Manifest mismatch: {key}')
    require(manifest.get('artifact_sha256') == expected_artifacts(seed), 'Runtime artifact manifest differs from frozen original')
    plan = read(path / 'eval/evaluation_plan.json')
    require(plan == {'task': 'turning_on_radio', 'mode': 'public_test', 'instance_indices': [10],
                     'instance_ids': [311], 'num_rollouts': 1, 'seed': seed}, 'Plan/instance/seed mismatch')
    identity = {'task': 'turning_on_radio', 'instance_id': 311, 'rollout_id': 0}
    require(manifest.get('planned_episodes') == [identity], 'Wrong planned episode identity')
    episodes = sorted((path / 'eval/json').glob('*.json'))
    require(len(episodes) == 1 and episodes[0].name == 'turning_on_radio_311_0.json', 'Missing/extra/wrong evaluator episode')
    row = read(episodes[0])
    require(all(row.get(k) == v for k, v in identity.items()), 'Wrong evaluator identity')
    require(type(row.get('success')) is bool and type(row.get('steps')) is int and 0 < row['steps'] <= 3225,
            'Invalid physical outcome/budget')
    require(row.get('policy_step_alignment') is True, 'Missing policy-step alignment')
    require(row.get('terminated') is True or row.get('truncated') is True, 'Episode has no terminal state')
    require(row.get('effective_policy_seed') == seed and row.get('effective_evaluator_seed') == seed, 'Wrong effective RNG seeds')
    q = row.get('q_score', {}).get('final')
    require(type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1, 'Invalid evaluator Q')
    require(not row['success'] or q == 1, 'Success/Q contradiction')
    completed = manifest['completed_episodes']
    require(len(completed) == 1 and completed[0]['steps'] == row['steps'] and completed[0]['success'] is row['success']
            and completed[0]['q_score'] == q and all(completed[0].get(k) == v for k, v in identity.items()),
            'Manifest/result terminal mismatch')
    metadata = read(path / 'eval/policy_metadata.json')
    require(metadata['effective_policy_seed'] == seed and metadata['policy_rng_reset_behavior'] == row['policy_rng_reset_behavior'],
            'Policy RNG handshake mismatch')
    resources = read(path / 'resource_summary.json')
    require(resources['returncode'] == 0 and resources['abort_reason'] is None, 'Infrastructure/resource abort')
    require(resources['monitor_policy'] == protocol['resource_policy'], 'Resource policy changed')
    require(read(path / 'policy_startup.json')['ready'] is True, 'Policy startup failed')
    traces = sorted(path.glob('policy_trace_*.jsonl'))
    require(len(traces) == 1 and traces[0].stat().st_size > 0, 'Missing/duplicate policy trace')
    trace_rows = [json.loads(line) for line in traces[0].read_text().splitlines()]
    require(sum(r.get('event') == 'v3_context' for r in trace_rows) == row['steps'], 'V3 trace/step coverage mismatch')
    log = (path / 'policy_server.log').read_text(errors='replace')
    require('policy.act failed' not in log and 'Traceback (most recent call last)' not in log, 'Policy exception masked by hold-on-error')
    poses = path / 'eval/object_poses/turning_on_radio_311_0.jsonl'
    require(poses.is_file() and poses.stat().st_size > 0, 'Missing privileged evidence log')
    if seed:
        imports = sorted((path / 'runtime_imports').glob('*.json'))
        require(len(imports) >= 3, 'Missing launcher/policy/evaluator import receipts')
        receipts = [read(p) for p in imports]
        require({'lilawam', 'behavior6'} <= {r['environment'] for r in receipts}, 'Missing interpreter receipt')
        require(any('harness.policy_server' not in r['modules'] and str(FROZEN / 'harness/policy_server.py') in r['argv'] for r in receipts),
                'Missing actual policy file entry receipt')
        for receipt in receipts:
            require(receipt['cwd'] == str(FROZEN) and receipt['task_id'] == '0', 'Runtime cwd/task drift')
            require(receipt['head_resolution'] == 720 and receipt['wrist_resolution'] == 224, 'Camera override drift')
            require(receipt['bootstrap_sha256'] == sha(RUNNER / 'bootstrap.py'), 'Interpreter transport changed')
            for module, source in receipt['modules'].items():
                require(Path(source).is_relative_to(FROZEN) and not Path(source).is_symlink(), f'Mutable project import: {module}')
    video = None
    if decode:
        spec = importlib.util.spec_from_file_location('repeat_video_checker', RUNNER / 'check_closed_loop_video.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        video = checker.check_run(path, pyav_python=PROJECT / '.conda-envs/behavior6/bin/python', timeout=180)
        require(video['all_verified'], 'Video incomplete/unreadable: ' + json.dumps(video))
    evidence = {}
    for candidate in path.rglob('*'):
        if candidate.is_file() and ('observations' not in candidate.parts):
            require(candidate.resolve().is_relative_to(path), f'Evidence symlink escapes run: {candidate}')
            evidence[str(candidate.relative_to(path))] = sha(candidate)
    return {'seed': seed, 'run_dir': str(path), 'status': 'complete', 'success': row['success'], 'q_score': q,
            'steps': row['steps'], 'physical_outcome': 'success' if row['success'] else 'failure',
            'official_score_valid': False, 'video': video, 'evidence_sha256': evidence,
            'protocol_sha256': sha(PROTOCOL), 'inventory_sha256': sha(INVENTORY)}


def receipt_state(seed, state_root=STATE):
    directory = Path(state_root) / run_name(seed)
    complete = directory / 'complete.json'
    if complete.exists():
        payload = read(complete)
        require(payload.get('seed') == seed and payload.get('status') == 'complete', 'Invalid terminal receipt')
        require(payload['protocol_sha256'] == sha(PROTOCOL) and payload['inventory_sha256'] == sha(INVENTORY), 'Receipt protocol drift')
        require(payload['run_dir'] == str(run_path(seed)), 'Receipt points to wrong run')
        for relative, digest in payload['evidence_sha256'].items():
            target = run_path(seed) / relative
            require(target.resolve().is_relative_to(run_path(seed)), 'Receipt evidence escape')
            require(sha(target) == digest, f'Completed evidence changed: {target}')
        return 'complete', payload
    require(not directory.exists(), f'Pending/error/incomplete receipt exists; no relaunch: {directory}')
    require(not run_path(seed).exists(), f'Existing output without complete receipt; refusing overwrite: {run_path(seed)}')
    return 'new', None


def nonempty_training_processes():
    """/proc scan, anchored to a Python process argv; never pgrep's own wrapper command text."""
    found = []
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = [x.decode(errors='replace') for x in file.read_bytes().split(b'\0') if x]
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        if args and re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(args[0]).name):
            if any(Path(arg).name in ('train_wam_v3.py', 'train_lilawam_behavior.py', 'cache_dino_features.py',
                                      'behavior_closed_loop.py', 'replay_demo_actions_feasibility.py') for arg in args[1:]):
                found.append({'pid': int(file.parent.name), 'argv': args})
    return found


@contextmanager
def chain_lock():
    RUNNER.mkdir(exist_ok=True)
    with (RUNNER / 'controller.flock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def acquire_gpu(owner):
    require(not nonempty_training_processes(), 'Training/simulation process already running')
    require(not GPU_LOCK.exists(), f'GPU lock exists; refusing to wait or steal: {GPU_LOCK}')
    # Only used after --execute, never during CPU dry-run. No lock release on infrastructure failure.
    spec = importlib.util.spec_from_file_location('frozen_gpu_lock', FROZEN / 'scripts/gpu_lock.py')
    lock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lock)
    lock.acquire(owner, max_wait=0, poll=0.25, max_used_mib=4000)
    require(not nonempty_training_processes(), 'Training/simulation appeared during admission; retaining GPU lock')
    return lock


def assert_units_stopped(pid):
    for unit in (f'wam-policy-{pid}', f'behavior-run-{pid}'):
        result = subprocess.run(['systemctl', '--user', 'show', unit, '--property=ActiveState', '--value'],
                                capture_output=True, text=True, timeout=20)
        require(result.returncode == 0 and result.stdout.strip() in ('inactive', 'failed'), f'Child unit not stopped: {unit}')


def run_one(seed, *, launch=subprocess.Popen, validator=validate_result):
    status, previous = receipt_state(seed)
    if status == 'complete':
        emit('seed_terminal_reused', seed=seed, success=previous['success'], q_score=previous['q_score'])
        return previous
    verify_bundle()
    verify_image()
    directory = STATE / run_name(seed)
    directory.mkdir(parents=True, exist_ok=False)
    pending = {'seed': seed, 'status': 'pending', 'run_dir': str(run_path(seed)), 'argv': command(seed),
               'protocol_sha256': sha(PROTOCOL), 'inventory_sha256': sha(INVENTORY), 'controller_pid': os.getpid()}
    publish_once(directory / 'pending.json', pending)
    child = None
    try:
        emit('seed_pending', seed=seed, run_dir=str(run_path(seed)))
        with (directory / 'launcher.log').open('x') as log:
            child = launch(command(seed), cwd=FROZEN, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                           stdout=log, stderr=subprocess.STDOUT)
            publish_once(directory / 'process.json', {'launcher_pid': child.pid,
                         'child_units': [f'wam-policy-{child.pid}', f'behavior-run-{child.pid}']})
            returncode = child.wait()  # wait for process termination; do not poll model state
        require(returncode == 0, f'Launcher infrastructure error: exit {returncode}')
        assert_units_stopped(child.pid)
        verify_bundle()
        verify_image()
        receipt = validator(run_path(seed), seed)
        publish_once(directory / 'complete.json', receipt)
        emit('seed_terminal', seed=seed, success=receipt['success'], q_score=receipt['q_score'], steps=receipt['steps'])
        return receipt
    except BaseException as error:
        if child is not None:
            # Stop only this launcher's exact units; never pkill and never free the GPU lock here.
            for unit in (f'wam-policy-{child.pid}', f'behavior-run-{child.pid}'):
                try:
                    subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=30)
                except (OSError, subprocess.SubprocessError):
                    pass
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        failure = {**pending, 'status': 'infrastructure_error', 'error': repr(error),
                   'gpu_lock_retained': True, 'action': 'Stop chain; preserve all evidence; manual inspection required, never automatic rerun.'}
        publish_once(directory / 'infrastructure_error.json', failure)
        emit('seed_terminal_infrastructure_error', seed=seed, error=repr(error), gpu_lock_retained=True)
        raise


def dry_run(seeds):
    validate_seeds(seeds)
    bundle = verify_bundle()
    provenance = verify_image()
    probe = cpu_probe()
    states = [{'seed': seed, 'state': receipt_state(seed)[0], 'argv': command(seed), 'cwd': str(FROZEN),
               'run_dir': str(run_path(seed))} for seed in seeds]
    return {'status': 'cpu_prepared_not_launched', 'gpu_or_simulator_started': False, 'nvidia_smi_called': False,
            'bundle_sha256': bundle, 'provenance': provenance, 'cpu_probe': probe, 'runs': states,
            'note': 'Execution still requires an explicit --execute invocation after review; same-instance stochastic repeats, not generalization.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--verify-seed0', action='store_true')
    mode.add_argument('--status', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 10)))
    args = parser.parse_args(argv)
    validate_seeds(args.seeds)
    if args.verify_seed0:
        verify_bundle()
        verify_image()
        payload = validate_result(Path(read(PROTOCOL)['reused_seed0']['run_dir']), 0)
        print(json.dumps(payload, indent=2))
        return
    if args.status:
        for seed in args.seeds:
            state, receipt = receipt_state(seed)
            emit('seed_status', seed=seed, state=state, success=receipt['success'] if receipt else None)
        return
    if not args.execute:
        print(json.dumps(dry_run(args.seeds), indent=2))
        return
    with chain_lock():
        dry_run(args.seeds)
        seed0 = validate_result(Path(read(PROTOCOL)['reused_seed0']['run_dir']), 0)
        emit('seed_terminal_reused', seed=0, success=seed0['success'], q_score=seed0['q_score'])
        outstanding = [seed for seed in args.seeds if receipt_state(seed)[0] != 'complete']
        if not outstanding:
            emit('chain_already_complete', requested_seeds=args.seeds, gpu_started=False)
            return
        owner = f'task0-25krepeat-v1-{os.getpid()}'
        lock = acquire_gpu(owner)
        def stopped(signum, frame):
            raise InterruptedError(f'Controller received signal {signum}')
        signal.signal(signal.SIGTERM, stopped)
        signal.signal(signal.SIGINT, stopped)
        try:
            receipts = [run_one(seed) for seed in args.seeds]
        except BaseException:
            emit('chain_halted', owner=owner, gpu_lock_retained=True)
            raise
        # Every launcher and both exact child units were stopped and fully verified before release.
        require(lock.release(owner), 'GPU lock owner changed; refusing success receipt')
        emit('chain_complete', reused_seed0=1, new_complete=len(receipts),
             successes=int(seed0['success']) + sum(int(r['success']) for r in receipts),
             denominator=1 + len(receipts), requested_seeds=args.seeds, official_score_valid=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit('controller_blocked', error=repr(error))
        raise SystemExit(2)
