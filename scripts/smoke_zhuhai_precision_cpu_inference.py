#!/usr/bin/env python3
"""One real TRAIN anchor, frozen Precision V2 loader and sampler, CPU only.

Never instantiate shared HybridFeatureCache/PreparationReuse, encode DINO,
read VAL/heldout arrays, fit statistics, run physics, or update shared logs.
Cache misses fail closed. The original loader's private /tmp normalization
JSON is temporary; the only persistent runtime output is the exclusive report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REVIEW_DIR = ROOT / 'outputs/zhuhai2026_tasks12/verification/precision_v2_overnight_20261009'
OUT = REVIEW_DIR / 'checkpoint_005000_cpu_inference.json'
TASK, STEP, ANCHOR = 'zhuhai_task2', 5000, 32
GIB = 1024**3


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def fingerprint(path):
    s = Path(path).lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, f'Regular nonlinked file required: {path}')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def readonly_bytes(path):
    """No atime updates on existing cache payloads/meta; no fallback generation."""
    path = Path(path)
    before = fingerprint(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NOATIME)
    with os.fdopen(fd, 'rb') as stream:
        s = os.fstat(stream.fileno())
        require((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) == before,
                f'Read publication race: {path}')
        data = stream.read()
    require(fingerprint(path) == before, f'File changed during read: {path}')
    return data


class ReadOnlyFrameBackend:
    """FrameFeatureStore backend: exact original compact identity + H-JEPA binding."""
    def __init__(self, root, binding, source_sha256, anchor):
        self.root = Path(root)
        require(self.root.is_dir() and self.root.resolve() == self.root, 'Existing nonsymlink cache root required')
        self.binding, self.source_sha256, self.anchor = binding, source_sha256, anchor
        self.entries = []
        self.bytes_read = 0

    def read_bytes(self, identity, generator):
        require(identity['source_sha256'] == self.source_sha256, 'Only the selected TRAIN source is allowed')
        frame = identity['frame_index']
        require(type(frame) is int and 0 <= frame <= self.anchor, 'Future/non-anchor cache input forbidden')
        bound = dict(identity, hjepa_binding=self.binding)
        encoded = json.dumps(bound, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = sha(encoded)
        meta_path, payload_path = self.root / (key + '.json'), self.root / (key + '.payload')
        try:
            meta_raw = readonly_bytes(meta_path)
            meta = json.loads(meta_raw)
            require(meta.get('key') == key and meta.get('identity') == json.loads(encoded), 'Feature identity mismatch')
            require(type(meta.get('size')) is int and 0 < meta['size'] <= 16 * 1024**2, 'Unbounded/invalid feature size')
            require(payload_path.stat().st_size == meta['size'], 'Feature payload size mismatch')
            data = readonly_bytes(payload_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Read-only cache miss for TRAIN frame {frame}; no generation/lock/index permitted') from exc
        require(len(data) == meta['size'] and sha(data) == meta.get('payload_sha256'), 'Feature payload SHA/size mismatch')
        self.bytes_read += len(data) + len(meta_raw)
        self.entries.append(dict(frame=frame, key=key, payload_sha256=sha(data), size=len(data), meta_sha256=sha(meta_raw)))
        return data  # generator deliberately never called, including on a miss.


def verify_sources(mapping):
    verified = {}
    for name, expected in mapping.items():
        path = Path(name)
        before = fingerprint(path)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                h.update(block)
        require(fingerprint(path) == before and h.hexdigest() == expected, f'Frozen source drift: {name}')
        verified[name] = expected
    return verified


class WorkspaceReadOnlyGuard:
    """Deny workspace writes except final report, plus ALL SQLite opens/locks/utime.

    Imports and the original loader may use process-private /tmp files. Python
    auditing does not cover every C extension call: our sole HDF5 handle is
    explicitly mode='r'. No online cache/physics/encoder object is constructed.
    """
    def __init__(self):
        self.rejected = []
        self.workspace_write_opens = []
        self.private_write_opens = []

    def deny(self, event, path):
        self.rejected.append(dict(event=event, path=str(path)))
        raise PermissionError(f'CPU smoke read-only guard: {event}: {path}')

    def __call__(self, event, args):
        if event in ('sqlite3.connect', 'fcntl.flock', 'fcntl.lockf', 'os.utime', 'socket.connect'):
            self.deny(event, args[0] if args else '')
        if event == 'open':
            path, mode, flags = args
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writing and isinstance(path, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(path)).absolute()
                resolved = path.resolve()
                if resolved.is_relative_to(ROOT):
                    if path != OUT or resolved != OUT:
                        self.deny(event, path)
                    self.workspace_write_opens.append(str(path))
                else:
                    self.private_write_opens.append(str(path))
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink'):
            for arg in args[:2]:
                if isinstance(arg, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(arg)).absolute()
                    if path.resolve().is_relative_to(ROOT):
                        self.deny(event, path)


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--repeat', action='store_true', help='Optional identical-seed second sample, same anchor')
    args = parser.parse_args(argv)
    if args.out.absolute() != OUT or args.out.resolve() != OUT or os.path.lexists(OUT):
        parser.error('Only a NEW checkpoint_005000_cpu_inference.json in the authorized directory may be written')
    guard = WorkspaceReadOnlyGuard()
    sys.addaudithook(guard)
    started = time.monotonic()
    report = dict(schema='precision-v2-real-5k-cpu-inference-smoke-v1', status='started',
                  command=' '.join(sys.argv), working_directory=str(Path.cwd()),
                  python=sys.executable, smoke_script_sha256=sha(Path(__file__).read_bytes()), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  task=TASK, checkpoint_step=STEP, anchor_index=ANCHOR,
                  stage='imports', shared_cache_or_preparation_instantiated=False,
                  training_or_physics_run=False, raw_rgb_read=False, heldout_or_val_arrays_read=False,
                  limits=['Single TRAIN anchor only; not physical success, validation accuracy, or CUDA validation.',
                          'Raw HDF whole-file SHA is inherited from the frozen manifest, not rescanned through image/future bytes.',
                          'No DINO encoding, action smoothing/rate limiting, IPC, robot, or simulator was executed.'])
    torch = None
    def mark(stage):
        report['stage'] = stage
        peak = rss_bytes()
        report.setdefault('stages', []).append(dict(stage=stage, elapsed_seconds=round(time.monotonic()-started, 3), peak_rss_bytes=peak))
        print(json.dumps(dict(stage=stage, elapsed_seconds=round(time.monotonic()-started, 3), peak_rss_gib=round(peak/GIB, 3))), flush=True)
        require(peak < 5*GIB, 'Peak RSS exceeded the 5GiB smoke target; stop CPU work')
        if torch is not None:
            assert torch.cuda.is_initialized() is False
    try:
        import numpy as np
        import h5py
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_initialized() is False
        from harness import zhuhai_precision_v2_training as precision
        from harness.zhuhai_dataset import FrameFeatureStore, PREPROCESS_SHA256
        from harness.zhuhai_demo_recording import JOINT_FIELDS
        from harness.wam_v3 import V3Context, history_frames
        from harness.wam_event_clock import EventClock
        mark('manifest_and_source_checks')
        directory = args.run.resolve() / 'train'
        manifest = json.loads(readonly_bytes(directory / 'manifest.json'))
        c = manifest['contract']
        require(manifest['contract_sha256'] == digest(c), 'Manifest digest mismatch')
        require(c['format'] == precision.FORMAT and c['mode'] == 'train', 'Wrong V2 contract')
        require(c['cache']['preprocess_sha256'] == PREPROCESS_SHA256, 'Preprocessing identity mismatch')
        audit = json.loads(readonly_bytes(REVIEW_DIR / 'checkpoint_005000_audit.json'))
        require(audit['status'] == 'verified' and audit['requested_step'] == STEP, 'Real 5k audit must already PASS')
        require(audit['manifest']['contract_sha256'] == manifest['contract_sha256'], 'Prior audit manifest differs')
        checkpoint = directory / 'checkpoint_step_005000.pt'
        checkpoint_stamp = fingerprint(checkpoint)
        receipt_path = checkpoint.with_name(checkpoint.name + '.sha256.json')
        receipt = json.loads(readonly_bytes(receipt_path))
        require(receipt['step'] == STEP and receipt['model_only'] is True
                and receipt['sha256'] == audit['model_only']['sha256'] and receipt['size'] == checkpoint_stamp[2], 'Milestone receipt/audit mismatch')
        train_hashes = verify_sources(c['source_files'])
        queue_plan = json.loads(readonly_bytes(REVIEW_DIR / 'frozen_queue_plan.json'))
        queue_hashes = verify_sources(queue_plan['source_hashes'])
        report['source_hashes'] = dict(training=train_hashes, queue=queue_hashes,
                                     training_count=len(train_hashes), queue_count=len(queue_hashes))
        report['checkpoint'] = dict(path=str(checkpoint), sha256=receipt['sha256'], size=receipt['size'],
                                    model_only=True, contract_sha256=manifest['contract_sha256'])
        row = c['splits'][TASK]['train'][0]  # Fixed before sampling; no outcome-based selection.
        require(row['split'] == 'train', 'Only TRAIN is permitted')
        source = Path(row['path']); source_stamp = fingerprint(source)
        require(all(row['path'] != r['path'] for split in ('val', 'heldout') for r in c['splits'][TASK][split]), 'Selected source overlaps non-TRAIN split')
        mark('read_past_lowdim_and_cached_features')
        with h5py.File(source, 'r') as handle:
            require(handle.attrs['state_dim'] == 16 and handle.attrs['action_dim'] == 16, 'HDF native16 schema mismatch')
            require(bool(handle.attrs['successful_expert']), 'TRAIN source is not admitted-successful')
            measured = handle['measured_state'][:ANCHOR+1].astype(np.float32)
            commanded = handle['target_command'][:ANCHOR+1].astype(np.float32)
            times = handle['physics_time'][:ANCHOR+1]
            q = np.concatenate([handle[f'state/{name}'][:ANCHOR+1].astype(np.float32) for name, width in JOINT_FIELDS], axis=1)
            previous_actions = np.concatenate([handle[f'action/{name}'][:ANCHOR].astype(np.float32) for name, width in JOINT_FIELDS], axis=1)
        require(measured.shape == commanded.shape == q.shape == (ANCHOR+1,16), 'Past native16 dimensions mismatch')
        require(np.array_equal(q, measured) and np.array_equal(previous_actions, commanded[1:]), 'Native16 state/previous-command alignment mismatch')
        require(np.isfinite(q).all() and np.isfinite(commanded).all() and np.allclose(np.diff(times), .1, rtol=0, atol=1e-7), 'Past stream nonfinite or not 10Hz')
        backend = ReadOnlyFrameBackend(c['cache']['root'], c['hjepa']['cache_binding'], row['sha256'], ANCHOR)
        def forbidden_producer(*unused):
            raise RuntimeError('DINO producer forbidden even on a cache miss')
        store = FrameFeatureStore(backend, forbidden_producer, encoder_sha256=c['cache']['encoder_sha256'])
        episode = SimpleNamespace(path=source, sha256=row['sha256'])
        cfg = c['config']; v3 = cfg['model']['v3']
        context = V3Context(v3['proprio_lags'], v3['memory_lags'], cfg['model']['views'], state_dim=16)
        clock = EventClock(**precision.CLOCK_CONFIG)
        pools = []
        for frame in range(ANCHOR+1):
            bits, pooled = store.get(episode, frame)
            context.push(q[frame], pooled)
            event = clock.update(q[frame], commanded[frame])
            pools.append(pooled)
        state, pool = context.inputs()
        state_frames = [ANCHOR] + history_frames(ANCHOR, 0, v3['proprio_lags'])
        memory_frames = [ANCHOR] + history_frames(ANCHOR, 0, v3['memory_lags'])
        require(np.array_equal(state, q[state_frames]), 'Original online state/history order mismatch')
        require(np.array_equal(pool, np.stack([pools[f] for f in memory_frames], axis=1)), 'Original memory history/view order mismatch')
        require(state.shape == (7,16) and pool.shape == (3,4,2048) and event.shape == (6,), 'Online input dimensions differ')
        report['anchor'] = dict(train_row=0, path=str(source), source_sha256_from_manifest=row['sha256'], seed=row['seed'],
                                family_id=row['family_id'], frame=ANCHOR, past_frame_range=[0,ANCHOR],
                                state_frames=state_frames, memory_frames=memory_frames,
                                source_whole_file_sha_rehashed=False, hdf_read_only=True,
                                datasets_read=['measured_state','target_command','physics_time'] +
                                    [f'{group}/{name}' for group in ('state','action') for name,width in JOINT_FIELDS],
                                action_read_range=[0,ANCHOR-1], future_input_frames=[])
        report['features'] = dict(backend='independent read-only FrameFeatureStore backend', entries=backend.entries,
                                  bytes_read=backend.bytes_read, feature_shape=list(bits.shape), feature_dtype=str(bits.dtype),
                                  binding_sha256=digest(c['hjepa']['cache_binding']), sqlite_opened=False,
                                  cache_lock_taken=False, cache_index_accessed=False, generator_called=False)
        mark('original_precision_load_policy_cpu')
        load_started = time.monotonic()
        wrapper, condition, actual = precision.load_policy(checkpoint, TASK, device='cpu')
        require(actual == c and np.array_equal(condition, np.asarray(c['conditions'][TASK], dtype=np.float32)), 'Loaded contract/VTT differs')
        require(condition.shape == (1024,) and np.isfinite(condition).all(), 'Invalid TRAIN VTT')
        model_state = wrapper.action_model.state_dict()
        expected = audit['model_only']['model_state_metadata']
        require(set(model_state) == set(expected), 'Strict-loaded model keys differ from prior audit')
        for key, value in model_state.items():
            require(value.device.type == 'cpu' and list(value.shape) == expected[key]['shape'] and str(value.dtype) == expected[key]['dtype'], 'Loaded tensor metadata differs: '+key)
        del model_state
        report['load'] = dict(strict_load=True, original_api='precision.load_policy(checkpoint, TASK, device=cpu)',
                             mechanism='Frozen build_wrapper calls action_model.load_state_dict(state_dict, strict=True)',
                             seconds=round(time.monotonic()-load_started,3), model_keys=len(expected),
                             parameters=sum(p.numel() for p in wrapper.action_model.parameters()),
                             all_model_tensors_cpu=True, projections_reset=False, torch=str(torch.__version__))
        feats = [torch.from_numpy(bits[:, layer].copy())[None] for layer in range(3)]
        tensor_state, tensor_pool = torch.from_numpy(state)[None], torch.from_numpy(pool)[None]
        tensor_condition, tensor_event = torch.from_numpy(condition)[None], torch.from_numpy(event.copy())[None]
        report['inputs'] = dict(feats=[list(f.shape) for f in feats], layers=[-12,-8,-4], views=cfg['model']['views'],
                                state=list(tensor_state.shape), pool=list(tensor_pool.shape), condition=list(tensor_condition.shape),
                                condition_sha256=sha(condition.tobytes()), state_sha256=sha(state.tobytes()), pool_sha256=sha(pool.tobytes()),
                                clock=list(tensor_event.shape), clock_values=event.tolist(), clock_layout=c['precision']['clock_layout'],
                                native16_order=c['data_contract']['native_order'], history_api='V3Context.push / inputs',
                                clock_api='EventClock.update(measured[t], target_command[t])', future_visual_inputs=False)
        observed_clocks = []
        hook = wrapper.action_model.precision.clock.register_forward_pre_hook(
            lambda module, values: observed_clocks.append(values[0].detach().cpu().clone()))
        seed_value = int(hashlib.sha256(f'{args.seed}:{ANCHOR}'.encode()).hexdigest()[:15],16)
        def sample():
            generator = torch.Generator(device='cpu').manual_seed(seed_value)
            with torch.inference_mode():
                return wrapper.sample_actions(feats, tensor_state, tensor_pool, tensor_condition,
                                              num_steps=10, generator=generator, event_state=tensor_event)[0]
        mark('original_10_step_euler_sampling')
        sampling_started = time.monotonic()
        actions = sample()
        require(tuple(actions.shape) == (1,32,16) and actions.device.type == 'cpu' and torch.isfinite(actions).all().item(), 'Invalid CPU action output')
        require(len(observed_clocks) == 10 and all(torch.equal(x, tensor_event) for x in observed_clocks), 'Clock did not reach all 10 original model forwards')
        report['output'] = dict(shape=list(actions[0].shape), dtype=str(actions.dtype), finite=True,
                                min=float(actions.min()), max=float(actions.max()), mean=float(actions.mean()), std=float(actions.std()),
                                first_action=actions[0,0].tolist(), sha256=sha(actions.contiguous().numpy().tobytes()),
                                sampler_seconds=round(time.monotonic()-sampling_started,3), num_euler_steps=10,
                                input_seed=args.seed, derived_generator_seed=seed_value, clock_forward_calls=10,
                                original_clock_values_seen=True, action_units='denormalized native16 absolute joint/gripper targets; no postprocessing')
        if args.repeat:
            repeat_started = time.monotonic()
            repeated = sample()
            require(torch.equal(actions, repeated), 'Same-seed CPU repeat is not bitwise reproducible')
            report['same_seed_repeat'] = dict(performed=True, bitwise_equal=True, seconds=round(time.monotonic()-repeat_started,3), same_anchor=True)
        else:
            report['same_seed_repeat'] = dict(performed=False)
        hook.remove()
        require(fingerprint(checkpoint) == checkpoint_stamp and fingerprint(source) == source_stamp, 'Source/checkpoint changed during smoke')
        require(json.loads(readonly_bytes(receipt_path)) == receipt, 'Milestone receipt changed')
        verify_sources(train_hashes); verify_sources(queue_hashes)
        mark('complete')
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if report.get('load', {}).get('strict_load') and report['stage'] == 'original_10_step_euler_sampling':
            report['interpretation'] = 'CPU sampling failed; this does not establish a CUDA/physical-evaluation failure.'
    finally:
        cuda = torch.cuda.is_initialized() if torch is not None else False
        assert cuda is False, 'CPU smoke initialized CUDA'
        report.update(cuda_initialized=False, total_seconds=round(time.monotonic()-started,3),
                      peak_rss_bytes=rss_bytes(), peak_rss_gib=round(rss_bytes()/GIB,3),
                      peak_rss_below_5gib=rss_bytes()<5*GIB,
                      thread_limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                      guard=dict(rejected=guard.rejected, private_write_opens=guard.private_write_opens,
                                 workspace_write_policy='Only exclusive final report; no cache/index/lock/source/MD writes'))
        payload = (json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd,'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=report['status'],out=str(OUT),peak_rss_gib=report['peak_rss_gib'],
                          seconds=report['total_seconds'],cuda_initialized=False,error=report.get('error')),ensure_ascii=False),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
