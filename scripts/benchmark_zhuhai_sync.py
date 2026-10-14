#!/usr/bin/env python3
"""Same-source target-action replay: reference vs render-only Fabric sync.

Without --execute this only validates CLI/HDF/provenance on CPU. Execution
requires the coordinator's existing GPU lock and an unused --out directory.
No expert/policy, smoothing, retargeting, pose restoration or score-threshold
changes are applied. PNGs are onboard observations, not videos. Both modes use
identical capture cadence; sparse capture is NOT a sync-only speedup vs the
original demonstration's every-control-frame RGB recording.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'outputs/zhuhai2026_tasks12/smoke/franka_task2_source1_20261009_v1'
FORMAT = 'zhuhai-native16-sync-replay-v1'
VIEWS = ('head_camera', 'left_camera', 'right_camera')
ARM = (*range(7), *range(8, 15))
GRIP = (7, 15)
PHYSICS_DT, CONTROL_DT = .004, .1
SETTING_KEYS = ('/physics/updateToUsd', '/physics/updateVelocitiesToUsd')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value, *, durable=False):
    """Atomic publication; final/error receipts additionally fsync file + directory.

    SimulationApp fast shutdown may call os._exit(), bypassing Python finally
    blocks and buffered writes. No required receipt may depend on close returning.
    """
    path = Path(path)
    content = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        handle.write(content)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    tmp.replace(path)
    if durable:
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def flush_existing(path):
    path = Path(path)
    if path.is_file():
        with path.open('rb') as handle:
            os.fsync(handle.fileno())


def log(message):
    from scripts.progress1_log import log as record
    record(message, actor='benchmark_zhuhai_sync')


def verify_owner(owner):
    from scripts.gpu_lock import owner as current
    if not owner or current() != owner:
        raise RuntimeError('benchmark requires the externally owned GPU lock; it never acquires/releases it')


def _matrix(group):
    import numpy as np
    if hasattr(group, 'shape'):
        return np.asarray(group[:])
    fields = ('left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states')
    return np.concatenate([group[k][:] for k in fields], axis=1)


@dataclass
class Episode:
    manifest_path: Path
    hdf_path: Path
    hashes: dict
    metadata: dict
    collection: dict
    source_row: dict
    state: object
    actions: object
    next_state: object
    times: object
    truth: list
    alignment_source: str

    def describe(self):
        return dict(manifest=str(self.manifest_path), hdf5=str(self.hdf_path), source_sha256=self.hashes,
            task=self.metadata['task'], seed=self.metadata['seed'], mode=self.metadata['mode'],
            family_id=self.metadata.get('family_id'), parent_source_seed=self.metadata.get('parent_source_seed'),
            action_count=len(self.actions), observation_count=len(self.times),
            action_dtype=str(self.actions.dtype), state_dtype=str(self.state.dtype),
            initial_recorded_state=self.state[0].tolist(), recorded_elapsed_seconds=float(self.times[-1]-self.times[0]),
            time_alignment='action[i] targets (t[i],t[i+1]]; compare AFTER execution to observation i+1, NOT state[i]',
            expected_next_state_source=self.alignment_source, recorded_final=self.truth[-1])


def read_episode(manifest_path, hdf_path):
    """Read only low-dimensional arrays/metadata/truth; never decode source RGB."""
    import h5py
    import numpy as np
    from harness import zhuhai_robot_spec as spec
    manifest_path, hdf_path = Path(manifest_path).resolve(), Path(hdf_path).resolve()
    hashes = dict(manifest=sha256(manifest_path), hdf5=sha256(hdf_path))
    manifest = json.loads(manifest_path.read_text())
    collection = manifest['contract']
    matches = [r for r in manifest['attempts'] if r.get('status') == 'success' and r.get('data_sha256') == hashes['hdf5']]
    if len(matches) != 1:
        raise ValueError('expected exactly one admitted successful manifest row matching the HDF content hash')
    row = matches[0]
    with h5py.File(hdf_path, 'r') as f:
        metadata = json.loads(f.attrs['metadata'])
        if (not bool(f.attrs.get('successful_expert', False)) or f.attrs.get('state_dim') != 16
                or f.attrs.get('action_dim') != 16 or f.attrs.get('state_semantics') != 'measured_joint_gripper'
                or f.attrs.get('action_semantics') != 'joint_position_target'):
            raise ValueError('successful measured-feedback native16 HDF required')
        if metadata.get('contract') != collection or metadata.get('seed') != row['seed']:
            raise ValueError('HDF seed/collection metadata does not match the admitted source')
        if (metadata.get('task') != collection['task'] or metadata.get('mode') != collection['mode']
                or type(metadata.get('seed')) is not int or metadata['seed'] < 0
                or collection.get('frequency_hz') != 10. or collection.get('physics_dt_seconds') != PHYSICS_DT):
            raise ValueError('task/seed/fixed 10Hz/.004s source protocol mismatch')
        if (collection.get('robot_contract') != spec.robot_contract()
                or collection.get('record_video') is not False or collection.get('third_view') is not False
                or collection.get('camera_count') != 3):
            raise ValueError('current official robot contract and three-onboard-camera/no-video protocol required')
        state, actions = _matrix(f['state']), _matrix(f['action'])
        if state.ndim != 2 or actions.shape != state.shape or state.shape[1] != 16 or not len(actions):
            raise ValueError('state/action must be nonempty aligned (N,16) arrays')
        n = len(actions)
        terminal = f['terminal_observation']
        terminal_state = np.asarray(terminal['measured_state'][:])
        if 'measured_state' in f:
            measured = np.asarray(f['measured_state'][:])
            if (measured.shape != (n+1,16) or not np.array_equal(measured[:-1],state)
                    or not np.array_equal(measured[-1],terminal_state)):
                raise ValueError('measured history / terminal alignment mismatch')
            next_state, alignment = measured[1:].copy(), 'measured_state[1:] (terminal independently checked)'
        else:
            next_state = np.concatenate([state[1:],terminal_state[None]],axis=0)
            alignment = 'state[1:] + terminal_observation/measured_state'
        if 'next_state' in f and not np.array_equal(_matrix(f['next_state']), next_state):
            raise ValueError('explicit next_state differs from observation i+1')
        if 'target_command' in f:
            commands = np.asarray(f['target_command'][:])
            if commands.shape != (n+1,16) or not np.array_equal(commands[1:],actions):
                raise ValueError('action must equal target_command[1:], never target_command[:-1]')
        times = np.asarray(f['physics_time'][:],np.float64)
        if (times.shape != (n+1,) or not np.isfinite(times).all() or times[0] < 0
                or not np.allclose(np.diff(times),CONTROL_DT,atol=1e-7,rtol=0)
                or not math.isclose(float(terminal['physics_time'][()]),times[-1],abs_tol=1e-7,rel_tol=0)):
            raise ValueError('source must retain N+1 correctly aligned 10Hz observations including terminal')
        if not all(np.isfinite(x).all() for x in (state,actions,next_state)):
            raise ValueError('nonfinite source state/action')
        bounds = np.asarray(spec.DUAL_JOINT_LIMITS)
        if (np.any(actions[:,ARM]<bounds[:,0]-1e-7) or np.any(actions[:,ARM]>bounds[:,1]+1e-7)
                or np.any(actions[:,GRIP]<0) or np.any(actions[:,GRIP]>1)):
            raise ValueError('archived target actions violate the unchanged official control contract')
        raw_truth = f['rule_evidence'].asstr()[:]
        if len(raw_truth) != n+1:
            raise ValueError('source truth must cover each observation, including terminal')
        truth = []
        for i, text in enumerate(raw_truth):
            evidence = json.loads(text)
            if not math.isclose(float(evidence['physics_time_seconds']),times[i],abs_tol=1e-7,rel_tol=0):
                raise ValueError('source scoring clock is misaligned')
            truth.append(dict(score=evidence['score'],physics_safety=evidence['physics_safety']))
        if truth[-1]['score'].get('full_task_success') is not True or truth[-1]['physics_safety'].get('passed') is not True:
            raise ValueError('selected source is not a complete safe successful demonstration')
    for value in (state, actions, next_state, times):
        value.setflags(write=False)
    return Episode(manifest_path,hdf_path,hashes,metadata,collection,row,state,actions,next_state,times,truth,alignment)


def source_snapshot(episode):
    paths = dict(episode.collection.get('scene_dependency_sha256', {}))
    scene = episode.collection['source_scene']; paths[scene['path']] = scene['sha256']
    current = {p: sha256(p) if Path(p).is_file() else None for p in paths}
    return dict(current_sha256=current, recorded_sha256=paths,
        drift=[dict(path=p,recorded=h,current=current[p]) for p,h in paths.items() if current[p] != h],
        caveat='All drift is disclosed, never erased. Expert planner is not called; runtime still constructs its FK helper. Compare both modes with matching CURRENT fingerprints.')


def state_error(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual,np.float64), np.asarray(expected,np.float64)
    if actual.shape != (16,) or expected.shape != (16,) or not np.isfinite(actual).all():
        raise ValueError('expected finite actual/recorded native16 state')
    delta = np.abs(actual-expected)
    return dict(max_joint_abs_rad=float(delta[list(ARM)].max()),
                max_gripper_abs_fraction=float(delta[list(GRIP)].max()),per_dimension_abs=delta.tolist())


class TensorReaders:
    """Read-only NumPy-backend PhysX views; no USD/Fabric fallback or tensor writes."""
    def __init__(self, env):
        self.env = env
        self.rigid = env.rigid
        self.arms = {arm: env.arms[arm]._articulation_view for arm in ('left','right')}
        self.calls = dict(body_states=0,actual_state=0)

    @staticmethod
    def handle(view):
        if not view.is_physics_handle_valid() or getattr(view,'_physics_view',None) is None:
            raise RuntimeError('invalid PhysX tensor handle; refusing stale USD/Fabric fallback')
        return view._physics_view

    def body_states(self):
        import numpy as np
        handle = self.handle(self.rigid)
        poses = np.asarray(handle.get_transforms()).copy()
        velocities = np.asarray(handle.get_velocities()).copy()
        if (poses.shape != (len(self.env.body_names),7) or velocities.shape != (len(self.env.body_names),6)
                or not np.isfinite(poses).all() or not np.isfinite(velocities).all()):
            raise ValueError('invalid PhysX rigid-body tensor geometry')
        self.calls['body_states'] += 1
        return {name:dict(position_world_m=poses[i,:3].tolist(), quaternion_wxyz=poses[i,[6,3,4,5]].tolist(),
                    linear_velocity_m_s=velocities[i,:3].tolist(),angular_velocity_rad_s=velocities[i,3:].tolist())
                for i,name in enumerate(self.env.body_names)}

    def actual_state(self):
        import numpy as np
        from harness.zhuhai_robot_spec import GRIPPER_FINGER_TRAVEL
        values = []
        for arm in ('left','right'):
            all_q = np.asarray(self.handle(self.arms[arm]).get_dof_positions()).copy()
            if all_q.ndim != 2 or all_q.shape[0] != 1 or not np.isfinite(all_q).all():
                raise ValueError('expected one valid PhysX articulation per arm')
            q = all_q[0,np.asarray(self.env.arm_indices[arm])]
            if q.shape != (9,):
                raise ValueError('expected seven Panda joints and two finger joints')
            values.extend(q[:7]); values.append(float(q[7:].mean()/GRIPPER_FINGER_TRAVEL))
        self.calls['actual_state'] += 1
        return np.asarray(values)


class SyncPreflightError(RuntimeError):
    def __init__(self, reason, diagnostics):
        self.reason_code, self.diagnostics = reason, diagnostics
        super().__init__(f'{reason}: {json.dumps(diagnostics, sort_keys=True)}; no physics action executed. '
                         'Do not run a longer prefix before this preflight is resolved; no automatic extension enable/fallback.')


class SyncPatch:
    """Instance-only instrumentation, installed AFTER identical scene setup/settling."""
    def __init__(self, env, settings, mode):
        self.env,self.settings,self.mode = env,settings,mode
        self.previous = {k:settings.get(k) for k in SETTING_KEYS}
        self.saved = []
        self.calls = dict(physics_only_steps=0,rendered_steps=0,fabric_true_physics_steps=0,fabric_false_physics_steps=0)
        self.readers = None
        self.reader_check = None
        self.render_source_error = None
        try:
            self.render_source = inspect.getsource(env.world.render)
        except (OSError, TypeError) as exc:
            self.render_source = ''
            self.render_source_error = str(exc)
        self.fabric_ready = callable(getattr(getattr(env.world,'_physx_fabric_interface',None),'force_update',None))
        self.extension_enabled = None
        self.extension_query_error = None
        manager = getattr(env.world, '_extension_manager', None)
        if callable(getattr(manager, 'is_extension_enabled', None)):
            try:
                self.extension_enabled = bool(manager.is_extension_enabled('omni.physx.fabric'))
            except Exception as exc:
                self.extension_query_error = str(exc)
        self.preflight = dict(fabric_interface_has_force_update=self.fabric_ready,
            fabric_extension_enabled=self.extension_enabled, extension_query_error=self.extension_query_error,
            render_contains_force_update=('force_update' in self.render_source),
            render_source_error=self.render_source_error,
            render_implementation_sha256=hashlib.sha256(self.render_source.encode()).hexdigest())

    def replace(self, obj, name, value):
        self.saved.append((obj,name,name in obj.__dict__,obj.__dict__.get(name)))
        setattr(obj,name,value)

    def install(self):
        import numpy as np
        original = self.env.world.step
        signature = inspect.signature(original)
        if 'render' not in signature.parameters or 'update_fabric' not in signature.parameters:
            raise RuntimeError('unsupported World.step signature')
        if self.mode == 'render_sync':
            if not self.fabric_ready:
                raise SyncPreflightError('fabric_interface_unavailable', self.preflight)
            if 'force_update' not in self.render_source:
                raise SyncPreflightError('render_force_update_unverified', self.preflight)
            readers = TensorReaders(self.env)
            old_state, old_bodies = self.env.actual_state(), self.env.body_states()
            new_state, new_bodies = readers.actual_state(), readers.body_states()
            self.reader_check = state_error(new_state,old_state)
            if not np.allclose(new_state,old_state,rtol=0,atol=1e-7):
                raise ValueError('initial direct joint tensor reader differs from original getter')
            if set(old_bodies) != set(new_bodies):
                raise ValueError('PhysX body order/name mismatch')
            for name in old_bodies:
                for field in old_bodies[name]:
                    if not np.allclose(old_bodies[name][field],new_bodies[name][field],rtol=0,atol=1e-7):
                        raise ValueError(f'initial direct body reader differs: {name}/{field}')
            self.readers = readers
            self.replace(self.env,'actual_state',readers.actual_state)
            self.replace(self.env,'body_states',readers.body_states)
        for key in SETTING_KEYS:
            self.settings.set(key,self.mode == 'reference')
            if self.settings.get(key) != (self.mode == 'reference'):
                raise RuntimeError(f'failed to set {key}')
        @functools.wraps(original)
        def step(*args,**kwargs):
            bound = signature.bind(*args,**kwargs)
            render = bound.arguments.get('render',signature.parameters['render'].default)
            if not render:
                bound.arguments['update_fabric'] = self.mode == 'reference'
                self.calls['physics_only_steps'] += 1
                self.calls['fabric_true_physics_steps' if self.mode == 'reference' else 'fabric_false_physics_steps'] += 1
            else:
                self.calls['rendered_steps'] += 1
            return original(*bound.args,**bound.kwargs)
        self.replace(self.env.world,'step',step)

    def describe(self):
        return dict(mode=self.mode,preflight=self.preflight,settings_before=self.previous,
            settings_active={k:self.settings.get(k) for k in SETTING_KEYS},calls=self.calls,
            direct_tensor_calls=None if self.readers is None else self.readers.calls,
            direct_reader_initial_check=self.reader_check,render_fabric_active=self.fabric_ready,
            render_implementation_sha256=hashlib.sha256(self.render_source.encode()).hexdigest(),
            installation='after common env.setup / settling / initial render, before action[0]',
            body_getter='unchanged valid-handle Prim API (already PhysX-backed)' if self.readers is None else 'direct PhysX get_transforms/get_velocities/get_dof_positions; xyzw -> wxyz',
            changed_core_files=False,poses_or_thresholds_changed=False)

    def restore(self):
        for obj,name,existed,old in reversed(self.saved):
            if existed:setattr(obj,name,old)
            else:delattr(obj,name)
        self.saved.clear()
        for key,value in self.previous.items():
            if value is not None:self.settings.set(key,value)


def save_onboard(out,label,step,images,physics_time):
    import numpy as np
    from PIL import Image
    if set(images) != set(VIEWS):
        raise ValueError('exactly head + two wrist onboard RGB images required')
    files = {}
    for view in VIEWS:
        image = np.asarray(images[view])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError('nonempty uint8 onboard RGB required')
        path = out/'onboard'/f'{label}_{view}.png'
        with path.open('xb') as f:Image.fromarray(image).save(f,format='PNG')
        files[view] = dict(path=str(path.relative_to(out)),sha256=sha256(path),shape=list(image.shape))
    return dict(label=label,control_step=step,physics_time_seconds=physics_time,files=files,kind='onboard PNG observations; not video')


def score_outcome(score):
    return {k:score.get(k) for k in ('track','mapping','base_score','base_score_max','full_task_success')}


def compare_reference(out,result,reference_out):
    """Only compare equal completed prefixes, source and capture protocol; never select a winner."""
    import numpy as np
    reference_out = Path(reference_out).resolve()
    reference = json.loads((reference_out/'result.json').read_text())
    if (reference.get('mode') != 'reference' or result['mode'] != 'render_sync'
            or reference.get('comparison_key') != result['comparison_key']):
        raise ValueError('reference source/seed/steps/capture/current-code fingerprint differs')
    if (reference.get('status') != 'completed' or result.get('status') != 'completed'
            or reference['completed_steps'] != result['completed_steps']):
        raise ValueError('partial/failed runs cannot produce a speedup claim')
    with (reference_out/'steps.jsonl').open() as f:
        prior = [json.loads(line) for line in f]
    with (out/'steps.jsonl').open() as f:
        current = [json.loads(line) for line in f]
    if len(prior) != result['completed_steps'] or len(current) != len(prior):
        raise ValueError('step log length differs from completion receipt')
    maximum = np.zeros(16)
    for old,new in zip(prior,current):
        if old['action_index'] != new['action_index'] or old['target_action'] != new['target_action']:
            raise ValueError('reference action sequence differs')
        maximum = np.maximum(maximum,np.abs(np.asarray(old['actual_state'])-new['actual_state']))
    return dict(reference_out=str(reference_out),reference_result_sha256=sha256(reference_out/'result.json'),
        max_joint_abs_rad=float(maximum[list(ARM)].max()),max_gripper_abs_fraction=float(maximum[list(GRIP)].max()),
        final_score_equal=score_outcome(reference['final_score']) == score_outcome(result['final_score']),
        exact_score_evidence_equal=reference['final_score'] == result['final_score'],
        final_safety_equal=reference['final_safety'] == result['final_safety'],
        replay_wall_speedup=reference['timing']['replay_wall_seconds']/result['timing']['replay_wall_seconds'],
        physics_step_wall_speedup=reference['timing']['physics_step_wall_seconds']/result['timing']['physics_step_wall_seconds'],
        same_capture_stride=True,adoption_decision='main-thread only; numeric drift is reported without inventing a new success tolerance')


def prepare_close(env, exit_code):
    """Use the supported SDK exit_code parameter without relying on its return.

    skip_cleanup=False still permits Kit fast shutdown. We do not monkey-patch
    launch configuration, enable extensions, or disable the SDK's shutdown
    watchdog. Only this app instance's close call receives the final exit code.
    """
    close = dict(state='pending_may_not_return', return_guaranteed=False,
        requested_exit_code=exit_code, exit_code_forwarded=False,
        fast_shutdown_modified=False, cleanup_duration_seconds=None)
    app = getattr(env, 'app', None)
    config = getattr(app, 'config', None)
    close['sdk_fast_shutdown_config'] = config.get('fast_shutdown') if isinstance(config, dict) else None
    original = getattr(app, 'close', None)
    if not callable(original):
        close['exit_code_note'] = 'no public SimulationApp.close on this environment; receipt remains authoritative'
        return close, None
    try:
        signature = inspect.signature(original)
        close['sdk_close_signature'] = str(signature)
        if 'exit_code' not in signature.parameters:
            close['exit_code_note'] = 'SDK lacks supported exit_code; process exit status is not guaranteed'
            return close, None
        had_instance_value = 'close' in getattr(app, '__dict__', {})
        previous_value = getattr(app, '__dict__', {}).get('close')
        @functools.wraps(original)
        def forward(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            bound.arguments['exit_code'] = exit_code
            return original(*bound.args, **bound.kwargs)
        app.close = forward
        close['exit_code_forwarded'] = True
        def restore():
            if had_instance_value:
                app.close = previous_value
            else:
                delattr(app, 'close')
        return close, restore
    except Exception as exc:
        close['exit_code_note'] = f'cannot install per-instance supported exit_code forwarding: {exc}'
        return close, None


def commit_before_close(out, result, protocol, exit_code):
    """All mandatory outcome/protocol/error/hash/timing data is durable on return."""
    result['intended_process_exit_code'] = exit_code
    result['finalization'] = dict(stage='pre_close_final', authoritative_replay_result=True,
        all_required_fields_committed_before_close=True, durability='atomic replace + file fsync + directory fsync',
        close_may_terminate_interpreter=True)
    protocol = dict(protocol, sync_protocol=result.get('sync_protocol'),
        final_outcome=dict(status=result['status'], error=result.get('error'),
            failure_phase=result.get('failure_phase'), reason_code=result.get('reason_code'),
            intended_process_exit_code=exit_code), finalization=result['finalization'],
        close=result['close'], timing_scope=result['timing']['total_wall_seconds_scope'])
    flush_existing(out/'steps.jsonl')
    write_json(out/'protocol.json', protocol, durable=True)
    write_json(out/'result.json', result, durable=True)
    # This print must also precede SimulationApp.close()/os._exit().
    print(json.dumps(dict(event='benchmark_result_committed_before_close', status=result['status'],
        mode=result['mode'], out=str(out), completed_steps=result['completed_steps'],
        intended_process_exit_code=exit_code, failure_phase=result.get('failure_phase'),
        reason_code=result.get('reason_code'), error=result.get('error'), close=result['close']),
        ensure_ascii=False), flush=True)
    if result.get('traceback'):
        print(result['traceback'], file=sys.stderr, flush=True)


def execute_replay(args,episode,*,env_factory,settings_factory,lease_check=verify_owner):
    import numpy as np
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)  # no overwrite/resume/retry selection, including old source directories
    (out/'onboard').mkdir()
    steps = len(episode.actions) if args.steps is None else args.steps
    snapshot = source_snapshot(episode)
    common = dict(source_sha256=episode.hashes,seed=episode.metadata['seed'],task=episode.metadata['task'],
        steps=steps,capture_stride=args.capture_stride,physics_dt=PHYSICS_DT,control_dt=CONTROL_DT,
        robot_contract=episode.collection['robot_contract'],current_source_sha256=snapshot['current_sha256'],
        benchmark_script_sha256=sha256(__file__),source_metadata_sha256=digest(episode.metadata))
    result = dict(format=FORMAT,status='starting',mode=args.mode,source=episode.describe(),source_metadata=episode.metadata,
        current_source_snapshot=snapshot,comparison_key=digest(common),comparison_protocol=common,
        completed_steps=0,planned_steps=steps,full_source_steps=len(episode.actions),full_task_replayed=False,
        max_joint_abs_rad=None,max_gripper_abs_fraction=None,final_score=None,final_safety=None,physics_elapsed_seconds=0.,
        recording=dict(video=False,third_view=False,capture_stride=args.capture_stride,required_initial_final_onboard_png=True),
        protocol_differences=dict(no_policy_or_expert=True,no_action_postprocessing=True,no_pose_restoration=True,
            original_source_rgb_every_control_frame=True,replay_sparse_capture_is_shared_by_both_modes=True,
            original_recorded_commands_dtype=str(episode.actions.dtype),
            warning='Do NOT label reduced RGB cadence as USD/Fabric sync speedup vs original collection.'),
        started_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,
        timing=dict(setup_wall_seconds=0.,physics_step_wall_seconds=0.,capture_wall_seconds=0.,png_write_wall_seconds=0.,replay_wall_seconds=0.),
        captures=[],physical_equivalence_proven=False,adoption_decision='not automatic')
    result['receipt_schema_version']=2
    result['phase']='before_environment_initialization'
    write_json(out/'result.json',result, durable=True)
    protocol=dict(common=common,mode=args.mode,source_metadata=episode.metadata,source_snapshot=snapshot,receipt_schema_version=2)
    write_json(out/'protocol.json',protocol, durable=True)
    env=patch=None; setup_done=False; final_saved=False; replay_start=None; original_handlers={}
    max_error=np.zeros(16); started=time.perf_counter(); code=2
    def capture(label,step,initial=False):
        nonlocal final_saved
        lease_check(args.gpu_lock_owner)
        t=time.perf_counter()
        if initial and getattr(env,'last_images',None) is not None:
            images=env.last_images
        else:
            images,_=env.capture()
        result['timing']['capture_wall_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        result['captures'].append(save_onboard(out,label,step,images,float(env.physics_time)))
        result['timing']['png_write_wall_seconds']+=time.perf_counter()-t
        if label=='final':final_saved=True
    def interrupted(signum,frame):
        raise InterruptedError(f'benchmark interrupted by signal {signum}')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):
            original_handlers[sig]=signal.signal(sig,interrupted)
        lease_check(args.gpu_lock_owner)
        result['phase']='environment_initialization'
        env=env_factory(gpu_lock_owner=args.gpu_lock_owner,output_dir=out)
        result['phase']='scene_setup'
        env.setup(task=episode.metadata['task'],seed=episode.metadata['seed'],
            randomized=episode.metadata['mode']=='randomized',mapping=episode.collection.get('life_mapping','body_text'),
            source_seed=episode.metadata.get('parent_source_seed'))
        setup_done=True; result['timing']['setup_wall_seconds']=time.perf_counter()-started
        if env.physics_dt!=PHYSICS_DT or env.robot_contract!=episode.collection['robot_contract'] or env.on_control_step is not None:
            raise ValueError('runtime dt/robot/hidden control callback differs; replay protocol refused')
        result['phase']='sync_patch_construction'
        settings=settings_factory()
        patch=SyncPatch(env,settings,args.mode)
        result['phase']='sync_preflight'
        result['sync_protocol']=patch.describe()
        write_json(out/'result.json',result, durable=True)
        patch.install()
        result['sync_protocol']=patch.describe()
        result['phase']='initial_observation'
        actual=np.asarray(env.actual_state()).copy(); initial_time=float(env.physics_time)
        result['initial_actual_state']=actual.tolist(); result['initial_source_error']=state_error(actual,episode.state[0])
        result['initial_rule_evidence']=jsonable(env.rule_state())
        capture('initial',0,initial=True)
        if result['initial_rule_evidence']['physics_safety'].get('passed') is not True:
            raise RuntimeError('initial mechanical safety gate failed')
        result['status']='running'; result['phase']='replaying_actions'
        write_json(out/'result.json',result, durable=True)
        replay_start=time.perf_counter()
        with (out/'steps.jsonl').open('x') as log_file:
            for i in range(steps):
                lease_check(args.gpu_lock_owner)
                target=episode.actions[i].copy()
                result['active_action_index']=i; result['active_target_action']=target.tolist()
                before=float(env.physics_time); t=time.perf_counter()
                returned=env.step_action(target,interval_seconds=CONTROL_DT)
                step_wall=time.perf_counter()-t
                result['timing']['physics_step_wall_seconds']+=step_wall
                evidence=returned if isinstance(returned,dict) else env.rule_state()
                actual=np.asarray(env.actual_state()).copy(); now=float(env.physics_time)
                error=state_error(actual,episode.next_state[i]); max_error=np.maximum(max_error,error['per_dimension_abs'])
                entry=dict(action_index=i,control_step=i+1,expected_hdf_observation_index=i+1,
                    target_action=target.tolist(),reported_target_action=jsonable(env.get_action()),actual_state=actual.tolist(),
                    expected_next_state=episode.next_state[i].tolist(),source_error=error,
                    physics_time_seconds=now,expected_source_time_seconds=float(episode.times[i+1]),
                    physics_step_wall_seconds=step_wall,score=evidence['score'],physics_safety=evidence['physics_safety'])
                log_file.write(json.dumps(jsonable(entry),allow_nan=False)+'\n');log_file.flush()
                result.update(completed_steps=i+1,final_actual_state=actual.tolist(),final_score=jsonable(evidence['score']),
                    final_safety=jsonable(evidence['physics_safety']),physics_elapsed_seconds=now-initial_time,
                    max_joint_abs_rad=float(max_error[list(ARM)].max()),max_gripper_abs_fraction=float(max_error[list(GRIP)].max()),
                    per_dimension_max_abs=max_error.tolist())
                result['timing']['replay_wall_seconds']=time.perf_counter()-replay_start
                write_json(out/'result.json',result)
                if not np.array_equal(np.asarray(env.get_action()),target):
                    raise ValueError('runtime changed archived target command; no clipping/resampling allowed')
                if not math.isclose(now-before,CONTROL_DT,abs_tol=1e-7,rel_tol=0):
                    raise ValueError('runtime physical clock diverged from unchanged 10Hz interval')
                if evidence['physics_safety'].get('passed') is not True or evidence['physics_safety'].get('events'):
                    raise RuntimeError('mechanical safety failure; failing step retained, no fallback or source replacement')
                if args.capture_stride and (i+1)%args.capture_stride==0 and i+1<steps:
                    capture(f'step_{i+1:06d}',i+1)
        result['phase']='final_capture'
        capture('final',steps)
        result['timing']['replay_wall_seconds']=time.perf_counter()-replay_start
        result['phase']='replay_finished'
        result.update(status='completed',full_task_replayed=(steps==len(episode.actions)),
            expected_source_prefix=episode.truth[steps],
            final_score_matches_source=(score_outcome(result['final_score'])==score_outcome(episode.truth[steps]['score'])),
            exact_source_score_evidence_equal=(result['final_score']==episode.truth[steps]['score']),
            final_safety_matches_source=(result['final_safety']==episode.truth[steps]['physics_safety']))
        code=0
        if not result['final_score_matches_source']:
            result['status']='completed_with_score_mismatch';code=2
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
            failure_phase=result.get('phase'),reason_code=getattr(exc,'reason_code',type(exc).__name__))
        if isinstance(exc,SyncPreflightError):
            result['sync_preflight_failure']=exc.diagnostics
        # An error must survive even if a final capture/cleanup subsequently terminates Kit.
        result['finalization']=dict(stage='exception_caught_before_cleanup',authoritative_replay_result=False)
        write_json(out/'failure.json',dict(status='failed',error=result['error'],traceback=result['traceback'],
            failure_phase=result['failure_phase'],reason_code=result['reason_code'],completed_steps=result['completed_steps'],
            sync_protocol=result.get('sync_protocol'),timing=result['timing']),durable=True)
        write_json(out/'result.json',result,durable=True)
        print(f"benchmark failure BEFORE close: {result['reason_code']}: {result['error']}",file=sys.stderr,flush=True)
        if replay_start is not None:result['timing']['replay_wall_seconds']=time.perf_counter()-replay_start
        if setup_done:
            try:
                result['failure_rule_evidence']=jsonable(env.rule_state())
                result['final_score']=jsonable(result['failure_rule_evidence']['score'])
                result['final_safety']=jsonable(result['failure_rule_evidence']['physics_safety'])
                result['physics_elapsed_seconds']=float(env.physics_time)-float(episode.times[0])
                failed_state=np.asarray(env.actual_state())
                if failed_state.shape==(16,) and np.isfinite(failed_state).all():
                    result['failure_actual_state']=failed_state.tolist()
                else:result['failure_state_invalid']=True
            except BaseException as snapshot_error:result['failure_snapshot_error']=str(snapshot_error)
    finally:
        # No env.close() is allowed until ALL mandatory fields and comparison are durable.
        result['phase']='finalizing_before_close'
        finalize_started=time.perf_counter()
        if setup_done and not final_saved:
            try:capture('final',result['completed_steps'])
            except BaseException as exc:result['final_capture_error']=f'{type(exc).__name__}: {exc}'
        if patch is not None:
            try:result['sync_protocol']=patch.describe()
            except BaseException as exc:result['sync_protocol_read_error']=str(exc);result['status']='failed';code=2
            try:patch.restore()
            except BaseException as exc:result['restore_error']=str(exc);result['status']='failed';code=2
        verification_started=time.perf_counter()
        try:
            result['source_hashes_unchanged']=(sha256(episode.manifest_path)==episode.hashes['manifest'] and sha256(episode.hdf_path)==episode.hashes['hdf5'])
            result['current_code_snapshot_unchanged']=source_snapshot(episode)['current_sha256']==snapshot['current_sha256']
        except BaseException as exc:
            result['final_hash_verification_error']=str(exc)
            result['source_hashes_unchanged']=False;result['current_code_snapshot_unchanged']=False
        result['timing']['final_hash_verification_wall_seconds']=time.perf_counter()-verification_started
        if not result['source_hashes_unchanged']:result['status']='failed';result['source_mutation_detected']=True;code=2
        if not result['current_code_snapshot_unchanged']:result['status']='failed';code=2
        if args.reference_out:
            try:result['reference_comparison']=compare_reference(out,result,args.reference_out)
            except Exception as exc:result['reference_comparison_error']=str(exc);code=2
        close_restore=None
        if env is not None:
            result['close'],close_restore=prepare_close(env,code)
        else:
            result['close']=dict(state='not_created',return_guaranteed=False,requested_exit_code=code)
        result['timing'].update(finalization_wall_seconds_before_commit=time.perf_counter()-finalize_started,
            total_wall_seconds=time.perf_counter()-started,
            total_wall_seconds_scope='through pre-close finalization; excludes receipt fsync and env.close',
            close_wall_seconds=None)
        result['phase']='pre_close_result_committed'
        commit_before_close(out,result,protocol,code)
        # At this point even os._exit() in env.close cannot lose the run outcome.
        for sig,previous in original_handlers.items():signal.signal(sig,previous)
        close_started=time.perf_counter()
        if env is not None:
            try:
                env.close()  # MAY NEVER RETURN. Everything below is optional enrichment only.
                result['close']['state']='returned'
            except BaseException as exc:
                result['close']['state']='raised'
                result['close_error']=f'{type(exc).__name__}: {exc}'
                result['status']='failed';code=2
            finally:
                if close_restore is not None:
                    try:close_restore()
                    except BaseException as exc:result['close_forwarder_restore_error']=str(exc)
        if env is not None:
            result['timing']['close_wall_seconds']=time.perf_counter()-close_started
            result['close']['cleanup_duration_seconds']=result['timing']['close_wall_seconds']
        result['timing']['total_wall_seconds']=time.perf_counter()-started
        result['timing']['total_wall_seconds_scope']='through env.close return; optional post-close observation'
        result['intended_process_exit_code']=code
        result['post_close_observed']=True
        write_json(out/'result.json',result,durable=True)
    return code,result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=DEFAULT_SOURCE/'manifest.json')
    p.add_argument('--hdf5','--episode',dest='hdf5',type=Path,default=DEFAULT_SOURCE/'data/episode_seed_0000921002.hdf5')
    p.add_argument('--mode',choices=('reference','render_sync'),default='reference')
    p.add_argument('--steps',type=int,help='positive control-step limit; omitted = ALL archived target actions')
    p.add_argument('--capture-stride',type=int,default=0,help='0: initial/final onboard PNG only; 1: every control observation')
    p.add_argument('--out',type=Path,help='required for execution; must not already exist')
    p.add_argument('--reference-out',type=Path,help='optional completed reference directory, same source/steps/capture/code')
    p.add_argument('--gpu-lock-owner')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--check-only',action='store_true')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.capture_stride<0 or (args.steps is not None and args.steps<1):p.error('steps must be positive; capture-stride nonnegative')
    if args.execute and (args.out is None or not args.gpu_lock_owner):p.error('--execute requires unique --out and --gpu-lock-owner')
    log(f'独立sync基准 {args.mode}：先只读核验source/HDF、10Hz及next-state对齐；execute={args.execute}')
    episode=read_episode(args.manifest,args.hdf5)
    if args.steps is not None and args.steps>len(episode.actions):p.error('--steps exceeds source action count')
    if not args.execute:
        print(json.dumps(dict(status='cpu_source_verified_no_simulator_started',mode=args.mode,source=episode.describe(),
            planned_steps=args.steps or len(episode.actions),capture_stride=args.capture_stride,
            current_source_snapshot=source_snapshot(episode)),ensure_ascii=False,indent=2))
        return 0
    if args.out.exists():raise FileExistsError('unique --out required; all previous results/source episodes are preserved')
    verify_owner(args.gpu_lock_owner)
    log(f'GPU锁由主线程持有；执行同源raw target replay mode={args.mode}，输出{args.out}；无姿态/规则更改，无视频')
    # These imports are unreachable in CPU check/CLI tests.
    from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
    def settings_factory():
        import carb
        return carb.settings.get_settings()
    code,result=execute_replay(args,episode,env_factory=IsaacZhuhaiEnv,settings_factory=settings_factory)
    print(json.dumps(dict(status=result['status'],mode=args.mode,out=str(args.out),completed_steps=result['completed_steps'],
        max_joint_abs_rad=result.get('max_joint_abs_rad'),final_score=result.get('final_score'),
        final_safety=result.get('final_safety'),timing=result['timing']),ensure_ascii=False))
    return code


if __name__=='__main__':
    raise SystemExit(main())
