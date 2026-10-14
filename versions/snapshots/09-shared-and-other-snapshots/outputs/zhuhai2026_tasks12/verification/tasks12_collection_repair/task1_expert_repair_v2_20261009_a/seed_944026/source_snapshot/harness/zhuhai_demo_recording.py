"""Native16 measured-state / target-command HDF5 archives, never videos.

Observations arrive after each completed 0.1s control interval. Consequently,
state[i] is the measured configuration at t[i], while action[i] is the target
reported at t[i+1], applied during (t[i], t[i+1]]. The last observation is kept
for auditing/future vision targets and never receives an invented action.
Legacy native14 recordings remain readable by their original code, not here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from harness.zhuhai_robot_spec import (ROBOT_DOF, ARM_DOF, GRIPPER_INDICES, ARM_INDICES,
    ACTION_CONTRACT, STATE_CONTRACT, DUAL_JOINT_LIMITS, validate_action as spec_validate_action)

STATE_DIM = ACTION_DIM = ROBOT_DOF
if (ROBOT_DOF != 16 or ARM_DOF != 7 or tuple(GRIPPER_INDICES) != (7, 15)
        or tuple(ARM_INDICES) != (*range(7), *range(8, 15))):
    raise ValueError('Native16 data cannot silently reinterpret another robot layout')
JOINT_FIELDS = (("left_arm_joint_states", 7), ("left_ee_joint_states", 1),
                ("right_arm_joint_states", 7), ("right_ee_joint_states", 1))
CAMERAS = {"cam_head": "head_camera", "cam_left_wrist": "left_camera",
           "cam_right_wrist": "right_camera"}
POLICY_VIEWS = tuple(CAMERAS)
SOURCE_FORMAT = "Zhuhai2026_native16_physical_expert"
STATE_SEMANTICS = "measured_joint_gripper"
ACTION_SEMANTICS = "joint_position_target"
DATA_CONTRACT = dict(schema="zhuhai-native16-measured-target-10hz-v1",
    robot_dof=16, arm_dof=7, state_dim=16, action_dim=16,
    native_order=["left_arm[7]", "left_gripper", "right_arm[7]", "right_gripper"],
    gripper_indices=[7, 15], arm_indices=list(ARM_INDICES), gripper_open=1., gripper_closed=0.,
    state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
    state_contract=STATE_CONTRACT, action_contract=ACTION_CONTRACT,
    state_alignment="state[i] = measured(t[i])",
    action_alignment="action[i] = target_command(t[i+1]), applied during (t[i], t[i+1]]",
    terminal_observation="retained without an invented subsequent action",
    measured_state_clipped_by_data_layer=False,
    physics_safety="explicit passed=True at every recorded observation and final evidence",
    frequency_hz=10., record_video=False, third_view=False,
    camera_names=list(CAMERAS.values()), camera_count=3)


def data_contract():
    """CPU-only schema, independent of asset existence or Isaac initialization."""
    return deepcopy(DATA_CONTRACT)


def data_contract_hash():
    return hashlib.sha256(json.dumps(DATA_CONTRACT, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _codec():
    from harness.robotwin_dataset import _codec as standalone_codec
    return standalone_codec()


def jsonable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def validate_native16(value, *, name="native16", command=False, joint_limits=None):
    """Validate without clipping/reordering; measured state is never a command."""
    import numpy as np
    array = np.asarray(value)
    if array.shape != (16,) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name} requires a finite real native16 vector; old 14D is not admitted')
    if (np.abs(array) > np.finfo(np.float32).max).any():
        raise ValueError(f'{name} cannot be represented as float32')
    if command:
        if np.any(array[list(GRIPPER_INDICES)] < 0) or np.any(array[list(GRIPPER_INDICES)] > 1):
            raise ValueError('Target grippers must be normalized [0,1], 1=open')
        limits = np.asarray(DUAL_JOINT_LIMITS if joint_limits is None else joint_limits, dtype=np.float64)
        if limits.shape != (14, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError('Two seven-joint arms require 14 finite ordered position limits')
        spec_validate_action(array, limits)
    return array.astype(np.float32, copy=True)


def _joint_fields(value, name):
    import numpy as np
    if not isinstance(value, dict):
        return validate_native16(value, name=name)
    vectors = []
    for hand in ('left', 'right'):
        arm = np.asarray(value[f'{hand}_arm'])
        gripper = np.asarray(value[f'{hand}_gripper'])
        if arm.shape != (7,) or gripper.shape not in ((), (1,)):
            raise ValueError(f'{name}: {hand} arm must have 7 joints and one gripper')
        vectors.extend([arm, gripper.reshape(1)])
    return validate_native16(np.concatenate(vectors), name=name)


def native_observation(observation, *, measured_state=None, target_action=None):
    """Allow measured-state aliases, but NEVER fall back to commanded state.

    Canonical policy keys are measured_joint_gripper_state and
    commanded_joint_gripper_state. The raw runtime can also expose state or
    joint_state, alongside joint_action with 7D left/right arm fields. Collector
    passes actual_state()/get_action() explicitly and cross-checks any raw copy.
    """
    import numpy as np
    if not isinstance(observation, dict):
        raise ValueError('Native16 observation must be a dictionary')
    measured = [] if measured_state is None else [_joint_fields(measured_state, 'actual_state')]
    for key in ('measured_joint_gripper_state', 'state', 'joint_state'):
        if key in observation:
            measured.append(_joint_fields(observation[key], key))
    if not measured:
        raise ValueError('Explicit measured_joint_gripper_state/state required; target commands are not state')
    if any(not np.allclose(value, measured[0], rtol=0, atol=1e-6) for value in measured[1:]):
        raise ValueError('Measured state copies disagree; do not mix observation timestamps')
    commands = [] if target_action is None else [validate_native16(target_action, name='get_action', command=True)]
    for key in ('commanded_joint_gripper_state', 'joint_action'):
        if key in observation:
            commands.append(validate_native16(_joint_fields(observation[key], key), name=key, command=True))
    if not commands:
        raise ValueError('Explicit native16 target command required')
    if any(not np.allclose(value, commands[0], rtol=0, atol=1e-6) for value in commands[1:]):
        raise ValueError('Target command copies disagree')
    if 'observation' in observation:
        cameras = observation['observation']
        images = {key: value['rgb'] for key, value in cameras.items()}
    elif 'rgb' in observation:
        images = observation['rgb']
    else:
        raise ValueError('Three onboard RGB observations required')
    if set(images) != set(CAMERAS.values()):
        raise ValueError('Exactly head + left/right wrist onboard observations are required')
    rgb = {}
    for key, image in images.items():
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError('Native onboard RGB must be nonempty uint8 HWC')
        rgb[key] = image.copy()
    return dict(rgb=rgb, measured_joint_gripper_state=measured[0],
                commanded_joint_gripper_state=commands[0])


def episode_statistics(state, action):
    """Episode diagnostics only: not TRAIN/VAL/heldout normalization fitting."""
    import numpy as np
    def moments(values, semantics):
        values = np.asarray(values, np.float64)
        return dict(dimension=16, semantics=semantics, count=len(values),
            mean=values.mean(axis=0).tolist(), std=values.std(axis=0).tolist(),
            minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist())
    return dict(scope='single_episode_diagnostics_not_policy_normalization',
        data_contract_sha256=data_contract_hash(), state=moments(state, STATE_SEMANTICS),
        action=moments(action, ACTION_SEMANTICS))


class StreamDemoRecorder:
    def __init__(self, path, *, metadata, frequency):
        import h5py
        import numpy as np
        if isinstance(frequency, bool) or float(frequency) != 10.:
            raise ValueError('Native16 archive requires fixed 10Hz control')
        metadata = deepcopy(metadata)
        if 'data_contract' in metadata and metadata['data_contract'] != DATA_CONTRACT:
            raise ValueError('Metadata data schema changed; old 14D is not admitted')
        contract = metadata.get('contract', {})
        if contract and (contract.get('data_contract') != DATA_CONTRACT or
                         contract.get('state_dim') != 16 or contract.get('action_dim') != 16 or
                         contract.get('state_semantics') != STATE_SEMANTICS or
                         contract.get('action_semantics') != ACTION_SEMANTICS):
            raise ValueError('Collection contract must explicitly declare native16 measured state / target action')
        metadata['data_contract'] = data_contract()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix('.hdf5.partial')
        if self.path.exists() or self.path.is_symlink() or self.temporary.exists() or self.temporary.is_symlink():
            raise FileExistsError(self.path)
        self.file = h5py.File(self.temporary, 'x')
        self.file.attrs['source_format'] = SOURCE_FORMAT
        self.file.attrs['record_video'] = False
        self.file.attrs['third_view'] = False
        for key, value in dict(state_dim=16, action_dim=16, robot_dof=16, arm_dof=7,
                state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS).items():
            self.file.attrs[key] = value
        self.file.attrs['gripper_indices'] = np.asarray(GRIPPER_INDICES, np.int32)
        self.file.attrs['data_contract'] = json.dumps(DATA_CONTRACT, sort_keys=True, allow_nan=False)
        self.file.attrs['data_contract_sha256'] = data_contract_hash()
        self.file.attrs['metadata'] = json.dumps(jsonable(metadata), ensure_ascii=False, allow_nan=False)
        self.file.create_dataset('data_format_version', data='v2.0')
        self.file.create_dataset('additional_info/frequency', data=10.)
        self.file.create_dataset('instructions', data=json.dumps([metadata['task']]))
        self.state = self._numeric('measured_state', (16,))
        self.commands = self._numeric('target_command', (16,))
        self.timestamps = self._numeric('physics_time', ())
        self.rules = self.file.create_dataset('rule_evidence', shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype('utf-8'), chunks=True)
        self.colors = {}
        self.length = 0
        self.closed = False
        self.statistics = None
        self.physics_safety_passed = True

    def _numeric(self, name, shape):
        return self.file.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape),
            dtype='float32' if shape else 'float64', chunks=True)

    @staticmethod
    def _append(ds, value):
        ds.resize(len(ds) + 1, axis=0)
        ds[-1] = value

    def capture(self, observation, physics_time, evidence, *, measured_state=None, target_action=None):
        import h5py
        import numpy as np
        if self.closed:
            raise RuntimeError('recorder is closed')
        obs = native_observation(observation, measured_state=measured_state, target_action=target_action)
        now = float(physics_time)
        if not np.isfinite(now) or now < 0 or (self.length and abs(now-float(self.timestamps[-1])-.1) > 1e-7):
            raise ValueError('Native16 observations must have uniform forward 10Hz physical timestamps')
        evidence_json = json.dumps(jsonable(evidence), ensure_ascii=False, allow_nan=False)
        safety = evidence.get('physics_safety', {})
        self.physics_safety_passed = self.physics_safety_passed and safety.get('passed') is True
        # Validate/encode all three views before advancing any dataset length.
        encoded_views = {}
        for name, native in CAMERAS.items():
            rgb = obs['rgb'][native]
            if name in self.colors and tuple(self.file[f'vision/{name}/shape'][:]) != rgb.shape:
                raise ValueError('RGB shape changed during episode')
            value = _codec().encode_image_bit(rgb)
            encoded_views[name] = (np.frombuffer(value, dtype=np.uint8) if isinstance(value, bytes)
                                  else np.asarray(value, dtype=np.uint8).reshape(-1))
            if name not in self.colors:
                group = self.file.require_group(f'vision/{name}')
                group.create_dataset('shape', data=np.asarray(rgb.shape, np.int32))
                self.colors[name] = group.create_dataset('colors', shape=(0,), maxshape=(None,),
                    chunks=True, dtype=h5py.vlen_dtype(np.dtype('uint8')))
        self._append(self.state, obs['measured_joint_gripper_state'])
        self._append(self.commands, obs['commanded_joint_gripper_state'])
        self._append(self.timestamps, now)
        self._append(self.rules, evidence_json)
        for name, encoded in encoded_views.items():
            self._append(self.colors[name], encoded)
        self.length += 1

    def finish(self, *, final_evidence, successful):
        if self.closed:
            raise RuntimeError('recorder is closed')
        if type(successful) is not bool:
            raise ValueError('successful must be an explicit bool, never a truthy status/string')
        if self.length < 2:
            raise ValueError('need at least two source observations')
        safety = final_evidence.get('physics_safety', {})
        safe = self.physics_safety_passed and safety.get('passed') is True
        if successful and not safe:
            raise ValueError('Unsafe/missing physics_safety cannot be labeled a successful demonstration')
        measured, commands = self.state[:], self.commands[:]
        terminal = self.file.create_group('terminal_observation')
        terminal.create_dataset('measured_state', data=measured[-1])
        terminal.create_dataset('target_command', data=commands[-1])
        terminal.create_dataset('physics_time', data=self.timestamps[-1])
        terminal.create_dataset('rule_evidence', data=self.rules[-1])
        for name, ds in self.colors.items():
            camera = terminal.require_group(f'vision/{name}')
            camera.create_dataset('colors', data=ds[-1])
            camera.create_dataset('shape', data=self.file[f'vision/{name}/shape'][:])
        offset = 0
        for field, width in JOINT_FIELDS:
            self.file.create_dataset(f'state/{field}', data=measured[:-1, offset:offset+width])
            self.file.create_dataset(f'action/{field}', data=commands[1:, offset:offset+width])
            offset += width
        # Full measured/command/time/rule streams remain intact for verification.
        # Only RGB is split into training observations and the terminal observation.
        for ds in self.colors.values():
            ds.resize(self.length-1, axis=0)
        self.statistics = episode_statistics(measured[:-1], commands[1:])
        self.file.attrs['episode_statistics'] = json.dumps(self.statistics, sort_keys=True, allow_nan=False)
        self.file.attrs['source_observation_count'] = self.length
        self.file.attrs['training_frame_count'] = self.length-1
        self.file.attrs['physics_safety_all_observations_passed'] = bool(safe)
        self.file.attrs['successful_expert'] = successful
        self.file.attrs['final_rule_evidence'] = json.dumps(jsonable(final_evidence), ensure_ascii=False, allow_nan=False)
        self.file.flush(); self.file.close(); self.closed = True
        # Unlike replace(), publication cannot overwrite an orphan/prior episode.
        os.link(self.temporary, self.path)
        self.temporary.unlink()
        return self.path

    def abort(self):
        if not self.closed:
            self.file.close(); self.closed = True
        return self.temporary  # Preserve clearly partial evidence, never index as data.
