"""Native16 episode-local training samples with bounded, on-demand DINO cache.

Only three camera RGB streams and measured joint/gripper state condition the policy. Privileged
rule evidence remains in source archives and never enters policy conditioning.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from harness.robotwin_dataset import _codec
from harness.zhuhai_demo_recording import (DATA_CONTRACT, SOURCE_FORMAT, STATE_DIM, ACTION_DIM,
    STATE_SEMANTICS, ACTION_SEMANTICS, GRIPPER_INDICES, JOINT_FIELDS, POLICY_VIEWS,
    data_contract, data_contract_hash, episode_statistics, ARM_INDICES, DUAL_JOINT_LIMITS)
from harness.robotwin_training_data import FEAT_SHAPE, POOL_SHAPE
from harness.wam_behavior import normalize_image
from harness.wam_v3 import history_frames

PREPROCESS = {"views": list(POLICY_VIEWS), "size": [224, 224],
              "resize": "cv2.INTER_AREA", "normalization": "ImageNet",
              "layers": [-12, -8, -4], "feature_schema": "zhuhai-dino-bf16-native16-v2",
              "source_schema": DATA_CONTRACT["schema"], "state_dim": STATE_DIM, "state_semantics": STATE_SEMANTICS}
PREPROCESS_SHA256 = hashlib.sha256(json.dumps(PREPROCESS, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _native16_lowdim(handle, group):
    if set(handle[group]) != {field for field, _ in JOINT_FIELDS}:
        raise ValueError('Native16 state/action must contain exactly two 7D arms and two grippers')
    arrays = []
    for field, width in JOINT_FIELDS:
        ds = handle[f'{group}/{field}']
        values = ds[:]
        if values.ndim != 2 or values.shape[1] != width or values.dtype.kind not in 'fiu':
            raise ValueError(f'{group}/{field}: expected native16 field width {width}; old 14D is forbidden')
        if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError('Native16 state/action contains nonfinite or unrepresentable values')
        arrays.append(values.astype(np.float32))
    if len({len(a) for a in arrays}) != 1:
        raise ValueError('Native16 fields have inconsistent frame counts')
    result = np.concatenate(arrays, axis=1)
    if group == 'action' and ((result[:, GRIPPER_INDICES] < 0).any() or (result[:, GRIPPER_INDICES] > 1).any()):
        raise ValueError('Target grippers must be [0,1], 1=open')
    if group == 'action':
        arm = result[:, ARM_INDICES]
        bounds = np.asarray(DUAL_JOINT_LIMITS)
        if (arm < bounds[:, 0]-1e-7).any() or (arm > bounds[:, 1]+1e-7).any():
            raise ValueError('Archived target action violates official dual-Panda joint limits')
    return result


class SourceEpisode:
    """Strict native16 measured-state archives; legacy/partial inputs fail closed."""
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    state_semantics = STATE_SEMANTICS
    action_semantics = ACTION_SEMANTICS

    def __init__(self, row):
        self.row = dict(row)
        source_path = Path(row['path'])
        if source_path.is_symlink() or source_path.name.endswith('.partial'):
            raise ValueError('Failed/partial or symlink demonstrations cannot enter training')
        self.path = source_path.resolve()
        if not self.path.is_file():
            raise ValueError('Source must be a regular HDF file')
        self.sha256 = sha256_file(self.path)
        expected = row.get('sha256') or row.get('data_sha256')
        if expected is not None and expected != self.sha256:
            raise ValueError('Source content differs from admitted demonstration hash')
        self._stat = (self.path.stat().st_size, self.path.stat().st_mtime_ns)
        self.data_contract = data_contract()
        for key, expected in dict(state_dim=16, action_dim=16,
                state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS).items():
            if key in row and row[key] != expected:
                raise ValueError(f'Admitted row {key} is not native16 measured-state data')
        with h5py.File(self.path, 'r') as handle:
            successful = handle.attrs.get('successful_expert')
            if not isinstance(successful, (bool, np.bool_)) or not bool(successful):
                raise ValueError('Failed/partial demonstrations cannot enter training')
            safe = handle.attrs.get('physics_safety_all_observations_passed')
            if not isinstance(safe, (bool, np.bool_)) or not bool(safe):
                raise ValueError('Missing/failed per-observation physics_safety cannot enter training')
            final_evidence = json.loads(handle.attrs.get('final_rule_evidence', '{}'))
            if final_evidence.get('physics_safety', {}).get('passed') is not True:
                raise ValueError('Final physics_safety must explicitly pass')
            if handle.attrs.get('source_format') != SOURCE_FORMAT:
                raise ValueError('Native16 source format required; old native14 must remain separate')
            for key, expected in dict(state_dim=16, action_dim=16, robot_dof=16, arm_dof=7,
                    state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS).items():
                if handle.attrs.get(key) != expected:
                    raise ValueError(f'Source {key} must explicitly declare native16 measured state / target action')
            if json.loads(handle.attrs['data_contract']) != DATA_CONTRACT or handle.attrs.get('data_contract_sha256') != data_contract_hash():
                raise ValueError('Source native16 data contract changed')
            if tuple(handle.attrs.get('gripper_indices', ())) != GRIPPER_INDICES:
                raise ValueError('Native16 gripper indices must be 7 and 15')
            if bool(handle.attrs.get('record_video', True)) or bool(handle.attrs.get('third_view', True)):
                raise ValueError('Video / third-view protocol mismatch')
            if handle['data_format_version'].asstr()[()] != 'v2.0' or float(handle['additional_info/frequency'][()]) != 10.:
                raise ValueError('Native16 v2.0 schema and 10Hz sampling are required')
            metadata = json.loads(handle.attrs['metadata'])
            if metadata.get('data_contract') != DATA_CONTRACT:
                raise ValueError('Embedded metadata has a different source data schema')
            collection = metadata.get('contract')
            if collection and (collection.get('data_contract') != DATA_CONTRACT or
                    collection.get('state_dim') != 16 or collection.get('action_dim') != 16 or
                    collection.get('state_semantics') != STATE_SEMANTICS or
                    collection.get('action_semantics') != ACTION_SEMANTICS):
                raise ValueError('Embedded collection contract is not native16 measured-state data')
            self.state = _native16_lowdim(handle, 'state')
            self.action = _native16_lowdim(handle, 'action')
            if self.state.shape != self.action.shape or self.state.shape[1:] != (16,) or not len(self.state):
                raise ValueError('Native16 source state/action alignment invalid')
            n = len(self.state)
            measured, commanded, timestamps = handle['measured_state'][:], handle['target_command'][:], handle['physics_time'][:]
            if measured.shape != (n+1, 16) or commanded.shape != (n+1, 16) or timestamps.shape != (n+1,):
                raise ValueError('Full native16 measured/target/timestamp audit streams required')
            if not np.isfinite(measured).all() or not np.isfinite(commanded).all() or not np.isfinite(timestamps).all():
                raise ValueError('Nonfinite native16 audit stream')
            if not np.array_equal(measured[:-1], self.state) or not np.array_equal(commanded[1:], self.action):
                raise ValueError('State must be measured[t]; action must target the next control interval')
            if timestamps[0] < 0 or not np.allclose(np.diff(timestamps), .1, rtol=0, atol=1e-7):
                raise ValueError('Native16 audit stream must retain uniform actual 10Hz timestamps')
            if len(handle['rule_evidence']) != n+1:
                raise ValueError('Physical rule evidence must cover every observation including terminal')
            if handle.attrs.get('training_frame_count') != n or handle.attrs.get('source_observation_count') != n+1:
                raise ValueError('Declared frame counts differ from archived measurements')
            if 'terminal_observation' not in handle:
                raise ValueError('Native16 terminal observation must be preserved without an invented action')
            self.has_terminal = True
            terminal = handle['terminal_observation']
            if not np.array_equal(terminal['measured_state'][:], measured[-1]) or not np.array_equal(terminal['target_command'][:], commanded[-1]) or float(terminal['physics_time'][()]) != timestamps[-1]:
                raise ValueError('Terminal measured state/command/time differs from audit stream')
            self.timestamps = timestamps.copy()
            self.statistics = json.loads(handle.attrs['episode_statistics'])
            if self.statistics != episode_statistics(self.state, self.action):
                raise ValueError('Episode 16D measured-state / target-action statistics differ from data')
            if set(handle['vision']) != set(POLICY_VIEWS) or set(terminal['vision']) != set(POLICY_VIEWS):
                raise ValueError('Exactly three onboard archive views are required')
            self.image_shapes = {}
            for view in POLICY_VIEWS:
                if len(handle[f'vision/{view}/colors']) != n:
                    raise ValueError('Source camera frame count mismatch')
                shape = tuple(handle[f'vision/{view}/shape'][:])
                if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) <= 0 or tuple(terminal[f'vision/{view}/shape'][:]) != shape:
                    raise ValueError('Native onboard image dimensions / terminal shape invalid')
                self.image_shapes[view] = shape
        self.observation_count = len(self.state)+1

    def __len__(self):
        return len(self.state)

    def rgb(self, frame):
        if not 0 <= frame < self.observation_count:
            raise IndexError(frame)
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._stat:
            raise ValueError('Source changed during training')
        with h5py.File(self.path, 'r') as handle:
            prefix = 'terminal_observation/' if frame == len(self) else ''
            result = []
            for view in POLICY_VIEWS:
                ds = handle[f'{prefix}vision/{view}/colors']
                image = _codec().decode_image_bit(ds[:] if prefix else ds[frame])
                if image.dtype != np.uint8 or image.shape != self.image_shapes[view]:
                    raise ValueError('Archived camera RGB payload has wrong type/shape')
                result.append(image)
            return result


class FrozenDinoProducer:
    """Load once under the GPU lock; no parameters receive gradients."""
    def __init__(self, wrapper, encoder):
        self.wrapper = wrapper
        self.wrapper.vision_encoder = encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, episode, frame):
        pixels = np.stack([normalize_image(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))
                           for rgb in episode.rgb(frame)])
        feats, pooled = self.wrapper.encode(torch.from_numpy(pixels)[None])
        array = torch.stack(feats, dim=2)[0].to(torch.bfloat16).contiguous()
        bits = array.view(torch.int16).cpu().numpy().view(np.uint16)
        pool = pooled[0].float().reshape(3, 2048).cpu().numpy()
        buffer = BytesIO()
        np.savez(buffer, feat=bits, pool=pool)
        return buffer.getvalue()


class FrameFeatureStore:
    def __init__(self, cache, producer, *, encoder_sha256):
        self.cache, self.producer = cache, producer
        self.encoder_sha256 = encoder_sha256

    def get(self, episode, frame):
        identity = {"source_sha256": episode.sha256, "encoder_sha256": self.encoder_sha256,
                    "checkpoint_sha256": self.encoder_sha256,
                    "preprocess_sha256": PREPROCESS_SHA256, "frame_index": int(frame),
                    "feature_schema_version": PREPROCESS["feature_schema"]}
        content = self.cache.read_bytes(identity, lambda: self.producer(episode, frame))
        with np.load(BytesIO(content), allow_pickle=False) as value:
            feat, pool = value["feat"].copy(), value["pool"].copy()
        if feat.shape != FEAT_SHAPE or feat.dtype != np.uint16:
            raise ValueError("Cached dense feature shape/codec mismatch")
        if pool.shape != POOL_SHAPE or pool.dtype != np.float32:
            raise ValueError("Cached history pool shape/dtype mismatch")
        if np.any((feat & 0x7f80) == 0x7f80) or not np.isfinite(pool).all():
            raise ValueError("Cached features contain nonfinite values")
        return feat, pool


class HybridTaskDataset:
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    state_semantics = STATE_SEMANTICS
    action_semantics = ACTION_SEMANTICS

    def __init__(self, task, rows, store, *, proprio_lags=(2, 4, 8, 16, 32),
                 memory_lags=(4, 16), chunk=32, future_offset=32, condition=None):
        self.task, self.store = task, store
        self.episodes = [SourceEpisode(row) for row in rows]
        if not self.episodes:
            raise ValueError("Empty split")
        self.ends = np.cumsum([len(ep) for ep in self.episodes]).tolist()
        self.proprio_lags, self.memory_lags = list(proprio_lags), list(memory_lags)
        self.chunk, self.future_offset = int(chunk), int(future_offset)
        self.condition = None if condition is None else np.asarray(condition, np.float32)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep_index = bisect_right(self.ends, index)
        frame = index - (self.ends[ep_index - 1] if ep_index else 0)
        episode = self.episodes[ep_index]
        if self.condition is None or self.condition.shape != (1024,):
            raise ValueError("TRAIN-only task prototype must be installed before sampling")
        feat, current_pool = self.store.get(episode, frame)
        pools = [current_pool]
        for old in history_frames(frame, 0, self.memory_lags):
            pools.append(current_pool if old == frame else self.store.get(episode, old)[1])
        state_frames = [frame] + history_frames(frame, 0, self.proprio_lags)
        valid = min(self.chunk, len(episode) - frame)
        actions = np.zeros((self.chunk, ACTION_DIM), np.float32)
        actions[:valid] = episode.action[frame:frame + valid]
        future_valid = frame + self.future_offset < episode.observation_count
        future = (self.store.get(episode, frame + self.future_offset)[0][:, -1, 5:]
                  if future_valid else np.zeros((3, 196, 1024), np.uint16))
        return {"state": episode.state[state_frames].copy(), "action_sequence": actions,
                "action_valid_mask": np.arange(self.chunk) < valid, "feats": feat,
                "pool": np.stack(pools, axis=1), "future_feats": future.copy(),
                "future_valid_mask": np.bool_(future_valid), "task_cond": self.condition.copy(),
                "phase": np.int64(-1), "active_hand": np.int64(-1), "dist": np.float32(0),
                "task": self.task, "episode": episode.path.name, "anchor": frame}

    def train_prototype(self):
        if any(ep.row.get("split") != "train" for ep in self.episodes):
            raise ValueError("Prototype must use TRAIN only")
        values = []
        for episode in self.episodes:
            first = self.store.get(episode, 0)[0][0, -1, 0]
            last = self.store.get(episode, episode.observation_count - 1)[0][0, -1, 0]
            to_float = lambda x: torch.from_numpy(x.view(np.int16).copy()).view(torch.bfloat16).float().numpy()
            values.append(to_float(last).astype(np.float64) - to_float(first).astype(np.float64))
        self.condition = np.mean(values, axis=0).astype(np.float32)
        if not np.isfinite(self.condition).all() or not np.linalg.norm(self.condition):
            raise ValueError("Invalid or zero task prototype")
        return self.condition.copy()


def grouped_split(rows, *, seed=20261008, counts=(80, 10, 10)):
    """Assign whole source families, including all randomized relatives."""
    families = sorted({str(row["family_id"]) for row in rows})
    if len(families) != sum(counts):
        raise ValueError(f"Expected {sum(counts)} source families; got {len(families)}")
    order = np.random.default_rng(seed).permutation(families).tolist()
    labels = {}
    start = 0
    for split, count in zip(("train", "val", "heldout"), counts):
        for family in order[start:start + count]:
            labels[family] = split
        start += count
    result = {split: [] for split in ("train", "val", "heldout")}
    for row in rows:
        split = labels[str(row["family_id"])]
        result[split].append({**row, "split": split})
    return result
