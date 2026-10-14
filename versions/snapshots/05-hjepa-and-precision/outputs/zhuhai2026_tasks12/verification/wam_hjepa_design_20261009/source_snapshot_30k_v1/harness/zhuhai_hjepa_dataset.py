"""Opt-in auxiliary targets over the unchanged native16 dataset.

Future RGB features and raw action residuals are training-only auxiliary fields.
No DINO model is constructed here; FrameFeatureStore retains the bf16 bit codec.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import numpy as np

from harness.robotwin_training_data import FEAT_SHAPE
from harness.zhuhai_dataset import HybridTaskDataset

IDM_STEPS = 4
IDM_DIM = 16
SHORT_HORIZON = 8
PATCH_SHAPE = (3, 196, 1024)
IDM_STATISTICS_SCHEMA = "zhuhai-hjepa-idm-statistics-v1"
_STAT_BLOCK_WINDOWS = 4096


def _last_real_observation(episode):
    """Never infer a terminal image merely from an oversized observation count."""
    last = len(episode) if getattr(episode, "has_terminal", False) else len(episode) - 1
    return min(last, int(episode.observation_count) - 1)


def _valid_idm_windows(episode):
    return max(0, min(len(episode), _last_real_observation(episode)) - IDM_STEPS + 1)


def _raw_residuals(action, current_state):
    # Cast BEFORE subtraction; never subtract separately normalized operands.
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = (np.asarray(action, dtype=np.float32)
                      - np.asarray(current_state, dtype=np.float32))
    except FloatingPointError as exc:
        raise ValueError("IDM residuals must be representable as float32") from exc
    if not np.isfinite(result).all():
        raise ValueError("IDM residuals must be finite")
    return result


class HybridHjepaTaskDataset(HybridTaskDataset):
    """Base-compatible sampling with independently optional 8-step/IDM fields.

    feats is the current frame; idm_future_feats contains ONLY t+1..t+4.
    Measured q_t is already state[0] (batched: state[:, 0]). The IDM target
    is float32 u[t+k] - q[t], k=0..3, not a future state. Disabled auxiliaries
    introduce neither new keys nor extra feature reads.
    """

    def __init__(self, *args, multi_horizon=False, use_idm=False, **kwargs):
        for name, value in (("multi_horizon", multi_horizon), ("use_idm", use_idm)):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool")
        super().__init__(*args, **kwargs)
        self.multi_horizon = bool(multi_horizon)
        self.use_idm = bool(use_idm)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not (self.multi_horizon or self.use_idm):
            return sample
        episode = self.episodes[bisect_right(self.ends, index)]
        frame = sample["anchor"]
        if self.multi_horizon:
            future_frame = frame + SHORT_HORIZON
            valid = future_frame <= _last_real_observation(episode)
            sample["future_feats_8"] = (
                self.store.get(episode, future_frame)[0][:, -1, 5:].copy()
                if valid else np.zeros(PATCH_SHAPE, dtype=np.uint16))
            sample["future_valid_mask_8"] = np.bool_(valid)
        if self.use_idm:
            valid = frame < _valid_idm_windows(episode)
            if valid:
                target = _raw_residuals(
                    episode.action[frame:frame + IDM_STEPS], sample["state"][0])
                future = np.stack([self.store.get(episode, frame + k)[0]
                                   for k in range(1, IDM_STEPS + 1)])
            else:
                # No partial reads, clamping, or crossing into the next episode.
                future = np.zeros((IDM_STEPS, *FEAT_SHAPE), dtype=np.uint16)
                target = np.zeros((IDM_STEPS, IDM_DIM), dtype=np.float32)
            sample["idm_future_feats"] = future
            sample["idm_target_raw"] = target
            sample["idm_valid_mask"] = np.bool_(valid)
        return sample


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def _train_entries(train_datasets):
    """Shared split preflight: no arrays, features or archive reads."""
    if isinstance(train_datasets, Mapping):
        datasets = list(train_datasets.values())
    elif isinstance(train_datasets, Sequence) and not isinstance(train_datasets, (str, bytes)):
        datasets = list(train_datasets)
    else:
        raise TypeError("Expected a dataset sequence or flat {task: TRAIN dataset} mapping")
    entries = []
    for dataset in datasets:
        if not hasattr(dataset, "episodes") or not hasattr(dataset, "task"):
            raise TypeError("Expected datasets, not rows or an all-splits mapping")
        for episode in dataset.episodes:
            if not isinstance(episode.row, Mapping) or episode.row.get("split") != "train":
                raise ValueError("IDM statistics require explicitly TRAIN episodes only")
            entries.append((str(dataset.task), episode))
    if not entries:
        raise ValueError("No TRAIN episodes supplied for IDM statistics")
    return entries


def fit_idm_statistics(
    train_datasets: Sequence[HybridTaskDataset] | Mapping[str, HybridTaskDataset],
) -> dict:
    """Fit auxiliary scales from explicitly TRAIN datasets, never features.

    Accept a dataset sequence or flat {task: dataset} mapping, NOT rows or an
    all-splits mapping. Check every split before reading ANY numeric array;
    VAL/heldout entries are rejected rather than silently filtered.

    Every complete four-action window with a real endpoint supplies four labels.
    Subtract in float32; merge population variance using bounded float64 Welford
    moments. Action range is max-min over ALL supplied TRAIN commands, including
    short episodes. Normalize only raw / scale16, with no centering or clipping.

    Sources bind admitted file hashes AND the float32 arrays actually used.
    train_sources_sha256 hashes the canonical train_sources list; statistics_sha256
    hashes the whole canonical result before its own key is added. Do not modify
    base statistics, VTT, conditions, archives, or feature stores.
    """
    entries = _train_entries(train_datasets)

    # Only after the complete split preflight may state/action/len be accessed.
    entries.sort(key=lambda item: (item[0], str(item[1].path), str(item[1].sha256)))
    sources, seen_paths, seen_hashes = [], set(), set()
    count, valid_windows = 0, 0
    mean, m2 = np.zeros(IDM_DIM, np.float64), np.zeros(IDM_DIM, np.float64)
    action_min = np.full(IDM_DIM, np.inf, np.float64)
    action_max = np.full(IDM_DIM, -np.inf, np.float64)
    for task, episode in entries:
        path, source_hash = str(Path(episode.path).resolve()), episode.sha256
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(c not in "0123456789abcdef" for c in source_hash)):
            raise ValueError("TRAIN source requires a verified SHA256")
        for key in ("sha256", "data_sha256"):
            expected = episode.row.get(key)
            if expected is not None and expected != source_hash:
                raise ValueError("TRAIN source hash differs from the admitted row")
        if episode.row.get("path") is not None and str(Path(episode.row["path"]).resolve()) != path:
            raise ValueError("TRAIN source path differs from the admitted row")
        if path in seen_paths or source_hash in seen_hashes:
            raise ValueError("Duplicate TRAIN source path/content in IDM statistics")
        seen_paths.add(path)
        seen_hashes.add(source_hash)
        arrays = []
        for name in ("state", "action"):
            value = np.asarray(getattr(episode, name))
            if (value.ndim != 2 or value.shape[1] != IDM_DIM or not len(value)
                    or value.dtype.kind not in "fiu" or not np.isfinite(value).all()
                    or (np.abs(value.astype(np.float64)) > np.finfo(np.float32).max).any()):
                raise ValueError("Finite representable native16 (N,16) TRAIN state/action required")
            arrays.append(value.astype(np.float32, copy=False))
        state, action = arrays
        if state.shape != action.shape or len(state) != len(episode):
            raise ValueError("TRAIN state/action must be aligned (N,16) arrays")
        windows = _valid_idm_windows(episode)
        valid_windows += windows
        action_min = np.minimum(action_min, action.min(axis=0).astype(np.float64))
        action_max = np.maximum(action_max, action.max(axis=0).astype(np.float64))
        for lo in range(0, windows, _STAT_BLOCK_WINDOWS):
            hi = min(lo + _STAT_BLOCK_WINDOWS, windows)
            indices = np.arange(lo, hi)[:, None] + np.arange(IDM_STEPS)
            labels = _raw_residuals(action[indices], state[lo:hi, None, :])
            values = labels.reshape(-1, IDM_DIM).astype(np.float64)
            n, block_mean = len(values), values.mean(axis=0)
            centered = values - block_mean
            block_m2 = np.einsum("ij,ij->j", centered, centered)
            delta, total = block_mean - mean, count + n
            m2 += block_m2 + delta * delta * (count * n / total)
            mean += delta * (n / total)
            count = total
        sources.append({
            "task": task, "path": path, "sha256": source_hash, "split": "train",
            "family_id": (str(episode.row["family_id"])
                          if episode.row.get("family_id") is not None else None),
            "seed": int(episode.row["seed"]) if episode.row.get("seed") is not None else None,
            "domain": (str(episode.row["domain"])
                       if episode.row.get("domain") is not None else None),
            "frames": len(state), "observation_count": int(episode.observation_count),
            "has_terminal": bool(getattr(episode, "has_terminal", False)),
            "valid_windows": windows,
            "state_sha256": _array_digest(state), "action_sha256": _array_digest(action),
        })
    if not count:
        raise ValueError("No valid four-step TRAIN windows with real endpoints")
    std = np.sqrt(np.maximum(m2 / count, 0.0))
    action_range = action_max - action_min
    scale = np.maximum(np.maximum(std, 0.01 * action_range), 1e-6)
    result = {
        "schema": IDM_STATISTICS_SCHEMA, "fit_split": "train", "heldout_used": False,
        "horizon_steps": IDM_STEPS, "action_dim": IDM_DIM,
        "label_formula": "u[t+k]-q[t], k=0..3; fixed measured q[t]",
        "target_dtype": "float32", "accumulator_dtype": "float64", "variance": "population",
        "normalization": "raw_div_scale_no_center_no_clip", "center": False, "clip": False,
        "scale_floor": {"train_action_range_multiplier": 0.01, "absolute": 1e-6},
        "scale16": scale.tolist(), "std16": std.tolist(),
        "action_range16": action_range.tolist(),
        "valid_windows": valid_windows, "valid_label_vectors": count,
        "train_sources": sources, "train_sources_sha256": _digest(sources),
    }
    result["statistics_sha256"] = _digest(result)
    return result
