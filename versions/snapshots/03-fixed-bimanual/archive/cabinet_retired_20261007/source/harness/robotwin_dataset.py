"""CPU-only RoboTwin v1.0 reader; no normalization, image cache or simulator.

Stored action[t] is already aligned to observation/state[t]; never shift again.
Only the three onboard cameras enter policy inputs. Chunks stay in one episode;
zero padding is NOT a target (action_valid_mask must mask the loss). All alternate
instructions belong to that episode, not independent split units. A future
trainer must fit normalization on TRAIN only, excluding val and heldout.
"""
from bisect import bisect_right
from functools import lru_cache
import importlib.util
import json
from operator import index
from pathlib import Path
import re

import h5py
import numpy as np

POLICY_VIEWS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINT_FIELDS = (("left_arm_joint_states", 6), ("left_ee_joint_states", 1),
                ("right_arm_joint_states", 6), ("right_ee_joint_states", 1))
ACTION_DIM = sum(width for _, width in JOINT_FIELDS)
SPLITS = ("train", "val", "heldout")


@lru_cache(maxsize=1)
def _codec():
    # Import only the official standalone codec, never RoboTwin's simulator.
    path = Path(__file__).resolve().parents[1] / "third_party/RoboTwin/data/decode_image_bit.py"
    spec = importlib.util.spec_from_file_location("_robotwin_image_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_rgb(value, image_shape=(240, 320, 3)):
    """Official decoder returns RGB for decoded, legacy and marked JPEG inputs."""
    image = _codec().decode_image_bit(value)
    if image.dtype != np.uint8 or image.shape != tuple(image_shape):
        raise ValueError(f"Expected uint8 RGB {tuple(image_shape)}, got {image.dtype} {image.shape}")
    return np.ascontiguousarray(image)


def _lowdim(file, group):
    arrays = []
    for name, width in JOINT_FIELDS:
        key = f"{group}/{name}"
        ds = file[key]
        if ds.ndim != 2 or ds.shape[1] != width or ds.dtype.kind not in "fiu":
            raise ValueError(f"{key}: expected numeric (T, {width}), got {ds.shape} {ds.dtype}")
        values = ds[:]
        if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError(f"{key}: nonfinite or unrepresentable values")
        arrays.append(values.astype(np.float32))
    if len({len(values) for values in arrays}) != 1:
        raise ValueError(f"{group}: inconsistent frame counts")
    return np.concatenate(arrays, axis=1)


class RoboTwinEpisode:
    """Keep small lowdim arrays in RAM; open HDF5 images only for each chunk.

    Construction validates all lowdim values, camera metadata and the first
    frame of each policy view. Later image payloads are validated when sampled.
    No persistent HDF5 handles are shared with data-loader workers.
    """
    def __init__(self, path, image_shape=(240, 320, 3), frequency=15):
        self.path = Path(path).resolve()
        self.episode_id = self.path.stem
        self.image_shape = tuple(image_shape)
        if len(self.image_shape) != 3 or self.image_shape[-1] != 3:
            raise ValueError("image_shape must be HWC RGB")
        try:
            with h5py.File(self.path, "r") as file:
                if file["data_format_version"].asstr()[()] != "v1.0":
                    raise ValueError("Unsupported data_format_version")
                self.frequency = float(file["additional_info/frequency"][()])
                if not np.isfinite(self.frequency) or self.frequency != frequency:
                    raise ValueError(f"Unexpected frequency: {self.frequency}")
                instructions = json.loads(file["instructions"].asstr()[()])
                if not isinstance(instructions, list) or not instructions or not all(
                        isinstance(text, str) and text.strip() for text in instructions):
                    raise ValueError("instructions must be a nonempty list of nonempty strings")
                self.instructions = tuple(instructions)
                self.state = _lowdim(file, "state")
                self.actions = _lowdim(file, "action")
                if self.state.shape != self.actions.shape or len(self.state) == 0:
                    raise ValueError("State/action frame counts differ or are empty")
                for view in POLICY_VIEWS:
                    camera = file[f"vision/{view}"]
                    colors = camera["colors"]
                    decoded = colors.ndim == 4 and colors.dtype == np.uint8
                    encoded = ((colors.ndim == 1 and colors.dtype.kind in "SO") or
                               (colors.ndim == 2 and colors.dtype == np.uint8))
                    if not (decoded or encoded) or colors.shape[0] != len(self):
                        raise ValueError(f"{view}: invalid colors layout or frame count")
                    if decoded and colors.shape[1:] != self.image_shape:
                        raise ValueError(f"{view}: wrong RGB shape")
                    if "shape" in camera and not np.array_equal(camera["shape"][:], self.image_shape):
                        raise ValueError(f"{view}: wrong declared image shape")
                    decode_rgb(colors[0], self.image_shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{self.path}: {exc}") from exc

    def __len__(self):
        return len(self.actions)

    def chunk(self, frame, chunk_size):
        frame, chunk_size = index(frame), index(chunk_size)
        if not 0 <= frame < len(self):
            raise IndexError(frame)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        valid = min(chunk_size, len(self) - frame)
        actions = np.zeros((chunk_size, ACTION_DIM), dtype=np.float32)
        actions[:valid] = self.actions[frame:frame + valid]
        with h5py.File(self.path, "r") as file:
            images = {view: decode_rgb(file[f"vision/{view}/colors"][frame], self.image_shape)
                      for view in POLICY_VIEWS}
        return {"episode_id": self.episode_id, "frame_index": frame,
                "instructions": self.instructions, "images": images,
                "state": self.state[frame].copy(), "actions": actions,
                "action_valid_mask": np.arange(chunk_size) < valid}


def make_split(data_root, seed=20261004, counts=(40, 5, 5)):
    """Deterministic episode-only manifest with one root and relative filenames."""
    root = Path(data_root).resolve()
    counts = tuple(index(n) for n in counts)
    if len(counts) != 3 or any(n <= 0 for n in counts):
        raise ValueError("Need positive train/val/heldout episode counts")
    paths = sorted(root.glob("episode_*.hdf5"))
    if len(paths) != sum(counts):
        raise ValueError(f"Expected {sum(counts)} episodes, found {len(paths)}")
    identities = set()
    for path in paths:
        if not re.fullmatch(r"episode_\d{7}\.hdf5", path.name) or not path.is_file():
            raise ValueError(f"Invalid episode: {path}")
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError(f"Episode must be a direct non-symlink file: {path}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise ValueError("Duplicate physical episode file")
        identities.add(identity)
    order = np.random.default_rng(index(seed)).permutation(len(paths))
    splits, start = {}, 0
    for name, count in zip(SPLITS, counts):
        splits[name] = sorted(paths[int(i)].name for i in order[start:start + count])
        start += count
    return {"schema_version": 1, "seed": index(seed), "data_root": str(root),
            "counts": dict(zip(SPLITS, counts)), "splits": splits}


def validate_split(manifest):
    """Fail closed on overlap, missing/extra/aliased episodes or changed assignment."""
    try:
        if manifest["schema_version"] != 1 or set(manifest["counts"]) != set(SPLITS):
            raise ValueError("Invalid split schema")
        expected = make_split(manifest["data_root"], manifest["seed"],
                              tuple(manifest["counts"][name] for name in SPLITS))
        if manifest != expected:
            raise ValueError("Split manifest differs from deterministic episode assignment")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid split manifest: {exc}") from exc
    return manifest


class RoboTwinDataset:
    """One split, indexed by stored observation frame (not by instruction).

    Does not apply the legacy 61D action masks or any feature normalization.
    """
    def __init__(self, manifest, split, chunk_size, **episode_kwargs):
        validate_split(manifest)
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.chunk_size = index(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        root = Path(manifest["data_root"])
        self.episodes = tuple(RoboTwinEpisode(root / name, **episode_kwargs)
                              for name in manifest["splits"][split])
        self.ends = np.cumsum([len(episode) for episode in self.episodes]).tolist()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, item):
        item = index(item)
        if not 0 <= item < len(self):
            raise IndexError(item)
        episode_index = bisect_right(self.ends, item)
        start = self.ends[episode_index - 1] if episode_index else 0
        return self.episodes[episode_index].chunk(item - start, self.chunk_size)
