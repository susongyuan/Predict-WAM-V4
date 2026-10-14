"""Native 14D RoboTwin cached input contract (no legacy segment/mask/shift).

RobotwinCachedDataset(manifest_or_path, split, cache_root, proprio_lags,
                      memory_lags, chunk_size=32, future_offset=32)
returns numpy arrays, compatible with torch default_collate:
  state (len(proprio_lags)+2,14), action_sequence (chunk_size,14) float32;
  action_valid_mask (chunk_size,) bool; feats (3,3,201,1024) uint16 BF16 bits;
  pool (3,len(memory_lags)+2,2048) float32 [current,lags...,episode0];
  future_feats (3,196,1024) uint16, future_valid_mask scalar bool;
  task_cond (1024,) zero float32, phase/active_hand=-1, dist=0.
All frames, including odd frames, are cached. Invalid future/actions are zeros.
Recommended 15Hz lags: proprio=[2,4,8,16,32], memory=[4,16].

Cache schema v1: <out>/<config SHA256>/cache_config.json describes immutable
encoder/preprocessing/source identity. Each <episode stem>-<HDF SHA256>/ has
meta.json plus feat.npy (T,3,3,201,1024) uint16 and pool.npy (T,3,2048) float32.
No decoded-image/JPEG/frame-index cache. cache_root accepts the config directory
or its parent (exactly one matching config required). Every requested episode
must have matching identity, complete metadata AND correctly sized typed files.

fit_train_statistics(..., out_dir) returns the two written absolute paths as
{normalization, temporal_stats}. normalization.json has robotwin.state/action
min/max/mean/std and top-level action_repr=absolute; temporal_stats.json has
views, lags, proprio_diff_std (len(proprio_lags)+1,14), memory_diff_std
(3,len(memory_lags)+1,2048), and train-only provenance. Population std >=1e-3.
"""
from bisect import bisect_right
import hashlib
import inspect
import io
import json
from operator import index
from pathlib import Path

import cv2
import h5py
import numpy as np

from harness.robotwin_dataset import POLICY_VIEWS, _codec, _lowdim, validate_split
from harness.wam_behavior import HEAD_CAMERA, IMAGENET_MEAN, IMAGENET_STD, normalize_image
from harness.wam_v3 import LEFT_WRIST, RIGHT_WRIST, history_frames, pool_from_hidden

ROOT = Path(__file__).resolve().parents[1]
ENCODER_PATH = ROOT / "models/dinov3-vitl16-pretrain-lvd1689m"
MODEL_VIEWS = [HEAD_CAMERA, LEFT_WRIST, RIGHT_WRIST]
MODEL_TO_CAMERA = dict(zip(MODEL_VIEWS, POLICY_VIEWS))
FEAT_LAYERS = [-12, -8, -4]
FEAT_SHAPE = (3, 3, 201, 1024)
POOL_SHAPE = (3, 2048)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json_bytes(value))
    tmp.replace(path)


def load_manifest(value):
    if not isinstance(value, dict):
        value = json.loads(Path(value).read_text())
    return validate_split(value)


def preprocess_rgb(rgb):
    """Official RGB decoding is upstream; square INTER_AREA then CPU normalization."""
    return normalize_image(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))


def cache_identity(manifest):
    """Compute local-only config identity; never load weights or touch a GPU."""
    root = Path(manifest["data_root"]).resolve()
    source = next((p / "SOURCE.json" for p in (root, *root.parents)
                   if (p / "SOURCE.json").is_file()), None)
    if source is None:
        raise ValueError(f"SOURCE.json not found above {root}")
    provenance = json.loads((ENCODER_PATH / "DOWNLOAD_PROVENANCE.json").read_text())
    prep = {"implementation": inspect.getsource(preprocess_rgb) + inspect.getsource(normalize_image),
            "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
            "opencv": cv2.__version__, "numpy": np.__version__, "size": [224, 224],
            "interpolation": "INTER_AREA", "normalize_device": "cpu", "color": "RGB"}
    return {"schema_version": 1, "source": json.loads(source.read_text()),
            "source_sha256": sha256_file(source), "data_root": str(root),
            "encoder_path": str(ENCODER_PATH), "encoder_revision": provenance["revision"],
            "encoder_sha256": provenance["files"]["model.safetensors"]["sha256"],
            "encoder_config_sha256": sha256_file(ENCODER_PATH / "config.json"),
            "codec_sha256": sha256_file(_codec().__file__), "preprocess_sha256": digest(prep),
            "preprocess": prep, "views": MODEL_VIEWS, "model_to_camera": MODEL_TO_CAMERA,
            "feat_layers": FEAT_LAYERS, "num_register_tokens": 4,
            "feature_shape": list(FEAT_SHAPE), "feature_dtype": "uint16-bfloat16-bits",
            "pool_shape": list(POOL_SHAPE), "pool_dtype": "float32",
            "pool_implementation_sha256": hashlib.sha256(inspect.getsource(pool_from_hidden).encode()).hexdigest(),
            "pool_layer": -4, "frame_stride": 1}


def episode_identity(path, config_id):
    """Only read the selected HDF, streaming its SHA; no images decoded here."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        n = int(f["state/left_arm_joint_states"].shape[0])
    if n <= 0:
        raise ValueError(f"empty episode {path}")
    return {"config_id": config_id, "episode": path.name, "hdf_sha256": sha256_file(path), "length": n}


def episode_directory(root, identity):
    return Path(root) / (Path(identity["episode"]).stem + "-" + identity["hdf_sha256"])


def payload_specs(identity):
    n = identity["length"]
    return {"feat": ((n, *FEAT_SHAPE), np.dtype("uint16")),
            "pool": ((n, *POOL_SHAPE), np.dtype("float32"))}


def npy_bytes(shape, dtype):
    """Exact logical bytes including NumPy's v1 header, not filesystem block allocation."""
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": np.dtype(dtype).str, "fortran_order": False, "shape": tuple(shape)})
    return header.tell() + int(np.prod(shape)) * np.dtype(dtype).itemsize


def completed_meta(identity):
    return {"complete": True, "identity": identity}


def validate_payload(root, identity, *, load_features=True):
    """Fail closed even when complete=true. Stats map only pool, inspect feat header."""
    directory = episode_directory(root, identity)
    try:
        if json.loads((directory / "meta.json").read_text()) != completed_meta(identity):
            raise ValueError("identity mismatch or incomplete metadata")
        arrays = {}
        for name, (shape, dtype) in payload_specs(identity).items():
            path = directory / f"{name}.npy"
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.shape != shape or arr.dtype != dtype or not arr.flags.c_contiguous:
                raise ValueError(f"wrong {name} shape/dtype/layout: {arr.shape}/{arr.dtype}")
            if path.stat().st_size != npy_bytes(shape, dtype):
                raise ValueError(f"wrong {name} file size")
            if name != "feat" or load_features:
                arrays[name] = arr
        return arrays
    except (OSError, ValueError, EOFError, KeyError) as exc:
        raise ValueError(f"invalid cache {directory}: {exc}") from exc


def resolve_cache(manifest, cache_root):
    identity = cache_identity(manifest)
    config_id = digest(identity)
    root = Path(cache_root).resolve()
    if not (root / "cache_config.json").exists():
        root = root / config_id
    try:
        if json.loads((root / "cache_config.json").read_text()) != identity:
            raise ValueError("config identity mismatch")
    except OSError as exc:
        raise ValueError(f"missing cache config {root}") from exc
    return root, config_id


def _lags(values):
    values = [index(v) for v in values]
    if any(v <= 0 for v in values) or len(set(values)) != len(values):
        raise ValueError("lags must be unique positive integers")
    return values


def read_lowdim(path):
    """Statistics/training never decode RGB. Reader's finite/order validation reused."""
    with h5py.File(path, "r") as f:
        state, action = _lowdim(f, "state"), _lowdim(f, "action")
    if not len(state) or state.shape != action.shape:
        raise ValueError(f"inconsistent state/action {path}")
    return state, action


class RobotwinCachedDataset:
    def __init__(self, split_manifest, split, cache_root, proprio_lags, memory_lags,
                 chunk_size=32, future_offset=32):
        self.manifest = load_manifest(split_manifest)
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown split {split}")
        self.split, self.views = split, list(MODEL_VIEWS)
        self.proprio_lags, self.memory_lags = _lags(proprio_lags), _lags(memory_lags)
        self.chunk_size, self.future_offset = index(chunk_size), index(future_offset)
        if self.chunk_size <= 0 or self.future_offset <= 0:
            raise ValueError("chunk_size/future_offset must be positive")
        self.cache_root, self.config_id = resolve_cache(self.manifest, cache_root)
        self.episodes, self.identities, self.lowdim = [], [], []
        for name in self.manifest["splits"][split]:
            path = Path(self.manifest["data_root"]) / name
            identity = episode_identity(path, self.config_id)
            validate_payload(self.cache_root, identity, load_features=False)
            state, action = read_lowdim(path)
            if len(state) != identity["length"]:
                raise ValueError("HDF frame count changed")
            self.episodes.append(name)
            self.identities.append(identity)
            self.lowdim.append((state, action))
        self.ends = np.cumsum([len(s) for s, _ in self.lowdim]).tolist()
        self._mapped = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mapped"] = {}  # worker opens its own read-only maps
        return state

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, item):
        item = index(item)
        if not 0 <= item < len(self):
            raise IndexError(item)
        ep = bisect_right(self.ends, item)
        frame = item - (self.ends[ep - 1] if ep else 0)
        if ep not in self._mapped:
            self._mapped[ep] = validate_payload(self.cache_root, self.identities[ep])
        arrays = self._mapped[ep]
        state, action = self.lowdim[ep]
        pframes = [frame] + history_frames(frame, 0, self.proprio_lags)
        mframes = [frame] + history_frames(frame, 0, self.memory_lags)
        valid = min(self.chunk_size, len(action) - frame)
        actions = np.zeros((self.chunk_size, 14), np.float32)
        actions[:valid] = action[frame:frame + valid]
        future_valid = frame + self.future_offset < len(state)
        future = (arrays["feat"][frame + self.future_offset, :, -1, 5:].copy() if future_valid
                  else np.zeros((3, 196, 1024), np.uint16))
        feats = arrays["feat"][frame].copy()
        pool = arrays["pool"][mframes].transpose(1, 0, 2).copy()
        # BF16 infinities/NaNs have all-one exponent; integer bit checks avoid a float copy.
        if (not np.isfinite(pool).all() or np.any((feats & 0x7f80) == 0x7f80)
                or np.any((future & 0x7f80) == 0x7f80)):
            raise ValueError("nonfinite cached features/pool")
        return {"state": state[pframes].copy(), "action_sequence": actions,
                "action_valid_mask": np.arange(self.chunk_size) < valid,
                "feats": feats, "pool": pool, "future_feats": future,
                "future_valid_mask": np.bool_(future_valid), "task_cond": np.zeros(1024, np.float32),
                "phase": -1, "active_hand": -1, "dist": np.float32(0),
                "episode": self.episodes[ep], "anchor": frame, "contract_start": 0}


class RunningMoments:
    """Float64 parallel-Welford population moments with finite input/output guards."""
    def __init__(self):
        self.n = 0

    def update(self, values):
        x = np.asarray(values, np.float64)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("empty or nonfinite statistics input")
        mean = x.mean(axis=0)
        m2 = np.sum((x - mean) ** 2, axis=0)
        lo, hi = x.min(axis=0), x.max(axis=0)
        if not self.n:
            self.mean, self.m2, self.lo, self.hi = mean, m2, lo, hi
        else:
            d = mean - self.mean
            self.m2 += m2 + d * d * self.n * len(x) / (self.n + len(x))
            self.mean += d * len(x) / (self.n + len(x))
            self.lo, self.hi = np.minimum(self.lo, lo), np.maximum(self.hi, hi)
        self.n += len(x)

    def result(self):
        if not self.n:
            raise ValueError("no TRAIN observations")
        std = np.maximum(np.sqrt(np.maximum(self.m2 / self.n, 0)), 1e-3)
        values = {"min": self.lo, "max": self.hi, "mean": self.mean, "std": std}
        if any(not np.isfinite(x).all() or (np.abs(x) > np.finfo(np.float32).max).any()
               for x in values.values()):
            raise ValueError("nonfinite/unrepresentable fitted statistics")
        return {k: v.tolist() for k, v in values.items()}


def fit_train_statistics(split_manifest, cache_root, proprio_lags, memory_lags, out_dir):
    """Fit all and only manifest TRAIN frames (40 for frozen production split).

    Only TRAIN HDF state/action values and pool payload values are sampled; no
    feature activations, validation/heldout payload or image decode is used.
    Moments merge in float64, in bounded 128-frame blocks. Provenance includes
    exact TRAIN HDF hashes, split SHA, config identity and frame count.
    """
    manifest = load_manifest(split_manifest)
    root, config_id = resolve_cache(manifest, cache_root)
    pl, ml = _lags(proprio_lags), _lags(memory_lags)
    st, act, prop, mem = (RunningMoments() for _ in range(4))
    identities = []
    for name in manifest["splits"]["train"]:
        path = Path(manifest["data_root"]) / name
        identity = episode_identity(path, config_id)
        pool = validate_payload(root, identity, load_features=False)["pool"]
        state, action = read_lowdim(path)
        identities.append(identity)
        if len(state) != identity["length"]:
            raise ValueError("HDF frame count changed")
        for lo in range(0, len(state), 128):
            hi = min(lo + 128, len(state))
            frames = range(lo, hi)
            st.update(state[lo:hi]); act.update(action[lo:hi])
            pframes = np.array([history_frames(t, 0, pl) for t in frames])
            mframes = np.array([history_frames(t, 0, ml) for t in frames])
            prop.update(state[lo:hi, None].astype(np.float64) - state[pframes].astype(np.float64))
            cur = np.asarray(pool[lo:hi], np.float64)
            old = np.asarray(pool[mframes], np.float64).transpose(0, 2, 1, 3)
            mem.update(cur[:, :, None] - old)
    provenance = {"fit_split": "train", "episodes": identities, "episode_count": len(identities),
                  "frame_count": st.n, "split_manifest_sha256": digest(manifest),
                  "cache_config_id": config_id, "variance": "population_float64_parallel_welford",
                  "history_start": "episode_frame_0", "std_floor": 1e-3}
    normalization = {"schema_version": 1, "action_repr": "absolute",
                     "robotwin": {"state": st.result(), "action": act.result()}, "provenance": provenance}
    temporal = {"schema_version": 1, "views": MODEL_VIEWS, "proprio_lags": pl, "memory_lags": ml,
                "proprio_diff_std": prop.result()["std"], "memory_diff_std": mem.result()["std"],
                "provenance": provenance}
    out_dir = Path(out_dir).resolve()
    paths = {"normalization": out_dir / "normalization.json", "temporal_stats": out_dir / "temporal_stats.json"}
    for key, data in (("normalization", normalization), ("temporal_stats", temporal)):
        if paths[key].exists() and paths[key].read_bytes() != json_bytes(data):
            raise ValueError(f"refusing to overwrite different statistics: {paths[key]}")
    for key, data in (("normalization", normalization), ("temporal_stats", temporal)):
        atomic_json(paths[key], data)
    return {k: str(v) for k, v in paths.items()}
