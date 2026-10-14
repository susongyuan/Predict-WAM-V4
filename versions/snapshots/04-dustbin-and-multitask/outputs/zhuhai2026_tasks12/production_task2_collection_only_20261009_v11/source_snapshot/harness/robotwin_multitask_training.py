"""TRAIN-only multi-task native14 WAM V3 + residual VTT training primitives.

No encoder, simulator, download, or legacy checkpoint is modified here. Task
conditions are fixed TRAIN DINO head CLS endpoint-minus-start prototypes; their
residual projection and the shared action network are trainable. Statistics are
shared across all TRAIN frames, never fitted to VAL/native evaluation episodes.

Task manifest: {"tasks": [{"task": "name", "split_manifest": "...",
                          "feature_cache": "..."}, ...]}.
Relative paths are workspace-relative, not task-manifest-relative.
load_multitask_policy(path, task, device) returns (wrapper, condition, contract).
The caller MUST reset its V3Context(state_dim=14) for each task and each episode and use
native14 absolute actions, the checkpoint views/lags and the returned condition.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import json
from operator import index
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Sampler

from harness import robotwin_training_data as data
from harness.robotwin_vtt_adapter import ResidualVTTProjection
from harness.wam_v3 import WamV3Wrapper, create_v3_action_model, history_frames, load_retargeted_v3_state

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "robotwin-multitask-native14-vtt-v1"
BANK_FORMAT = "robotwin-multitask-train-vtt-v1"


def resolve(path):
    return (ROOT / path).resolve()


def close_maps(arrays):
    for value in arrays.values():
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


class BoundedCachedDataset(data.RobotwinCachedDataset):
    """Inherited episode-local slicing, explicit LRU close, copied output only.

    The bound is per task/split, not per frame. num_workers=0 is enforced by the
    CLI: 5 tasks * 2 splits * max_mapped_episodes (default 1) live episode maps.
    Lowdim state/action arrays remain resident and are small relative to DINO.
    """
    def __init__(self, *args, max_mapped_episodes=1, **kwargs):
        self.max_mapped_episodes = index(max_mapped_episodes)
        if self.max_mapped_episodes < 1:
            raise ValueError("max_mapped_episodes must be positive")
        super().__init__(*args, **kwargs)
        if not self.ends:
            raise ValueError(f"empty {self.split} dataset")
        self._mapped = OrderedDict()

    def __getstate__(self):
        result = super().__getstate__()
        result["_mapped"] = OrderedDict()
        return result

    def __getitem__(self, item):
        item = index(item)
        if not 0 <= item < len(self):
            raise IndexError(item)
        ep = bisect_right(self.ends, item)
        if ep not in self._mapped:
            while len(self._mapped) >= self.max_mapped_episodes:
                _, arrays = self._mapped.popitem(last=False)
                close_maps(arrays)
        result = super().__getitem__(item)
        self._mapped.move_to_end(ep)
        return result

    def close(self):
        for arrays in self._mapped.values():
            close_maps(arrays)
        self._mapped.clear()


def load_task_specs(value):
    if not isinstance(value, (list, dict)):
        value = json.loads(resolve(value).read_text())
    rows = value.get("tasks") if isinstance(value, dict) else value
    if not isinstance(rows, list) or not 2 <= len(rows) <= 50:
        raise ValueError("tasks must list 2..50 tasks (5 is the evaluation target)")
    result, names = [], set()
    for row in rows:
        name = row["task"]
        if not isinstance(name, str) or not name or name in names or any(c in name for c in "/\\:"):
            raise ValueError("task names must be unique nonempty path-free strings")
        names.add(name)
        result.append({"task": name, **{k: str(resolve(row[k])) for k in ("split_manifest", "feature_cache")}})
    return result


def build_datasets(specs, cfg, *, max_mapped_episodes=1):
    v3 = cfg["model"]["v3"]
    datasets = {"train": {}, "val": {}}
    reference, roots = None, set()
    fields = ("encoder_sha256", "encoder_config_sha256", "preprocess_sha256", "codec_sha256",
              "views", "feat_layers", "feature_shape", "pool_shape", "pool_implementation_sha256",
              "num_register_tokens", "frame_stride")
    try:
        for row in specs:
            for split in datasets:
                ds = BoundedCachedDataset(row["split_manifest"], split, row["feature_cache"],
                    v3["proprio_lags"], v3["memory_lags"], chunk_size=cfg["common"]["action_chunk_size"],
                    future_offset=cfg["dataset"]["future_frame_offset"], max_mapped_episodes=max_mapped_episodes)
                datasets[split][row["task"]] = ds
            ds = datasets["train"][row["task"]]
            root = str(Path(ds.manifest["data_root"]).resolve())
            if root in roots:
                raise ValueError("different tasks cannot alias the same data_root")
            roots.add(root)
            cache = json.loads((ds.cache_root / "cache_config.json").read_text())
            if cache["source"]["task"] != row["task"]:
                raise ValueError(f"task/cache SOURCE mismatch: {row['task']}")
            identity = {k: cache[k] for k in fields}
            if reference is not None and reference != identity:
                raise ValueError("all tasks must share frozen encoder, preprocessing and cache geometry")
            reference = identity
            if cache["views"] != cfg["model"]["views"]:
                raise ValueError("cache views differ from model")
        return datasets
    except BaseException:
        for split in datasets.values():
            for ds in split.values():
                ds.close()
        raise


def train_provenance(datasets):
    if any(ds.split != "train" for ds in datasets.values()):
        raise ValueError("fitting may only use TRAIN datasets")
    return {"fit_split": "train", "history_start": "episode_frame_0", "heldout_used": False,
        "tasks": {task: {"episodes": ds.identities, "episode_count": len(ds.identities),
            "frame_count": len(ds), "split_manifest_sha256": data.digest(ds.manifest),
            "cache_config_id": ds.config_id} for task, ds in datasets.items()}}


def fit_shared_statistics(datasets, check=lambda: None):
    provenance = train_provenance(datasets)
    st, act, prop, mem = (data.RunningMoments() for _ in range(4))
    first = next(iter(datasets.values()))
    for ds in datasets.values():
        if (ds.proprio_lags, ds.memory_lags, ds.views) != (first.proprio_lags, first.memory_lags, first.views):
            raise ValueError("shared statistics require identical history/view contracts")
        for identity, (state, action) in zip(ds.identities, ds.lowdim):
            arrays = data.validate_payload(ds.cache_root, identity, load_features=False)
            try:
                pool = arrays["pool"]
                for lo in range(0, len(state), 128):
                    check()
                    hi = min(lo + 128, len(state))
                    pframes = np.array([history_frames(t, 0, ds.proprio_lags) for t in range(lo, hi)])
                    mframes = np.array([history_frames(t, 0, ds.memory_lags) for t in range(lo, hi)])
                    st.update(state[lo:hi]); act.update(action[lo:hi])
                    prop.update(state[lo:hi, None].astype(np.float64) - state[pframes].astype(np.float64))
                    cur = np.asarray(pool[lo:hi], np.float64)
                    old = np.asarray(pool[mframes], np.float64).transpose(0, 2, 1, 3)
                    mem.update(cur[:, :, None] - old)
            finally:
                close_maps(arrays)
    provenance.update(frame_count=st.n, weighting="uniform_all_TRAIN_frames",
                      variance="population_float64_parallel_welford", std_floor=1e-3)
    return {"normalization": {"schema_version": 1, "action_repr": "absolute",
                "robotwin": {"state": st.result(), "action": act.result()}, "provenance": provenance},
            "temporal_stats": {"schema_version": 1, "views": first.views,
                "proprio_lags": first.proprio_lags, "memory_lags": first.memory_lags,
                "proprio_diff_std": prop.result()["std"], "memory_diff_std": mem.result()["std"],
                "provenance": provenance}}


def build_train_bank(datasets, check=lambda: None):
    """All TRAIN demos contribute; no VAL/heldout endpoint is opened here."""
    provenance = train_provenance(datasets)
    conditions, records = {}, {}
    for task, ds in datasets.items():
        deltas, records[task] = [], []
        for identity in ds.identities:
            check()
            arrays = data.validate_payload(ds.cache_root, identity)
            try:
                if identity["length"] < 2:
                    raise ValueError("VTT needs at least two frames per TRAIN episode")
                bits = np.array(arrays["feat"][[0, -1], 0, -1, 0], copy=True).view(np.int16)
                cls = torch.from_numpy(bits).view(torch.bfloat16).float().numpy()
                if not np.isfinite(cls).all():
                    raise ValueError("nonfinite TRAIN endpoint CLS")
                deltas.append(cls[1].astype(np.float64) - cls[0])
                records[task].append({"episode": identity["episode"], "frames": [0, identity["length"] - 1]})
            finally:
                close_maps(arrays)
        conditions[task] = np.mean(deltas, axis=0).astype(np.float32).tolist()
    bank = {"format": BANK_FORMAT, "source_split": "train", "provenance": provenance,
        "representation": "mean TRAIN DINO hidden_states[-4] head CLS endpoint minus frame0",
        "conditions": conditions, "records": records, "inference": "explicit_task_prototype_no_eval_demonstrations"}
    validate_bank(bank, list(datasets))
    return bank


def validate_bank(bank, tasks=None):
    if (bank.get("format") != BANK_FORMAT or bank.get("source_split") != "train"
            or bank.get("provenance", {}).get("fit_split") != "train"):
        raise ValueError("condition bank must be TRAIN-only with provenance")
    if tasks is not None and list(bank["conditions"]) != list(tasks):
        # JSON sorting may reorder keys, so task membership, not insertion order, is immutable.
        if set(bank["conditions"]) != set(tasks):
            raise ValueError("bank task membership mismatch")
    values = []
    for task, value in bank["conditions"].items():
        x = np.asarray(value, np.float32)
        if (x.shape != (1024,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-6
                or np.linalg.norm(x - x.mean()) < 1e-6):
            raise ValueError(f"zero/nonfinite/LayerNorm-degenerate VTT for {task}")
        for other, old in values:
            # Distinct raw vectors can collapse under LayerNorm (e.g. an offset).
            a = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
            b = (old - old.mean()) / np.sqrt(old.var() + 1e-5)
            if np.allclose(a, b, rtol=1e-5, atol=1e-6):
                raise ValueError(f"indistinguishable VTT task prototypes: {task}, {other}")
        values.append((task, x))
    if not values:
        raise ValueError("empty VTT bank")


class MultitaskDataset:
    def __init__(self, datasets, bank):
        self.datasets, self.tasks = datasets, list(datasets)
        validate_bank(bank, self.tasks)
        self.conditions = {task: np.asarray(bank["conditions"][task], np.float32) for task in self.tasks}
        self.ends = np.cumsum([len(datasets[t]) for t in self.tasks]).tolist()

    def __len__(self):
        return self.ends[-1]

    def locate(self, item):
        item = index(item)
        if not 0 <= item < len(self):
            raise IndexError(item)
        task_index = bisect_right(self.ends, item)
        return self.tasks[task_index], item - (self.ends[task_index - 1] if task_index else 0)

    def __getitem__(self, item):
        task, local = self.locate(item)
        result = self.datasets[task][local]
        result.update(task=task, task_cond=self.conditions[task].copy(),
                      episode=f"{task}:{result['episode']}")
        return result


class UniformTaskSampler(Sampler):
    """Replacement sampling: P(task)=1/K, P(frame|task)=1/N_task.

    Dedicated generator, deterministic by seed/epoch; never global torch RNG.
    Only integer indices are generated; no materialized weight vector or maps.
    """
    def __init__(self, lengths, num_samples, seed):
        self.lengths = [index(v) for v in lengths]
        self.num_samples, self.seed = index(num_samples), index(seed)
        if not self.lengths or min(self.lengths) < 1 or self.num_samples < 1:
            raise ValueError("sampler needs positive lengths/sample count")
        self.offsets = np.cumsum([0] + self.lengths[:-1]).tolist()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            task = int(rng.integers(len(self.lengths)))
            yield self.offsets[task] + int(rng.integers(self.lengths[task]))


def install_joint_projection(model):
    if isinstance(model.task_cond_proj, ResidualVTTProjection):
        raise ValueError("VTT projection is already installed")
    # Do not call the old adapter helper: its default hidden=768 is hard-coded.
    model.requires_grad_(True)
    model.task_cond_proj = ResidualVTTProjection(model.task_cond_proj, hidden=model.hidden_dim)
    model.task_cond_proj.residual.requires_grad_(True)
    return model.task_cond_proj


def build_wrapper(cfg, statistics, *, device="cpu", state_dict=None, initialization=None):
    from omegaconf import OmegaConf
    model = create_v3_action_model(OmegaConf.create(cfg), dino_hidden_size=1024,
                                  num_dino_layers=3, task_cond_dim=1024, patch_size=16)
    notes = {"initialization": "new_model"}
    if initialization is not None:
        sd = initialization["model_state_dict"]
        if sd["action_proj.weight"].shape == model.action_proj.weight.shape:
            model.load_state_dict(sd, strict=True)
            notes = {"initialization": "strict_native14_WAM_V3", "source_step": initialization.get("step")}
        else:
            notes = {"initialization": "retargeted_WAM_V3", **load_retargeted_v3_state(model, sd)}
    install_joint_projection(model)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        notes = {"initialization": "strict_multitask_checkpoint"}
    # Legacy wrapper accepts a path. A private temporary bridge lets a checkpoint
    # be self-contained without introducing another persistent stats convention.
    with tempfile.TemporaryDirectory(prefix="robotwin-multitask-stats-") as directory:
        path = Path(directory) / "normalization.json"
        data.atomic_json(path, statistics["normalization"])
        wrapper = WamV3Wrapper(model, views=cfg["model"]["views"], stats=statistics["temporal_stats"],
            norm_stats_path=path, normalization_key="robotwin", proprio_mask="native_all14",
            proprio_keep_override=np.ones(14), train_cfg=cfg["training"], vision_encoder=None,
            feat_layers=(-12, -8, -4), num_register_tokens=4, device=device).to(device)
    return wrapper, notes


def gradient_evidence(wrapper):
    """Record actual backward paths, not just requires_grad flags."""
    projection = wrapper.action_model.task_cond_proj
    if not isinstance(projection, ResidualVTTProjection) or not projection.enabled:
        raise ValueError("enabled residual VTT projection required")
    if wrapper.vision_encoder is not None:
        raise ValueError("cached trainer must not construct a DINO encoder")
    def grad_norm(parameter):
        return 0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm())
    evidence = {"vtt_projection_grad_norm": grad_norm(projection.residual.weight),
                "shared_action_grad_norm": grad_norm(wrapper.action_model.action_proj.weight),
                "frozen_base_has_grad": any(p.grad is not None for p in projection.base.parameters()),
                "dino_encoder_loaded": False, "dino_trainable_parameters": 0}
    if evidence["frozen_base_has_grad"] or any(p.requires_grad for p in projection.base.parameters()):
        raise ValueError("zero-condition base must stay frozen")
    if not all(np.isfinite(evidence[k]) for k in ("vtt_projection_grad_norm", "shared_action_grad_norm")):
        raise FloatingPointError("nonfinite joint gradient")
    return evidence


def load_multitask_policy(checkpoint_path, task, device="cpu"):
    """Verified self-contained load. Unknown tasks fail; no zero-condition fallback.

    Returns wrapper (eval), numpy float32 condition (1024,), immutable contract.
    Load from this new API, NOT the single-task worker's strict old checkpoint loader.
    """
    from scripts.train_robotwin_multitask_wam import load_checkpoint
    checkpoint = load_checkpoint(resolve(checkpoint_path), resume=False)
    contract = checkpoint["contract"]
    validate_bank(contract["condition_bank"], [row["task"] for row in contract["tasks"]])
    if task not in contract["condition_bank"]["conditions"]:
        raise ValueError(f"unknown checkpoint task {task!r}")
    wrapper, _ = build_wrapper(contract["config"], contract["statistics"], device=device,
                               state_dict=checkpoint["model_state_dict"])
    wrapper.eval()
    condition = np.asarray(contract["condition_bank"]["conditions"][task], np.float32).copy()
    return wrapper, condition, contract
