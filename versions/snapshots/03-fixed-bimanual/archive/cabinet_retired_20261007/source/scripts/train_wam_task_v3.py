#!/usr/bin/env python3
"""Task-isolated WAM V3 adaptation. Reuses the existing model, dataset and accepted train_loop.

No legacy Task0 approval/continuation chain is used or modified. Every invocation trains at most
5k updates of the same independent 50k schedule, then returns for externally managed physical
evaluation. --resume-source restores last/AdamW/scheduler/RNG/cursor; output must be a NEW directory.
--check-only performs CPU data/schema/provenance checks (no encoder/model/GPU). --inspect-init
loads the existing initialization on CPU and verifies exact V3 structure; this is not task success.
Large feature arrays are header/size/metadata/stat checked, with boundary rows checked for finiteness;
this is explicitly NOT a fresh full-payload hash. Sources must remain immutable during training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_wam_v3 as core

DEFAULT_CONFIG = "configs/lilawam_task090_multiview_manip_v3_50k_train.yaml"
MODE = "task_adaptation_50k"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--run-name")
    p.add_argument("--max-steps", type=int, default=5000, help="absolute completed updates; at most 5000 per invocation")
    p.add_argument("--resume-source")
    p.add_argument("--resume-source-sha256")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--inspect-init", action="store_true")
    return p.parse_args(argv)


def validate_config(cfg, args):
    d, t, c, m = cfg["dataset"], cfg["training"], cfg["common"], cfg["model"]
    task = d.get("task_id")
    if type(task) is not int or not 0 <= task <= 999:
        raise ValueError("task_id must be a nonnegative integer registered in official metadata")
    paths = {"root": ROOT / "data/behavior-1k/2026-challenge-demos"}
    data = paths["root"]
    paths.update(split_path=data / f"split_task_{task:03d}.json", stats_path=data / f"train_stats_task{task:03d}.json",
                 skill_segments=data / f"skill_segments_task{task:03d}.json", vtt_dir=ROOT / f"data/skill_vtt_task{task:03d}",
                 lowdim_cache=ROOT / ("data/lowdim_cache" if task == 13 else f"data/lowdim_cache_task{task:03d}"),
                 feature_cache=ROOT / f"data/dino_cache_task{task:03d}",
                 contract_targets=ROOT / f"data/skill_vtt_task{task:03d}/contract_targets.json")
    # Compare lexical AND resolved paths: even a canonical-name symlink may not alias another task.
    for key, path in paths.items():
        given = ROOT / d[key]
        if given.absolute() != path or given.resolve() != path:
            raise ValueError(f"task-isolated {key} must be {path}, not {given}")
    v3path = data / f"wam_v3_stats_task{task:03d}.json"
    if (ROOT / m["v3"]["stats_path"]).absolute() != v3path or v3path.resolve() != v3path:
        raise ValueError("V3 stats must have the task-isolated canonical path")
    catalog = [json.loads(line) for line in (data / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    matches = [row for row in catalog if row["task_index"] == task]
    if len(matches) != 1 or matches[0]["task_name"] != d.get("task_name"):
        raise ValueError("task_id/name differ from official metadata")
    fixed = [(d.get("scope"), "manipulation"), (d.get("cond_mode"), "contract_proto"),
             (d.get("in_segment"), True), (d.get("instance_vtt_prob"), 0.0), (d.get("frame_stride"), 2),
             (d.get("future_frame_offset"), 32), (d.get("image_size"), [224, 224]),
             (d.get("skip_unsupported_contracts"), True), (c.get("action_dim"), 23), (c.get("state_dim"), 61),
             (c.get("action_chunk_size"), 32), (c.get("proprio_len"), 7),
             (m.get("proprio_mask"), "positions_no_gripper"), (m.get("action_repr", "absolute"), "absolute"),
             (m.get("use_task_cond"), True), (m["vision_encoder"]["feat_layers"], [-12, -8, -4]),
             (m["v3"]["proprio_lags"], [4, 8, 16, 32, 64]), (m["v3"]["memory_lags"], [8, 32]),
             (t.get("max_steps"), 50000), (t.get("schedule_steps"), 50000), (t.get("save_every"), 5000),
             (t.get("validate_every"), 5000), (t.get("fine_validation_samples"), 256),
             (t.get("checkpoint_metric"), core.METRIC), (cfg["inference"].get("samples"), 1),
             (cfg["manifest"].get("task_id"), task), (cfg["manifest"].get("early_stopping"), False)]
    if any(a != b for a, b in fixed) or t.get("early_stopping") or t.get("image_augmentation") or t.get("resume"):
        raise ValueError("unsupported WAM V3 cached manipulation training contract")
    if m["views"] != ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
                      "observation.rgb.right_realsense_link_camera_0"]:
        raise ValueError("expected existing head/left/right view order")
    if not re.fullmatch(r"[0-9a-f]{64}", t.get("init_from_sha256", "")):
        raise ValueError("initialization must be explicitly hashed")
    if not re.fullmatch(r"[0-9a-f]{40}", d.get("revision", "")):
        raise ValueError("dataset requires a fixed repository revision")
    if args.num_workers < 0 or not 1 <= args.max_steps <= 50000 or args.max_steps % 5000:
        raise ValueError("absolute target must be 5000,10000,...,50000")
    if any(type(t[k]) is not int or t[k] < 1 for k in ("batch_size", "grad_accum_steps", "samples_per_epoch", "validation_samples")):
        raise ValueError("invalid positive training count")
    s = cfg["system"]
    if not (0 < s["duty_cycle"] <= 1 and 0 < s["cuda_memory_fraction"] <= 1 and 0 < s["ram_abort_percent"] <= 85
            and 0 < s["vram_abort_mib"] <= 15300 and 1 <= s["cpu_threads"] <= max(1, int((os.cpu_count() or 1) * .85))):
        raise ValueError("invalid resource limits")
    if not args.check_only and not args.inspect_init:
        if not args.run_name or not args.run_name.startswith(f"task{task:03d}_v3_50k_"):
            raise ValueError("independent taskNNN_v3_50k_ run-name required")
        if args.device != "cuda":
            raise ValueError("production adaptation is CUDA-only; use --check-only / --inspect-init for CPU")
    return task


def select_task(task):
    previous = os.environ.get("P2_TASK_ID")
    if previous is not None and previous != str(task):
        raise ValueError("P2_TASK_ID differs from config; refusing cross-task imports")
    loaded = sys.modules.get("harness.task_profile")
    if loaded is not None and loaded.TASK_ID != task:
        raise ValueError("task_profile already imported for another task")
    os.environ["P2_TASK_ID"] = str(task)
    from harness import task_profile as tp
    if tp.TASK_ID != task:
        raise ValueError("task_profile task mismatch")
    return tp


def required_paths(cfg, tp):
    d = cfg["dataset"]
    return [tp.PROVENANCE, tp.SPLIT, tp.SEGMENTS, tp.TRAIN_STATS, tp.LOWDIM / "index.json",
            core.resolve(cfg["model"]["v3"]["stats_path"]), core.resolve(d["feature_cache"]) / "cache_config.json",
            core.resolve(d["contract_targets"]), *[tp.VTT_DIR / n for n in
             ("meta.json", "prototypes.npy", "task_level.npy", "instances.npy", "instances_index.json")]]


class InputLedger(core.SourceLedger):
    """Small sources have full hashes; large DINO payloads have explicit stat-only identities."""
    def __init__(self, check=lambda: None):
        super().__init__(check)
        self.payloads = {}

    def payload(self, path):
        path = str(Path(path).resolve())
        stamp = core.stat_identity(path)
        self.payloads[path] = stamp
        self.stamps[path] = stamp


def checked_array(path, shape, dtype, ledger, *, bits=False):
    """Read header and edge rows, not the entire multi-hundred-GB feature cache."""
    before = core.stat_identity(path)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype) or arr.offset + arr.nbytes != before["size"]:
        raise ValueError(f"array shape/dtype/file length mismatch: {path}")
    if len(arr):
        edge = np.array(arr[[0, -1]], copy=True)
        value = torch.from_numpy(edge.view(np.int16)).view(torch.bfloat16) if bits else edge
        core.finite(value, str(path))
    if before != core.stat_identity(path):
        raise ValueError(f"array changed during check: {path}")
    ledger.payload(path)
    return arr


def audit_data(cfg, tp, ledger):
    """Task identity from official episode metadata, not from a guessed episode-number range."""
    import pyarrow.parquet as pq
    from scripts.cache_dino_features import manipulation_frames
    d, m = cfg["dataset"], cfg["model"]
    task = d["task_id"]
    def read(path):
        ledger.add(path)
        return json.loads(Path(path).read_text())
    provenance = read(tp.PROVENANCE)
    if any(provenance.get(k) != v for k, v in {"task_id": task, "revision": d["revision"],
                                               "repo_id": "behavior-1k/2026-challenge-demos", "chunk": tp.CHUNK}.items()):
        raise ValueError("download provenance differs from task/revision")
    split = read(tp.SPLIT)
    if split.get("task_id") != task or split.get("chunk") != tp.CHUNK:
        raise ValueError("split task/chunk mismatch")
    train, val = split["train_episode_indices"], split["validation_episode_indices"]
    if (not train or not val or any(type(e) is not int for e in train + val)
            or len(set(train + val)) != len(train) + len(val)):
        raise ValueError("empty, duplicate or overlapping split")
    meta = {}
    for path in sorted(tp.EPISODES_META.glob("*.parquet")):
        ledger.add(path)
        for row in pq.read_table(path, columns=["episode_index", "task_index", "length", "data/chunk_index", "data/file_index"]).to_pylist():
            ep = row["episode_index"]
            if ep in meta or row["task_index"] != task or row["data/chunk_index"] != task:
                raise ValueError("duplicate or cross-task official episode metadata")
            meta[ep] = row
    if set(meta) != set(train + val):
        raise ValueError("split does not cover exactly this task's official episode metadata")
    split_sha = ledger.files[str(tp.SPLIT.resolve())]
    ns, segments, index = read(tp.TRAIN_STATS), read(tp.SEGMENTS), read(tp.LOWDIM / "index.json")
    p = ns["provenance"]
    if (p.get("task_id") != task or p.get("revision") != d["revision"] or p.get("split") != "train"
            or p.get("split_sha256") != split_sha or set(p.get("episode_indices", [])) != set(train)):
        raise ValueError("normalization stats are not this task's TRAIN split")
    if (index.get("task_id") != task or index.get("revision") != d["revision"] or index.get("split_sha256") != split_sha
            or segments.get("task_id") != task or segments.get("dataset_revision") != d["revision"]
            or segments.get("split_sha256") != split_sha
            or set(map(int, index["episodes"])) != set(meta) or set(map(int, segments["episodes"])) != set(meta)):
        raise ValueError("lowdim/segments task/revision/episode inventory differs from split")
    for key, width in (("action", 23), ("state", 61)):
        lo, hi = [np.asarray(ns["behavior1k"][key][s]) for s in ("min", "max")]
        if lo.shape != (width,) or hi.shape != (width,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError("invalid action/state normalization schema")
    vtt = read(tp.VTT_DIR / "meta.json")
    if (vtt.get("task_id") != task or vtt.get("prototypes_use_validation") is not False
            or vtt.get("skill_segments_sha256") != ledger.files[str(tp.SEGMENTS.resolve())]):
        raise ValueError("VTT task/split/segments provenance mismatch")
    for name in ("prototypes.npy", "task_level.npy", "instances.npy", "instances_index.json"):
        ledger.add(tp.VTT_DIR / name)
    target = read(core.resolve(d["contract_targets"]))
    target_sha = ledger.files[str(core.resolve(d["contract_targets"]))]
    if d.get("contract_targets_sha256") not in (None, target_sha):
        raise ValueError("contract target digest mismatch")
    if (target.get("source_split") != "train" or target.get("uses_validation_vectors") is not False
            or target.get("uses_validation_scalars") is not False or set(target.get("train_episodes", [])) != set(train)):
        raise ValueError("contract targets are not this task's TRAIN-only targets")
    stats = read(core.resolve(m["v3"]["stats_path"]))
    if (stats.get("task_id") != task or stats.get("split") != "train only" or stats.get("split_sha256") != split_sha
            or stats.get("train_episodes") != len(train) or stats.get("views") != m["views"]
            or stats.get("proprio_lags") != m["v3"]["proprio_lags"] or stats.get("memory_lags") != m["v3"]["memory_lags"]
            or core.resolve(stats.get("feature_cache", "")) != core.resolve(d["feature_cache"])):
        raise ValueError("V3 history stats task/split/cache mismatch (proprio-only is insufficient)")
    for name, shape in (("proprio_diff_std", (6, 61)), ("memory_diff_std", (3, 3, 2048))):
        a = np.asarray(stats.get(name))
        if a.shape != shape or not np.isfinite(a).all() or (a <= 0).any():
            raise ValueError(f"invalid full V3 history stats: {name}")
    cache = core.resolve(d["feature_cache"])
    cache_cfg = read(cache / "cache_config.json")
    for key, want in {"task_id": task, "views": m["views"], "feat_layers": m["vision_encoder"]["feat_layers"],
                      "stride": 2, "tokens_per_image": 201, "encoder_revision": m["vision_encoder"]["revision"]}.items():
        if cache_cfg.get(key) != want:
            raise ValueError(f"DINO cache config mismatch: {key}")
    if core.resolve(cache_cfg["encoder"]) != core.resolve(m["vision_encoder"]["checkpoint_path"]):
        raise ValueError("DINO encoder path mismatch")
    # Parquet targets/reward labels are small compared with RGB/features; pin all relevant files.
    files = {tp.DATA / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet" for row in meta.values()}
    row_frames = {}
    for path in sorted(files):
        ledger.add(path)
        raw = pq.read_table(path, columns=["task_index", "episode_index", "frame_index"])
        tasks, episodes, frames_raw = [raw.column(k).to_numpy() for k in ("task_index", "episode_index", "frame_index")]
        if not np.all(tasks == task) or not set(map(int, episodes)).issubset(meta):
            raise ValueError(f"raw parquet contains cross-task rows: {path}")
        for ep in np.unique(episodes):
            row_frames.setdefault(int(ep), []).append(frames_raw[episodes == ep])
    if set(row_frames) != set(meta):
        raise ValueError("raw parquet episode inventory incomplete")
    for ep, parts in row_frames.items():
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(meta[ep]["length"])):
            raise ValueError(f"raw parquet duplicate/missing frame indices in episode {ep}")
    total_frames = 0
    for ep, row in sorted(meta.items()):
        ledger.check()
        length = row["length"]
        info = index["episodes"][str(ep)]
        seg = segments["episodes"][str(ep)]
        wanted_split = "train" if ep in train else "validation"
        if any(x.get("split") != wanted_split or x.get("length") != length for x in (info, seg)):
            raise ValueError(f"episode {ep} split/length mismatch")
        for key, width in (("action", 23), ("state", 61)):
            path = tp.LOWDIM / f"episode_{ep}_{key}.npy"
            ledger.add(path)
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.shape != (length, width) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f"invalid lowdim array: {path}")
            if core.hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() != info[f"{key}_sha256"]:
                raise ValueError(f"lowdim payload digest mismatch: {path}")
        frames = manipulation_frames(seg["segments"], length, 2)
        total_frames += len(frames)
        for view in m["views"]:
            folder = cache / view
            stem = f"episode_{ep}"
            rgb_path = tp.FRAMES / view / f"{stem}.meta.json"
            rgb = read(rgb_path)
            cm = read(folder / f"{stem}.meta.json")
            if (rgb.get("complete") is not True or rgb.get("length") != length or rgb.get("stride") != 2
                    or cm.get("complete") is not True or cm.get("episode") != ep or cm.get("view") != view
                    or cm.get("source_frame_meta_sha256") != ledger.files[str(rgb_path.resolve())]
                    or cm.get("feat_layers") != m["vision_encoder"]["feat_layers"] or cm.get("stride") != 2
                    or cm.get("rows") != len(frames) or cm.get("feat_shape") != [len(frames), 3, 201, 1024]):
                raise ValueError(f"incomplete/stale DINO source metadata: {folder / stem}")
            fp = folder / f"{stem}.frames.npy"
            ledger.add(fp)
            cached_frames = np.load(fp, allow_pickle=False)
            if cached_frames.dtype != np.int64 or not np.array_equal(cached_frames, frames):
                raise ValueError(f"DINO frame inventory mismatch: {fp}")
            checked_array(folder / f"{stem}.feat.npy", (len(frames), 3, 201, 1024), np.uint16, ledger, bits=True)
            checked_array(folder / f"{stem}.pool.npy", (len(frames), 2, 1024), np.float32, ledger)
    ledger.unchanged()
    return stats, target_sha, split, {"task_id": task, "episodes": len(meta), "frames_per_view": total_frames,
        "feature_bytes": sum(v["size"] for v in ledger.payloads.values()),
        "assurance": "metadata hashes, exact inventory/header/size, finite edge rows and immutable stat identities; NOT full feature-payload hashes"}


def build_action_model(cfg):
    from omegaconf import OmegaConf
    from harness.wam_v3 import create_v3_action_model
    enc = json.loads((core.resolve(cfg["model"]["vision_encoder"]["checkpoint_path"]) / "config.json").read_text())
    model = create_v3_action_model(OmegaConf.create(cfg), dino_hidden_size=enc["hidden_size"], num_dino_layers=3,
                                   task_cond_dim=enc["hidden_size"], patch_size=enc["patch_size"])
    return model, enc


def initialize_model(cfg, model):
    path, digest = core.resolve(cfg["training"]["init_from"]), cfg["training"]["init_from_sha256"]
    before = core.stat_identity(path)
    core.require_hash(path, digest)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("step") != cfg["training"]["initialization_step"]:
        raise ValueError("initialization checkpoint step differs from config")
    from harness.wam_v3 import load_v3_state
    notes = load_v3_state(model, checkpoint["model_state_dict"])
    if before != core.stat_identity(path):
        raise ValueError("initialization changed while loading")
    return {**notes, "source": str(path), "sha256": digest, "source_task_id": checkpoint["config"]["dataset"]["task_id"],
            "source_step": checkpoint["step"], "fresh_optimizer": True, "adaptation_start_step": 0}


def load_resume(path, cfg, device, expected_sha=None):
    path = core.resolve(path)
    digest = expected_sha or json.loads(Path(str(path) + ".sha256.json").read_text())["sha256"]
    core.require_hash(path, digest)
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = ck.get("resume_state") or {}
    required = {"optimizer_state_dict", "scheduler_state_dict", "rng", "cursor", "best", "step"}
    if (ck.get("resume_format") != "wam-v3-last-v2" or ck.get("resumable") is not True
            or state.get("safe_optimizer_boundary") is not True or not required.issubset(state)
            or ck.get("mode") != MODE or core.digest_json(ck.get("config")) != core.digest_json(cfg)
            or ck.get("initialization_sha256") != cfg["training"]["init_from_sha256"]
            or state.get("device_type") != device or state.get("step") != ck.get("step")
            or state["scheduler_state_dict"].get("last_epoch") != ck["step"]):
        raise ValueError("resume requires this Task90-style adaptation's complete safe last/config/device, not legacy/model-only")
    return ck, digest


def main(argv=None):
    args = parse_args(argv)
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(core.resolve(args.config)), resolve=True)
    task = validate_config(cfg, args)
    tp = select_task(task)
    torch.set_num_threads(min(2 if args.check_only or args.inspect_init else cfg["system"]["cpu_threads"], os.cpu_count() or 1))
    if args.inspect_init:
        model, _ = build_action_model(cfg)
        notes = initialize_model(cfg, model)
        print(json.dumps({"status": "initialization_compatible_not_training_ready", "task_id": task,
                          "parameters": sum(p.numel() for p in model.parameters()), "cuda_initialized": torch.cuda.is_initialized(), **notes}))
        return 0
    missing = [str(p) for p in required_paths(cfg, tp) if not p.is_file()]
    if missing:
        print(json.dumps({"status": "blocked", "task_id": task, "missing": missing}))
        return 2
    if not args.check_only:
        policy = json.loads((ROOT / "configs/behavior_environment.json").read_text())["resource_policy"]
        from scripts.gpu_lock import owner
        if not policy["gpu_execution_enabled"] or owner() != f"wam-train-{args.run_name}":
            raise ValueError("main session must acquire exclusive GPU lock as wam-train-<run-name>; no automatic GPU work")
    out = None if args.check_only else core.new_run(args.run_name)
    log = None if out is None else (out / "log.jsonl").open("x", buffering=1)
    def emit(row):
        line = json.dumps({"time": time.time(), "mode": MODE, **row}, allow_nan=False)
        if log:
            log.write(line + "\n")
        print(line, flush=True)
    guard = core.ResourceGuard(cfg["system"], "cpu" if args.check_only else args.device, emit)
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for s in old:
        signal.signal(s, lambda n, _f: setattr(guard, "signal", n))
    try:
        guard.check(force=True)
        ledger = InputLedger(guard.check)
        ledger.add(core.resolve(args.config))
        for path in (Path(__file__), ROOT / "scripts/train_wam_v3.py", ROOT / "scripts/train_lilawam_behavior.py",
                     ROOT / "scripts/cache_dino_features.py", ROOT / "harness/wam_v3.py", ROOT / "harness/wam_behavior.py",
                     ROOT / "harness/task_profile.py", ROOT / "harness/orchestrator.py", ROOT / "harness/vtt_monitor.py",
                     ROOT / "harness/r1pro_kinematics.py", ROOT / "configs/r1pro_eef_offsets.json"):
            ledger.add(path)
        from harness.r1pro_kinematics import URDF
        ledger.add(URDF)
        ledger.add(core.resolve(cfg["model"]["vision_encoder"]["checkpoint_path"]) / "config.json")
        for path in sorted((ROOT / "third_party/LiLa-WAM/models").rglob("*.py")):
            ledger.add(path)
        stats, target_sha, split, assurance = audit_data(cfg, tp, ledger)
        from harness.wam_v3 import WamV3Dataset, WamV3Wrapper, reward_frames
        from harness.wam_behavior import hold_baseline
        from scripts.train_lilawam_behavior import motion_sampling_weights
        dsargs = dict(feature_cache=core.resolve(cfg["dataset"]["feature_cache"]), views=cfg["model"]["views"],
                      proprio_lags=cfg["model"]["v3"]["proprio_lags"], memory_lags=cfg["model"]["v3"]["memory_lags"],
                      contract_targets=str(core.resolve(cfg["dataset"]["contract_targets"])), contract_targets_sha256=target_sha,
                      seed=cfg["training"]["seed"], rewards=reward_frames())
        train, val = WamV3Dataset("train", **dsargs), WamV3Dataset("validation", **dsargs)
        core.assert_splits(train, val, split)
        for ds in (train, val):
            # Do not silently remove a rare manipulation contract from the new task.
            unsupported = [r for r in ds.contract_filter["unsupported_segments"]
                           if ds.skills.episodes[r["episode"]]["segments"][r["segment"]]["skill"] != "move to"]
            if unsupported or ds.anchor_filter["empty_episodes"]:
                raise ValueError(f"missing manipulation targets or empty episodes in {ds.split}: {unsupported[:3]}")
            sample = ds[0]
            for key in ("state", "pool", "task_cond", "action_sequence"):
                core.finite(sample[key], f"{ds.split} {key}")
        t = cfg["training"]
        plan = core.fixed_plan(val, t["validation_seed"], t["validation_samples"], t["fine_validation_samples"])
        if args.check_only:
            emit({"status": "data_ready_cpu_only", "task_id": task, "train_anchors": len(train), "validation_anchors": len(val),
                  "initialization_not_checked": True, "physical_success_not_evaluated": True, "cache": assurance})
            return 0
        ck, resume_sha = load_resume(args.resume_source, cfg, args.device, args.resume_source_sha256) if args.resume_source else (None, None)
        start = ck["step"] if ck else 0
        if not 0 < args.max_steps - start <= 5000:
            raise ValueError("one invocation must add 1..5000 updates; absolute targets never reset the 50k schedule")
        seed = t["seed"]
        random.seed(seed); np.random.seed(seed % 2**32); torch.random.default_generator.manual_seed(seed)
        weights, sampler_stats = motion_sampling_weights(train.anchors, **t["motion_weighting"])
        core.finite(weights, "sampling weights")
        if (weights <= 0).any():
            raise ValueError("nonpositive sampling weights")
        def loader(epoch):
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), t["samples_per_epoch"], replacement=True,
                                            generator=torch.Generator().manual_seed(core.noise_seed(seed, epoch, 0, "train-sampler-epoch")))
            return DataLoader(train, batch_size=t["batch_size"], sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=False, prefetch_factor=1 if args.num_workers else None,
                              generator=torch.Generator().manual_seed(core.noise_seed(seed, epoch, 0, "loader-workers")))
        contract = {"config": cfg, "sources": ledger.files, "feature_payload_stats": ledger.payloads,
                    "validation_plan": plan, "train_anchors": train.anchors, "sampler": "weighted_epoch_seed_v1", "workers": args.num_workers}
        contract_sha = core.digest_json(contract)
        if ck and ck.get("training_contract_sha256") != contract_sha:
            raise ValueError("resume task/data/model/source/split/feature identities changed")
        metadata = {"config": cfg, "mode": MODE, "run_name": args.run_name, "task_id": task,
                    "result_kind": "independent_task_50k_wam_executor", "initialization_sha256": t["init_from_sha256"],
                    "training_contract_sha256": contract_sha, "v3_stats": stats, "cond_mode": "contract_proto",
                    "contract_targets_sha256": target_sha, "inference_samples": 1,
                    "action_execution_horizon": cfg["common"]["action_execution_horizon"], "early_stopping": False,
                    "resume_source": str(core.resolve(args.resume_source)) if ck else None, "resume_source_sha256": resume_sha}
        core.atomic_json(out / "validation_plan.json", plan)
        core.atomic_json(out / "train_labels.json", {"coverage_by_skill": core.anchor_records(train)[1], "label_report": train.label_report})
        core.atomic_json(out / "manifest.json", {**metadata, "source_sha256": ledger.files, "feature_payload_stats": ledger.payloads,
                          "cache_assurance": assurance, "sampler": sampler_stats, "train_episodes": train.episodes,
                          "validation_episodes": val.episodes, "requested_updates": args.max_steps, "steps_start_at": start,
                          "schedule_steps": 50000, "fresh_optimizer": ck is None, "args": vars(args), "offline_proxy_only": True})
        metadata["manifest_sha256"] = core.sha256(out / "manifest.json")
        core.atomic_json(out / "status.json", {"status": "preflight", "task_id": task, "step": start})
        torch.cuda.set_per_process_memory_fraction(cfg["system"]["cuda_memory_fraction"])
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        model, enc = build_action_model(cfg)
        if ck:
            model.load_state_dict(ck["model_state_dict"], strict=True)
            notes = {"resume_strict": True, "step": start}
        else:
            notes = initialize_model(cfg, model)
        wrapper = WamV3Wrapper(model, views=cfg["model"]["views"], stats=stats,
                               norm_stats_path=tp.TRAIN_STATS, proprio_mask=cfg["model"]["proprio_mask"], train_cfg=t,
                               vision_encoder=None, feat_layers=cfg["model"]["vision_encoder"]["feat_layers"],
                               num_register_tokens=enc["num_register_tokens"], device=args.device).to(args.device)
        core.atomic_json(out / "initialization.json", notes)
        optimizer, scheduler = core.make_optimizer(wrapper.action_model, t)
        from harness.r1pro_kinematics import R1ProKinematics
        kin = R1ProKinematics()
        best_artifact = core.inherit_best_checkpoint(out, ck, core.resolve(args.resume_source)) if ck else None
        def validate(step):
            ledger.unchanged()
            report, rows = core.evaluate(wrapper, val, plan, kin=kin, hold_fn=hold_baseline, lambdas=t["auxiliary_loss"],
                                         num_steps=cfg["common"]["num_inference_steps"], check=guard.check, throttle=guard.throttle)
            core.atomic_json(out / f"validation_{step:06d}.json", {"step": step, "report": report, "rows": rows})
            return report
        def save(tag, step, extra):
            nonlocal best_artifact
            if tag == "last":
                extra = {**extra, "best_checkpoint": best_artifact}
            path = core.save_checkpoint(out, tag, wrapper.action_model, step, {**metadata, **extra})
            if tag == "best":
                best_artifact = {"path": path, "sha256": json.loads(Path(path + ".sha256.json").read_text())["sha256"], "step": step}
            emit({"event": "checkpoint", "tag": tag, "step": step, "path": path})
            return path
        result = core.train_loop(wrapper, optimizer, scheduler, loader, max_steps=args.max_steps, accum=t["grad_accum_steps"],
                                 grad_clip=t["grad_clip_norm"], lambdas=t["auxiliary_loss"], validate_fn=validate, save_fn=save,
                                 emit=emit, check=guard.check, throttle=guard.throttle, resources=lambda: guard.latest,
                                 every=5000, resume_state=ck["resume_state"] if ck else None)
        core.atomic_json(out / "summary.json", {**result, "offline_proxy_only": True, "automatic_next_segment": False})
        core.atomic_json(out / "status.json", {"status": "segment_complete", "step": result["completed_updates"],
                                               "awaiting_external_physical_evaluation": True, "automatic_next_segment": False})
        return 0
    except BaseException as exc:
        if out:
            core.atomic_json(out / "status.json", {"status": "aborted", "reason": type(exc).__name__, "error": str(exc)})
        if isinstance(exc, core.StopRequested):
            return 128 + guard.signal if guard.signal else 3
        raise
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
        if log:
            log.close()


if __name__ == "__main__":
    sys.exit(main())
