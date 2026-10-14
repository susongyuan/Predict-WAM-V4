#!/usr/bin/env python3
"""Native 14D rigid RoboTwin pilot; cached features only, never encoder/simulator.

Default --mode plan performs CPU provenance checks without constructing a model.
--mode smoke defaults to two updates (batch 1); --mode train is explicit, with the
unaltered 50k schedule. --max-steps may stop a train invocation at an absolute
boundary; --resume continues checkpoint_last in the SAME directory. Smoke cannot
be promoted into train. CUDA requires a matching externally held --gpu-lock-owner.
Only TRAIN fits statistics; fixed VAL action errors are diagnostic, never selection.
No goal understanding, lid-opening, deformable, physical-success or system claims.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Import-safe numerical/checkpoint primitives only. Never call the legacy CLI,
# validate_config, train_loop, data/label builders, or its fine_eef_m checkpoint.
from scripts.train_wam_v3 import (BatchStream, ResourceGuard, StopRequested,
    capture_rng, cpu_tree, isolated_rng, make_optimizer, restore_rng)

DEFAULT_CONFIG = "configs/fixed_bimanual/robotwin_place_cans_wam.yaml"
SPLIT_PATH = "outputs/validation/robotwin_dataset_20261004/split_manifest.json"
SPLIT_SHA = "7d02319e8b9ce4cd883b1e6e9a7e4ed68b3462c63cda18211ce1c43e1bb3f8bc"
INIT_PATH = "outputs/training/task090_v3_50k_20260930_s050000/checkpoint_step_050000.pt"
INIT_SHA = "159290c2d08e12cc06c19eb41ea464d4914d1278e28e43ae5135c133532707ef"
FORMAT = "robotwin-native14-v1"


def resolve(path):
    return (ROOT / path).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def file_identity(path, content=False):
    path = Path(path).resolve()
    s = path.stat()
    out = {"path": str(path), "size": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino}
    if content:
        out["sha256"] = sha256(path)
    return out


def validate_recipe(cfg):
    """This entrypoint has one architecture/recipe, not a legacy task selector."""
    expected = {
        "common.action_dim": 14, "common.state_dim": 14, "common.action_chunk_size": 32,
        "common.proprio_len": 7, "common.num_inference_steps": 10,
        "dataset.frame_stride": 1,
        "dataset.image_size": [224, 224], "dataset.future_frame_offset": 32,
        "dataset.cond_mode": "zero_single_task",
        "model.num_registers": 2, "model.use_task_cond": True, "model.proprio_mask": "native_all14",
        "model.vision_encoder.feat_layers": [-12, -8, -4],
        "model.vision_encoder.include_cls_register": True, "model.vision_encoder.fusion_mode": "concat",
        "model.vision_encoder.concat.proj_type": "linear", "model.vision_encoder.concat.pre_norm": True,
        "model.vision_encoder.concat.out_dim": None,
        "model.future_feat.enabled": True, "model.future_feat.target_layer": -4,
        "model.future_feat.num_decoder_layers": 4, "model.future_feat.num_heads": 4,
        "model.action_expert.depth": 12, "model.action_expert.hidden_size": 768,
        "model.action_expert.num_heads": 8, "model.action_expert.vlm_adapter_num_queries": 64,
        "model.action_expert.adapter_depth": 4, "model.v3.phase_hidden": 512,
        "model.v3.proprio_lags": [2, 4, 8, 16, 32], "model.v3.memory_lags": [4, 16],
        "training.init_from_sha256": INIT_SHA, "training.seed": 20261004,
        "training.validation_seed": 20261004, "training.batch_size": 16,
        "training.grad_accum_steps": 2, "training.max_steps": 50000, "training.schedule_steps": 50000,
        "training.learning_rate": 5e-5, "training.lr_min": 5e-6, "training.warmup_steps": 2000,
        "training.betas": [0.9, 0.99], "training.weight_decay": 0.01,
        "training.grad_clip_norm": 1.0, "training.time_mu": 0.0, "training.time_sigma": 1.0,
        "training.auxiliary_loss": {"future": 0.5, "gate_kl": 0.0, "phase_ce": 0.0, "log_dist": 0.0},
        "training.save_every": 5000, "training.validate_every": 5000,
        "training.validation_samples": 512,
        "system.cpu_abort_percent": 85, "system.ram_abort_percent": 85,
        "system.vram_abort_mib": 15300,
    }
    for key, wanted in expected.items():
        value = cfg
        for part in key.split("."):
            value = value[part]
        if value != wanted:
            raise ValueError(f"fixed pilot recipe {key}: {value!r} != {wanted!r}")
    task = cfg["dataset"]["task_name"]
    fingerprint = cfg["dataset"]["split_sha256"]
    if not isinstance(task, str) or not task or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in task):
        raise ValueError("invalid task identifier")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("dataset requires an explicit frozen split SHA256")
    if resolve(cfg["training"]["init_from"]) != resolve(INIT_PATH):
        raise ValueError("only the pinned 50k model-only initialization is permitted")
    if not 1 <= cfg["system"]["cpu_threads"] <= 8:
        raise ValueError("cpu_threads must be in [1,8]")
    if not 0 < cfg["system"]["cuda_memory_fraction"] <= 1:
        raise ValueError("invalid cuda memory fraction")
    resume_every = cfg["training"].get("resume_save_every", 1000)
    if type(resume_every) is not int or not 1 <= resume_every <= 5000:
        raise ValueError("resume_save_every must be an integer in [1,5000]")


def initialization_receipt(cfg):
    """Reuse existing SHA receipt; deliberately do not repeatedly hash 888MB weights.

    This proves receipt/path/size consistency, not a new full content hash. Inputs
    are trusted local immutable artifacts; stat identity is bound into the run.
    """
    path = resolve(cfg["training"]["init_from"])
    receipt_path = path.with_name(path.name + ".sha256.json")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("sha256") != INIT_SHA or receipt.get("step") != 50000
            or receipt.get("model_only") is not True or receipt.get("size") != path.stat().st_size):
        raise ValueError("initialization receipt does not match the pinned model")
    return {"weights": file_identity(path), "receipt": file_identity(receipt_path, True),
            "sha256": INIT_SHA, "assurance": "existing SHA receipt + path/size/stat; no fresh large-weight hash"}


def preflight(cfg):
    from harness.robotwin_training_data import MODEL_VIEWS, RobotwinCachedDataset, cache_identity, load_manifest
    validate_recipe(cfg)
    if cfg["model"]["views"] != list(MODEL_VIEWS):
        raise ValueError("native three-view order differs")
    split = resolve(cfg["dataset"]["split_manifest"])
    if sha256(split) != cfg["dataset"]["split_sha256"]:
        raise ValueError("frozen split bytes changed")
    if cache_identity(load_manifest(split))["source"].get("task") != cfg["dataset"]["task_name"]:
        raise ValueError("configured task differs from dataset source receipt")
    source = initialization_receipt(cfg)
    ec = json.loads((resolve(cfg["model"]["vision_encoder"]["checkpoint_path"]) / "config.json").read_text())
    if (ec["hidden_size"], ec["patch_size"], ec["num_register_tokens"]) != (1024, 16, 4):
        raise ValueError("encoder CONFIG is not local DINOv3 L/16 reg4")
    v3 = cfg["model"]["v3"]
    datasets = {name: RobotwinCachedDataset(split, name, resolve(cfg["dataset"]["feature_cache"]),
        v3["proprio_lags"], v3["memory_lags"], chunk_size=32, future_offset=32) for name in ("train", "val")}
    # Heldout is not opened, fitted, sampled, or used to choose a checkpoint.
    return datasets, source


def stats_contract(cfg, datasets, out, *, resume):
    from harness.robotwin_training_data import fit_train_statistics, digest as data_digest
    names = {k: str(out / f"{k}.json") for k in ("normalization", "temporal_stats")}
    v3 = cfg["model"]["v3"]
    if not resume:
        names = fit_train_statistics(resolve(cfg["dataset"]["split_manifest"]),
            resolve(cfg["dataset"]["feature_cache"]), v3["proprio_lags"], v3["memory_lags"], out)
    values = {k: json.loads(Path(p).read_text()) for k, p in names.items()}
    train = datasets["train"]
    for item in values.values():
        p = item["provenance"]
        if (p["fit_split"] != "train" or p["episodes"] != train.identities
                or p["episode_count"] != len(train.identities) or p["split_manifest_sha256"] != data_digest(train.manifest)
                or p["cache_config_id"] != train.config_id):
            raise ValueError("statistics must match exactly the frozen TRAIN/cache identities")
    temporal = values["temporal_stats"]
    if temporal["proprio_lags"] != v3["proprio_lags"] or temporal["memory_lags"] != v3["memory_lags"]:
        raise ValueError("statistics history lags differ")
    return names, values


def immutable_contract(cfg, args, datasets, source, stats_paths):
    cache_files = []
    from harness.robotwin_training_data import episode_directory
    for ds in datasets.values():
        for identity in ds.identities:
            directory = episode_directory(ds.cache_root, identity)
            for name in ("meta.json", "feat.npy", "pool.npy"):
                cache_files.append(file_identity(directory / name, name == "meta.json"))
    source_files = [Path(__file__), ROOT / "scripts/train_wam_v3.py",
        ROOT / "harness/robotwin_dataset.py", ROOT / "harness/robotwin_training_data.py",
        ROOT / "harness/wam_v3.py", ROOT / "harness/wam_behavior.py"]
    source_files += sorted((ROOT / "third_party/LiLa-WAM/models").rglob("*.py"))
    return {"format": FORMAT, "config": cfg, "config_sha256": sha256(resolve(args.config)),
        "split_sha256": sha256(resolve(cfg["dataset"]["split_manifest"])),
        "initialization": source, "cache_config_id": datasets["train"].config_id,
        "cache_config": file_identity(datasets["train"].cache_root / "cache_config.json", True),
        "episodes": {k: ds.identities for k, ds in datasets.items()}, "cache_files": cache_files,
        "cache_assurance": "validated provenance, headers, sizes, per-use finite checks; payload stats bound, not content hashes",
        "stats": {k: file_identity(p, True) for k, p in stats_paths.items()},
        "source_hashes": {str(p.resolve()): sha256(p) for p in source_files},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        "execution": {"mode": args.mode, "device": args.device,
            "batch_size": 16 if args.mode == "train" else args.smoke_batch_size,
            "accum": 2 if args.mode == "train" else 1,
            "cuda_memory_fraction": args.cuda_memory_fraction or cfg["system"]["cuda_memory_fraction"]}}


def check_cache_unchanged(contract):
    for item in [contract["cache_config"], *contract["cache_files"]]:
        if file_identity(item["path"], "sha256" in item) != item:
            raise ValueError(f"immutable cache changed: {item['path']}")


def build_wrapper(cfg, stats_paths, stats, device, checkpoint=None):
    from omegaconf import OmegaConf
    from harness.wam_v3 import WamV3Wrapper, create_v3_action_model, load_retargeted_v3_state
    model = create_v3_action_model(OmegaConf.create(cfg), dino_hidden_size=1024,
        num_dino_layers=3, task_cond_dim=1024, patch_size=16)
    if checkpoint is None:
        ck = torch.load(resolve(cfg["training"]["init_from"]), map_location="cpu", weights_only=False, mmap=True)
        if ck.get("step") != 50000 or ck.get("model_only") is not True:
            raise ValueError("initialization is not the pinned 50k model-only checkpoint")
        notes = load_retargeted_v3_state(model, ck["model_state_dict"])
        del ck
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        notes = {"strict_resume": True, "step": checkpoint["step"]}
    wrapper = WamV3Wrapper(model, views=cfg["model"]["views"], stats=stats["temporal_stats"],
        norm_stats_path=stats_paths["normalization"], normalization_key="robotwin",
        proprio_mask="native_all14", proprio_keep_override=np.ones(14), train_cfg=cfg["training"],
        vision_encoder=None, feat_layers=(-12, -8, -4), num_register_tokens=4, device=device)
    return wrapper.to(device), notes


def save_checkpoint(out, tag, wrapper, step, contract, optimizer=None, scheduler=None, cursor=None):
    """Atomic model-only checkpoints and exact safe-boundary last; never a 'best'."""
    last = tag == "last"
    if last and (optimizer is None or scheduler is None or cursor is None):
        raise ValueError("last requires complete optimizer/scheduler/cursor/RNG")
    payload = {"format": FORMAT, "step": step, "model_only": not last,
        "model_state_dict": cpu_tree(wrapper.action_model.state_dict()), "contract": contract,
        "contract_sha256": digest(contract), "checkpoint_metric": "diagnostic_only_no_selection"}
    if last:
        payload["resume_state"] = cpu_tree({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": capture_rng(wrapper.device), "cursor": cursor, "safe_optimizer_boundary": True})
    path = out / f"checkpoint_{tag}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_json(path.with_name(path.name + ".sha256.json"),
        {"sha256": sha256(path), "size": path.stat().st_size, "step": step, "model_only": not last})
    return path


def load_checkpoint(path, contract, *, resume=True):
    path = Path(path)
    receipt = json.loads(path.with_name(path.name + ".sha256.json").read_text())
    if receipt["size"] != path.stat().st_size or receipt["sha256"] != sha256(path):
        raise ValueError("checkpoint receipt mismatch before load")
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if (ck.get("format") != FORMAT or ck.get("contract") != contract
            or ck.get("contract_sha256") != digest(contract) or receipt["step"] != ck.get("step")
            or receipt["model_only"] != ck.get("model_only")):
        raise ValueError("checkpoint immutable contract differs")
    if resume:
        st = ck.get("resume_state", {})
        if (ck.get("model_only") is not False or st.get("safe_optimizer_boundary") is not True
                or not {"optimizer", "scheduler", "rng", "cursor"} <= st.keys()
                or st["scheduler"].get("last_epoch") != ck["step"]):
            raise ValueError("resume requires a complete safe checkpoint_last")
    return ck


def action_metrics(prediction, target, valid, wrapper):
    """Native order: L six joint radians, L gripper, R six radians, R gripper."""
    if prediction.shape != target.shape or prediction.shape[-1] != 14 or valid.shape != target.shape[:2]:
        raise ValueError("wrong native action/mask shape")
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError("validation needs valid action rows")
    p, t = prediction[valid].float(), target[valid].float()
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError("nonfinite valid validation actions")
    err = p - t
    norm = wrapper.normalize_action(p) - wrapper.normalize_action(t)
    out = {"normalized_mse": float(norm.square().mean()), "valid_action_rows": len(p)}
    for name, columns in (("left_arm_rad", slice(0, 6)), ("right_arm_rad", slice(7, 13)),
                          ("left_gripper_units", slice(6, 7)), ("right_gripper_units", slice(13, 14))):
        e = err[:, columns]
        out[name + "_mae"] = float(e.abs().mean())
        out[name + "_mse"] = float(e.square().mean())
    return out


def validation_indices(length, seed, count):
    if length < 1:
        raise ValueError("empty validation dataset")
    return np.random.default_rng(seed).choice(length, min(length, count), replace=False).tolist()


def validation_anchors(dataset, indices):
    """Resolve stable anchors without reading/copying multi-megabyte feature rows."""
    anchors = []
    for index in indices:
        ep = bisect_right(dataset.ends, index)
        anchors.append({"episode": dataset.episodes[ep],
                        "anchor": index - (dataset.ends[ep - 1] if ep else 0)})
    return anchors


@torch.no_grad()
def evaluate(wrapper, dataset, indices, seed, check=lambda: None):
    from harness.wam_v3 import bf16_from_bits
    if dataset.split != "val":
        raise ValueError("offline validation must only use val, never heldout/train")
    was_training = wrapper.training
    sums, rows = {}, 0
    wrapper.eval()
    try:
        with isolated_rng(seed, wrapper.device):
            for index in indices:
                check()
                item = dataset[index]
                batch = default_collate([item])
                # One deterministic noise per episode/anchor, independent of call order.
                sample_seed = int(hashlib.sha256(f"{seed}:{item['episode']}:{item['anchor']}".encode()).hexdigest()[:15], 16)
                gen = torch.Generator(device=wrapper.device).manual_seed(sample_seed)
                feats = bf16_from_bits(batch["feats"].to(wrapper.device))
                if wrapper.device.type == "cpu":
                    feats = feats.float()
                pred, _ = wrapper.sample_actions([feats[:, :, i] for i in range(3)], batch["state"], batch["pool"],
                    batch["task_cond"], num_steps=10, generator=gen)
                m = action_metrics(pred, batch["action_sequence"].to(wrapper.device),
                    batch["action_valid_mask"].to(wrapper.device), wrapper)
                n = m.pop("valid_action_rows")
                rows += n
                for key, value in m.items():
                    sums[key] = sums.get(key, 0.0) + n * value
    finally:
        wrapper.train(was_training)
    result = {k: v / rows for k, v in sums.items()}
    for name in ("left_arm_rad", "right_arm_rad", "left_gripper_units", "right_gripper_units"):
        result[name + "_rmse"] = result.pop(name + "_mse") ** 0.5
    return {"split": "val", "samples": len(indices), "valid_action_rows": rows, **result,
            "selection": "none", "physical_evaluation": False}


class Guard(ResourceGuard):
    """Existing RAM/VRAM guard plus system CPU85, external lock, and peak logging."""
    def __init__(self, cfg, device, emit, owner=None):
        super().__init__(cfg, device, emit)
        self.owner, self.peaks = owner, {}
        self.cpu_previous = self.cpu_ticks()

    @staticmethod
    def cpu_ticks():
        values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        return sum(values), values[3] + values[4]

    def check(self, force=False):
        before = self.last
        super().check(force)
        if before == self.last:
            return
        total, idle = self.cpu_ticks()
        dt, di = total - self.cpu_previous[0], idle - self.cpu_previous[1]
        self.cpu_previous = total, idle
        self.latest["cpu_percent"] = 100 * (1 - di / dt) if dt else 0.0
        for key, value in self.latest.items():
            self.peaks[key] = max(self.peaks.get(key, 0), value)
        if self.latest["cpu_percent"] >= self.cfg["cpu_abort_percent"]:
            raise StopRequested(f"CPU limit: {self.latest}")
        if self.device.type == "cuda":
            from scripts.gpu_lock import owner
            if not self.owner or owner() != self.owner:
                raise StopRequested("matching external GPU lock is required and must remain held")


def checkpoint_due(step, target, resume_every):
    """Resumable last may be more frequent; fixed 5k milestones cannot be skipped."""
    return step % 5000 == 0 or step % resume_every == 0 or step == target


def execute(cfg, args, out, datasets, source):
    torch.set_num_threads(cfg["system"]["cpu_threads"])
    def emit(value):
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), **value}, allow_nan=False) + "\n")
    guard = Guard(cfg["system"], args.device, emit, args.gpu_lock_owner)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: setattr(guard, "signal", signum))
    step, target, wrapper = 0, (args.max_steps or 50000) if args.mode == "train" else args.smoke_steps, None
    try:
        guard.check(force=True)
        if args.device == "cuda":
            fraction = args.cuda_memory_fraction or cfg["system"]["cuda_memory_fraction"]
            torch.cuda.set_per_process_memory_fraction(fraction)
            torch.cuda.reset_peak_memory_stats()
        paths, stats = stats_contract(cfg, datasets, out, resume=args.resume)
        contract = immutable_contract(cfg, args, datasets, source, paths)
        manifest_path = out / "manifest.json"
        if args.resume:
            if json.loads(manifest_path.read_text())["contract"] != contract:
                raise ValueError("run immutable config/split/cache/stats/source/execution changed")
            ck = load_checkpoint(out / "checkpoint_last.pt", contract)
            step = ck["step"]
            if target <= step:
                raise ValueError("resume target must be greater than the saved step")
        else:
            ck = None
            atomic_json(manifest_path, {"contract": contract, "contract_sha256": digest(contract),
                "claims": "native14 rigid single-task cached WAM pilot only; no goal/physical/system success"})
        random.seed(cfg["training"]["seed"])
        np.random.seed(cfg["training"]["seed"])
        torch.manual_seed(cfg["training"]["seed"])
        wrapper, notes = build_wrapper(cfg, paths, stats, args.device, ck)
        guard.check(force=True)
        emit({"event": "initialization", "step": step, "details": notes, "fresh_optimizer": not args.resume})
        optimizer, scheduler = make_optimizer(wrapper.action_model, cfg["training"])
        execution = contract["execution"]
        def loader(epoch):
            g = torch.Generator().manual_seed(cfg["training"]["seed"] + epoch)
            return DataLoader(datasets["train"], batch_size=execution["batch_size"], shuffle=True,
                generator=g, num_workers=0, drop_last=False, pin_memory=args.device == "cuda")
        stream = BatchStream(loader, args.device, cursor=ck["resume_state"]["cursor"] if ck else None, check=guard.check)
        if ck:
            optimizer.load_state_dict(ck["resume_state"]["optimizer"])
            scheduler.load_state_dict(ck["resume_state"]["scheduler"])
            restore_rng(ck["resume_state"]["rng"], args.device)
            del ck
        indices = validation_indices(len(datasets["val"]), cfg["training"]["validation_seed"],
            1 if args.mode == "smoke" else cfg["training"]["validation_samples"])
        plan = {"split": "val", "seed": cfg["training"]["validation_seed"], "indices": indices,
            "anchors": validation_anchors(datasets["val"], indices)}
        if args.resume and json.loads((out / "validation_plan.json").read_text()) != plan:
            raise ValueError("fixed validation plan changed")
        if not args.resume:
            atomic_json(out / "validation_plan.json", plan)
        initial_projection = wrapper.action_model.action_proj.weight.detach().cpu().clone() if args.mode == "smoke" else None
        wrapper.train()
        # Save only completed optimizer boundaries. An interrupted partial update
        # never overwrites last: resume replays from the previous durable boundary.
        if not args.resume:
            save_checkpoint(out, "last", wrapper, step, contract, optimizer, scheduler, stream.state())
        while step < target:
            guard.check()
            optimizer.zero_grad(set_to_none=True)
            sums = {}
            for _ in range(execution["accum"]):
                batch = stream.next()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    loss, info, _ = wrapper(batch, cfg["training"]["auxiliary_loss"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss / execution["accum"]).backward()
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        sums[key] = sums.get(key, 0.0) + value / execution["accum"]
            grad = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), cfg["training"]["grad_clip_norm"], error_if_nonfinite=True)
            guard.check()
            optimizer.step()
            scheduler.step()
            step += 1
            emit({"event": "optimizer_step", "step": step, "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad), "losses": sums, "resources": guard.latest})
            if checkpoint_due(step, target, cfg["training"].get("resume_save_every", 1000)):
                check_cache_unchanged(contract)
                save_checkpoint(out, "last", wrapper, step, contract, optimizer, scheduler, stream.state())
                if step % 5000 == 0 or args.mode == "smoke":
                    path = save_checkpoint(out, f"step_{step:06d}", wrapper, step, contract)
                    if args.mode == "smoke":
                        reloaded = load_checkpoint(path, contract, resume=False)
                        if any(not torch.equal(value.detach().cpu(), reloaded["model_state_dict"][key])
                               for key, value in wrapper.action_model.state_dict().items()):
                            raise ValueError("smoke save/reload tensor mismatch")
                        wrapper.action_model.load_state_dict(reloaded["model_state_dict"], strict=True)
                        del reloaded
                        if torch.equal(initial_projection, wrapper.action_model.action_proj.weight.detach().cpu()):
                            raise ValueError("smoke did not change action projection parameters")
                        emit({"event": "real_cached_smoke_save_reload", "step": step, "passed": True})
                    metrics = evaluate(wrapper, datasets["val"], indices, cfg["training"]["validation_seed"], guard.check)
                    atomic_json(out / f"validation_{step:06d}.json", metrics)
                    emit({"event": "offline_validation", "step": step, **metrics})
                guard.check(force=True)
            atomic_json(out / "status.json", {"status": "running", "mode": args.mode, "step": step,
                "target": target, "schedule_budget": 50000, "peak_resources": guard.peaks})
        status = {"status": "complete" if target == 50000 else "bounded_complete", "mode": args.mode,
            "step": step, "target": target, "schedule_budget": 50000, "peak_resources": guard.peaks,
            "physical_evaluation": False, "heldout_used": False}
        atomic_json(out / "status.json", status)
        return status
    except BaseException as exc:
        atomic_json(out / "status.json", {"status": "stopped" if isinstance(exc, StopRequested) else "failed",
            "step_observed": step, "error": f"{type(exc).__name__}: {exc}", "peak_resources": guard.peaks,
            "resume": "only existing checkpoint_last; incomplete updates were not saved"})
        raise
    finally:
        atomic_json(out / "peak_resources.json", guard.peaks)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--mode", choices=("plan", "smoke", "train"), default="plan")
    p.add_argument("--out", help="new run directory; same directory only with --resume")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--max-steps", type=int, help="train absolute stop target, never changes the 50k schedule")
    p.add_argument("--smoke-steps", type=int, choices=(1, 2), default=2)
    p.add_argument("--smoke-batch-size", type=int, choices=(1, 2), default=1)
    p.add_argument("--gpu-lock-owner", help="external owner; trainer never acquires/releases the lock")
    p.add_argument("--cuda-memory-fraction", type=float)
    args = p.parse_args(argv)
    if args.mode != "plan" and not args.out:
        p.error("--out is required for smoke/train")
    if args.resume and (args.mode == "plan" or not args.out):
        p.error("--resume requires smoke/train and the existing --out directory")
    if args.max_steps is not None and (args.mode != "train" or not 1 <= args.max_steps <= 50000):
        p.error("--max-steps is train-only, within [1,50000]")
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        p.error("--cuda-memory-fraction must be in (0,1]")
    if args.device == "cuda" and args.mode != "plan" and not args.gpu_lock_owner:
        p.error("CUDA execution requires --gpu-lock-owner held externally")
    return args


def main(argv=None):
    from omegaconf import OmegaConf
    args = parse_args(argv)
    cfg = OmegaConf.to_container(OmegaConf.load(resolve(args.config)), resolve=True)
    out = resolve(args.out) if args.out else None
    if out:
        if args.resume:
            if not out.is_dir() or not (out / "checkpoint_last.pt").is_file():
                raise ValueError("--resume requires an existing run with checkpoint_last.pt")
        else:
            out.mkdir(parents=True, exist_ok=False)
    try:
        datasets, source = preflight(cfg)
    except Exception as exc:
        result = {"status": "blocked_preflight", "mode": args.mode, "model_constructed": False,
            "error": f"{type(exc).__name__}: {exc}", "random_feature_fallback": False,
            "encoder_loaded": False, "simulator_started": False}
        if out:
            atomic_json(out / ("plan.json" if args.mode == "plan" else "status.json"), result)
        print(json.dumps(result, indent=2))
        return 2
    if args.mode == "plan":
        result = {"status": "cache_preflight_passed_not_a_smoke", "mode": "plan", "model_constructed": False,
            "dataset_frames": {k: len(v) for k, v in datasets.items()}, "config": cfg,
            "initialization": source, "heldout_used": False, "schedule_budget": 50000,
            "next": "Explicit real cached smoke under externally scheduled resources, then explicit train"}
        if out:
            atomic_json(out / "plan.json", result)
    else:
        result = execute(cfg, args, out, datasets, source)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
