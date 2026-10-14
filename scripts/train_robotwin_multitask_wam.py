#!/usr/bin/env python3
"""Explicit cached native14 multi-task WAM V3 + VTT training, never DINO training.

--config reuses the native WAM YAML architecture. Set dataset.cond_mode to
train_task_vtt; training.max_steps/schedule_steps define a NEW fixed cosine
budget (default 30000 if omitted). --max-steps is only an absolute early stop,
not a schedule change. --tasks JSON lists task/split_manifest/feature_cache.

Default --mode plan is CPU metadata-only. --mode smoke is two updates with a
fresh optimizer in a separate directory and cannot resume into --mode train.
CUDA requires the externally held pipeline --gpu-lock-owner. VAL diagnostics
never select checkpoints. No simulator, videos, downloads or old baseline edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from harness import robotwin_multitask_training as mt
from harness import robotwin_training_data as data
from scripts.train_robotwin_wam import (Guard, atomic_json, digest, evaluate, file_identity,
                                        sha256, validation_indices, validation_anchors)
from scripts.train_wam_v3 import (BatchStream, StopRequested, capture_rng, cpu_tree,
                                  make_optimizer, restore_rng)

FORMAT = mt.FORMAT


def effective_config(value):
    cfg = copy.deepcopy(value)
    t = cfg["training"]
    t.setdefault("max_steps", 30000)
    t.setdefault("schedule_steps", t["max_steps"])
    t.setdefault("save_every", 5000)
    t.setdefault("validate_every", 5000)
    t.setdefault("resume_save_every", 1000)
    t.setdefault("validation_samples", 16)
    t.setdefault("sampler_epoch_batches", 128)
    expected = {"action_dim": 14, "state_dim": 14, "action_chunk_size": 32, "num_inference_steps": 10}
    for key, wanted in expected.items():
        if cfg["common"][key] != wanted:
            raise ValueError(f"common.{key} must be {wanted}")
    if cfg["common"]["proprio_len"] != len(cfg["model"]["v3"]["proprio_lags"]) + 2:
        raise ValueError("proprio history geometry mismatch")
    if cfg["dataset"].get("cond_mode") != "train_task_vtt":
        raise ValueError("dataset.cond_mode must explicitly be train_task_vtt")
    if (cfg["dataset"]["future_frame_offset"] != 32 or cfg["dataset"].get("frame_stride", 1) != 1
            or cfg["model"]["views"] != data.MODEL_VIEWS or not cfg["model"]["use_task_cond"]
            or cfg["model"].get("proprio_mask") != "native_all14"
            or cfg["model"]["vision_encoder"]["feat_layers"] != [-12, -8, -4]):
        raise ValueError("native14/frozen DINO cache contract mismatch")
    for key in ("max_steps", "schedule_steps", "batch_size", "grad_accum_steps", "save_every",
                "validate_every", "resume_save_every", "validation_samples", "sampler_epoch_batches"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if t["schedule_steps"] != t["max_steps"]:
        raise ValueError("schedule_steps must match the new fixed multi-task budget")
    if type(t["warmup_steps"]) is not int or not 0 <= t["warmup_steps"] < t["schedule_steps"]:
        raise ValueError("warmup_steps must be within [0, schedule_steps)")
    if not 0 < t["lr_min"] <= t["learning_rate"] or not t["grad_clip_norm"] > 0:
        raise ValueError("invalid learning rates/gradient clipping")
    if any(t["auxiliary_loss"].get(k, 0) != 0 for k in ("gate_kl", "phase_ce", "log_dist")):
        raise ValueError("this dataset has no supervised phase/distance/gate labels")
    if len(t.get("init_from_sha256", "")) != 64:
        raise ValueError("initialization requires explicit SHA256")
    if not 0 < cfg["system"]["cuda_memory_fraction"] <= 1:
        raise ValueError("invalid CUDA memory fraction")
    return cfg


def preflight(cfg, specs, max_mapped_episodes=1):
    path = mt.resolve(cfg["training"]["init_from"])
    source = file_identity(path, True)
    if source["sha256"] != cfg["training"]["init_from_sha256"]:
        raise ValueError("initialization SHA256 differs")
    # File hashing and cache provenance only; no tensor weights, encoder or CUDA.
    return mt.build_datasets(specs, cfg, max_mapped_episodes=max_mapped_episodes), source


def fit_artifacts(datasets, out, resume=False, check=lambda: None):
    provenance = mt.train_provenance(datasets["train"])
    if resume:
        statistics = json.loads((out / "statistics.json").read_text())
        bank = json.loads((out / "condition_bank.json").read_text())
        for value in [*statistics.values(), bank]:
            if any(value["provenance"].get(k) != v for k, v in provenance.items()):
                raise ValueError("TRAIN-only fitting provenance changed on resume")
        mt.validate_bank(bank, datasets["train"])
    else:
        statistics = mt.fit_shared_statistics(datasets["train"], check)
        bank = mt.build_train_bank(datasets["train"], check)
        atomic_json(out / "statistics.json", statistics)
        atomic_json(out / "condition_bank.json", bank)
    return statistics, bank


def immutable_contract(cfg, args, specs, datasets, source, statistics, bank):
    files = []
    for split in datasets.values():
        for ds in split.values():
            files.append(file_identity(ds.cache_root / "cache_config.json", True))
            for identity in ds.identities:
                directory = data.episode_directory(ds.cache_root, identity)
                files += [file_identity(directory / name, name == "meta.json")
                          for name in ("meta.json", "feat.npy", "pool.npy")]
    sources = [Path(__file__), Path(mt.__file__), Path(data.__file__),
               ROOT / "scripts/train_robotwin_wam.py", ROOT / "scripts/train_wam_v3.py",
               ROOT / "harness/wam_v3.py", ROOT / "harness/wam_behavior.py",
               ROOT / "harness/robotwin_dataset.py", ROOT / "harness/robotwin_vtt_adapter.py"]
    sources += sorted((ROOT / "third_party/LiLa-WAM/models").rglob("*.py"))
    train = args.mode == "train"
    return {"format": FORMAT, "config": cfg, "tasks": specs, "initialization": source,
        "condition_bank": bank, "statistics": statistics,
        "initialization_disclosure": "warm-start WAM V3; native14 loads strictly without resetting heads; "
            "put_bottles_dustbin checkpoint has prior single-task training advantage; not from scratch or a fair baseline",
        "sampling": "uniform task with replacement; uniform frame within task; fixed TRAIN task prototypes",
        "cache_files": files, "cache_assurance": "metadata/header/size/provenance/stat binding, not payload content hashes",
        "source_hashes": {str(p.resolve()): sha256(p) for p in sources},
        "episodes": {split: {task: ds.identities for task, ds in rows.items()} for split, rows in datasets.items()},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        "execution": {"mode": args.mode, "device": args.device,
            "batch_size": cfg["training"]["batch_size"] if train else args.smoke_batch_size,
            "accum": cfg["training"]["grad_accum_steps"] if train else 1,
            "max_mapped_episodes_per_task_split": args.max_mapped_episodes,
            "num_workers": 0, "cuda_memory_fraction": args.cuda_memory_fraction or cfg["system"]["cuda_memory_fraction"]},
        "inference": {"action_repr": "absolute", "action_dim": 14, "state_dim": 14,
            "views": cfg["model"]["views"], "proprio_lags": cfg["model"]["v3"]["proprio_lags"],
            "memory_lags": cfg["model"]["v3"]["memory_lags"], "normalization": "shared_all_TRAIN_frames",
            "condition": "lookup explicit task in embedded condition_bank, no zero fallback",
            "reset_history": "every episode and every task switch; never share buffers across simultaneous episodes",
            "encoder": "frozen DINOv3-L/16 reg4 hidden layers [-12,-8,-4]",
            "checkpoint_selection": "fixed_final_budget_only_VAL_never_selects",
            "native_eval_episodes_per_task": 10, "record_video": False,
            "evaluation_only_third_view_camera": False, "retain_actions_scores_onboard_observations": True},
        "heldout_used": False, "encoder_loaded_for_training": False,
        "trainable": "shared action model + residual VTT projection; frozen original zero-condition base"}


def check_cache_unchanged(contract):
    for item in contract["cache_files"]:
        if file_identity(item["path"], "sha256" in item) != item:
            raise ValueError(f"immutable cache changed: {item['path']}")


def save_checkpoint(out, tag, wrapper, step, contract, optimizer=None, scheduler=None, cursor=None,
                    evidence=None, task_counts=None):
    last = tag == "last"
    if last and (optimizer is None or scheduler is None or cursor is None):
        raise ValueError("checkpoint_last requires complete safe-boundary continuation state")
    payload = {"format": FORMAT, "step": step, "model_only": not last,
        "model_state_dict": cpu_tree(wrapper.action_model.state_dict()),
        "contract": contract, "contract_sha256": digest(contract),
        "gradient_evidence": evidence or {}, "task_sample_counts": dict(task_counts or {}),
        "checkpoint_metric": "fixed_budget_no_VAL_selection"}
    if last:
        payload["resume_state"] = cpu_tree({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": capture_rng(wrapper.device), "cursor": cursor, "safe_optimizer_boundary": True})
    path = Path(out) / f"checkpoint_{tag}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_json(path.with_name(path.name + ".sha256.json"),
                {"sha256": sha256(path), "size": path.stat().st_size, "step": step, "model_only": not last})
    return path


def load_checkpoint(path, contract=None, *, resume=True):
    path = Path(path)
    receipt = json.loads(path.with_name(path.name + ".sha256.json").read_text())
    if receipt["size"] != path.stat().st_size or receipt["sha256"] != sha256(path):
        raise ValueError("checkpoint receipt mismatch")
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if (ck.get("format") != FORMAT or ck.get("contract_sha256") != digest(ck.get("contract"))
            or receipt["step"] != ck.get("step") or receipt["model_only"] != ck.get("model_only")
            or (contract is not None and ck["contract"] != contract)):
        raise ValueError("checkpoint immutable contract mismatch")
    if resume:
        state = ck.get("resume_state", {})
        if (ck.get("model_only") is not False or state.get("safe_optimizer_boundary") is not True
                or not {"optimizer", "scheduler", "rng", "cursor"} <= state.keys()
                or state["scheduler"].get("last_epoch") != ck["step"]):
            raise ValueError("resume requires complete safe optimizer boundary")
    return ck


def validation_plan(datasets, seed, count):
    result = {}
    for i, (task, ds) in enumerate(datasets["val"].items()):
        indices = validation_indices(len(ds), seed + i, count)
        result[task] = {"indices": indices, "anchors": validation_anchors(ds, indices), "seed": seed + i}
    return {"split": "val", "tasks": result, "selection": "none", "heldout_used": False}


def evaluate_tasks(wrapper, datasets, bank, plan, check=lambda: None):
    results = {}
    for task, row in plan["tasks"].items():
        ds = datasets["val"][task]
        # Reuse the established deterministic flow sampler/valid-mask metrics;
        # inject the SAME TRAIN prototype used for training and native inference.
        view = mt.MultitaskDataset({task: ds}, {**bank, "conditions": {task: bank["conditions"][task]}})
        view.split = "val"
        try:
            results[task] = evaluate(wrapper, view, row["indices"], row["seed"], check)
        finally:
            ds.close()
    return {"per_task": results, "macro_normalized_mse": float(np.mean([r["normalized_mse"] for r in results.values()])),
            "selection": "none", "physical_evaluation": False}


def execute(cfg, args, out, specs, datasets, source):
    torch.set_num_threads(cfg["system"]["cpu_threads"])
    def emit(value):
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), **value}, allow_nan=False) + "\n")
    guard = Guard(cfg["system"], args.device, emit, args.gpu_lock_owner)
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: setattr(guard, "signal", signum))
    budget = cfg["training"]["max_steps"]
    target = (args.max_steps or budget) if args.mode == "train" else args.smoke_steps
    step, wrapper = 0, None
    try:
        guard.check(force=True)
        if args.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction or cfg["system"]["cuda_memory_fraction"])
            torch.cuda.reset_peak_memory_stats()
        statistics, bank = fit_artifacts(datasets, out, args.resume, guard.check)
        contract = immutable_contract(cfg, args, specs, datasets, source, statistics, bank)
        if args.resume:
            if json.loads((out / "manifest.json").read_text())["contract"] != contract:
                raise ValueError("run config/data/cache/stats/bank/source/execution contract changed")
            ck = load_checkpoint(out / "checkpoint_last.pt", contract)
            step = ck["step"]
            if target <= step:
                raise ValueError("resume target must exceed saved step")
        else:
            ck = None
            atomic_json(out / "manifest.json", {"contract": contract, "contract_sha256": digest(contract)})
        t = cfg["training"]
        random.seed(t["seed"]); np.random.seed(t["seed"]); torch.manual_seed(t["seed"])
        init = None if ck else torch.load(source["path"], map_location="cpu", weights_only=False, mmap=True)
        wrapper, notes = mt.build_wrapper(cfg, statistics, device=args.device,
            state_dict=ck["model_state_dict"] if ck else None, initialization=init)
        del init
        guard.check(force=True)
        optimizer, scheduler = make_optimizer(wrapper.action_model, t)
        train = mt.MultitaskDataset(datasets["train"], bank)
        execution = contract["execution"]
        def loader(epoch):
            seed = t["seed"] + epoch
            sampler = mt.UniformTaskSampler([len(ds) for ds in datasets["train"].values()],
                execution["batch_size"] * t["sampler_epoch_batches"], seed)
            return DataLoader(train, batch_size=execution["batch_size"], sampler=sampler,
                generator=torch.Generator().manual_seed(seed), num_workers=0,
                pin_memory=args.device == "cuda", drop_last=False)
        stream = BatchStream(loader, args.device, cursor=ck["resume_state"]["cursor"] if ck else None, check=guard.check)
        evidence = dict(ck.get("gradient_evidence", {})) if ck else {}
        counts = Counter(ck.get("task_sample_counts", {})) if ck else Counter()
        if ck:
            optimizer.load_state_dict(ck["resume_state"]["optimizer"])
            scheduler.load_state_dict(ck["resume_state"]["scheduler"])
            restore_rng(ck["resume_state"]["rng"], args.device)
            del ck
        plan = validation_plan(datasets, t["validation_seed"], 1 if args.mode == "smoke" else t["validation_samples"])
        if args.resume and json.loads((out / "validation_plan.json").read_text()) != plan:
            raise ValueError("fixed validation plan changed")
        if not args.resume:
            atomic_json(out / "validation_plan.json", plan)
        emit({"event": "initialization", "step": step, "details": notes, "fresh_optimizer": not args.resume,
              "nonzero_vtt_norms": {task: float(np.linalg.norm(v)) for task, v in bank["conditions"].items()},
              "encoder_loaded": False, "budget": budget})
        def save(tag):
            return save_checkpoint(out, tag, wrapper, step, contract, optimizer, scheduler,
                                   stream.state(), evidence, counts)
        if not args.resume:
            save("last")
        wrapper.train()
        initial = {key: value.detach().cpu().clone() for key, value in wrapper.action_model.state_dict().items()
                   if key in ("action_proj.weight", "task_cond_proj.residual.weight")} if args.mode == "smoke" else None
        while step < target:
            guard.check()
            optimizer.zero_grad(set_to_none=True)
            sums = {}
            for _ in range(execution["accum"]):
                batch = stream.next()
                if not torch.all(torch.linalg.vector_norm(batch["task_cond"], dim=-1) > 0):
                    raise ValueError("zero VTT entered training")
                counts.update(batch["task"])
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    loss, info, _ = wrapper(batch, t["auxiliary_loss"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss / execution["accum"]).backward()
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        sums[key] = sums.get(key, 0.) + value / execution["accum"]
            observed = mt.gradient_evidence(wrapper)
            for key in ("vtt_projection_grad_norm", "shared_action_grad_norm"):
                evidence[key + "_max"] = max(evidence.get(key + "_max", 0.), observed[key])
            evidence.update(observed)
            if step + 1 >= min(2, target) and not all(evidence[k + "_max"] > 0 for k in
                                                        ("vtt_projection_grad_norm", "shared_action_grad_norm")):
                raise ValueError("no joint VTT/shared-action backward evidence by second update")
            grad = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), t["grad_clip_norm"], error_if_nonfinite=True)
            guard.check()
            optimizer.step(); scheduler.step(); step += 1
            emit({"event": "optimizer_step", "step": step, "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad), "joint_gradient": observed, "losses": sums,
                "task_sample_counts": dict(counts), "resources": guard.latest})
            if step % t["save_every"] == 0 or step % t["resume_save_every"] == 0 or step == target:
                check_cache_unchanged(contract)
                save("last")
                if step % t["save_every"] == 0 or step == target:
                    path = save(f"step_{step:06d}")
                    if args.mode == "smoke":
                        loaded = load_checkpoint(path, contract, resume=False)
                        if any(not torch.equal(v.detach().cpu(), loaded["model_state_dict"][k])
                               for k, v in wrapper.action_model.state_dict().items()):
                            raise ValueError("smoke checkpoint roundtrip differs")
                        del loaded
            if step % t["validate_every"] == 0 or step == target:
                for ds in datasets["train"].values():
                    ds.close()
                metrics = evaluate_tasks(wrapper, datasets, bank, plan, guard.check)
                emit({"event": "validation", "step": step, **metrics})
                atomic_json(out / f"validation_step_{step:06d}.json", metrics)
            if step % 25 == 0 or step == target:
                atomic_json(out / "status.json", {"status": "running", "step": step, "target": target,
                    "budget": budget, "task_sample_counts": dict(counts), "gradient_evidence": evidence})
        if initial is not None:
            changes = {k: not torch.equal(v, wrapper.action_model.state_dict()[k].detach().cpu()) for k, v in initial.items()}
            if not all(changes.values()):
                raise ValueError(f"smoke did not update both action and VTT weights: {changes}")
            emit({"event": "smoke_joint_updates_verified", "changed": changes})
        status = {"status": "smoke_passed" if args.mode == "smoke" else
                    ("completed_fixed_budget" if step == budget else "stopped_at_requested_boundary"),
            "step": step, "target": target, "schedule_budget": budget, "gradient_evidence": evidence,
            "task_sample_counts": dict(counts), "peak_resources": guard.peaks,
            "physical_evaluation": False, "heldout_used": False,
            "evaluation_checkpoint": str(out / f"checkpoint_step_{step:06d}.pt") if args.mode == "train" and step == budget else None}
        atomic_json(out / "status.json", status)
        return status
    except BaseException as exc:
        atomic_json(out / "status.json", {"status": "stopped" if isinstance(exc, StopRequested) else "failed",
            "step_observed": step, "error": f"{type(exc).__name__}: {exc}", "peak_resources": guard.peaks,
            "resume": "only checkpoint_last; incomplete optimizer updates are never saved"})
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for split in datasets.values():
            for ds in split.values():
                ds.close()
        atomic_json(out / "peak_resources.json", guard.peaks)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True, help="task/split_manifest/feature_cache JSON")
    parser.add_argument("--mode", choices=("plan", "smoke", "train"), default="plan")
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-steps", type=int, help="absolute early stop, does NOT change cosine schedule")
    parser.add_argument("--smoke-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--smoke-batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-mapped-episodes", type=int, default=1, help="per task/split read-only mmap LRU bound")
    parser.add_argument("--gpu-lock-owner")
    parser.add_argument("--cuda-memory-fraction", type=float)
    args = parser.parse_args(argv)
    if args.mode != "plan" and not args.out:
        parser.error("--out required for execution")
    if args.resume and (args.mode == "plan" or not args.out):
        parser.error("--resume requires smoke/train and same --out")
    if args.max_steps is not None and (args.mode != "train" or args.max_steps < 1):
        parser.error("--max-steps is a positive train-only absolute target")
    if args.device == "cuda" and args.mode != "plan" and not args.gpu_lock_owner:
        parser.error("CUDA requires externally held --gpu-lock-owner")
    if args.max_mapped_episodes < 1:
        parser.error("--max-mapped-episodes must be positive")
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        parser.error("--cuda-memory-fraction must be in (0,1]")
    return args


def main(argv=None):
    from omegaconf import OmegaConf
    args = parse_args(argv)
    cfg = effective_config(OmegaConf.to_container(OmegaConf.load(mt.resolve(args.config)), resolve=True))
    specs = mt.load_task_specs(args.tasks)
    if args.max_steps is not None and args.max_steps > cfg["training"]["max_steps"]:
        raise ValueError("--max-steps exceeds fixed schedule budget")
    out = mt.resolve(args.out) if args.out else None
    if out:
        if args.resume:
            if not (out / "checkpoint_last.pt").is_file():
                raise ValueError("resume requires existing checkpoint_last.pt")
        else:
            out.mkdir(parents=True, exist_ok=False)
    try:
        datasets, source = preflight(cfg, specs, args.max_mapped_episodes)
    except Exception as exc:
        result = {"status": "blocked_preflight", "mode": args.mode,
            "error": f"{type(exc).__name__}: {exc}", "model_constructed": False,
            "encoder_loaded": False, "random_feature_fallback": False}
        if out:
            atomic_json(out / "status.json", result)
        print(json.dumps(result, indent=2))
        return 2
    if args.mode == "plan":
        result = {"status": "cache_preflight_passed_not_a_smoke", "mode": "plan",
            "dataset_frames": {s: {t: len(ds) for t, ds in rows.items()} for s, rows in datasets.items()},
            "schedule_budget": cfg["training"]["max_steps"], "initialization": source,
            "model_constructed": False, "encoder_loaded": False, "heldout_used": False}
        if out:
            atomic_json(out / "plan.json", result)
        for rows in datasets.values():
            for ds in rows.values():
                ds.close()
    else:
        result = execute(cfg, args, out, specs, datasets, source)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
