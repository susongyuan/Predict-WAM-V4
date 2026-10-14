#!/usr/bin/env python3
"""Independent cached-feature Task-0 V3 trainer. No encoder construction, simulator or early stopping.

Default: bounded 2,000-update pilot. --max-steps can only shorten the selected mode.
--mode tiny: fixed <=32 TRAIN anchors, gradient/parameter-change and fixed-noise overfit diagnostics.
--mode long: configured 30k/50k/100k budget, and --long-run-approval FILE is mandatory. Approval JSON must contain
  approved: true, config_sha256, initialization_sha256, checks: {cache_audit, cpu_checks,
  tiny_overfit, pilot_2k, independent_review: {passed: true, path: ..., sha256: ...}}.
The approval is a human-reviewed attestation, NOT generated/promoted by this trainer.

Periodic/best/baseline/final checkpoints are model-only; checkpoint_last includes optimizer, scheduler,
global RNGs and an epoch/batch cursor at the last completed optimizer boundary. --resume-source LAST
always creates a NEW run directory, restores the same schedule/step and requires matching config/data.
--max-steps is an absolute optimizer-step target, not a number of additional steps (e.g. stop at 5000,
then resume the SAME mode/config to 10000 ... total budget). Physical evaluations between segments
are managed externally; this trainer never starts them or auto-queues another segment. Partial accumulation
is replayed from its boundary. Optimizer-update failures are marked unsafe and cannot auto-resume.
--cache-ledger FILE (default <feature_cache>/sha256_ledger_v1.json) hashes payloads ONCE. Reuse rehashes
config/metadata and requires exact path/size/mtime_ns/ctime_ns/inode/device matches for ALL payloads.
Reuse is explicitly stat-validated historical SHA256 assurance, NOT a fresh payload hash. Changed
receipts/files fail closed; choose a new ledger path to perform a new full hash.
Sampling is one noise per (seed, episode, anchor), independent of batching/order. Future *action* step
31 FK in metres selects best; flow is report-only. K=1 is unrelated to execution horizon (usually 8).
Tiny/pilot outputs are diagnostics, never long-run results or physical success claims.

--continuation-plan JSON is a separate, explicitly approved 30k+20k stage from the pinned 30k last.
It keeps the original config/cache ledger/history unchanged, preserves full AdamW/RNG/cursor/global
scheduler state, and applies a separately recorded post-30k LR callable. Production runs must stop at
35000/40000/45000/50000 (one <=5k segment per invocation). An independently approved gpu_smoke_only
scope allows <=10 added updates from the original parent; smoke is not promoted automatically.
Old 30k approval is insufficient. Only exact trainer/test source migrations with verified backups are
allowed; other source/model/data/stat/sampler drift fails closed. 30k+20k is NOT an original 50k run.

CPU tests: PYTHONDONTWRITEBYTECODE=1 .conda-envs/lilawam/bin/python scripts/test_train_wam_v3.py
This module is import-safe: it does not select task_profile, initialize CUDA or start work on import.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_SHA256 = "f0ac9b1a9209fcdc358ac14a15566034d8a651071d19788ed0464879319948a9"
DEFAULT_INIT = "outputs/training/task0_manip_v2_20260928/checkpoint_es_best.pt"
DEFAULT_CONFIG = "configs/lilawam_task000_multiview_manip_v3_train.yaml"
PHASES = ("far", "near", "contact", "post")
METRIC = "fine_eef_m"
NEW_MODULES = ("mem_proj", "mem_slot_emb", "type_emb_mem", "phase_net", "gate_head", "phase_head", "dist_head")
# A distinct, opt-in second stage, never a reinterpretation of the original 30k schedule.
CONTINUATION_PARENT = "outputs/training/task0_v3_30k_20260929_s030000/checkpoint_last.pt"
CONTINUATION_PARENT_SHA = "3fe60483a4ee7087105b528ca787d84d790b4bf3f14d80e6ea5882b0bf56f75b"
CONTINUATION_MANIFEST_SHA = "41991e62b0e134e241b4a34142df7356e14b6a6c5c51b7bc0565cd17dd966a89"
CONTINUATION_SCHEDULE = {"start_step": 30000, "additional_steps": 20000, "warmup_steps": 500,
                         "start_lr": 5e-6, "peak_lr": 2e-5, "final_lr": 5e-6,
                         "kind": "linear_then_cosine_completed_global_updates"}
CONTINUATION_PREFIX = "task0_v3_cont20k_20260929_"


def resolve(path):
    return (ROOT / path).resolve()


def stat_identity(path):
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
            "inode": st.st_ino, "device": st.st_dev}


def sha256(path, tick=None):
    h = hashlib.sha256()
    path = Path(path)
    before = stat_identity(path)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
            if tick:
                tick()
    after = stat_identity(path)
    if before != after:
        raise RuntimeError(f"source changed while hashing: {path}")
    return h.hexdigest()


def require_hash(path, expected, tick=None):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"missing/invalid SHA256 for {path}")
    actual = sha256(path, tick)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: expected {expected}, got {actual}")
    return actual


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def new_run(run_name, root=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", run_name):
        raise ValueError("run-name must be a single safe directory name, not a path")
    root = Path(root or ROOT / "outputs/training")
    root.mkdir(parents=True, exist_ok=True)
    out = root / run_name
    out.mkdir(exist_ok=False)  # existing/failed runs and symlinks must never be reused
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-from", default=DEFAULT_INIT)
    source.add_argument("--resume-source", help="resumable checkpoint_last; new --run-name required")
    p.add_argument("--continuation-plan", help="explicit hashed 30k+20k stage; original config remains unchanged")
    p.add_argument("--resume-source-sha256", help="optional explicit hash; otherwise require checkpoint's .sha256.json")
    p.add_argument("--cache-ledger", help="reusable full-payload hash sidecar; default in feature_cache")
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=("tiny", "pilot", "long"), default="pilot")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--tiny-samples", type=int, default=8)
    p.add_argument("--long-run-approval")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--num-workers", type=int, default=0, help="cache is large; avoid unbounded worker prefetch")
    return p.parse_args(argv)


def validate_config(cfg, args):
    t, d, c = cfg["training"], cfg["dataset"], cfg["common"]
    fixed = [(d.get("task_id"), 0, "task_id"), (d.get("scope"), "manipulation", "scope"),
             (d.get("cond_mode"), "contract_proto", "cond_mode"), (d.get("in_segment"), True, "in_segment"),
             (d.get("instance_vtt_prob"), 0.0, "instance_vtt_prob"), (d.get("frame_stride"), 2, "stride"),
             (d.get("future_frame_offset"), 32, "future offset"), (c.get("action_chunk_size"), 32, "chunk"),
             (c.get("action_dim"), 23, "action_dim"), (c.get("state_dim"), 61, "state_dim"),
             (cfg["inference"].get("samples"), 1, "inference.samples (NOT execution horizon)"),
             (t.get("save_every"), 5000, "save_every"), (t.get("validate_every"), 5000, "validate_every"),
             (t.get("fine_validation_samples"), 256, "fine_validation_samples"),
             (t.get("checkpoint_metric"), METRIC, "checkpoint_metric"),
             (t.get("init_from_sha256"), V2_SHA256, "pinned V2 esbest source")]
    for actual, wanted, name in fixed:
        if actual != wanted:
            raise ValueError(f"{name}: expected {wanted!r}, got {actual!r}")
    budget = t.get("max_steps")
    if budget not in (30000, 50000, 100000) or t.get("schedule_steps") != budget:
        raise ValueError("training.max_steps must be 30000, 50000 or 100000; schedule_steps must equal that budget")
    if t.get("early_stopping") not in (None, False) or cfg.get("manifest", {}).get("early_stopping", False) is not False:
        raise ValueError("early stopping is forbidden")
    if (cfg["model"].get("proprio_mask") != "positions_no_gripper"
            or cfg["model"].get("action_repr", "absolute") != "absolute"
            or cfg["model"]["vision_encoder"]["feat_layers"] != [-12, -8, -4]
            or not cfg["model"].get("use_task_cond", False)
            or d.get("skip_unsupported_contracts") is not True
            or d.get("image_size") != [224, 224]):
        raise ValueError("unsupported cached-feature/conditioning/action/proprio contract")
    if t.get("image_augmentation") or t.get("resume"):
        raise ValueError("cached features: no image augmentation; use CLI --resume-source for a verified continuation")
    for k in ("batch_size", "grad_accum_steps", "samples_per_epoch", "validation_samples"):
        if int(t[k]) < 1:
            raise ValueError(f"{k} must be positive")
    if not 1 <= int(c["num_inference_steps"]) or not 1 <= int(c["action_execution_horizon"]) <= 32:
        raise ValueError("invalid inference steps / execution horizon")
    if args.num_workers < 0 or not 1 <= args.tiny_samples <= 32:
        raise ValueError("invalid workers / tiny sample count")
    cap = int(budget) if args.mode == "long" else 2000
    if getattr(args, "continuation_plan", None):
        if args.mode != "long" or not args.resume_source or budget != 30000:
            raise ValueError("continuation requires long mode, original 30k config and --resume-source")
        if not args.run_name.startswith(CONTINUATION_PREFIX) or args.max_steps is None:
            raise ValueError("continuation requires independent run prefix and explicit absolute --max-steps")
        cap = 50000
        if not 30000 < args.max_steps <= cap:
            raise ValueError("continuation target must be in (30000, 50000]")
    steps = args.max_steps if args.max_steps is not None else (200 if args.mode == "tiny" else cap)
    if not 1 <= steps <= cap:
        raise ValueError(f"{args.mode} --max-steps must be in [1, {cap}]; no pilot auto-promotion")
    if args.mode == "long" and not args.long_run_approval:
        raise ValueError("long mode requires explicit --long-run-approval; pilot never queues a long run")
    s = cfg["system"]
    if not 0 < s["duty_cycle"] <= 1 or not 0 < s["cuda_memory_fraction"] <= 1:
        raise ValueError("invalid duty cycle / memory fraction")
    return steps


def read_continuation_plan(path, cfg, config_path):
    """Validate a narrow migration specification, not an approval or a mutable config overlay."""
    path = resolve(path)
    plan_sha = sha256(path)
    plan = json.loads(path.read_text())
    require_hash(path, plan_sha)
    if (set(plan) != {"format", "parent", "schedule", "run_prefix", "source_migrations", "reason"}
            or plan["format"] != "wam-v3-continuation-20k-v1"
            or plan["schedule"] != CONTINUATION_SCHEDULE or plan["run_prefix"] != CONTINUATION_PREFIX
            or not plan["reason"]):
        raise ValueError("invalid continuation plan/schema/schedule/budget")
    parent = plan["parent"]
    if (set(parent) != {"checkpoint", "sha256", "manifest", "manifest_sha256", "config", "config_sha256",
                        "training_contract_sha256"}
            or resolve(parent["checkpoint"]) != resolve(CONTINUATION_PARENT)
            or parent["sha256"] != CONTINUATION_PARENT_SHA
            or resolve(parent["manifest"]) != resolve(CONTINUATION_PARENT).parent / "manifest.json"
            or parent["manifest_sha256"] != CONTINUATION_MANIFEST_SHA):
        raise ValueError("continuation parent identity differs from pinned 30k last")
    require_hash(resolve(parent["manifest"]), parent["manifest_sha256"])
    manifest = json.loads(resolve(parent["manifest"]).read_text())
    require_hash(resolve(parent["manifest"]), parent["manifest_sha256"])
    if (resolve(config_path) != resolve(parent["config"])
            or resolve(parent["config"]) != resolve(manifest["args"]["config"])
            or parent["config_sha256"] != manifest["source_sha256"][str(resolve(config_path))]
            or cfg != manifest["config"]
            or parent["training_contract_sha256"] != manifest["training_contract_sha256"]):
        raise ValueError("continuation config/model/data/stats or parent contract changed")
    require_hash(resolve(config_path), parent["config_sha256"])
    allowed = {"scripts/train_wam_v3.py", "scripts/test_train_wam_v3.py"}
    seen = set()
    for row in plan["source_migrations"]:
        if (set(row) != {"path", "from_sha256", "to_sha256", "backup", "reason"}
                or row["path"] not in allowed or row["path"] in seen or not row["reason"]):
            raise ValueError("unapproved or duplicate source migration; only trainer/test may migrate")
        seen.add(row["path"])
        if manifest["source_sha256"].get(str(resolve(row["path"]))) != row["from_sha256"]:
            raise ValueError("source migration does not match pinned parent source")
        require_hash(resolve(row["backup"]), row["from_sha256"])
        require_hash(resolve(row["path"]), row["to_sha256"])
    if "scripts/train_wam_v3.py" not in seen:
        raise ValueError("explicit trainer source migration is required")
    return {"path": str(path), "sha256": plan_sha, "plan": plan, "parent_manifest": manifest}


def continuation_identity(context):
    plan = context["plan"]
    return {"plan_sha256": context["sha256"], "parent_checkpoint_sha256": plan["parent"]["sha256"],
            "parent_training_contract_sha256": plan["parent"]["training_contract_sha256"],
            "schedule": plan["schedule"], "not_equivalent_to_original_50k": True}


def verify_continuation_checkpoint(ck, source, source_sha, context):
    """Keep all model/AdamW/RNG/cursor state intact; validate ancestry and LR before construction."""
    parent = context["plan"]["parent"]
    identity = continuation_identity(context)
    if ck.get("continuation") is None:
        if (resolve(source) != resolve(parent["checkpoint"]) or source_sha != parent["sha256"]
                or ck["step"] != 30000 or ck.get("manifest_sha256") != parent["manifest_sha256"]
                or ck.get("training_contract_sha256") != parent["training_contract_sha256"]):
            raise ValueError("continuation must start from the pinned parent checkpoint_last")
    elif ck["continuation"] != identity or not 30000 <= ck["step"] <= 50000:
        raise ValueError("continuation ancestry/plan changed")
    state = ck["resume_state"]
    groups = state["optimizer_state_dict"]["param_groups"]
    expected_lr = continuation_lr(ck["step"], context["plan"]["schedule"])
    if (state.get("safe_optimizer_boundary") is not True or len(groups) != 1
            or not state["optimizer_state_dict"]["state"]
            or state["scheduler_state_dict"].get("base_lrs") != [ck["config"]["training"]["learning_rate"]]
            or not math.isclose(groups[0]["lr"], expected_lr, rel_tol=1e-12, abs_tol=0)
            or state["scheduler_state_dict"].get("_last_lr") != [groups[0]["lr"]]
            or any(float(v["step"]) != ck["step"] for v in state["optimizer_state_dict"]["state"].values())):
        raise ValueError("continuation optimizer counts/LR/boundary are inconsistent; no fresh optimizer allowed")


def continuation_contract(contract, context, checkpoint):
    """Rebuild the exact parent signature with ONLY the explicitly attested source substitutions."""
    parent = context["parent_manifest"]
    sources = dict(contract["sources"])
    for row in context["plan"]["source_migrations"]:
        name = str(resolve(row["path"]))
        if sources.get(name) != row["to_sha256"]:
            raise ValueError("current source SHA differs from explicit migration")
        sources[name] = row["from_sha256"]
    # Parent run-only ancestry and cache receipt are not semantic sources (same legacy rule).
    expected = {k: v for k, v in parent["source_sha256"].items()
                if k not in {str(resolve(parent["args"]["init_from"])),
                             str(resolve(parent["args"]["resume_source"])),
                             str(resolve(parent["cache_assurance"]["ledger"]))}}
    if sources != expected:
        changed = sorted(k for k in set(sources) | set(expected) if sources.get(k) != expected.get(k))
        raise ValueError(f"unapproved source/data/stat inventory or SHA change: {changed[:8]}")
    if digest_json({**contract, "sources": sources}) != context["plan"]["parent"]["training_contract_sha256"]:
        raise ValueError("parent model/data/stats/plan/sampler contract differs after whitelisted migration")
    digest = digest_json({"contract": contract, "continuation": continuation_identity(context)})
    if checkpoint.get("continuation") is not None and checkpoint["training_contract_sha256"] != digest:
        raise ValueError("resumed continuation training contract differs")
    return digest


def verify_continuation_approval(approval, args, context):
    if (context is None or approval.get("continuation") != continuation_identity(context)
            or approval.get("additional_steps") != 20000 or approval.get("global_budget") != 50000):
        raise ValueError("new continuation approval required; original 30k approval cannot extend budget")
    kind = approval.get("scope")
    if kind == "gpu_smoke_only":
        if not 30000 < args.max_steps <= 30010:
            raise ValueError("smoke approval only permits at most ten additional updates from parent")
        checks = ("continuation_cpu", "continuation_review")
    elif kind == "four_5k_segments":
        if args.max_steps not in (35000, 40000, 45000, 50000):
            raise ValueError("continuation production targets must be 35000/40000/45000/50000")
        checks = ("continuation_cpu", "continuation_review", "continuation_gpu_smoke")
    else:
        raise ValueError("explicit continuation approval scope missing")
    for name in checks:
        check = approval.get("checks", {}).get(name, {})
        if check.get("passed") is not True:
            raise ValueError(f"continuation prerequisite not approved: {name}")
        require_hash(resolve(check["path"]), check.get("sha256"))


def verify_long_approval(args, config_sha, continuation=None):
    if args.mode != "long":
        return None
    path = resolve(args.long_run_approval)
    approval = json.loads(path.read_text())
    if (approval.get("approved") is not True or approval.get("config_sha256") != config_sha
            or approval.get("initialization_sha256") != V2_SHA256):
        raise ValueError("long-run approval is absent or for different config/source")
    for name in ("cache_audit", "cpu_checks", "tiny_overfit", "pilot_2k", "independent_review"):
        check = approval.get("checks", {}).get(name, {})
        if check.get("passed") is not True:
            raise ValueError(f"long-run prerequisite not approved: {name}")
        require_hash(resolve(check["path"]), check.get("sha256"))
    if getattr(args, "continuation_plan", None):
        verify_continuation_approval(approval, args, continuation)
    return {"path": str(path), "sha256": sha256(path), "attestation": approval}


def assert_splits(train, val, split):
    te, ve = set(map(int, train.episodes)), set(map(int, val.episodes))
    if not te or not ve or te & ve:
        raise ValueError("empty or overlapping train/validation episodes")
    if te != set(split["train_episode_indices"]) or ve != set(split["validation_episode_indices"]):
        raise ValueError("dataset episodes do not equal the pinned train/validation split")
    for ds, episodes in ((train, te), (val, ve)):
        anchors = [(int(ep), int(a)) for ep, a in ds.anchors]
        if len(set(anchors)) != len(anchors) or any(ep not in episodes for ep, _ in anchors):
            raise ValueError("duplicate anchors or anchor outside its split")


def anchor_records(dataset):
    """Metadata-only census of the ENTIRE split. Never fetch features or consult model predictions."""
    records, coverage = [], {}
    for i, (ep, a) in enumerate(dataset.anchors):
        ep, a = int(ep), int(a)
        seg = dataset.skills.segment(ep, a)
        skill = str(seg.get("skill") or "<missing-skill>")
        ph, di, ha = dataset.labels[ep]
        phase, hand, dist = int(ph[a]), int(ha[a]), float(di[a])
        labelled = phase in range(4) and hand in (0, 1) and math.isfinite(dist)
        fine = labelled and phase in (1, 2)
        r = {"index": i, "episode": ep, "anchor": a, "skill": skill, "phase": phase, "active_hand": hand,
             "dist_m": dist if math.isfinite(dist) else None, "fine": fine,
             "contract_start": int(seg["start"])}
        records.append(r)
        row = coverage.setdefault(skill, {"anchors": 0, "labelled": 0, "missing_labels": 0, "fine_candidates": 0,
                                         "phases": dict.fromkeys(PHASES, 0)})
        row["anchors"] += 1
        row["labelled" if labelled else "missing_labels"] += 1
        row["fine_candidates"] += int(fine)
        if phase in range(4):
            row["phases"][PHASES[phase]] += 1
    return records, coverage


def fixed_plan(dataset, seed, general_n=512, fine_n=256, *, tiny=False):
    records, coverage = anchor_records(dataset)
    eligible = [r for r in records if r["fine"]]  # filter full split BEFORE selecting any general subset
    if not tiny and len(eligible) < fine_n:
        raise ValueError(f"only {len(eligible)} near/contact anchors in full validation; require exactly {fine_n}")
    if not eligible:
        raise ValueError("no labelled near/contact anchors; cannot select FK best")
    def select(pool, n, salt):
        # Stable keyed ordering is also independent of dataset enumeration and Python's randomized hash().
        return sorted(pool, key=lambda r: (noise_seed(seed, r["episode"], r["anchor"], salt), r["episode"], r["anchor"]))[:n]
    general = select(records, min(general_n, len(records)), "subset-general")
    fine = select(eligible, min(fine_n, len(eligible)), "subset-fine")
    return {"version": 1, "split": "train_diagnostic" if tiny else "validation", "seed": int(seed),
            "selection": "full split demo phase in {near,contact} AND known active hand, then fixed keyed selection",
            "metric": "mean L2(FK(predicted action[31]), FK(demo action[31])) metres, active hand, base frame",
            "samples_per_anchor": 1, "terminal_step": 31, "general": general, "fine": fine,
            "full_split_anchors": len(records), "full_split_fine_candidates": len(eligible), "coverage_by_skill": coverage,
            "full_census_sha256": digest_json(records),
            # Canonicalize only NEW plans. Keep digest_json unchanged for legacy source/resume signatures.
            "label_report": {str(ep): report for ep, report in getattr(dataset, "label_report", {}).items()}}


def noise_seed(seed, episode, anchor, purpose="sample-k1"):
    s = f"wam-v3-v1:{purpose}:{int(seed)}:{int(episode)}:{int(anchor)}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") % (2**63 - 1)


@contextmanager
def isolated_rng(seed, device):
    device = torch.device(device)
    py, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def finite(value, name):
    if isinstance(value, torch.Tensor):
        ok = bool(torch.isfinite(value).all())
    else:
        ok = bool(np.isfinite(value).all())
    if not ok:
        raise FloatingPointError(f"non-finite {name}")


def action_eef(kin, action, hand):
    """Reuse numpy URDF FK. Action targets, not observation eef columns; trunk + chosen arm."""
    if hand not in (0, 1):
        raise ValueError("FK requires known active hand")
    a = np.asarray(action, dtype=np.float64)
    finite(a, "FK action")
    arm = "left" if hand == 0 else "right"
    p, _ = kin.fk(arm, np.r_[a[3:7], a[7:14] if hand == 0 else a[15:22]])
    finite(p, "FK position")
    return np.asarray(p)


class ProjectionMoments:
    """Across-anchor variance per (view, memory slot, hidden dim), BEFORE positional/type embeddings.

    A constant but different vector in each slot must have ZERO across-anchor variance (not false diversity).
    Accumulate float64 CPU sufficient statistics, one projection per anchor, not per flow integration step.
    """
    def __init__(self):
        self.n, self.total, self.square = 0, None, None

    def add(self, value):
        x = value.detach().double().cpu()
        finite(x, "memory projection")
        if self.total is None:
            self.total, self.square = torch.zeros_like(x), torch.zeros_like(x)
        self.n += 1
        self.total += x
        self.square += x.square()

    def result(self):
        if not self.n:
            return {"anchors": 0, "variance_mean": None}
        var = (self.square / self.n - (self.total / self.n).square()).clamp_min(0)
        return {"anchors": self.n, "variance_mean": float(var.mean()), "variance_max": float(var.max()),
                "variance_by_view_slot": var.mean(-1).tolist(),
                "fraction_dims_variance_below_1e-8": float((var < 1e-8).double().mean()),
                "warning": "fewer than two anchors; variance not informative" if self.n < 2 else None}


@torch.no_grad()
def evaluate(wrapper, dataset, plan, *, kin, hold_fn, lambdas, num_steps, check=lambda: None, throttle=lambda busy: 0.0):
    """One sample per anchor, fixed flow RNG separately. No adaptive anchor selection and no K averaging."""
    device, previous = torch.device(wrapper.device), wrapper.training
    general = {(r["episode"], r["anchor"]) for r in plan["general"]}
    fine_keys = {(r["episode"], r["anchor"]) for r in plan["fine"]}
    records = {(r["episode"], r["anchor"]): r for r in plan["general"] + plan["fine"]}
    rows, projection = [], {}
    captured = {}
    def capture(_module, _inputs, output):
        captured["value"] = output.detach()  # overwrite repeated Euler evaluations, do not count them repeatedly
    hook = wrapper.action_model.mem_proj.register_forward_hook(capture)
    wrapper.eval()
    started = time.monotonic()
    try:
        with isolated_rng(plan["seed"], device):
            for key, rec in sorted(records.items()):
                check()
                sample_started = time.monotonic()
                batch = default_collate([dataset[rec["index"]]])
                if (int(batch["episode"][0]), int(batch["anchor"][0])) != key:
                    raise ValueError("dataset anchor identity drifted since plan registration")
                if int(batch["phase"][0]) != rec["phase"] or int(batch["active_hand"][0]) != rec["active_hand"]:
                    raise ValueError("demo labels drifted since plan registration")
                target = batch["action_sequence"].float()
                finite(target, "validation action")
                row = {**rec, "is_general": key in general, "is_fine": key in fine_keys}
                if key in general:
                    with isolated_rng(noise_seed(plan["seed"], *key, "flow-report"), device):
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            loss, info, _ = wrapper(batch, lambdas)
                        finite(loss, "validation loss")
                        row["flow"] = {k: float(v) for k, v in info.items() if isinstance(v, (float, int))}
                        for k, v in row["flow"].items():
                            finite(v, f"validation {k}")
                        row["loss_total"] = float(loss)
                bits = batch["feats"].to(device)
                features = bits.view(torch.bfloat16)
                features = [features[:, :, i] for i in range(features.shape[2])]
                gen = torch.Generator(device=device).manual_seed(noise_seed(plan["seed"], *key))
                captured.clear()
                pred, outputs = wrapper.sample_actions(features, batch["state"], batch["pool"], batch["task_cond"],
                                                       num_steps=num_steps, generator=gen)
                if pred.shape != (1, 32, 23):
                    raise ValueError(f"K=1 requires one (1,32,23) chunk, got {tuple(pred.shape)}")
                finite(pred, "sampled action")
                pred = pred.float().cpu()
                held = hold_fn(batch["state"].float(), chunk_len=32).cpu()
                finite(held, "hold action")
                row["action_mae"] = float((pred - target).abs().mean())
                row["hold_action_mae"] = float((held - target).abs().mean())
                if key in fine_keys:
                    gt = action_eef(kin, target[0, 31].numpy(), rec["active_hand"])
                    row[METRIC] = float(np.linalg.norm(action_eef(kin, pred[0, 31].numpy(), rec["active_hand"]) - gt))
                    row["fine_hold_eef_m"] = float(np.linalg.norm(action_eef(kin, held[0, 31].numpy(), rec["active_hand"]) - gt))
                gate = outputs["gate"][0].float().cpu()
                logits = outputs["phase_logits"][0].float().cpu()
                finite(gate, "gate"); finite(logits, "phase logits")
                if (gate < 0).any() or not torch.isclose(gate.sum(), torch.tensor(1.0), atol=1e-5):
                    raise ValueError("invalid gate probabilities")
                row["gate"] = gate.tolist()
                row["gate_entropy_nats"] = float(-(gate * gate.clamp_min(1e-30).log()).sum())
                row["phase_pred"] = int(logits.argmax())
                if "value" not in captured:
                    raise RuntimeError("mem_proj hook not called; cannot report projection collapse diagnostics")
                for skill in ("__all__", rec["skill"]):
                    projection.setdefault(skill, ProjectionMoments()).add(captured["value"][0])
                rows.append(row)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                throttle(time.monotonic() - sample_started)
    finally:
        hook.remove()
        wrapper.train(previous)
    def summary(rs):
        fr = [r for r in rs if r["is_fine"]]
        gr = [r for r in rs if r["is_general"]]
        labelled = [r for r in rs if r["phase"] in range(4) and r["active_hand"] in (0, 1)]
        mean = lambda values: float(np.mean(values)) if values else None
        fm = mean([r[METRIC] for r in fr]); hm = mean([r["fine_hold_eef_m"] for r in fr])
        confusion = np.zeros((4, 4), dtype=np.int64)
        for r in labelled:
            confusion[r["phase"], r["phase_pred"]] += 1
        return {"samples": len(rs), "fine_samples": len(fr), METRIC: fm, "fine_hold_eef_m": hm,
                "fine_pred_minus_hold_m": fm - hm if fm is not None else None,
                "action_mae": mean([r["action_mae"] for r in gr]), "hold_action_mae": mean([r["hold_action_mae"] for r in gr]),
                "flow": {k: mean([r["flow"][k] for r in gr if k in r["flow"]]) for k in sorted({k for r in gr for k in r["flow"]})},
                "loss_total": mean([r["loss_total"] for r in gr]),
                "gate_mean": np.mean([r["gate"] for r in rs], axis=0).tolist() if rs else None,
                "gate_entropy_nats": mean([r["gate_entropy_nats"] for r in rs]),
                "phase_labels": dict(Counter(PHASES[r["phase"]] for r in labelled)),
                "phase_predictions": dict(Counter(PHASES[r["phase_pred"]] for r in rs)),
                "phase_accuracy": mean([float(r["phase"] == r["phase_pred"]) for r in labelled]),
                "phase_confusion": confusion.tolist(), "missing_labels": len(rs) - len(labelled)}
    report = summary(rows)
    report.update({"samples_per_anchor": 1, "terminal_step": 31, "split": plan["split"], "noise_rule": "sha256(seed,episode,anchor,purpose)",
                   "plan_sha256": digest_json(plan), "seconds": time.monotonic() - started,
                   "memory_projection": projection["__all__"].result(), "by_skill": {}})
    for skill in plan["coverage_by_skill"]:
        report["by_skill"][skill] = {**summary([r for r in rows if r["skill"] == skill]),
                                     "full_split_coverage": plan["coverage_by_skill"][skill],
                                     "memory_projection": projection.get(skill, ProjectionMoments()).result()}
    if report["fine_samples"] != len(plan["fine"]):
        raise ValueError("fine validation did not evaluate the exact registered set")
    finite(report[METRIC], METRIC)
    return report, rows


@dataclass
class Best:
    value: float | None = None
    step: int | None = None

    def update(self, report, step):
        value = float(report[METRIC])  # deliberately no flow, patience, min_delta or hold-based selection
        finite(value, METRIC)
        if self.value is None or value < self.value:
            self.value, self.step = value, int(step)
            return True
        return False


class StopRequested(RuntimeError):
    pass


class ResourceGuard:
    """Synchronous bounded telemetry. Repeated failures abort, rather than silently disabling the guard."""
    def __init__(self, config, device, emit):
        self.cfg, self.device, self.emit = config, torch.device(device), emit
        self.last, self.failures, self.signal = 0.0, 0, None
        self.latest = {}

    def check(self, force=False):
        if self.signal is not None:
            raise StopRequested(f"signal {self.signal}")
        now = time.monotonic()
        if not force and now - self.last < 2:
            return
        self.last = now
        try:
            mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
            rss = {k: v.strip() for k, v in (line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())}
            self.latest = {"ram_percent": 100 * (1 - mem["MemAvailable"] / mem["MemTotal"]), "rss_kib": int(rss["VmRSS"].split()[0])}
            if self.device.type == "cuda":
                q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5, check=True)
                values = [[float(x) for x in line.split(",")] for line in q.stdout.strip().splitlines()]
                self.latest.update(vram_global_mib=max(x[0] for x in values), gpu_util=max(x[1] for x in values))
                if torch.cuda.is_initialized():
                    self.latest.update(torch_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                                       torch_reserved_mib=torch.cuda.memory_reserved() / 2**20)
            self.failures = 0
        except Exception as exc:
            self.failures += 1
            self.emit({"event": "telemetry_error", "consecutive": self.failures, "error": str(exc)})
            if self.failures >= 3:
                raise StopRequested("resource telemetry failed three times") from exc
            return
        if self.latest["ram_percent"] >= self.cfg["ram_abort_percent"]:
            raise StopRequested(f"RAM limit: {self.latest}")
        if self.latest.get("vram_global_mib", 0) >= self.cfg["vram_abort_mib"]:
            raise StopRequested(f"VRAM limit: {self.latest}")

    def throttle(self, busy):
        if self.device.type != "cuda":
            return 0.0
        duty = self.cfg["duty_cycle"]
        seconds = max(self.cfg.get("min_sleep_s", 0), busy * (1 - duty) / duty)
        if self.latest.get("gpu_util", 0) > self.cfg.get("gpu_util_target", 85):
            seconds *= 2
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            self.check()
            time.sleep(min(0.2, max(0, until - time.monotonic())))
        return seconds


def continuation_lr(completed_step, schedule):
    if schedule != CONTINUATION_SCHEDULE:
        raise ValueError("unapproved continuation schedule")
    added = completed_step - schedule["start_step"]
    if not 0 <= added <= schedule["additional_steps"]:
        raise ValueError("continuation LR outside its approved cumulative budget")
    if added <= schedule["warmup_steps"]:
        return schedule["start_lr"] + (schedule["peak_lr"] - schedule["start_lr"]) * added / schedule["warmup_steps"]
    progress = (added - schedule["warmup_steps"]) / (schedule["additional_steps"] - schedule["warmup_steps"])
    return schedule["final_lr"] + (schedule["peak_lr"] - schedule["final_lr"]) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, cfg, continuation_schedule=None):
    t = cfg
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], betas=tuple(t["betas"]), weight_decay=t["weight_decay"])
    warm, total, ratio = int(t["warmup_steps"]), int(t["schedule_steps"]), t["lr_min"] / t["learning_rate"]
    def multiplier(step):
        if continuation_schedule is not None and step >= continuation_schedule["start_step"]:
            return continuation_lr(step, continuation_schedule) / t["learning_rate"]
        if warm and step < warm:
            return 0.01 + 0.99 * step / warm
        progress = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
        return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def capture_rng(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": None}
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.device(device).type == "cuda":
        if state["torch_cuda"] is None or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG device topology mismatch")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    elif state["torch_cuda"] is not None:
        raise ValueError("CUDA -> CPU resume is not an exact boundary continuation")


class BatchStream:
    """Deterministic epoch sampler + consumed-batch cursor. Worker prefetch is NOT the cursor.

    Loader factories must recreate a deterministic epoch using dedicated generators. WamV3Dataset is
    deterministic (contract_proto, no augmentation). Reconstruct/skips do not advance training RNGs.
    """
    def __init__(self, loader, device, cursor=None, check=lambda: None):
        self.loader, self.device, self.check = loader, device, check
        self.epoch = int((cursor or {}).get("epoch", 1))
        self.batch_skip = int((cursor or {}).get("batch_skip", 0))
        if self.epoch < 1 or self.batch_skip < 0:
            raise ValueError("invalid resume batch cursor")
        self.iterator = None

    def state(self):
        return {"epoch": self.epoch, "batch_skip": self.batch_skip,
                "rule": "reseed weighted sampler by epoch; skip consumed batches; deterministic cached dataset"}

    def next(self):
        while True:
            self.check()
            if self.iterator is None:
                with isolated_rng(0, self.device):
                    loader = self.loader(self.epoch) if callable(self.loader) else self.loader
                    self.iterator = iter(loader)
                    for _ in range(self.batch_skip):
                        self.check()
                        try:
                            next(self.iterator)
                        except StopIteration:
                            raise ValueError("resume batch cursor exceeds epoch length") from None
            try:
                batch = next(self.iterator)
            except StopIteration:
                if self.batch_skip == 0:
                    raise ValueError("training loader has no batches")
                self.epoch += 1
                self.batch_skip = 0
                self.iterator = None
                continue
            self.batch_skip += 1
            return batch


def save_checkpoint(out, tag, model, step, metadata):
    path = Path(out) / f"checkpoint_{tag}.pt"
    tmp = path.with_name(path.name + ".tmp")
    metadata = dict(metadata)
    resume_state = metadata.pop("resume_state", None)
    if tag != "last" and resume_state is not None:
        raise ValueError("only checkpoint_last may contain optimizer/RNG continuation state")
    payload = {**metadata, "model_state_dict": cpu_tree(model.state_dict()), "step": int(step),
               "model_only": tag != "last", "optimizer_state_saved": resume_state is not None,
               "checkpoint_metric": METRIC}
    if tag == "last":
        # Construction/pre-optimizer errors can only provide a clearly NON-resumable last.
        payload["resume_format"] = "wam-v3-last-v2"
        payload["resume_state"] = cpu_tree(resume_state)
        payload["resumable"] = resume_state is not None and resume_state["safe_optimizer_boundary"]
    torch.save(payload, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # Hash is verified before unpickling a resume source; crash between files fails closed.
    atomic_json(path.with_name(path.name + ".sha256.json"), {"sha256": sha256(path), "step": int(step),
                                                          "size": path.stat().st_size, "model_only": tag != "last"})
    return str(path)


def load_resume_source(path, cfg, mode, device, expected_sha=None):
    path = Path(path).resolve()
    if expected_sha is None:
        receipt = json.loads(path.with_name(path.name + ".sha256.json").read_text())
        expected_sha = receipt.get("sha256")
    require_hash(path, expected_sha)  # BEFORE pickle; only locally trusted, explicitly hashed artifacts
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    require_hash(path, expected_sha)
    if ck.get("resume_format") != "wam-v3-last-v2" or ck.get("resumable") is not True:
        raise ValueError("resume requires a safe optimizer-boundary checkpoint_last, not a model-only checkpoint")
    state = ck.get("resume_state", {})
    required = {"optimizer_state_dict", "scheduler_state_dict", "rng", "cursor", "best", "step"}
    if not required.issubset(state) or state["step"] != ck["step"]:
        raise ValueError("incomplete/inconsistent continuation state")
    if digest_json(ck["config"]) != digest_json(cfg) or ck.get("mode") != mode:
        raise ValueError("resume config/mode differs; pilot/tiny may not promote to a long run")
    if ck.get("initialization_sha256") != V2_SHA256 or state.get("device_type") != torch.device(device).type:
        raise ValueError("resume initialization/device contract mismatch")
    if state["scheduler_state_dict"].get("last_epoch") != ck["step"]:
        raise ValueError("resume scheduler/update count mismatch")
    return ck, expected_sha


def inherit_best_checkpoint(out, checkpoint, resume_path):
    """Retain a verified model-only historical best in the NEW directory even if resume does not improve."""
    best = checkpoint["resume_state"]["best"]
    if best["step"] is None:
        return None
    recorded = checkpoint.get("best_checkpoint")
    source = Path(recorded["path"]) if recorded else Path(resume_path).parent / "checkpoint_best.pt"
    receipt = json.loads(source.with_name(source.name + ".sha256.json").read_text())
    digest = recorded["sha256"] if recorded else receipt["sha256"]
    if receipt.get("step") != best["step"] or receipt.get("model_only") is not True:
        raise ValueError("historical best artifact does not match resume best step/type")
    require_hash(source, digest)
    target = Path(out) / "checkpoint_best.pt"
    if target.exists():
        raise FileExistsError("cannot overwrite a new run's existing best during inheritance")
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp)
    require_hash(tmp, digest)
    os.replace(tmp, target)
    atomic_json(target.with_name(target.name + ".sha256.json"), {**receipt, "sha256": digest})
    return {"path": str(target.resolve()), "sha256": digest, "step": best["step"], "inherited_from": str(source.resolve())}


def restore_training_state(wrapper, optimizer, scheduler, state):
    # The model is loaded strictly separately, never passed through a legacy V2 migration.
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng(state["rng"], wrapper.device)  # AFTER model/optimizer construction and loader reconstruction


def train_loop(wrapper, optimizer, scheduler, loader, *, max_steps, accum, grad_clip, lambdas,
               validate_fn, save_fn, emit, check=lambda: None, throttle=lambda busy: 0.0,
               resources=lambda: {}, every=5000, resume_state=None):
    """Count completed optimizer updates. Checkpoint last replays partial accumulation from its boundary.

    Normal validation cannot stop the budget. A failure during optimizer.step/scheduler.step or a
    nonfinite post-update parameter invalidates resumability (a partially mutated optimizer is unsafe).
    """
    dev = torch.device(wrapper.device)
    step = int(resume_state["step"]) if resume_state else 0
    if step > max_steps:
        raise ValueError("--max-steps is an absolute target and cannot precede the resume step")
    if resume_state:
        restore_training_state(wrapper, optimizer, scheduler, resume_state)
    best = Best(**resume_state["best"]) if resume_state else Best()
    initial = resume_state.get("initial_validation") if resume_state else None
    final = resume_state.get("final_validation") if resume_state else None
    grad_seen = defaultdict(float, resume_state.get("new_module_grad_norm_max", {}) if resume_state else {})
    stream = BatchStream(loader, dev, resume_state["cursor"] if resume_state else None, check)
    micro, unsafe_update = 0, False
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    boundary_rng, boundary_cursor = capture_rng(dev), stream.state()

    def continuation():
        return {"step": step, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "rng": boundary_rng, "cursor": boundary_cursor, "best": {"value": best.value, "step": best.step},
                "initial_validation": initial, "final_validation": final, "new_module_grad_norm_max": dict(grad_seen),
                "safe_optimizer_boundary": not unsafe_update, "device_type": dev.type,
                "discarded_microbatches_replayed": micro, "semantics": "replay from last completed optimizer boundary",
                "bitwise_scope": "CPU synthetic replay tested; GPU kernel determinism is not guaranteed"}

    def last(extra):
        save_fn("last", step, {**extra, "resume_state": continuation()})

    def validate_and_save(baseline=None):
        nonlocal initial, final
        check()
        report = validate_fn(step)
        if baseline:
            if baseline == "baseline":
                initial = report
            save_fn(baseline, step, {"validation": report, "baseline": True})
        final = report
        if best.update(report, step):
            save_fn("best", step, {"validation": report, "best_step": best.step, "best_value": best.value})
        emit({"event": "validation", "step": step, "best_step": best.step, "best_value": best.value, **report})
    try:
        validate_and_save("resume_baseline" if resume_state else "baseline")
        # Persist a complete step-0/resume boundary before any gradient work.
        boundary_rng = capture_rng(dev)
        last({"reason": "resume_boundary" if resume_state else "initial_boundary"})
        wrapper.train()
        while step < max_steps:
            busy_start = time.monotonic()
            for _ in range(accum):
                batch = stream.next()
                check()
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    loss, info, outputs = wrapper(batch, lambdas)
                finite(loss, "training loss")
                for name, value in info.items():
                    if isinstance(value, (int, float, list)):
                        finite(value, f"training {name}")
                (loss / accum).backward()
                micro += 1
                del outputs
            grad_items = [(name.split(".")[0], p.grad.detach().norm()) for name, p in wrapper.action_model.named_parameters()
                          if p.grad is not None and name.split(".")[0] in NEW_MODULES]
            if grad_items:
                norms = torch.stack([v for _, v in grad_items]).detach().cpu()
                finite(norms, "new module gradients")
                for (name, _), n in zip(grad_items, norms.tolist()):
                    grad_seen[name] = max(grad_seen[name], n)
            norm = torch.nn.utils.clip_grad_norm_(wrapper.action_model.parameters(), grad_clip, error_if_nonfinite=True)
            check()
            unsafe_update = True
            optimizer.step()
            step += 1
            scheduler.step()
            torch.nn.utils.get_total_norm([p.detach() for p in wrapper.action_model.parameters()],
                                          error_if_nonfinite=True, foreach=True)
            unsafe_update = False
            micro = 0
            optimizer.zero_grad(set_to_none=True)
            boundary_rng, boundary_cursor = capture_rng(dev), stream.state()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            busy = time.monotonic() - busy_start
            sleep = throttle(busy)
            if step == 1 or step % 20 == 0 or step == max_steps:
                emit({"event": "train", "step": step, "epoch": stream.epoch, "batch_cursor": stream.batch_skip,
                      "loss": float(loss.detach()), "losses": info, "gradient_norm": float(norm),
                      "new_module_grad_norm_max": dict(grad_seen), "lr_next": optimizer.param_groups[0]["lr"],
                      "busy_s": busy, "sleep_s": sleep, "elapsed_s": time.monotonic() - started,
                      "updates_per_s": (step - (resume_state["step"] if resume_state else 0)) / (time.monotonic() - started), **resources()})
            del loss
            if step % every == 0:
                save_fn(f"step_{step:06d}", step, {"epoch": stream.epoch})
                validate_and_save()
                last({"best_step": best.step, "best_value": best.value})
        if step % every:
            validate_and_save()
        result = {"completed_updates": step, "epoch": stream.epoch, "best_step": best.step, "best_value": best.value,
                  "initial_validation": initial, "final_validation": final, "new_module_grad_norm_max": dict(grad_seen),
                  "elapsed_s": time.monotonic() - started, "early_stopping": False,
                  "resumed_from_step": resume_state["step"] if resume_state else None}
        save_fn("final", step, result)
        last({"reason": "budget_complete", "best_step": best.step, "best_value": best.value})
        return result
    except BaseException as exc:
        try:
            last({"reason": type(exc).__name__, "error": str(exc), "epoch": stream.epoch,
                  "discarded_microbatches": micro, "best_step": best.step, "best_value": best.value, "aborted": True})
        except BaseException as save_error:
            emit({"event": "emergency_save_failed", "step": step, "error": str(save_error)})
        emit({"event": "abort", "step": step, "reason": type(exc).__name__, "error": str(exc)})
        raise


class SourceLedger:
    """Hash immutable inputs, then fail closed on metadata drift at validation boundaries."""
    def __init__(self, check=lambda: None):
        self.files, self.stamps, self.check = {}, {}, check

    def add(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            stamp = stat_identity(path)
            value = sha256(path, self.check)
            if stat_identity(path) != stamp:
                raise RuntimeError(f"source changed while recording digest: {path}")
            if expected is not None and value != expected:
                raise ValueError(f"SHA256 mismatch: {path}")
            self.files[key] = value
            self.stamps[key] = stamp
        elif expected is not None and self.files[key] != expected:
            raise ValueError(f"conflicting source SHA256: {path}")
        return self.files[key]

    def unchanged(self):
        for name, old in self.stamps.items():
            self.check()
            if stat_identity(name) != old:
                raise RuntimeError(f"source drift during run: {name}")


class CacheLedger:
    """A historical full-content digest receipt, NOT a claim of fresh byte hashing on reuse."""
    FORMAT = "wam-v3-cache-sha256-v1"

    def __init__(self, path, source_ledger):
        self.path, self.sources = Path(path).resolve(), source_ledger

    def _reuse(self, paths, bindings):
        receipt_hash = sha256(self.path, self.sources.check)
        receipt = json.loads(self.path.read_text())
        if receipt.get("format") != self.FORMAT or receipt.get("bindings") != bindings:
            raise ValueError("cache ledger config/meta hash mismatch; use a NEW ledger path to rehash")
        rows = receipt.get("files", [])
        if digest_json(rows) != receipt.get("files_sha256"):
            raise ValueError("cache ledger record digest mismatch")
        indexed = {row["path"]: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(paths):
            raise ValueError("cache ledger payload inventory mismatch")
        # Validate the ENTIRE inventory before accepting even one cached digest.
        for name in paths:
            self.sources.check()
            row = indexed[name]
            current = stat_identity(name)
            if (current != {k: row.get(k) for k in current}
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))):
                raise ValueError(f"cache ledger stat/hash invalidated: {name}; choose a new ledger to rehash")
        require_hash(self.path, receipt_hash, self.sources.check)
        for name, row in indexed.items():
            self.sources.files[name] = row["sha256"]
            self.sources.stamps[name] = {k: row[k] for k in ("size", "mtime_ns", "ctime_ns", "inode", "device")}
        self.sources.add(self.path, receipt_hash)
        return {"assurance": "reused_historical_full_sha256_all_stats_match_config_meta_rehashed",
                "payloads_rehashed_this_run": 0, "payloads_stat_verified": len(paths), "ledger": str(self.path),
                "ledger_sha256": receipt_hash, "not_fresh_payload_hash": True}

    def verify_or_create(self, paths, bindings):
        paths = sorted({str(Path(p).resolve()) for p in paths})
        if self.path.exists():
            return self._reuse(paths, bindings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        try:
            if self.path.exists():
                return self._reuse(paths, bindings)
            rows = []
            for name in paths:
                digest = self.sources.add(name)
                rows.append({"path": name, **self.sources.stamps[name], "sha256": digest})
            self.sources.unchanged()  # metadata/data must not drift during the potentially lengthy first hash
            atomic_json(self.path, {"format": self.FORMAT, "created_unix": time.time(), "bindings": bindings,
                                    "files": rows, "files_sha256": digest_json(rows)})
            receipt_hash = self.sources.add(self.path)
            return {"assurance": "full_payload_sha256_this_run", "payloads_rehashed_this_run": len(paths),
                    "payloads_stat_verified": len(paths), "ledger": str(self.path), "ledger_sha256": receipt_hash,
                    "not_fresh_payload_hash": False}
        finally:
            lock.unlink()


def audit_sources(cfg, train, val, ledger, emit, cache_ledger_path):
    """Read-only CPU audit; no cache regeneration, stats fitting or image encoder invocation."""
    from harness import task_profile as tp
    from scripts.cache_dino_features import manipulation_frames
    from harness.r1pro_kinematics import URDF, OFFSETS
    d, v3, enc = cfg["dataset"], cfg["model"]["v3"], cfg["model"]["vision_encoder"]
    expected = {"root": tp.DATA, "stats_path": tp.TRAIN_STATS, "skill_segments": tp.SEGMENTS, "vtt_dir": tp.VTT_DIR}
    if tp.TASK_ID != 0 or any(resolve(d[k]) != p.resolve() for k, p in expected.items()):
        raise ValueError("config paths / P2_TASK_ID do not match Task 0")
    split_hash = ledger.add(tp.SPLIT)
    split = json.loads(tp.SPLIT.read_text())
    assert_splits(train, val, split)
    index = json.loads((tp.LOWDIM / "index.json").read_text())
    if index.get("split_sha256") != split_hash or index.get("task_id") != 0:
        raise ValueError("lowdim index split/task provenance mismatch")
    stats_path = resolve(v3["stats_path"])
    stats = json.loads(stats_path.read_text())
    if (stats.get("split_sha256") != split_hash or stats.get("split") != "train only" or stats.get("task_id") != 0
            or stats.get("train_episodes") != len(train.episodes)
            or stats.get("views") != cfg["model"]["views"] or stats.get("proprio_lags") != v3["proprio_lags"]
            or stats.get("memory_lags") != v3["memory_lags"] or resolve(stats["feature_cache"]) != resolve(d["feature_cache"])):
        raise ValueError("V3 statistics provenance/lag/view mismatch")
    ledger.add(resolve(d["contract_targets"]), d["contract_targets_sha256"])
    sources = [tp.SEGMENTS, tp.TRAIN_STATS, tp.LOWDIM / "index.json", stats_path, URDF, OFFSETS,
               resolve(enc["checkpoint_path"]) / "config.json", ROOT / "configs/behavior_environment.json"]
    sources += [tp.VTT_DIR / name for name in ("prototypes.npy", "instances.npy", "instances_index.json", "meta.json", "task_level.npy")]
    # Reward labels are obtained from parquet: pin those too, not only the cached actions/state.
    sources += sorted((tp.DATA / "data" / tp.CHUNK).glob("*.parquet"))
    for path in sources:
        ledger.add(path)
    cache = resolve(d["feature_cache"])
    ledger.add(cache / "cache_config.json")
    cache_cfg = json.loads((cache / "cache_config.json").read_text())
    for key, want in (("task_id", 0), ("views", cfg["model"]["views"]), ("feat_layers", enc["feat_layers"]),
                      ("stride", 2), ("encoder_revision", enc["revision"]), ("tokens_per_image", 201)):
        if cache_cfg.get(key) != want:
            raise ValueError(f"feature cache config mismatch: {key}")
    if resolve(cache_cfg["encoder"]) != resolve(enc["checkpoint_path"]) or not enc["include_cls_register"]:
        raise ValueError("feature cache encoder/CLS configuration mismatch")
    started = time.monotonic()
    payload_paths = []
    # Bind the exact resolved training config plus cache/encoder/RGB metadata, freshly hashed every run.
    bindings = {"resolved_training_config": digest_json(cfg),
                str(cache / "cache_config.json"): ledger.add(cache / "cache_config.json"),
                str(resolve(enc["checkpoint_path"]) / "config.json"):
                    ledger.add(resolve(enc["checkpoint_path"]) / "config.json")}
    for number, ep in enumerate(sorted(train.episodes + val.episodes), 1):
        ep = int(ep)
        info = index["episodes"][str(ep)]
        for kind in ("action", "state"):
            path = tp.LOWDIM / f"episode_{ep}_{kind}.npy"
            # cache_behavior_lowdim hashes ndarray payload, NOT the .npy header/file.
            ledger.add(path)
            arr = np.load(path, mmap_mode="r")
            if hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest() != info[f"{kind}_sha256"]:
                raise ValueError(f"lowdim array payload hash mismatch: {path}")
            del arr
        frames_expected = manipulation_frames(train.skills.episodes[ep]["segments"], info["length"], 2)
        for view in cfg["model"]["views"]:
            folder = cache / view
            stem = f"episode_{ep}"
            meta_path = folder / f"{stem}.meta.json"
            ledger.add(meta_path)
            meta = json.loads(meta_path.read_text())
            src_path = tp.FRAMES / view / f"{stem}.meta.json"
            src_hash = ledger.add(src_path)
            bindings[str(meta_path)] = ledger.add(meta_path)
            bindings[str(src_path)] = src_hash
            if (meta.get("complete") is not True or meta.get("episode") != ep or meta.get("view") != view
                    or meta.get("source_frame_meta_sha256") != src_hash or meta.get("feat_layers") != enc["feat_layers"]
                    or meta.get("stride") != 2):
                raise ValueError(f"cache provenance mismatch: {meta_path}")
            frames = np.load(folder / f"{stem}.frames.npy", mmap_mode="r")
            feat = np.load(folder / f"{stem}.feat.npy", mmap_mode="r")
            pool = np.load(folder / f"{stem}.pool.npy", mmap_mode="r")
            if (frames.dtype != np.int64 or not np.array_equal(frames, frames_expected)
                    or feat.dtype != np.uint16 or feat.shape != (len(frames), 3, 201, 1024)
                    or pool.dtype != np.float32 or pool.shape != (len(frames), 2, 1024)
                    or meta.get("rows") != len(frames) or meta.get("feat_shape") != list(feat.shape)):
                raise ValueError(f"cache frames/shape/dtype mismatch: {folder / stem}")
            del frames, feat, pool
            for suffix in ("frames.npy", "pool.npy", "feat.npy"):
                payload_paths.append(folder / f"{stem}.{suffix}")
        emit({"event": "source_audit_episode", "episodes_done": number, "episode": ep,
              "elapsed_s": time.monotonic() - started, "hashed_files": len(ledger.files)})
    assurance = CacheLedger(cache_ledger_path, ledger).verify_or_create(payload_paths, bindings)
    emit({"event": "cache_assurance", **assurance, "seconds": time.monotonic() - started})
    return stats, assurance


def build_cached_wrapper(cfg, stats, init_path, source_hash, device, resume_checkpoint=None):
    """Build only action model from local encoder CONFIG, never load frozen encoder weights."""
    from omegaconf import OmegaConf
    from harness.wam_v3 import create_v3_action_model, load_v3_state, WamV3Wrapper
    config = OmegaConf.create(cfg)
    ec = json.loads((resolve(cfg["model"]["vision_encoder"]["checkpoint_path"]) / "config.json").read_text())
    if resume_checkpoint is None:
        require_hash(init_path, source_hash)  # verify BEFORE torch.load (pickle)
    model = create_v3_action_model(config, dino_hidden_size=ec["hidden_size"],
                                   num_dino_layers=len(cfg["model"]["vision_encoder"]["feat_layers"]),
                                   task_cond_dim=ec["hidden_size"], patch_size=ec["patch_size"])
    if resume_checkpoint is None:
        checkpoint = torch.load(init_path, map_location="cpu", mmap=True, weights_only=False)
        if int(checkpoint.get("step", -1)) != 500:
            raise ValueError("expected pinned V2 esbest step 500")
        notes = load_v3_state(model, checkpoint["model_state_dict"])
        del checkpoint
        require_hash(init_path, source_hash)
    else:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        notes = {"resume_strict": True, "step": resume_checkpoint["step"]}
    wrapper = WamV3Wrapper(model, views=cfg["model"]["views"], stats=stats,
                          norm_stats_path=resolve(cfg["dataset"]["stats_path"]), proprio_mask=cfg["model"]["proprio_mask"],
                          train_cfg=cfg["training"], vision_encoder=None,
                          feat_layers=cfg["model"]["vision_encoder"]["feat_layers"],
                          num_register_tokens=ec["num_register_tokens"], device=device)
    return wrapper, notes  # caller assigns wrapper before moving it, so transfer OOM can still save last


def main(argv=None):
    args = parse_args(argv)
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(resolve(args.config)), resolve=True)
    max_steps = validate_config(cfg, args)
    if os.environ.get("P2_TASK_ID", "0") != "0":
        raise ValueError("this independent trainer requires P2_TASK_ID=0")
    os.environ["P2_TASK_ID"] = "0"  # BEFORE task-dependent imports; import itself remains side-effect free
    from harness import task_profile
    if task_profile.TASK_ID != 0:
        raise ValueError("task_profile was already imported for a different task")
    from harness.wam_v3 import WamV3Dataset, reward_frames
    from harness.wam_behavior import hold_baseline
    from harness.r1pro_kinematics import R1ProKinematics
    from scripts.train_lilawam_behavior import motion_sampling_weights
    config_hash = sha256(resolve(args.config))
    continuation = read_continuation_plan(args.continuation_plan, cfg, args.config) if args.continuation_plan else None
    approval = verify_long_approval(args, config_hash, continuation)
    # Check source pin even before creating a run or unpickling any checkpoint.
    resume_ck, resume_sha = (None, None)
    if args.resume_source:
        resume_ck, resume_sha = load_resume_source(resolve(args.resume_source), cfg, args.mode, args.device,
                                                    args.resume_source_sha256)
        if resume_ck["step"] > max_steps:
            raise ValueError("--max-steps is an absolute target below the resume source step")
        if continuation:
            verify_continuation_checkpoint(resume_ck, args.resume_source, resume_sha, continuation)
            scope = approval["attestation"]["scope"]
            if scope == "gpu_smoke_only" and resume_sha != CONTINUATION_PARENT_SHA:
                raise ValueError("GPU smoke must use the pinned parent, never extend a smoke or production run")
            if not 0 < max_steps - resume_ck["step"] <= 5000:
                raise ValueError("continuation permits one segment of at most 5000 updates, never automatic full 20k")
    else:
        require_hash(resolve(args.init_from), V2_SHA256)
    if args.device == "cuda":
        policy = json.loads((ROOT / "configs/behavior_environment.json").read_text())["resource_policy"]
        if not policy["gpu_execution_enabled"]:
            raise ValueError("project policy disables GPU execution")
        from scripts.gpu_lock import owner
        if owner() != f"wam-train-{args.run_name}":
            raise ValueError("acquire external GPU lock as wam-train-<run-name> first; trainer never auto-queues GPU work")
    out = new_run(args.run_name)
    log = (out / "log.jsonl").open("x", buffering=1)
    def emit(row):
        line = json.dumps({"time": time.time(), "mode": args.mode, **row}, allow_nan=False)
        log.write(line + "\n"); log.flush()
        print(line, flush=True)
    guard = ResourceGuard(cfg["system"], args.device, emit)
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for s in old_handlers:
        signal.signal(s, lambda signum, _frame: setattr(guard, "signal", signum))
    wrapper = None
    loop_started = False
    metadata = {"config": cfg, "mode": args.mode, "run_name": args.run_name,
                "result_kind": "long_training" if args.mode == "long" else "diagnostic_not_long_training",
                "initialization_sha256": V2_SHA256, "cond_mode": "contract_proto", "inference_samples": 1,
                "action_execution_horizon": cfg["common"]["action_execution_horizon"], "early_stopping": False,
                "resume_source": str(resolve(args.resume_source)) if args.resume_source else None,
                "resume_source_sha256": resume_sha}
    if continuation:
        metadata["continuation"] = continuation_identity(continuation)
        metadata["result_kind"] = ("continuation_gpu_smoke_diagnostic" if approval["attestation"]["scope"] == "gpu_smoke_only"
                                   else "distinct_30k_plus_20k_continuation")
    try:
        guard.check(force=True)
        emit({"event": "start", "max_steps": max_steps, "schedule_steps": cfg["training"]["schedule_steps"],
              "fresh_optimizer": resume_ck is None, "resume_step": resume_ck["step"] if resume_ck else None,
              "inference_samples": 1, "execution_horizon": cfg["common"]["action_execution_horizon"],
              "source": str(resolve(args.resume_source or args.init_from)), "initialization_sha256": V2_SHA256,
              "cache_audit": "full hash once; reusable stat-validated receipt with fresh config/meta hashes", **guard.latest})
        atomic_json(out / "status.json", {"status": "preflight", **metadata})
        ledger = SourceLedger(guard.check)
        ledger.add(resolve(args.config), config_hash)
        if resume_ck is None:
            ledger.add(resolve(args.init_from), V2_SHA256)
        for path in (Path(__file__), ROOT / "scripts/test_train_wam_v3.py", ROOT / "harness/wam_v3.py",
                     ROOT / "harness/wam_behavior.py", ROOT / "harness/task_profile.py", ROOT / "harness/r1pro_kinematics.py",
                     ROOT / "harness/orchestrator.py", ROOT / "scripts/train_lilawam_behavior.py", ROOT / "scripts/cache_dino_features.py"):
            ledger.add(path)
        for path in sorted((ROOT / "third_party/LiLa-WAM/models").rglob("*.py")):
            ledger.add(path)
        torch.set_num_threads(int(cfg["system"]["cpu_threads"]))
        seed = int(cfg["training"]["seed"])
        torch.random.default_generator.manual_seed(seed)
        random.seed(seed); np.random.seed(seed % 2**32)
        d, v3 = cfg["dataset"], cfg["model"]["v3"]
        ds_args = {"feature_cache": resolve(d["feature_cache"]), "views": cfg["model"]["views"],
                   "proprio_lags": v3["proprio_lags"], "memory_lags": v3["memory_lags"],
                   "contract_targets": str(resolve(d["contract_targets"])), "contract_targets_sha256": d["contract_targets_sha256"],
                   "seed": seed, "rewards": reward_frames()}
        train, val = WamV3Dataset("train", **ds_args), WamV3Dataset("validation", **ds_args)
        cache_ledger_path = resolve(args.cache_ledger) if args.cache_ledger else resolve(d["feature_cache"]) / "sha256_ledger_v1.json"
        stats, cache_assurance = audit_sources(cfg, train, val, ledger, emit, cache_ledger_path)
        vs = int(cfg["training"]["validation_seed"])
        plan = fixed_plan(val, vs, cfg["training"]["validation_samples"], 256)
        train_records, train_coverage = anchor_records(train)
        atomic_json(out / "validation_plan.json", plan)
        atomic_json(out / "train_labels.json", {"coverage_by_skill": train_coverage, "label_report": train.label_report})
        train_indices = list(range(len(train)))
        eval_ds, eval_plan = val, plan
        if args.mode == "tiny":
            tiny_plan = fixed_plan(train, vs, args.tiny_samples, args.tiny_samples, tiny=True)
            # Fixed near/contact TRAIN anchors give phase/gate/dist supervision to every new module.
            train_indices = [r["index"] for r in tiny_plan["fine"]]
            tiny_plan["general"] = list(tiny_plan["fine"])
            eval_ds, eval_plan = train, tiny_plan
            atomic_json(out / "tiny_train_plan.json", tiny_plan)
        weights, sampler_stats = motion_sampling_weights([train.anchors[i] for i in train_indices],
                                                         **cfg["training"]["motion_weighting"])
        finite(weights, "sampling weights")
        if (weights <= 0).any():
            raise ValueError("sampling weights must be positive")
        batch_size = min(cfg["training"]["batch_size"], len(train_indices))
        def loader(epoch):
            epoch_seed = noise_seed(seed, epoch, 0, "train-sampler-epoch")
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                            num_samples=int(cfg["training"]["samples_per_epoch"]), replacement=True,
                                            generator=torch.Generator().manual_seed(epoch_seed))
            return DataLoader(Subset(train, train_indices), batch_size=batch_size, sampler=sampler, drop_last=False,
                              num_workers=args.num_workers, pin_memory=args.device == "cuda",
                              persistent_workers=False, prefetch_factor=1 if args.num_workers else None,
                              generator=torch.Generator().manual_seed(noise_seed(seed, epoch, 0, "loader-workers")))
        # Bind semantic inputs independently of receipt filename/assurance and run-specific ancestry.
        semantic_sources = {k: v for k, v in ledger.files.items()
                            if k != str(cache_ledger_path) and k != str(resolve(args.init_from))}
        contract = {"config": cfg, "evaluation_plan": eval_plan, "train_indices": train_indices,
                    "sources": semantic_sources, "sampler_rule": "weighted_epoch_seed_v1", "workers": args.num_workers}
        metadata["training_contract_sha256"] = (continuation_contract(contract, continuation, resume_ck)
                                                 if continuation else digest_json(contract))
        if not continuation and resume_ck and resume_ck.get("training_contract_sha256") != metadata["training_contract_sha256"]:
            raise ValueError("resume model/data/stats/plan/sampler/source contract differs")
        if continuation:
            # These attestations do not alter the semantic source inventory; still guard them against mid-run drift.
            ledger.add(continuation["path"], continuation["sha256"])
            ledger.add(resolve(continuation["plan"]["parent"]["manifest"]), CONTINUATION_MANIFEST_SHA)
            ledger.add(resolve(args.long_run_approval), approval["sha256"])
            for row in continuation["plan"]["source_migrations"]:
                ledger.add(resolve(row["backup"]), row["from_sha256"])
            atomic_json(out / "continuation_migration.json", {"plan": continuation["plan"],
                        "identity": continuation_identity(continuation), "resume_source_sha256": resume_sha,
                        "from_contract_sha256": resume_ck["training_contract_sha256"],
                        "to_contract_sha256": metadata["training_contract_sha256"],
                        "state_transform": "none; exact AdamW/RNG/cursor and scheduler counters restored; new post-30k LR callable only",
                        "optimizer_reset": False})
        if resume_ck:
            ledger.add(resolve(args.resume_source), resume_sha)
        best_artifact = inherit_best_checkpoint(out, resume_ck, resolve(args.resume_source)) if resume_ck else None
        manifest = {**metadata, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "args": vars(args), "requested_updates": max_steps, "fresh_optimizer": resume_ck is None, "approval": approval,
                    "cache_assurance": cache_assurance, "inherited_best": best_artifact, "sampler_epoch_seed_rule": "sha256(seed,epoch,0,train-sampler-epoch)",
                    "source_sha256": ledger.files, "validation_plan_sha256": digest_json(plan),
                    "evaluation_plan_sha256": digest_json(eval_plan), "sampler": sampler_stats,
                    "train_episodes": train.episodes, "validation_episodes": val.episodes,
                    "train_anchor_sha256": digest_json(train_records), "train_indices_sha256": digest_json(train_indices),
                    "initialization_step": 500, "steps_start_at": resume_ck["step"] if resume_ck else 0, "encoder_constructed": False,
                    "torch_version": str(torch.__version__), "python_version": sys.version,
                    "offline_proxy_only": True, "projection_diagnostic": "across-anchor per-view/slot variance before embeddings"}
        ledger.unchanged()
        atomic_json(out / "manifest.json", manifest)
        metadata["manifest_sha256"] = sha256(out / "manifest.json")
        metadata["v3_stats"] = stats
        atomic_json(out / "config.json", cfg)
        if args.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(cfg["system"]["cuda_memory_fraction"])
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
        guard.check(force=True)
        wrapper, notes = build_cached_wrapper(cfg, stats, resolve(args.init_from), V2_SHA256, args.device, resume_ck)
        wrapper.to(args.device)
        atomic_json(out / "initialization.json", {"migration": notes, "new_modules_seed": seed})
        optimizer, scheduler = make_optimizer(wrapper.action_model, cfg["training"],
                                               continuation["plan"]["schedule"] if continuation else None)
        before = (resume_ck.get("tiny_initial_parameters") if resume_ck and args.mode == "tiny" else None)
        if before is None:
            before = {k: p.detach().cpu().clone() for k, p in wrapper.action_model.named_parameters()
                      if args.mode == "tiny" and k.split(".")[0] in NEW_MODULES}
        kin = R1ProKinematics()
        def validation(step):
            ledger.unchanged()
            report, rows = evaluate(wrapper, eval_ds, eval_plan, kin=kin, hold_fn=hold_baseline,
                                    lambdas=cfg["training"]["auxiliary_loss"], num_steps=cfg["common"]["num_inference_steps"],
                                    check=guard.check, throttle=guard.throttle)
            atomic_json(out / f"validation_{step:06d}.json", {"mode": args.mode, "step": step, "report": report, "rows": rows})
            return report
        def save(tag, step, extra):
            nonlocal best_artifact
            extra = dict(extra)
            if tag == "last":
                extra["best_checkpoint"] = best_artifact
            if tag == "last" and args.mode == "tiny":
                extra["tiny_initial_parameters"] = before
            path = save_checkpoint(out, tag, wrapper.action_model, step, {**metadata, **extra})
            if tag == "best":
                receipt = json.loads(Path(path + ".sha256.json").read_text())
                best_artifact = {"path": path, "sha256": receipt["sha256"], "step": step}
            emit({"event": "checkpoint", "tag": tag, "step": step, "path": path})
            return path
        loop_started = True
        result = train_loop(wrapper, optimizer, scheduler, loader, max_steps=max_steps,
                            accum=cfg["training"]["grad_accum_steps"], grad_clip=cfg["training"]["grad_clip_norm"],
                            lambdas=cfg["training"]["auxiliary_loss"], validate_fn=validation, save_fn=save, emit=emit,
                            check=guard.check, throttle=guard.throttle, resources=lambda: guard.latest,
                            resume_state=resume_ck["resume_state"] if resume_ck else None)
        if args.mode == "tiny":
            changes = defaultdict(float)
            for k, p in wrapper.action_model.named_parameters():
                if k in before:
                    changes[k.split(".")[0]] += float((p.detach().cpu() - before[k]).double().square().sum())
            grad_ok = all(result["new_module_grad_norm_max"].get(k, 0) > 0 for k in NEW_MODULES)
            changed = all(changes.get(k, 0) > 0 for k in NEW_MODULES)
            initial, final = result["initial_validation"], result["final_validation"]
            loss_improved = final["loss_total"] < 0.9 * initial["loss_total"]
            fk_improved = final[METRIC] < initial[METRIC]
            result["tiny_sanity"] = {"passed": grad_ok and changed and loss_improved and fk_improved,
                                     "new_module_gradients": grad_ok, "new_module_parameters_changed": changed,
                                     "parameter_delta_l2": {k: math.sqrt(v) for k, v in changes.items()},
                                     "fixed_loss_reduced_10pct": loss_improved, "fixed_k1_fk_improved": fk_improved,
                                     "not_validation_generalization": True}
        result.update(mode=args.mode, result_kind=metadata["result_kind"], automatic_long_run=False)
        atomic_json(out / "summary.json", result)
        atomic_json(out / "status.json", {"status": "budget_complete", "step": result["completed_updates"], "mode": args.mode,
                                           "tiny_sanity_passed": result.get("tiny_sanity", {}).get("passed")})
        emit({"event": "complete", "step": result["completed_updates"], "automatic_long_run": False, **guard.latest})
        return 0 if args.mode != "tiny" or result["tiny_sanity"]["passed"] else 4
    except BaseException as exc:
        if wrapper is not None and not loop_started:
            save_checkpoint(out, "last", wrapper.action_model, resume_ck["step"] if resume_ck else 0,
                            {**metadata, "aborted": True, "error": str(exc), "resume_state": None})
        atomic_json(out / "status.json", {"status": "aborted", "mode": args.mode, "reason": type(exc).__name__, "error": str(exc),
                                           "model_available": wrapper is not None})
        emit({"event": "run_failed", "reason": type(exc).__name__, "error": str(exc)})
        if isinstance(exc, StopRequested):
            return 128 + guard.signal if guard.signal else 3
        raise
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
        log.close()


if __name__ == "__main__":
    sys.exit(main())
