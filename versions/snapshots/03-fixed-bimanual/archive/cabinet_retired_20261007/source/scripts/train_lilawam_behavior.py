#!/usr/bin/env python3
"""Train LiLa-WAM on BEHAVIOR Task 13 (R1Pro, 61D state -> 32x23D chunk) under strict resource limits.

- Upstream model code unmodified; BEHAVIOR-specific glue in harness/wam_behavior.py.
- fp32 master weights + bf16 autocast; frozen DINOv3 in bf16.
- Built-in guard: GPU duty cycle (sleeps so average utilisation stays under `gpu_util_target`),
  torch VRAM fraction cap, global VRAM / RAM abort thresholds. This does not hard-cap Vulkan/desktop use.
- Validation on a fixed subset of the 20 held-out episodes: flow loss, sampled-chunk MAE per action
  group vs a copy-state hold baseline, future-feature cosine. Checkpoints keep the action model only
  (DINOv3 is frozen and reproducible from its SHA-256).
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa: E402
from harness.wam_behavior import (ACTION_GROUPS, BehaviorChunkDataset, collate, build_models, group_errors, hold_baseline,  # noqa
                                  load_action_state, model_views, model_proprio_lags)


class Guard(threading.Thread):
    """Samples nvidia-smi + /proc/meminfo; exposes recent GPU util and raises a stop flag on limits."""

    def __init__(self, vram_limit_mib, ram_limit_pct, period=2.0):
        super().__init__(daemon=True)
        self.vram_limit, self.ram_limit, self.period = vram_limit_mib, ram_limit_pct, period
        self.util = 0.0; self.vram = 0; self.ram = 0.0; self.stop_reason = None; self.samples = []

    def run(self):
        while True:
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.util, self.vram = float(q[0]), int(q[1])
                info = dict(line.split(":")[0:2] for line in open("/proc/meminfo").read().splitlines()[:3])
                total = int(info["MemTotal"].split()[0]); avail = int(info["MemAvailable"].split()[0])
                self.ram = 100.0 * (total - avail) / total
                self.samples.append((time.time(), self.util, self.vram, self.ram)); self.samples = self.samples[-300:]
                if self.vram >= self.vram_limit:
                    self.stop_reason = f"global VRAM {self.vram} MiB >= {self.vram_limit}"
                if self.ram >= self.ram_limit:
                    self.stop_reason = f"global RAM {self.ram:.1f}% >= {self.ram_limit}"
            except Exception as e:  # keep training if telemetry hiccups; abort if it dies repeatedly
                self.samples.append((time.time(), -1, -1, -1))
            time.sleep(self.period)

    def recent_util(self, seconds=20):
        now = time.time(); vals = [u for t, u, _, _ in self.samples if now - t <= seconds and u >= 0]
        return float(np.mean(vals)) if vals else 0.0


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(out, cfg, args, views, source_sha256, train_ds, val_ds, init_missing_keys):
    """Start-of-run provenance (opt-in via a top-level `manifest:` config block); refuses to overwrite."""
    path = out / "manifest.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; use a new run name")
    sources = [ROOT / args.config, Path(__file__).resolve(), ROOT / "harness/wam_behavior.py", ROOT / "harness/orchestrator.py",
               ROOT / "harness/frame_cache.py", ROOT / cfg.dataset.stats_path, ROOT / cfg.dataset.skill_segments,
               task_profile.SPLIT, task_profile.LOWDIM / "index.json"]
    if cfg.model.get("proprio_history_stats", None) is not None:
        sources.append(ROOT / cfg.model.proprio_history_stats)
    if cfg.dataset.get("cond_mode") == "contract_proto":
        sources.append(ROOT / cfg.dataset.contract_targets)
    sources += [ROOT / cfg.dataset.vtt_dir / name for name in ("prototypes.npy", "instances.npy", "instances_index.json", "meta.json")]
    frame_caches = {view: hashlib.sha256("".join(
        (task_profile.FRAMES / view / f"episode_{ep}.meta.json").read_text() for ep in sorted(train_ds.episodes + val_ds.episodes)
    ).encode()).hexdigest() for view in views}
    manifest = {**OmegaConf.to_container(cfg.manifest, resolve=True),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_name": args.run_name,
                "unit": f"wam-train-{args.run_name}", "main_pid_at_start": os.getpid(),
                "initialization_checkpoint": args.init_from, "initialization_sha256": source_sha256,
                "initialization_missing_keys": init_missing_keys, "views": views, "future_feature_camera": views[0], "proprio_lags": model_proprio_lags(cfg),
                "max_steps": int(cfg.training.max_steps), "batch_size": int(cfg.training.batch_size), "seed": int(cfg.training.seed),
                "validation_seed": int(cfg.training.get("validation_seed", 0)),
                "checkpoint_metric": cfg.training.get("checkpoint_metric", "val_loss_mse"),
                "cond_mode": cfg.dataset.cond_mode, "scope": cfg.dataset.get("scope", "all"),
                "contract_targets_sha256": train_ds.contract_targets_sha256,
                "train_episodes": len(train_ds.episodes), "validation_episodes": len(val_ds.episodes),
                "train_anchors": len(train_ds), "validation_anchors": len(val_ds),
                "report_data_used": False, "task_success_evaluated": False,
                "source_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sources},
                "frame_cache_meta_sha256": frame_caches}
    path.write_text(json.dumps(manifest, indent=2) + "\n")


# Gripper command columns (+1 open / -1 closed in the demos). Reported only; never used for checkpoint selection.
GRIPPER_COLUMNS = {"left": ACTION_GROUPS["left_gripper"][0], "right": ACTION_GROUPS["right_gripper"][0]}


def gripper_sign_counts(estimate, target):
    """Per-hand open/close sign agreement between an estimated and the demonstrated (B, T, 23) chunk.

    Returns sums (callers divide): `agree` = sum over samples of the per-sample fraction of the T steps whose sign
    matches; for EVENT samples (the demo command changes sign inside the chunk: a grasp or a release) `events` counts
    them and `event_end_agree` counts those whose final-step sign matches the demo's final sign (timing proxy)."""
    out = {}
    for hand, col in GRIPPER_COLUMNS.items():
        est, tgt = torch.sign(estimate[..., col].float()), torch.sign(target[..., col].float())
        event = (tgt != tgt[:, :1]).any(dim=1)
        out[hand] = {"agree": float((est == tgt).float().mean(dim=1).sum()), "events": int(event.sum()),
                     "event_end_agree": int(((est[:, -1] == tgt[:, -1]) & event).sum())}
    return out


def checkpoint_score(metrics, metric="val_loss_mse"):
    if metric == "val_loss_mse":
        return metrics[metric]
    if metric == "upper_body_mae":
        widths = {k: ACTION_GROUPS[k][1] - ACTION_GROUPS[k][0] for k in ("trunk", "left_arm", "right_arm")}
        return sum(metrics["val_mae"][k] * d for k, d in widths.items()) / sum(widths.values())
    raise ValueError(f"Unknown checkpoint metric {metric!r}")


@torch.no_grad()
def validate(wrapper, vla, loader, device, num_inference_steps, gen_seed=0):
    """Fixed evaluation noise/timesteps and sampler seed, without advancing training RNGs."""
    device = torch.device(device)
    dataset = loader.dataset
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    skill_names = {cid: name for name, cid in dataset.skills.classes.items()}
    was_training = wrapper.training
    wrapper.eval()
    losses, ff_losses, stats = [], [], {}
    gen = torch.Generator(device=device).manual_seed(gen_seed)
    try:
        # Upstream flow loss samples randn_like(x1) AND t even in eval mode.
        # Include DataLoader iteration so its worker seeds cannot consume the training CPU RNG either.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.random.default_generator.manual_seed(gen_seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(gen_seed)
            for batch in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, info = wrapper(batch)
                losses.append(float(info["loss_mse"])); ff_losses.append(float(info["loss_future_feat"]))
                pred, _, _ = wrapper.sample_actions(batch["pixel_values"], batch["state"], batch.get("task_cond"), num_inference_steps, gen)
                target = batch["action_sequence"].to(device)
                held = hold_baseline(batch["state"].to(device))
                selections = [(None, slice(None))] + [(skill_names[c], (batch["skill_class"] == c).to(device))
                                                       for c in batch["skill_class"].unique().tolist()]
                for skill, mask in selections:
                    row = stats.setdefault(skill, {"val_samples": 0, "val_mae": {}, "val_hold_mae": {},
                                                   "val_gripper_sign": {}, "val_hold_gripper_sign": {}})
                    count = len(target[mask]); row["val_samples"] += count
                    for key, estimate in (("val_mae", pred.float()), ("val_hold_mae", held)):
                        for group, value in group_errors(estimate[mask], target[mask]).items():
                            row[key][group] = row[key].get(group, 0.0) + value * count
                    for key, estimate in (("val_gripper_sign", pred.float()), ("val_hold_gripper_sign", held)):
                        for hand, counts in gripper_sign_counts(estimate[mask], target[mask]).items():
                            acc = row[key].setdefault(hand, dict.fromkeys(counts, 0))
                            for name, value in counts.items():
                                acc[name] += value
    finally:
        wrapper.train(was_training)
    if not stats:
        raise ValueError("Validation has no samples")
    for row in stats.values():
        for key in ("val_mae", "val_hold_mae"):
            row[key] = {k: v / row["val_samples"] for k, v in row[key].items()}
        for key in ("val_gripper_sign", "val_hold_gripper_sign"):  # reported only (grasp timing), not selection
            row[key] = {hand: {"sign_agree": c["agree"] / row["val_samples"], "event_samples": c["events"],
                               "event_end_sign_agree": (c["event_end_agree"] / c["events"]) if c["events"] else None}
                        for hand, c in row[key].items()}
    result = {"val_loss_mse": float(np.mean(losses)), "val_loss_future_feat": float(np.mean(ff_losses)),
              **stats.pop(None), "val_by_skill": stats, "val_seed": int(gen_seed)}
    result["val_upper_body_mae"] = checkpoint_score(result, "upper_body_mae")
    return result


# Upper-body (trunk + both arms) action columns and the matching proprio position columns.
_A_UP = np.r_[3:7, 7:14, 15:22]
_S_UP = np.r_[53:57, 3:10, 28:35]


def motion_sampling_weights(anchors, power, clip, gripper_event_weight=1.0):
    """Per-anchor sampling weight = clip((m / median m) ** power, *clip), m = ||a[t+31, upper] - q_t|| (18D, rad): the
    demo chunk's net upper-body motion. Rare fast motions (e.g. the reach under a lid lip) are otherwise a tiny share of
    uniformly sampled chunks. `gripper_event_weight` multiplies the weight of chunks whose demo gripper command changes
    sign (either hand): the arm is slow around a grasp/release, so motion weighting alone halves their share (Task 1:
    9.1% -> 5.0%). Uses train anchors only; returns (float64 weights, stats)."""
    from harness.wam_behavior import LOWDIM
    m = np.empty(len(anchors), np.float64)
    flip = np.zeros(len(anchors), bool)
    by_ep = {}
    for i, (ep, a) in enumerate(anchors):
        by_ep.setdefault(int(ep), []).append((i, int(a)))
    for ep, items in by_ep.items():
        act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        idx = np.array([a for _, a in items])
        end = np.minimum(idx + 31, len(act) - 1)
        m[[i for i, _ in items]] = np.linalg.norm(np.asarray(act[end][:, _A_UP]) - np.asarray(st[idx][:, _S_UP]), axis=1)
        if gripper_event_weight != 1.0:
            changes = np.r_[0, np.cumsum(np.any(np.diff(np.sign(np.asarray(act[:, [14, 22]])), axis=0) != 0, axis=1))]
            flip[[i for i, _ in items]] = changes[end] > changes[idx]
    med = float(np.median(m))
    w = np.clip((m / max(med, 1e-6)) ** power, clip[0], clip[1])
    motion_only = w.copy()
    w = w * np.where(flip, float(gripper_event_weight), 1.0)
    top = m >= np.quantile(m, 0.95)
    stats = {"power": power, "clip": list(clip), "median_motion_rad": med,
             "motion_quantiles_rad": {q: float(np.quantile(m, q)) for q in (0.5, 0.9, 0.95, 0.99)},
             "weight_mean": float(w.mean()), "top5pct_share_uniform": 0.05,
             "top5pct_share_weighted": float(w[top].sum() / w.sum())}
    if gripper_event_weight != 1.0:
        stats.update({"gripper_event_weight": float(gripper_event_weight), "gripper_event_share_uniform": float(flip.mean()),
                      "gripper_event_share_motion_only": float(motion_only[flip].sum() / motion_only.sum()),
                      "gripper_event_share_weighted": float(w[flip].sum() / w.sum())})
    return w, stats


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def augment_pixels(px, aug, gen):
    """Train-time augmentation of ImageNet-normalized pixels, (B,3,H,W) or (B,V,3,H,W), on px.device.

    Geometric: crop-and-resize (side scale U(crop_scale_min, 1)) with a random in-bounds shift, only for the view
    indices in `geometric_views` (default [0] = head). The wrist cameras see the gripper at a fixed pixel position and
    that gripper-object geometry is the grasp cue, so they are never warped. Photometric (brightness / contrast /
    saturation factors U(1-x, 1+x)) for every view: also covers the JPEG-demo vs raw-render gap. Validation and
    deployment never call this; the future-feature target frame is not augmented."""
    multi = px.dim() == 5
    x = px if multi else px.unsqueeze(1)
    B, V, C, H, W = x.shape
    dev = x.device
    mean, std = _IMAGENET_MEAN.to(dev), _IMAGENET_STD.to(dev)
    x = x.reshape(B * V, C, H, W).float() * std + mean
    n = B * V

    def u(lo, hi, m=n):
        return lo + (hi - lo) * torch.rand(m, device=dev, generator=gen)

    smin = float(aug.get("crop_scale_min", 1.0))
    geo = torch.zeros(B, V, dtype=torch.bool, device=dev)
    for v in aug.get("geometric_views", [0]):
        if int(v) < V:
            geo[:, int(v)] = True
    geo = geo.reshape(n)
    if smin < 1.0 and bool(geo.any()):
        m = int(geo.sum())
        s = u(smin, 1.0, m)
        theta = torch.zeros(m, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 1, 1] = s, s
        theta[:, 0, 2], theta[:, 1, 2] = (1 - s) * u(-1.0, 1.0, m), (1 - s) * u(-1.0, 1.0, m)
        grid = torch.nn.functional.affine_grid(theta, (m, C, H, W), align_corners=False)
        x[geo] = torch.nn.functional.grid_sample(x[geo], grid, mode="bilinear", padding_mode="border", align_corners=False)
    if float(aug.get("brightness", 0.0)) > 0:
        b = float(aug["brightness"])
        x = x * u(1 - b, 1 + b).view(n, 1, 1, 1)
    if float(aug.get("contrast", 0.0)) > 0:
        c = float(aug["contrast"])
        gray_mean = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean(dim=(1, 2)).view(n, 1, 1, 1)
        x = (x - gray_mean) * u(1 - c, 1 + c).view(n, 1, 1, 1) + gray_mean
    if float(aug.get("saturation", 0.0)) > 0:
        s = float(aug["saturation"])
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = (x - gray) * u(1 - s, 1 + s).view(n, 1, 1, 1) + gray
    x = ((x.clamp(0.0, 1.0) - mean) / std).reshape(B, V, C, H, W)
    return x if multi else x[:, 0]


def early_stopping_update(es, value, step, min_rel_delta):
    """Keras-style bookkeeping: an improvement must beat the best by >= min_rel_delta (relative). Returns True if
    this validation is the new best (the caller saves checkpoint_es_best), else increments the no-improvement count."""
    if es["best"] is None or value < es["best"] * (1.0 - min_rel_delta):
        es.update(best=float(value), best_step=int(step), bad=0)
        return True
    es["bad"] += 1
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lilawam_behavior_train.yaml")
    ap.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--max-steps", type=int); ap.add_argument("--batch-size", type=int)
    ap.add_argument("--samples-per-epoch", type=int); ap.add_argument("--val-samples", type=int)
    ap.add_argument("--validate-every", type=int); ap.add_argument("--save-every", type=int)
    ap.add_argument("--cond-mode", choices=["skill_proto", "contract_proto", "task_level", "none"])
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--resume"); source.add_argument("--init-from")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    resource_policy = json.loads((ROOT / "configs/behavior_environment.json").read_text())["resource_policy"]
    if not resource_policy["gpu_execution_enabled"]:
        raise SystemExit(resource_policy["gpu_execution_block_reason"])
    cfg = OmegaConf.load(ROOT / args.config)
    t = cfg.training
    checkpoint_metric = t.get("checkpoint_metric", "val_loss_mse")
    if checkpoint_metric not in ("val_loss_mse", "upper_body_mae"):
        ap.error(f"Unknown checkpoint metric {checkpoint_metric!r}")
    validation_seed = int(t.get("validation_seed", 0))
    for key, val in (("max_steps", args.max_steps), ("batch_size", args.batch_size), ("samples_per_epoch", args.samples_per_epoch),
                     ("validation_samples", args.val_samples), ("validate_every", args.validate_every), ("save_every", args.save_every)):
        if val is not None:
            t[key] = val
    cond_mode = args.cond_mode or cfg.dataset.get("cond_mode", "skill_proto")
    cfg.dataset.cond_mode = cond_mode
    if cond_mode == "contract_proto" and not (args.init_from or args.resume):
        ap.error("contract_proto adaptation requires --init-from or --resume")
    checkpoint_source = args.init_from or args.resume
    source_sha256 = None
    if checkpoint_source:
        digest = hashlib.sha256()
        with Path(checkpoint_source).open("rb") as checkpoint_file:
            for block in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        source_sha256 = digest.hexdigest()
        if args.init_from and t.get("init_from_sha256") not in (None, source_sha256):
            ap.error("--init-from checkpoint SHA256 differs from the pinned configuration")
        t["init_from" if args.init_from else "resume"] = checkpoint_source
        t["init_from_sha256" if args.init_from else "resume_sha256"] = source_sha256
    if cond_mode == "none":
        cfg.model.use_task_cond = False
    out = ROOT / "outputs/training" / args.run_name
    if args.init_from and Path(args.init_from).resolve().parent == out.resolve():
        ap.error("Adaptation output must be separate from the initialization checkpoint directory")
    out.mkdir(parents=True, exist_ok=True)
    sysc = cfg.system
    torch.set_num_threads(int(sysc.cpu_threads)); torch.manual_seed(int(t.seed)); np.random.seed(int(t.seed))
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(float(sysc.cuda_memory_fraction))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    guard = Guard(int(sysc.get("vram_abort_mib", 13312)), float(sysc.get("ram_abort_percent", 82))); guard.start()
    log = (out / "log.jsonl").open("a")

    def emit(rec):
        rec["time"] = time.time(); log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)

    image_size = int(cfg.dataset.image_size[0]); assert image_size == int(cfg.dataset.image_size[1]) == 224
    # The dataset reads task_profile (P2_TASK_ID) paths; the config must name the same task's artifacts.
    cfg_task = int(cfg.dataset.get("task_id", 13))
    expected = {"stats_path": task_profile.TRAIN_STATS, "skill_segments": task_profile.SEGMENTS, "vtt_dir": task_profile.VTT_DIR}
    mismatched = {k: str(cfg.dataset[k]) for k, v in expected.items() if (ROOT / cfg.dataset[k]).resolve() != v.resolve()}
    if cfg_task != task_profile.TASK_ID or mismatched:
        raise SystemExit(f"Config task {cfg_task} / paths {mismatched} do not match P2_TASK_ID={task_profile.TASK_ID}")
    contract_args = {"contract_targets": cfg.dataset.get("contract_targets", str((task_profile.VTT_DIR / "contract_targets.json").relative_to(ROOT))),
                     "contract_targets_sha256": cfg.dataset.get("contract_targets_sha256"),
                     "skip_unsupported_contracts": bool(cfg.dataset.get("skip_unsupported_contracts", False))} if cond_mode == "contract_proto" else {}
    views = model_views(cfg)
    dataset_args = {"scope": cfg.dataset.get("scope", "all"), "in_segment": bool(cfg.dataset.get("in_segment", False)),
                    "chunk": int(cfg.common.action_chunk_size), **contract_args}
    if len(views) > 1:  # single-view runs keep the exact legacy dataset call
        dataset_args["views"] = views
    lags = model_proprio_lags(cfg)
    if lags != [0]:  # legacy runs keep the exact legacy dataset call
        dataset_args["proprio_lags"] = lags
    train_ds = BehaviorChunkDataset("train", stride=int(cfg.dataset.frame_stride), future_offset=int(cfg.dataset.future_frame_offset),
                                    image_size=image_size, cond_mode=cond_mode, instance_prob=float(cfg.dataset.get("instance_vtt_prob", 0.2)), seed=int(t.seed), **dataset_args)
    val_ds = BehaviorChunkDataset("validation", stride=int(cfg.dataset.frame_stride), future_offset=int(cfg.dataset.future_frame_offset),
                                  image_size=image_size, cond_mode=cond_mode, instance_prob=0.0, seed=int(t.seed), **dataset_args)
    if cond_mode == "contract_proto":
        if train_ds.contract_targets_sha256 != val_ds.contract_targets_sha256:
            raise ValueError("Train and validation loaded different contract target files")
        cfg.dataset.contract_targets_sha256 = train_ds.contract_targets_sha256
    OmegaConf.save(cfg, out / "config.yaml")
    val_idx = np.random.default_rng(int(t.seed)).choice(len(val_ds), size=min(int(t.validation_samples), len(val_ds)), replace=False)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, sorted(val_idx.tolist())), batch_size=int(t.batch_size), shuffle=False,
                            num_workers=2, collate_fn=collate, persistent_workers=False)
    motion_weighting = t.get("motion_weighting", None)
    # Optional (absent = legacy behaviour): early stopping on a validation metric, train-time image augmentation,
    # linear LR warmup, validation before the first step. A resumed run re-seeds the sampler with seed + step so it
    # does not replay the draws of the run's beginning (legacy resumes without these options keep the plain seed).
    early_stopping = t.get("early_stopping", None)
    image_augmentation = t.get("image_augmentation", None)
    warmup_steps = int(t.get("warmup_steps", 0))
    new_options = early_stopping is not None or image_augmentation is not None or warmup_steps > 0
    resume_step = int(torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)["step"]) if args.resume else 0
    sampler_seed = int(t.seed) + (resume_step if new_options else 0)
    if early_stopping is not None and early_stopping.get("metric", "val_loss_mse") not in ("val_loss_mse", "upper_body_mae"):
        ap.error(f"Unknown early-stopping metric {early_stopping.get('metric')!r}")
    if motion_weighting is None:
        sampler = RandomSampler(train_ds, replacement=True, num_samples=int(t.samples_per_epoch), generator=torch.Generator().manual_seed(sampler_seed))
        motion_weight_stats = None
    else:
        weights, motion_weight_stats = motion_sampling_weights(train_ds.anchors, float(motion_weighting.power),
                                                               tuple(float(x) for x in motion_weighting.clip),
                                                               float(motion_weighting.get("gripper_event_weight", 1.0)))
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=int(t.samples_per_epoch), replacement=True,
                                        generator=torch.Generator().manual_seed(sampler_seed))
        emit({"event": "motion_weighting", **motion_weight_stats})
    train_loader = DataLoader(train_ds, batch_size=int(t.batch_size), sampler=sampler, num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers else None)
    emit({"event": "data", "train_anchors": len(train_ds), "train_episodes": len(train_ds.episodes), "val_anchors": len(val_ds),
          "val_subset": int(len(val_idx)), "cond_mode": cond_mode, "steps_per_epoch": len(train_loader),
          "contract_targets_sha256": train_ds.contract_targets_sha256,
          "train_contract_filter": train_ds.contract_filter, "validation_contract_filter": val_ds.contract_filter,
          "train_anchor_filter": train_ds.anchor_filter, "validation_anchor_filter": val_ds.anchor_filter,
          "checkpoint_metric": checkpoint_metric, "validation_seed": validation_seed, "views": views,
          "validation_subset_sha256": hashlib.sha256(np.asarray([val_ds.anchors[i] for i in sorted(val_idx)], dtype="<i8").tobytes()).hexdigest()})

    wrapper, vla = build_models(cfg, device)
    action_model = wrapper.action_model
    n_params = sum(p.numel() for p in action_model.parameters())
    opt = torch.optim.AdamW(action_model.parameters(), lr=float(t.learning_rate), betas=tuple(t.betas), weight_decay=float(t.weight_decay))
    if warmup_steps > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(t.schedule_steps) - warmup_steps), eta_min=float(t.lr_min))],
            milestones=[warmup_steps])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(t.schedule_steps), eta_min=float(t.lr_min))
    step = 0; epoch = 0
    es = {"best": None, "best_step": None, "bad": 0}
    initialization_step = None
    init_missing_keys = []
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False, mmap=True)
        init_missing_keys = load_action_state(action_model, ck["model_state_dict"])  # multi-view: only view_emb may be new
        initialization_step = ck.get("step")
        del ck  # optimizer/scheduler are fresh; don't retain the source checkpoint's optimizer tensors
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if ck.get("cond_mode", ck["config"]["dataset"].get("cond_mode")) != cond_mode:
            raise ValueError("Changed conditioning requires --init-from, not --resume")
        for key, default in (("scope", "all"), ("in_segment", False), ("instance_vtt_prob", 0.2)):
            if ck["config"]["dataset"].get(key, default) != cfg.dataset.get(key, default):
                raise ValueError("Changed dataset filtering/conditioning requires --init-from, not --resume")
        if cond_mode == "contract_proto" and ck["config"]["dataset"].get("contract_targets_sha256") != train_ds.contract_targets_sha256:
            raise ValueError("Cannot resume with changed contract target vectors")
        action_model.load_state_dict(ck["model_state_dict"], strict=True)
        opt.load_state_dict(ck["optimizer_state_dict"]); sched.load_state_dict(ck["scheduler_state_dict"]); step, epoch = ck["step"], ck["epoch"]
        if ck.get("early_stopping") is not None:
            es.update(ck["early_stopping"])
        del ck
    emit({"event": "model", "action_model_params": n_params, "trainable": sum(p.numel() for p in action_model.parameters() if p.requires_grad),
          "init_from": args.init_from, "resume": args.resume, "checkpoint_source_sha256": source_sha256,
          "initialization_step": initialization_step, "start_step": step, "views": views,
          "init_missing_keys": init_missing_keys})
    if cfg.get("manifest"):
        write_manifest(out, cfg, args, views, source_sha256, train_ds, val_ds, init_missing_keys)

    def save(tag, extra):
        path = out / f"checkpoint_{tag}.pt"
        es_state = {"early_stopping": dict(es)} if early_stopping is not None else {}
        torch.save({"model_state_dict": action_model.state_dict(), "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
                    "step": step, "epoch": epoch, "config": OmegaConf.to_container(cfg, resolve=True), "cond_mode": cond_mode,
                    "checkpoint_metric": checkpoint_metric, **es_state, **extra}, path)
        return str(path)

    def early_stop_check(v):
        """After a validation: update early stopping, save checkpoint_es_best on a new best; True = stop now."""
        if early_stopping is None:
            return False
        metric = early_stopping.get("metric", "val_loss_mse")
        value = v[metric] if metric == "val_loss_mse" else checkpoint_score(v, metric)
        if early_stopping_update(es, value, step, float(early_stopping.get("min_rel_delta", 0.0))):
            emit({"event": "es_best", "step": step, "metric": metric, "value": value,
                  "path": save("es_best", {"val": v, "es_metric": metric, "es_value": value})})
        else:
            emit({"event": "es_no_improvement", "step": step, "metric": metric, "value": value, "best": es["best"],
                  "best_step": es["best_step"], "bad": es["bad"], "patience": int(early_stopping.get("patience", 1))})
        return es["bad"] >= int(early_stopping.get("patience", 1))

    import signal
    stop_flag = {"signal": None}
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_flag.__setitem__("signal", signum))  # save + exit at step boundary
    duty, min_sleep = float(sysc.duty_cycle), float(sysc.min_sleep_s); util_target = float(sysc.get("gpu_util_target", 80))
    accum = int(t.grad_accum_steps); max_steps = int(t.max_steps); best = None
    torch.cuda.reset_peak_memory_stats()
    wrapper.train(); t_start = time.time(); done = False
    assert not wrapper.vision_encoder.training, "Frozen DINO must remain in eval mode"
    emit({"event": "training_mode", "vision_encoder_training": wrapper.vision_encoder.training,
          "action_model_training": action_model.training,
          "early_stopping": None if early_stopping is None else OmegaConf.to_container(early_stopping, resolve=True),
          "image_augmentation": None if image_augmentation is None else OmegaConf.to_container(image_augmentation, resolve=True),
          "warmup_steps": warmup_steps, "sampler_seed": sampler_seed,
          "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), ROOT / "harness/wam_behavior.py", ROOT / "harness/orchestrator.py")}})
    aug_gen = None
    if image_augmentation is not None:
        aug_gen = torch.Generator(device=device).manual_seed(int(t.seed) + 1 + step)
    if bool(t.get("validate_at_start", False)) and step == 0:
        # Baseline of the initialization on the same fixed validation subset (early stopping starts from it).
        v = validate(wrapper, vla, val_loader, device, int(cfg.common.num_inference_steps), validation_seed)
        emit({"event": "val", "step": step, "epoch": epoch, **v})
        if early_stop_check(v):
            done = True
    while not done:
        epoch += 1
        for i, batch in enumerate(train_loader):
            if guard.stop_reason:
                emit({"event": "abort", "reason": guard.stop_reason, "step": step}); save("abort", {}); return 3
            if stop_flag["signal"] is not None:
                emit({"event": "sigterm_save", "step": step, "path": save("last", {})}); return 0
            t0 = time.time()
            if image_augmentation is not None:
                batch["pixel_values"] = augment_pixels(batch["pixel_values"].to(device, non_blocking=True), image_augmentation, aug_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, info = wrapper(batch)
                loss = loss / accum
            loss.backward()
            if (i + 1) % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(action_model.parameters(), float(t.grad_clip_norm))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                torch.cuda.synchronize(); busy = time.time() - t0
                # Duty cycle: keep GPU busy fraction <= duty; tighten if the global average exceeds util_target.
                sleep = max(min_sleep, busy * (1 - duty) / duty)
                if guard.recent_util() > util_target:
                    sleep *= 2
                time.sleep(sleep)
                if step % 20 == 0 or step == 1:
                    emit({"event": "train", "step": step, "epoch": epoch, "loss_mse": float(info["loss_mse"]), "loss_future_feat": float(info["loss_future_feat"]),
                          "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "step_s": round(busy, 3), "sleep_s": round(sleep, 3),
                          "gpu_util_recent": round(guard.recent_util(), 1), "vram_global_mib": guard.vram, "ram_pct": round(guard.ram, 1),
                          "torch_peak_mib": int(torch.cuda.max_memory_allocated() / 2**20), "elapsed_min": round((time.time() - t_start) / 60, 1)})
                if step % int(t.validate_every) == 0:
                    v = validate(wrapper, vla, val_loader, device, int(cfg.common.num_inference_steps), validation_seed)
                    emit({"event": "val", "step": step, "epoch": epoch, **v})
                    score = checkpoint_score(v, checkpoint_metric)
                    if best is None or score < best:
                        best = score; emit({"event": "best", "step": step, "val_loss_mse": v["val_loss_mse"],
                                            "checkpoint_metric": checkpoint_metric, "checkpoint_score": score,
                                            "path": save("best", {"val": v, "checkpoint_score": score})})
                    if early_stop_check(v):
                        emit({"event": "early_stop", "step": step, "best": es["best"], "best_step": es["best_step"],
                              "bad": es["bad"], "path": save("last", {})})
                        done = True; break
                if step % int(t.save_every) == 0:
                    emit({"event": "save", "step": step, "path": save("last", {})})
                if step >= max_steps:
                    done = True; break
        emit({"event": "epoch_end", "epoch": epoch, "step": step})
    v = validate(wrapper, vla, val_loader, device, int(cfg.common.num_inference_steps), validation_seed)
    emit({"event": "final_val", "step": step, **v, "path": save("final", {"val": v}),
          "torch_peak_mib": int(torch.cuda.max_memory_allocated() / 2**20), "elapsed_min": round((time.time() - t_start) / 60, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
