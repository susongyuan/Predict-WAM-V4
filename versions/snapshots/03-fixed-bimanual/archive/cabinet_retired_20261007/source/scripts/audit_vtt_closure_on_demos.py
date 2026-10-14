#!/usr/bin/env python3
"""Audit of the deployed VTT skill-closure rule on the human DEMONSTRATIONS (evaluation-method audit; no simulator).

Question: replaying the exact rule of harness.orchestrator.ClosureAdapter.update (progress / alignment of
CLS(t) - CLS(start) against the contract condition vector u, done thresholds + stall clock + budget of
harness.vtt_monitor.default_configs(exec_horizon=32, budget_factor=1.5) with the orchestrator's per-key
max_steps = int(1.5 * median demo frames)) over the demonstration frames themselves, at control-step (frame)
resolution, how often would the rule declare the demonstration "stalled / over budget" before it finishes?
A rule that fails the demonstrations cannot be used to judge the policy.

Per manipulation segment (skill != "move to") of the 20 validation episodes and the first N train episodes:
frames every --every frames from segment start to end-1 (last cached frame always included), CLS from the frozen
DINOv3 encoder, progress = <CLS(t)-CLS(start), u/|u|>/|u|, alignment = cos(CLS(t)-CLS(start), u); the rule is
replayed step by step (step = frame offset from the segment start, sample-and-hold between sampled frames).

Then a demo-calibrated stall_steps per skill class = p95 of TRAIN plateau lengths (longest run of steps without
best_progress improving by stall_delta), capped at the class max_steps, is re-simulated on validation.

Outputs (refuses to overwrite an existing --out-dir): report.json, per_segment.jsonl (with the sampled
progress/alignment curves, so other rules can be re-simulated without re-encoding), README.md.
"""
import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import DATA, FRAMES, HEAD_CAMERA, LOWDIM, VTT_DIR, SkillTable, normalize_image, load_upstream  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.vtt_monitor import precondition_key, default_configs  # noqa: E402
from harness.orchestrator import budgets_from_program  # noqa: E402

NAV = "move to"
FPS = 30
CROSSINGS = (0.1, 0.3, 0.5, 0.7)
CURVE_FRACTIONS = [round(0.1 * i, 1) for i in range(11)]


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_frames(start, end, stride, length, every):
    """Cached frames every `every` frames from the segment start to end-1; the last cached frame is always included."""
    assert every % stride == 0
    last_cached = ((length - 1) // stride) * stride
    first = min(((start + stride - 1) // stride) * stride, last_cached)
    last = min(((end - 1) // stride) * stride, last_cached)
    if last < first:
        last = first
    frames = list(range(first, last + 1, every))
    if frames[-1] != last:
        frames.append(last)
    return frames


# ---------------------------------------------------------------------------------------------------------------
# the deployed rule, replayed exactly (ClosureAdapter.signals / update)
# ---------------------------------------------------------------------------------------------------------------
def signals(cls, start_cls, unit, norm):
    r = cls.astype(np.float64) - start_cls.astype(np.float64)
    rn = np.linalg.norm(r, axis=-1)
    proj = r @ unit
    progress = proj / (norm + 1e-8)
    alignment = np.where(rn > 1e-6, proj / (rn + 1e-8), 0.0)
    return progress, alignment


def step_signals(prog, align, offsets, length):
    """Sample-and-hold at control-step resolution: step s (1..length) uses the last sample with offset <= s."""
    steps = np.arange(1, length + 1)
    k = np.searchsorted(offsets, steps, side="right") - 1
    return prog[k], align[k]


def simulate(p, a, cfg):
    """Replay ClosureAdapter.update over steps 1..len(p). Returns the first True/False steps (each evaluated as if
    nothing had terminated the contract), the False reason, and the improvement steps of the stall clock."""
    best, best_step = -1e9, 0
    first_true = first_false = reason = None
    improvements = []
    for s in range(1, len(p) + 1):
        ps, as_ = float(p[s - 1]), float(a[s - 1])
        if ps > best + cfg.stall_delta:
            best, best_step = ps, s
            improvements.append(s)
        if s >= cfg.min_steps and ps >= cfg.done_progress and as_ >= cfg.done_alignment:
            if first_true is None:
                first_true = s
        elif first_false is None:
            if cfg.max_steps is not None and s >= cfg.max_steps:
                first_false, reason = s, "max_steps"
            elif s - best_step >= cfg.stall_steps and s >= cfg.min_steps:
                first_false, reason = s, "stall"
    L = len(p)
    gaps = np.diff(improvements).tolist() + [L - improvements[-1]] if improvements else [L]
    return {"first_true": first_true, "first_false": first_false, "false_reason": reason,
            "false_failure": bool(first_false is not None and (first_true is None or first_false < first_true)),
            "plateau_max": int(max(gaps)), "plateau_final": int(L - improvements[-1]) if improvements else L,
            "n_improvements": len(improvements)}


def crossings(p, L):
    out = {}
    for thr in CROSSINGS:
        idx = np.flatnonzero(p > thr)
        out[str(thr)] = (None if idx.size == 0 else round(float(idx[0] + 1) / L, 4))
    return out


def curve(p, a, L):
    prog, ali = [], []
    for f in CURVE_FRACTIONS:
        s = int(round(f * L))
        prog.append(0.0 if s == 0 else float(p[min(s, L) - 1]))
        ali.append(0.0 if s == 0 else float(a[min(s, L) - 1]))
    return prog, ali


# ---------------------------------------------------------------------------------------------------------------
# encoder with background JPEG decoding and a GPU duty-cycle throttle (shared GPU, <= 85 % utilisation rule)
# ---------------------------------------------------------------------------------------------------------------
class Encoder:
    def __init__(self, dino, batch, workers, duty):
        import torch
        self.torch = torch
        _, runner = load_upstream()
        self.enc, self.hidden, self.n_reg, self.patch = runner.ModelFactory.create_vision_encoder(str(ROOT / dino), torch.bfloat16, "cuda")
        self.enc.eval()
        self.batch, self.workers, self.duty = batch, workers, duty

    def encode(self, jobs, log_every=50):
        """jobs: list of (ep, frame) grouped by episode -> (N, hidden) float32 CLS in the same order."""
        torch = self.torch
        out = np.zeros((len(jobs), self.hidden), np.float32)
        q = queue.Queue(maxsize=6)
        err = []

        def producer():
            try:
                pool = ThreadPoolExecutor(self.workers)
                packs = {}
                for i in range(0, len(jobs), self.batch):
                    chunk = jobs[i:i + self.batch]
                    needed = {ep for ep, _ in chunk}
                    for ep in list(packs):
                        if ep not in needed:
                            del packs[ep]                      # a batch may span two episodes; keep every pack it needs
                    for ep in needed:
                        if ep not in packs:
                            packs[ep] = FramePack(FRAMES / HEAD_CAMERA, ep)
                            packs[ep]._map()
                    imgs = list(pool.map(lambda j: normalize_image(packs[j[0]].rgb(j[1])), chunk))
                    q.put((i, np.stack(imgs)))
                pool.shutdown()
            except Exception as e:  # noqa: BLE001
                err.append(e)
            q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        t0, n_batches, gpu_time = time.time(), 0, 0.0
        with torch.no_grad():
            while True:
                item = q.get()
                if item is None:
                    break
                i, x = item
                tb = time.time()
                px = torch.from_numpy(x).to("cuda", torch.bfloat16)
                cls = self.enc(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :].float()
                out[i:i + len(x)] = cls.cpu().numpy()
                dt = time.time() - tb
                gpu_time += dt
                n_batches += 1
                if self.duty < 1.0:
                    time.sleep(dt * (1.0 / self.duty - 1.0))
                if n_batches % log_every == 0:
                    done = i + len(x)
                    el = time.time() - t0
                    print(f"  encoded {done}/{len(jobs)} frames  {done / el:.0f} img/s  eta {(len(jobs) - done) / (done / el):.0f}s", flush=True)
        if err:
            raise err[0]
        return out, {"seconds": round(time.time() - t0, 1), "gpu_seconds": round(gpu_time, 1), "frames": len(jobs)}


# ---------------------------------------------------------------------------------------------------------------
def pct(x, q):
    return None if len(x) == 0 else float(np.percentile(x, q))


def med(x):
    x = [v for v in x if v is not None]
    return None if not x else float(np.median(x))


def rnd(v, n=4):
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: rnd(x, n) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, n) for x in v]
    if isinstance(v, (float, np.floating)):
        return round(float(v), n)
    return v


def summarize(rows, sim_field="deployed"):
    """Aggregate per-segment records into the per-key / per-class statistics of the report."""
    n = len(rows)
    sims = [r[sim_field] for r in rows]
    ff = [s["false_failure"] for s in sims]
    reasons = defaultdict(int)
    for s in sims:
        if s["false_failure"]:
            reasons[s["false_reason"]] += 1
    done = [s["first_true"] is not None for s in sims]
    plateaus = [s["plateau_max"] for s in sims]
    out = {
        "n": n,
        "false_failure_rate": float(np.mean(ff)) if n else None,
        "n_false_failure": int(sum(ff)),
        "false_failure_reasons": dict(reasons),
        "done_rate": float(np.mean(done)) if n else None,
        "done_frac_median": med([s["first_true"] / r["length"] for s, r in zip(sims, rows) if s["first_true"] is not None]),
        "false_frac_median": med([s["first_false"] / r["length"] for s, r in zip(sims, rows) if s["false_failure"]]),
        "any_false_before_end_rate": float(np.mean([s["first_false"] is not None for s in sims])) if n else None,
        "plateau_steps": {"p50": pct(plateaus, 50), "p90": pct(plateaus, 90), "p95": pct(plateaus, 95),
                          "max": (max(plateaus) if plateaus else None)},
        "length_median": med([r["length"] for r in rows]),
        "budget_overrun_rate": float(np.mean([r["length"] >= r["max_steps"] for r in rows])) if n else None,
        "first_cross_frac_median": {t: med([r["first_cross"][t] for r in rows]) for t in map(str, CROSSINGS)},
        "first_cross_rate": {t: (float(np.mean([r["first_cross"][t] is not None for r in rows])) if n else None) for t in map(str, CROSSINGS)},
        "final_progress_median": med([r["final_progress"] for r in rows]),
        "final_alignment_median": med([r["final_alignment"] for r in rows]),
        "max_alignment_median": med([r["max_alignment"] for r in rows]),
        "max_progress_median": med([r["max_progress"] for r in rows]),
    }
    return rnd(out)


def median_curve(rows, field):
    if not rows:
        return None
    return rnd(np.median(np.array([r[field] for r in rows]), axis=0).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dino", default="models/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--every", type=int, default=8, help="frame sampling period inside a segment (multiple of the cache stride)")
    ap.add_argument("--train-episodes", type=int, default=40, help="number of TRAIN episodes (first by id)")
    ap.add_argument("--val-episodes", type=int, default=None, help="limit validation episodes (debug only)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4, help="JPEG decode threads")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cuda-fraction", type=float, default=0.25)
    ap.add_argument("--gpu-duty", type=float, default=0.8, help="GPU duty cycle (sleep between batches; shared GPU)")
    ap.add_argument("--exec-horizon", type=int, default=32)
    ap.add_argument("--budget-factor", type=float, default=1.5)
    ap.add_argument("--calib-percentile", type=float, default=95.0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/validation/vtt_closure_demo_audit_20260923")
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir.exists():
        sys.exit(f"refusing to overwrite existing output dir {out_dir}")
    import torch
    torch.set_num_threads(args.threads)
    torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)
    t_start = time.time()

    # ---- rule under audit ---------------------------------------------------------------------------------------
    contracts_path = VTT_DIR / "contract_targets.json"
    contracts = json.loads(contracts_path.read_text())["contracts"]
    class_cfg = default_configs(exec_horizon=args.exec_horizon, budget_factor=args.budget_factor)   # class id -> ClosureConfig
    budgets = budgets_from_program()
    table = SkillTable()
    classes = table.classes
    class_name = {v: k for k, v in classes.items()}

    def key_cfg(key, skill, stall_override=None):
        cfg = class_cfg[classes[skill]]
        budget = budgets.get(key, budgets.get(skill, 900))
        cfg = replace(cfg, max_steps=int(budget * args.budget_factor))
        if stall_override is not None:
            cfg = replace(cfg, stall_steps=int(stall_override))
        return cfg

    index = json.loads((LOWDIM / "index.json").read_text())
    val_eps = sorted(e for e in table.episodes if index["episodes"][str(e)]["split"] == "validation")
    train_eps = sorted(e for e in table.episodes if index["episodes"][str(e)]["split"] == "train")[:args.train_episodes]
    if args.val_episodes is not None:
        val_eps = val_eps[:args.val_episodes]
    episodes = [(e, "validation") for e in val_eps] + [(e, "train") for e in train_eps]

    # ---- segments and frame jobs --------------------------------------------------------------------------------
    segments, jobs, skipped = [], [], []
    for ep, split in episodes:
        info = table.episodes[ep]
        try:
            pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        except Exception as e:  # noqa: BLE001
            skipped.append({"episode": ep, "reason": f"frame cache: {e}"})
            continue
        for i, s in enumerate(info["segments"]):
            if s["skill"] == NAV:
                continue
            key = precondition_key(s["skill"], s["objects"])
            if key not in contracts:
                skipped.append({"episode": ep, "segment": i, "key": key, "reason": "no contract vector"})
                continue
            frames = sample_frames(s["start"], s["end"], pack.stride, pack.length, args.every)
            segments.append({"episode": ep, "split": split, "segment": i, "key": key, "skill": s["skill"],
                             "class": int(s["class"]), "start": int(s["start"]), "end": int(s["end"]),
                             "length": int(s["end"] - s["start"]), "frames": frames, "job0": len(jobs)})
            jobs.extend((ep, f) for f in frames)
    print(f"episodes {len(episodes)} (validation {len(val_eps)}, train {len(train_eps)}); manipulation segments {len(segments)}; "
          f"frames to encode {len(jobs)}; skipped {len(skipped)}", flush=True)

    # ---- encode ---------------------------------------------------------------------------------------------------
    enc = Encoder(args.dino, args.batch, args.workers, args.gpu_duty)
    cls_all, enc_stats = enc.encode(jobs)
    print(f"encoding done: {enc_stats}", flush=True)
    del enc
    torch.cuda.empty_cache()

    # ---- per-segment signals and rule replay ----------------------------------------------------------------------
    units = {}
    for key, c in contracts.items():
        v = np.asarray(c["vector"], np.float64)
        nv = float(np.linalg.norm(v))
        units[key] = (v / (nv + 1e-8), nv)
    rows = []
    for seg in segments:
        n = len(seg["frames"])
        cls = cls_all[seg["job0"]:seg["job0"] + n]
        unit, norm = units[seg["key"]]
        prog, ali = signals(cls, cls[0], unit, norm)
        offsets = np.asarray(seg["frames"]) - seg["start"]
        L = seg["length"]
        p, a = step_signals(prog, ali, offsets, L)
        cfg = key_cfg(seg["key"], seg["skill"])
        sim = simulate(p, a, cfg)
        cprog, cali = curve(p, a, L)
        rows.append({**{k: v for k, v in seg.items() if k != "job0"},
                     "offsets": offsets.tolist(), "progress": rnd(prog.tolist()), "alignment": rnd(ali.tolist()),
                     "max_steps": cfg.max_steps, "stall_steps": cfg.stall_steps, "min_steps": cfg.min_steps,
                     "deployed": {**sim, "first_true_frac": (None if sim["first_true"] is None else round(sim["first_true"] / L, 4)),
                                  "first_false_frac": (None if sim["first_false"] is None else round(sim["first_false"] / L, 4))},
                     "first_cross": crossings(p, L),
                     "final_progress": round(float(prog[-1]), 4), "final_alignment": round(float(ali[-1]), 4),
                     "max_alignment": round(float(ali.max()), 4), "max_progress": round(float(prog.max()), 4),
                     "curve_progress": rnd(cprog), "curve_alignment": rnd(cali)})

    # ---- calibration: per class stall_steps = p95 of TRAIN plateaus, capped at the class max_steps ----------------
    proposed = {}
    for cid, cfg in class_cfg.items():
        name = class_name[cid]
        if name == NAV:
            continue
        plateaus = [r["deployed"]["plateau_max"] for r in rows if r["split"] == "train" and r["class"] == cid]
        if not plateaus:
            continue
        p95 = float(np.percentile(plateaus, args.calib_percentile))
        cap = int(cfg.max_steps)
        proposed[name] = {"stall_steps": int(min(int(np.ceil(p95)), cap)), "train_plateau_p95": round(p95, 1), "cap_class_max_steps": cap,
                          "capped": bool(np.ceil(p95) > cap), "deployed_stall_steps": int(cfg.stall_steps), "n_train_segments": len(plateaus)}
    for r in rows:
        prop = proposed.get(r["skill"])
        if prop is None:
            r["proposed"] = None
            continue
        offsets = np.asarray(r["offsets"])
        p, a = step_signals(np.asarray(r["progress"]), np.asarray(r["alignment"]), offsets, r["length"])
        sim = simulate(p, a, key_cfg(r["key"], r["skill"], stall_override=prop["stall_steps"]))
        r["proposed"] = {**sim, "first_true_frac": (None if sim["first_true"] is None else round(sim["first_true"] / r["length"], 4)),
                         "first_false_frac": (None if sim["first_false"] is None else round(sim["first_false"] / r["length"], 4))}

    # ---- aggregate ------------------------------------------------------------------------------------------------
    keys = sorted({r["key"] for r in rows})
    per_key, per_class, proposed_eval, curves = {}, {}, {}, {}
    for split in ("validation", "train"):
        per_key[split] = {k: summarize([r for r in rows if r["split"] == split and r["key"] == k]) for k in keys
                          if any(r["split"] == split and r["key"] == k for r in rows)}
        per_class[split] = {c: summarize([r for r in rows if r["split"] == split and r["skill"] == c]) for c in proposed}
        proposed_eval[split] = {"per_key": {k: summarize([r for r in rows if r["split"] == split and r["key"] == k and r["proposed"] is not None], "proposed")
                                            for k in keys if any(r["split"] == split and r["key"] == k for r in rows)},
                                "per_class": {c: summarize([r for r in rows if r["split"] == split and r["skill"] == c], "proposed") for c in proposed}}
        curves[split] = {k: {"median_progress": median_curve([r for r in rows if r["split"] == split and r["key"] == k], "curve_progress"),
                             "median_alignment": median_curve([r for r in rows if r["split"] == split and r["key"] == k], "curve_alignment")}
                         for k in keys}
    overall = {split: {"deployed": summarize([r for r in rows if r["split"] == split]),
                       "proposed": summarize([r for r in rows if r["split"] == split and r["proposed"] is not None], "proposed")}
               for split in ("validation", "train")}

    report = {
        "question": "Would the deployed VTT closure rule (ClosureAdapter.update) declare the human demonstrations stalled/failed before they finish?",
        "provenance": {
            "dino": args.dino,
            "dino_model_sha256": json.loads((ROOT / args.dino / "DOWNLOAD_PROVENANCE.json").read_text())["files"]["model.safetensors"]["sha256"],
            "contract_targets": str(contracts_path.relative_to(ROOT)), "contract_targets_sha256": sha256_of(contracts_path),
            "skill_segments_sha256": sha256_of(DATA / "skill_segments_task013.json"),
            "skill_program": "configs/task013_skill_program.json", "skill_program_sha256": sha256_of(ROOT / "configs/task013_skill_program.json"),
            "camera": HEAD_CAMERA, "feature": "DINOv3 last_hidden_state CLS (bf16 encoder, float32 signals, float64 rule)",
            "frame_stride": 2, "sample_every_frames": args.every, "fps": FPS,
            "start_cls": "first sampled frame of the segment", "control_step": "one frame; sample-and-hold between sampled frames",
            "rule_source": "harness/orchestrator.py ClosureAdapter.update + harness/vtt_monitor.py default_configs",
            "episodes": {"validation": val_eps, "train": train_eps}, "skipped": skipped,
            "encoding": enc_stats, "seconds_total": round(time.time() - t_start, 1),
            "script": "scripts/audit_vtt_closure_on_demos.py", "no_simulator": True,
        },
        "rule": {
            "exec_horizon": args.exec_horizon, "budget_factor": args.budget_factor,
            "class_configs": {class_name[c]: asdict(cfg) for c, cfg in class_cfg.items()},
            "per_key_max_steps": {k: key_cfg(k, k.split("(")[0]).max_steps for k in keys},
            "per_key_budget_frames": {k: budgets.get(k) for k in keys},
            "verdict": "True: steps>=min_steps and progress>=done_progress and alignment>=done_alignment; "
                       "False: steps>=max_steps or (steps-best_step>=stall_steps and steps>=min_steps); best_step = last step with progress > best+stall_delta",
            "false_failure": "False fires before True fires and before the demonstration segment ends",
            "plateau": "longest run of steps without best_progress improving by stall_delta (a stall fires iff plateau >= stall_steps, modulo min_steps and a True verdict)",
        },
        "n_segments": {"validation": sum(r["split"] == "validation" for r in rows), "train": sum(r["split"] == "train" for r in rows)},
        "overall": overall,
        "per_key": per_key,
        "per_class": per_class,
        "proposed_rule": {"definition": f"stall_steps(class) = ceil(p{args.calib_percentile:g} of TRAIN plateau_max), capped at the class max_steps of "
                                        f"default_configs(budget_factor={args.budget_factor}); done thresholds, min_steps, per-key max_steps unchanged",
                          "stall_steps": proposed, "evaluation": proposed_eval},
        "curves": {"fractions": CURVE_FRACTIONS, "by_split_key": curves,
                   "open_lid_validation_median_progress": curves["validation"].get("open lid(car)", {}).get("median_progress"),
                   "open_lid_validation_median_alignment": curves["validation"].get("open lid(car)", {}).get("median_alignment"),
                   "open_lid_train_median_progress": curves["train"].get("open lid(car)", {}).get("median_progress")},
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    with (out_dir / "per_segment.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "frames"}) + "\n")
    (out_dir / "README.md").write_text(readme(report))
    print(json.dumps({"validation_false_failure_rate": overall["validation"]["deployed"]["false_failure_rate"],
                      "validation_false_failure_rate_proposed": overall["validation"]["proposed"]["false_failure_rate"],
                      "proposed_stall_steps": {c: v["stall_steps"] for c, v in proposed.items()},
                      "open_lid_val_curve": report["curves"]["open_lid_validation_median_progress"],
                      "seconds": report["provenance"]["seconds_total"]}, indent=1), flush=True)


def fmt(v, n=2):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def readme(rep):
    lines = ["# VTT closure rule audited on the human demonstrations (2026-09-23)", "",
             "Evaluation-method audit, no simulator: the exact deployed skill-closure rule (`ClosureAdapter.update`, configs from "
             f"`default_configs(exec_horizon={rep['rule']['exec_horizon']}, budget_factor={rep['rule']['budget_factor']})`, per-key "
             "`max_steps = int(1.5 * median demo frames)`) is replayed over the demonstration frames of every manipulation segment "
             f"({rep['n_segments']['validation']} validation segments / {len(rep['provenance']['episodes']['validation'])} episodes, "
             f"{rep['n_segments']['train']} train segments / "
             f"{len(rep['provenance']['episodes']['train'])} episodes), one control step per frame, DINOv3 CLS every "
             f"{rep['provenance']['sample_every_frames']} frames with sample-and-hold. `false_failure` = the rule returns False (stall or budget) "
             "before it returns True and before the demonstration ends; a rule that fails the demonstrations is not a valid judge of the policy.", ""]
    for split in ("validation", "train"):
        pk, pp = rep["per_key"][split], rep["proposed_rule"]["evaluation"][split]["per_key"]
        lines += [f"## {split}: deployed rule vs demo-calibrated stall_steps", "",
                  "| key | n | len p50 | max_steps | stall_steps | FF rate | stall/budget | done rate | done@frac | plateau p50/p95/max | final prog/align | FF rate (proposed) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, s in pk.items():
            pr = pp.get(k, {})
            cls = k.split("(")[0]
            rs = s["false_failure_reasons"]
            pl = s["plateau_steps"]
            lines.append(f"| {k} | {s['n']} | {fmt(s['length_median'], 0)} | {rep['rule']['per_key_max_steps'][k]} | "
                         f"{rep['rule']['class_configs'][cls]['stall_steps']} | {fmt(s['false_failure_rate'])} | {rs.get('stall', 0)}/{rs.get('max_steps', 0)} | "
                         f"{fmt(s['done_rate'])} | {fmt(s['done_frac_median'])} | {fmt(pl['p50'], 0)}/{fmt(pl['p95'], 0)}/{fmt(pl['max'], 0)} | "
                         f"{fmt(s['final_progress_median'])}/{fmt(s['final_alignment_median'])} | {fmt(pr.get('false_failure_rate'))} |")
        o = rep["overall"][split]
        lines += ["", f"All {split} segments: false_failure_rate deployed {fmt(o['deployed']['false_failure_rate'])} "
                      f"(reasons {o['deployed']['false_failure_reasons']}), proposed {fmt(o['proposed']['false_failure_rate'])} "
                      f"(reasons {o['proposed']['false_failure_reasons']}); done_rate deployed {fmt(o['deployed']['done_rate'])}.", ""]
    lines += ["## Proposed stall_steps per skill class (p95 of TRAIN plateau lengths, capped at class max_steps)", "",
              "| class | deployed stall_steps | train plateau p95 | class max_steps | proposed stall_steps | val FF rate deployed -> proposed |", "|---|---|---|---|---|---|"]
    for c, v in rep["proposed_rule"]["stall_steps"].items():
        d = rep["per_class"]["validation"].get(c, {}).get("false_failure_rate")
        p = rep["proposed_rule"]["evaluation"]["validation"]["per_class"].get(c, {}).get("false_failure_rate")
        lines.append(f"| {c} | {v['deployed_stall_steps']} | {fmt(v['train_plateau_p95'], 0)} | {v['cap_class_max_steps']} | {v['stall_steps']}"
                     f"{' (capped)' if v['capped'] else ''} | {fmt(d)} -> {fmt(p)} |")
    cv = rep["curves"]
    lines += ["", "## open lid(car): median progress / alignment over validation demonstrations vs fraction of the segment", "",
              "| fraction | " + " | ".join(f"{f:.1f}" for f in cv["fractions"]) + " |", "|---|" + "---|" * len(cv["fractions"]),
              "| progress (val) | " + " | ".join(fmt(v) for v in (cv["open_lid_validation_median_progress"] or [])) + " |",
              "| alignment (val) | " + " | ".join(fmt(v) for v in (cv["open_lid_validation_median_alignment"] or [])) + " |",
              "| progress (train) | " + " | ".join(fmt(v) for v in (cv["open_lid_train_median_progress"] or [])) + " |", ""]
    lines += ["## Conclusion", ""] + conclusion(rep) + ["",
              "Provenance: " + json.dumps({k: rep["provenance"][k] for k in ("dino", "dino_model_sha256", "contract_targets_sha256", "sample_every_frames")}),
              "Files: report.json (all numbers), per_segment.jsonl (per-segment verdicts and the sampled progress/alignment curves), this README."]
    return "\n".join(lines) + "\n"


def conclusion(rep):
    v = rep["overall"]["validation"]
    ol = rep["per_key"]["validation"].get("open lid(car)")
    olp = rep["proposed_rule"]["evaluation"]["validation"]["per_key"].get("open lid(car)")
    ff, ffp = v["deployed"]["false_failure_rate"], v["proposed"]["false_failure_rate"]
    invalid = ff is not None and ff >= 0.2
    out = []
    out.append(f"1. Deployed rule on validation demonstrations: false_failure_rate {fmt(ff)} over {v['deployed']['n']} segments "
               f"(reasons {v['deployed']['false_failure_reasons']}); the rule {'IS INVALID' if invalid else 'is not invalidated'} as a demo-level judge: "
               f"it would {'abort' if invalid else 'rarely abort'} successful human executions before they finish.")
    if ol is not None:
        out.append(f"2. open lid(car): false_failure_rate {fmt(ol['false_failure_rate'])} (stall {ol['false_failure_reasons'].get('stall', 0)}, budget "
                   f"{ol['false_failure_reasons'].get('max_steps', 0)} of {ol['n']}), done_rate {fmt(ol['done_rate'])} at median fraction {fmt(ol['done_frac_median'])}, "
                   f"plateau p95 {fmt(ol['plateau_steps']['p95'], 0)} steps vs deployed stall_steps {rep['rule']['class_configs']['open lid']['stall_steps']}; "
                   f"final progress/alignment {fmt(ol['final_progress_median'])}/{fmt(ol['final_alignment_median'])}.")
    worst = sorted(((k, s["false_failure_rate"]) for k, s in rep["per_key"]["validation"].items()), key=lambda t: -(t[1] or 0))[:3]
    out.append("3. Worst validation keys under the deployed rule: " + ", ".join(f"{k} {fmt(r)}" for k, r in worst) + ".")
    out.append(f"4. Demo-calibrated stall_steps (p95 of train plateaus, capped): " +
               ", ".join(f"{c} {v_['stall_steps']}" for c, v_ in rep["proposed_rule"]["stall_steps"].items()) +
               f"; validation false_failure_rate {fmt(ff)} -> {fmt(ffp)} (remaining reasons {v['proposed']['false_failure_reasons']}"
               + (f"; open lid(car) {fmt(ol['false_failure_rate'])} -> {fmt(olp['false_failure_rate'])}" if ol and olp else "") + ").")
    out.append("5. Remaining budget (max_steps) failures are demonstrations longer than 1.5x the median and cannot be fixed by the stall clock; "
               "the calibrated stall rule only removes the stall-driven false failures, and closure True (done_rate) still depends on the provisional "
               "done thresholds, which this audit does not refit.")
    return out


if __name__ == "__main__":
    main()
