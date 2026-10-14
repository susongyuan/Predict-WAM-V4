#!/usr/bin/env python3
"""Fit-vs-generalization diagnostic for a Task-0 WAM checkpoint: the admission metrics on TRAIN anchors.

Question answered: is a failed held-out admission (outputs/validation/task0_manip_admission_20260927) caused by
UNDERFITTING (the model is also no better than copy-hold on demonstrations it was trained on -> more / longer training
is the lever) or by a GENERALIZATION GAP (it fits the training demos but not held-out ones -> more epochs are not the
lever)? Everything except the episode split is the registered admission procedure: the same anchor rules, K, seed
rule, deployed sampler, metrics and thresholds (scripts/check_task0_manip_admission.py and the Task-1 helpers it
imports are reused unchanged; the thresholds are read from the registered gate file).

Episodes: every 9th TRAIN episode of the sorted TRAIN list (20 of 180), fixed before any result.
Diagnostic only: training-split numbers never admit a checkpoint and are not a held-out or closed-loop result.

  P2_TASK_ID=0 python scripts/check_task0_fit_gap.py --gate outputs/validation/task0_manip_admission_20260927/gate.json \
      --checkpoint outputs/training/task0_manip_v1_20260927/checkpoint_final.pt --out DIR --duty-cycle 0.8
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_task0_manip_admission", ROOT / "scripts/check_task0_manip_admission.py")
T0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T0)
T1 = T0.T1

EPISODE_STRIDE = 9


def train_anchor_sets():
    """Registered anchor rules on TRAIN episodes: every EPISODE_STRIDE-th of the sorted TRAIN list."""
    all_eps, sets = T1.anchor_sets(T0.CONTRACTS, split="train", max_episodes=10 ** 6)
    episodes = all_eps[::EPISODE_STRIDE]
    keep = set(episodes)
    for key, s in sets.items():
        for kind in ("fixed", "event", "skipped"):
            s[kind] = [r for r in s[kind] if r["episode"] in keep]
        s["segments"] = len({(r["episode"], r["segment"]) for r in s["fixed"]})
    return episodes, sets, T0.press_reach_sets(sets, T0.toggle_frames())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", type=Path, required=True, help="registered Task-0 gate (thresholds only)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--duty-cycle", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true", help="CPU: list the TRAIN anchors and exit")
    args = ap.parse_args()
    if task_profile.TASK_ID != T0.TASK_ID:
        ap.error("run with P2_TASK_ID=0")
    gate = json.loads(args.gate.read_text())
    if gate.get("format") != T0.GATE_FORMAT or args.samples != gate["sampling"]["K"]:
        ap.error("needs the registered Task-0 gate and its K")
    episodes, sets, pr = train_anchor_sets()
    index = json.loads((T0.LOWDIM / "index.json").read_text())
    assert all(index["episodes"][str(e)]["split"] == "train" for e in episodes) and not set(episodes) & set(gate["episodes"])
    counts = {k: {"fixed": len(v["fixed"]), "event": len(v["event"])} for k, v in sets.items()}
    counts["press_reach"] = {"approach": len(pr["approach"]), "arrival": len(pr["arrival"])}
    if args.dry_run:
        print(json.dumps({"episodes": episodes, "counts": counts}, indent=1))
        return 0
    targets_path = ROOT / gate["contract_targets"]
    if T1.file_sha256(targets_path) != gate["contract_targets_sha256"]:
        raise ValueError("contract targets changed since the gate was registered")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg, wrapper, step, views = T1.load_model(args.checkpoint, args.device)
    num_steps, size = int(cfg.common.num_inference_steps), int(cfg.dataset.image_size[0])
    targets = json.loads(targets_path.read_text())
    torch.cuda.reset_peak_memory_stats()
    summary = {"diagnostic": "TRAIN-split fit check with the admission metrics; NOT an admission, NOT held-out, NOT closed loop",
               "split": "train", "episodes": episodes, "episode_rule": f"sorted TRAIN episodes[::{EPISODE_STRIDE}]",
               "counts": counts, "checkpoint": args.checkpoint, "checkpoint_step": step,
               "checkpoint_sha256": T1.file_sha256(args.checkpoint), "gate_thresholds_from": str(args.gate),
               "gate_sha256": T1.file_sha256(args.gate), "samples_K": args.samples, "flow_steps": num_steps,
               "source_sha256": {str(p.relative_to(ROOT)): T1.file_sha256(p) for p in
                                 (Path(__file__).resolve(), ROOT / "scripts/check_task0_manip_admission.py",
                                  ROOT / "scripts/check_task1_manip_admission.py", ROOT / "harness/wam_behavior.py")},
               "contracts": {}}
    approach_keys = {(r["episode"], r["anchor"]) for r in pr["approach"]}
    all_rows = {}
    for key in gate["contracts"]:
        vector = torch.tensor(targets["contracts"][key]["vector"], dtype=torch.float32)[None]
        rows = {"fixed": [], "event": [], "approach": [], "arrival": []}
        for kind in ["fixed", "event"] + (["arrival"] if key == T0.PRESS else []):
            packs = {}
            for rec in (pr["arrival"] if kind == "arrival" else sets[key][kind]):
                ep, a = rec["episode"], rec["anchor"]
                if ep not in packs:
                    packs = {ep: [T0.FramePack(T0.FRAMES / v, ep) for v in views]}
                act = np.load(T0.LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
                st = np.load(T0.LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
                demo, state = np.array(act[a:a + 32], np.float32), np.array(st[a], np.float32)
                busy = time.monotonic()
                S = T1.sample_k(wrapper, T1.view_pixels(packs[ep], a, size), state, vector, args.samples, 1000 * ep + a, num_steps)
                if args.duty_cycle < 1.0:
                    time.sleep((time.monotonic() - busy) * (1.0 - args.duty_cycle) / args.duty_cycle)
                if not np.isfinite(S).all():
                    raise ValueError(f"non-finite samples at {key} episode {ep} anchor {a}")
                row, M = T1.anchor_metrics(S, demo, state)
                if kind == "fixed":
                    rows["fixed"].append({**rec, **row})
                    if key == T0.PRESS and (ep, a) in approach_keys:
                        rows["approach"].append({**rec, "reach": T0.approach_metrics(S, M, demo, state)})
                elif kind == "event":
                    rows["event"].append({**rec, **row, "event": T1.event_metrics(M, demo, rec["hand"])})
                else:
                    rows["arrival"].append({**rec, "arrival": T0.arrival_metrics(S, M, demo, state, rec["step"])})
        verdict = T0.evaluate_contract(gate, key, rows["fixed"], rows["event"], args.samples, rows["approach"], rows["arrival"])
        summary["contracts"][key] = verdict
        all_rows[key] = rows
        g = verdict["groups"]
        print(f"[{key}] TRAIN fixed {len(rows['fixed'])} event {len(rows['event'])} | " + " ".join(
            f"{n}: cos {g[n]['intent_cos_mean']:.2f} ratio {g[n]['intent_norm_ratio']:.2f} "
            f"dMAE {g[n]['mae_pred_minus_hold']:+.4f}/{g[n]['noise_floor']:.4f}" for n in T1.UPPER)
            + (f" | arrival dE {verdict['press_reach']['arrival'].get('e_pred_minus_hold_m', float('nan')):+.4f} m"
               if key == T0.PRESS else "") + f" | would_admit={verdict['admitted']}", flush=True)
    summary["seconds"] = round(time.monotonic() - started, 1)
    summary["torch_peak_mib"] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    (args.out / "rows.json").write_text(json.dumps(all_rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "seconds": summary["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
