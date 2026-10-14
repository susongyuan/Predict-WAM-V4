#!/usr/bin/env python3
"""Score the open-lid post-condition gate on physically labelled Isaac-6.1 native frames from the demo-action replay.

Labels come from the evaluator-side joint log of the SAME replay step (j_trunk; open >= 5% of range = 0.1265 rad):
positive = open, hard negative = 0.01 <= j < 0.1265 (lid lifting but not open), negative = j < 0.01. Uses the
runtime-equivalent CPU encoder from scripts/check_postcond_real_negatives.py. Calibration evidence only: a train
demo on a train instance; nothing is fed to any policy. The deployed threshold is reported, not changed.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from harness.vtt_monitor import PatchPostconditionGate
from harness.wam_behavior import VTT_DIR
from check_postcond_real_negatives import RuntimeEncoder, jpeg90_roundtrip, auroc

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--replay", type=Path, required=True)
ap.add_argument("--key", default="open lid(car)")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
gate = PatchPostconditionGate(VTT_DIR); thr = float(gate.threshold(args.key))
upper = None; j = {}
for line in open(next((args.replay / "eval/object_joints").glob("*.jsonl"))):
    d = json.loads(line)
    if d.get("type") == "header":
        upper = d["entities"]["car.n.01_1"]["upper"][0]
    elif "step" in d:
        j[d["step"]] = d["joints"]["car.n.01_1"][0]
open_thr = 0.05 * upper
enc = RuntimeEncoder(ROOT / gate.meta["dino"])
rows = []
for p in sorted((args.replay / "native_frames/head").glob("step_*.png")):
    t = int(p.stem.split("_")[1])
    if t not in j:
        continue
    rgb = np.asarray(Image.open(p))[..., :3]
    s = float(gate.score(enc.patches(rgb), args.key)); sj = float(gate.score(enc.patches(jpeg90_roundtrip(rgb)), args.key))
    lab = "open" if j[t] >= open_thr else ("lifting" if j[t] >= 0.01 else "closed")
    rows.append({"step": t, "j_trunk": j[t], "label": lab, "score": s, "score_jpeg90": sj})
def block(lab, key="score"):
    v = np.array([r[key] for r in rows if r["label"] == lab])
    return {"n": int(len(v)), "accepted": int((v >= thr).sum()), "rate": float((v >= thr).mean()) if len(v) else None,
            "min": float(v.min()) if len(v) else None, "median": float(np.median(v)) if len(v) else None,
            "max": float(v.max()) if len(v) else None}
rep = {"key": args.key, "deployed_threshold": thr, "open_threshold_rad": open_thr, "replay": str(args.replay),
       "labels": "evaluator joint log at the same replay step", "policy_result": False,
       "by_label": {l: block(l) for l in ("open", "lifting", "closed")},
       "by_label_jpeg90": {l: block(l, "score_jpeg90") for l in ("open", "lifting", "closed")},
       "auroc_open_vs_notopen": auroc([r["score"] for r in rows if r["label"] == "open"],
                                      [r["score"] for r in rows if r["label"] != "open"]),
       "first_open_step": min((r["step"] for r in rows if r["label"] == "open"), default=None),
       "first_gate_accept_step": min((r["step"] for r in rows if r["score"] >= thr), default=None)}
(args.out / "rows.json").write_text(json.dumps(rows) + "\n"); (args.out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep, indent=2))
print("step j score:", [(r["step"], round(r["j_trunk"], 3), round(r["score"], 3)) for r in rows if 880 <= r["step"] <= 1100])
