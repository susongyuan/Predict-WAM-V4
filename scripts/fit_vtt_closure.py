#!/usr/bin/env python3
"""Fit per-skill VTT-closure thresholds from closed-loop development traces (instances 10-19 only).

Input: policy_trace_*.jsonl files (events: progress/closure) from outputs/behavior/closed_loop/<dev runs>,
plus a per-run label file `closure_labels.json` = {"<episode>": {"<index>": "done"|"not_done"}} produced by
reviewing the rollout videos (human/dev-set labels, never simulator ground truth read by the policy).
Output: configs/vtt_closure_thresholds.json with, per skill class, the (done_progress, done_alignment) pair
maximising labelled accuracy, plus the counts it was fitted on. Report profile runs must load this file frozen.

    python3 scripts/fit_vtt_closure.py outputs/behavior/closed_loop/s1_dev*  --out configs/vtt_closure_thresholds.json
"""
import argparse, glob, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="configs/vtt_closure_thresholds.json")
    args = ap.parse_args()
    samples = {}  # skill -> list of (progress, alignment, label)
    for pattern in args.runs:
        for run in glob.glob(pattern):
            run = Path(run)
            labels_path = run / "closure_labels.json"
            if not labels_path.exists():
                print(f"skip {run}: no closure_labels.json (label the video first)")
                continue
            labels = json.loads(labels_path.read_text())
            for trace in run.glob("policy_trace_*.jsonl"):
                for line in trace.read_text().splitlines():
                    r = json.loads(line)
                    if r.get("event") != "progress":
                        continue
                    lab = labels.get(str(r["episode"]), {}).get(str(r["index"]))
                    if lab in ("done", "not_done"):
                        samples.setdefault(r["skill"], []).append((r["progress"], r["alignment"], lab == "done"))
    out = {}
    for skill, rows in samples.items():
        p = np.array([x[0] for x in rows]); a = np.array([x[1] for x in rows]); y = np.array([x[2] for x in rows])
        best = None
        for tp in np.unique(np.round(p, 2)):
            for ta in np.unique(np.round(a, 2)):
                acc = float(((p >= tp) & (a >= ta)) == y).mean()
                if best is None or acc > best[0]:
                    best = (acc, float(tp), float(ta))
        out[skill] = {"done_progress": best[1], "done_alignment": best[2], "labelled_accuracy": best[0],
                      "n_samples": int(len(rows)), "n_done": int(y.sum())}
        print(skill, out[skill])
    if out:
        Path(args.out).write_text(json.dumps({"source": "development instances 10-19 closed-loop traces + video labels",
                                              "thresholds": out}, indent=1) + "\n")
        print("wrote", args.out)
    else:
        print("no labelled samples; nothing written")


if __name__ == "__main__":
    main()
