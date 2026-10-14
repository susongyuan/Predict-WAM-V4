#!/usr/bin/env python3
"""Paired development-only action diagnostic; never a robot success or RL comparison.

One command per candidate (or `check_openlid_replay.py --admit-against SOURCE`):
    compare_manipulation_predictions.py SOURCE_DIR CANDIDATE_DIR --out CANDIDATE_DIR/paired_admission.json
Both dirs hold `summary.json` + `rows.json` from scripts/check_openlid_replay.py with identical anchors/seeds.
The pre-registered thresholds (PREREGISTERED) are written first in the output file, before any result.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, median, quantiles

GROUPS = {"trunk": 4, "left_arm": 7, "right_arm": 7}
MIN_RELATIVE_IMPROVEMENT = 0.05   # 18D weighted MAE must drop >= 5 % vs source
MAX_ARM_REGRESSION = 0.05         # neither arm's MAE may rise > 5 % vs source
INTENT_COS_MEAN_MIN = 0.0         # intent-cosine criteria (report-only, not part of the MAE gate)
INTENT_FRAC_POSITIVE_MIN = 0.6
INTENT_NORM_RATIO_MIN = 0.5       # "substantial fraction" of demo intent magnitude; fixed 2026-09-24 before stage6

PREREGISTERED = {
    "registered": "progress.md 2026-09-23 19:14 (stage5 manipulation pilot admission rule) and 2026-09-24 14:15 "
                  "plan item 2 (stage6 reuses the same gate); thresholds fixed before any candidate result",
    "protocol": "20 validation episodes x 10 in-segment fractions of `open lid(car)`, K=8 flow-sample mean, "
                "per-anchor seed 1000*episode+anchor; candidate and source rows must pair on identical anchors",
    "metric": "absolute-radian MAE over the 32-step chunk, 18D weighted over trunk(4)+left_arm(7)+right_arm(7)",
    "gate": {"weighted_18d_relative_improvement_min": MIN_RELATIVE_IMPROVEMENT,
             "per_arm_relative_regression_max": MAX_ARM_REGRESSION,
             "passed_if": "candidate_18d <= (1-0.05)*source_18d AND each arm candidate <= (1+0.05)*source"},
    "reported_not_gating": {
        "vs_hold": "each group candidate MAE vs copy-state hold baseline (`all_upper_body_groups_beat_hold`)",
        "intent_cosine": {"cos_mean_gt": INTENT_COS_MEAN_MIN, "frac_positive_min": INTENT_FRAC_POSITIVE_MIN,
                          "intent_norm_pred_over_demo_min": INTENT_NORM_RATIO_MIN,
                          "note": "primary diagnostic per the eval methodology audit; the norm-ratio threshold "
                                  "was fixed 2026-09-24, not in the original 19:14 registration"},
        "episode_bootstrap": "95% percentile CI of candidate-source 18D MAE, 2000 resamples of 20 episodes, seed 0",
        "strict_including_hold": "legacy stage5 script verdict = gate AND all groups beat hold"},
}


def compare(source, candidate):
    a, b = ({(r["episode"], r["anchor"]): r for r in rows} for rows in (source, candidate))
    if not a or a.keys() != b.keys() or len(a) != len(source) or len(b) != len(candidate):
        raise ValueError("Prediction rows must have identical, unique, nonempty episode/anchor sets")
    for key in a:
        if any(a[key][field] != b[key][field] for field in ("fraction", "segment")):
            raise ValueError(f"Segment/fraction mismatch at {key}")
        for g in GROUPS:
            x, y = a[key]["groups"][g], b[key]["groups"][g]
            if not all(math.isfinite(v) and v >= 0 for v in (x["mae_pred"], y["mae_pred"], x["mae_hold"], y["mae_hold"])):
                raise ValueError(f"Invalid error at {key}/{g}")
            if abs(x["mae_hold"] - y["mae_hold"]) > 1e-8:
                raise ValueError(f"Hold baseline differs at {key}/{g}: data/metric mismatch")
    groups = {g: {"source_mae": mean(r["groups"][g]["mae_pred"] for r in a.values()),
                  "candidate_mae": mean(r["groups"][g]["mae_pred"] for r in b.values()),
                  "hold_mae": mean(r["groups"][g]["mae_hold"] for r in a.values())} for g in GROUPS}
    weighted = {k: sum(groups[g][k] * n for g, n in GROUPS.items()) / sum(GROUPS.values())
                for k in ("source_mae", "candidate_mae", "hold_mae")}
    improved = weighted["candidate_mae"] <= (1 - MIN_RELATIVE_IMPROVEMENT) * weighted["source_mae"]
    no_arm_regression = all(groups[g]["candidate_mae"] <= (1 + MAX_ARM_REGRESSION) * groups[g]["source_mae"]
                            for g in ("left_arm", "right_arm"))
    beats_hold = all(groups[g]["candidate_mae"] < groups[g]["hold_mae"] for g in GROUPS)
    by_fraction = {str(f): {g: {"source_mae": mean(r["groups"][g]["mae_pred"] for r in a.values() if r["fraction"] == f),
                                 "candidate_mae": mean(r["groups"][g]["mae_pred"] for r in b.values() if r["fraction"] == f),
                                 "hold_mae": mean(r["groups"][g]["mae_hold"] for r in a.values() if r["fraction"] == f)}
                             for g in GROUPS} for f in sorted({r["fraction"] for r in a.values()})}
    def error(row, metric):
        return sum(row["groups"][g][metric] * n for g, n in GROUPS.items()) / sum(GROUPS.values())
    episode_deltas = [mean(error(b[k], "mae_pred") - error(a[k], "mae_pred") for k in a if k[0] == ep)
                      for ep in sorted({ep for ep, _ in a})]
    rng = random.Random(0)
    boot = [mean(rng.choices(episode_deltas, k=len(episode_deltas))) for _ in range(2000)]
    q = quantiles(boot, n=40, method="inclusive")
    motion_cut = median(error(r, "mae_hold") for r in a.values())
    motion_groups = {}
    for label, moving in (("lower_demo_motion", False), ("higher_demo_motion", True)):
        keys = [k for k in a if (error(a[k], "mae_hold") > motion_cut) == moving]
        if keys:
            motion_groups[label] = {"anchors": len(keys), "source_mae": mean(error(a[k], "mae_pred") for k in keys),
                                    "candidate_mae": mean(error(b[k], "mae_pred") for k in keys),
                                    "hold_mae": mean(error(a[k], "mae_hold") for k in keys)}
    intent = {}
    for label, rows in (("source", a), ("candidate", b)):
        intent[label] = {}
        for g in GROUPS:
            if "intent_cos" not in next(iter(rows.values()))["groups"][g]:
                continue
            cos = [r["groups"][g]["intent_cos"] for r in rows.values()]
            finite = [c for c in cos if math.isfinite(c)]
            norm_pred = mean(r["groups"][g]["intent_norm_pred"] for r in rows.values())
            norm_demo = mean(r["groups"][g]["intent_norm_demo"] for r in rows.values())
            m = intent[label][g] = {
                "cos_mean": mean(finite) if finite else float("nan"),
                "frac_positive": sum(c > 0 for c in finite) / len(cos),  # same definition as summary.json
                "intent_norm_pred": norm_pred, "intent_norm_demo": norm_demo,
                "norm_ratio_pred_over_demo": norm_pred / max(norm_demo, 1e-12)}
            m["criteria_met"] = bool(m["cos_mean"] > INTENT_COS_MEAN_MIN and m["frac_positive"] >= INTENT_FRAC_POSITIVE_MIN
                                     and m["norm_ratio_pred_over_demo"] >= INTENT_NORM_RATIO_MIN)
    for f, fr in by_fraction.items():
        for g in GROUPS:
            rs = [r for r in b.values() if str(r["fraction"]) == f]
            if "intent_cos" in rs[0]["groups"][g]:
                vals = [r["groups"][g]["intent_cos"] for r in rs if math.isfinite(r["groups"][g]["intent_cos"])]
                fr[g]["candidate_intent_cos"] = mean(vals) if vals else float("nan")
    return {"anchors": len(a), "episodes": len({ep for ep, _ in a}), "groups": groups, "weighted_upper_body": weighted,
            "paired_episode_bootstrap": {"seed": 0, "resamples": 2000, "unit": "episode, not correlated frames",
                                         "candidate_minus_source_rad": mean(episode_deltas), "percentile_95_interval_rad": [q[0], q[-1]],
                                         "episodes_improved": sum(d < 0 for d in episode_deltas)},
            "diagnostic_motion_bands": {"hold_mae_median_rad": motion_cut, "groups": motion_groups,
                                         "used_for_admission": False},
            "relative_improvement": 1 - weighted["candidate_mae"] / max(weighted["source_mae"], 1e-12),
            "gate": {"improved_at_least_5_percent": improved, "no_arm_regression_above_5_percent": no_arm_regression,
                     "passed": improved and no_arm_regression,
                     "all_upper_body_groups_beat_hold": beats_hold,
                     "strict_passed_including_hold": improved and no_arm_regression and beats_hold},
            "intent_cosine": {"used_for_admission": False, **intent,
                              "candidate_all_groups_meet_criteria": all(v["criteria_met"] for v in intent.get("candidate", {}).values()) if intent.get("candidate") else None},
            "by_fraction": by_fraction, "task_success_evaluated": False,
            "scope": "paired validation-demonstration predictions only; no physical success, generalization or RL claim"}


def self_test():
    def rows(pred):
        return [{"episode": 1, "anchor": 2, "fraction": 0.2, "segment": [0, 100],
                 "groups": {g: {"mae_pred": pred, "mae_hold": 1.0} for g in GROUPS}}]
    checked = compare(rows(1.0), rows(0.9))
    assert checked["gate"]["passed"]
    assert abs(checked["paired_episode_bootstrap"]["percentile_95_interval_rad"][0] + 0.1) < 1e-12
    assert not compare(rows(1.0), rows(1.2))["gate"]["passed"]
    changed = rows(0.9); changed[0]["anchor"] = 4
    try:
        compare(rows(1.0), changed)
    except ValueError:
        pass
    else:
        raise AssertionError("Unpaired anchors accepted")
    print("PASS (CPU synthetic): paired anchors, hold baseline identity and pre-registered offline admission")


def admit(source, candidate, out):
    """Validate protocol identity, compute the paired report, write it to `out` (thresholds first)."""
    source, candidate, out = Path(source), Path(candidate), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing comparison: {out}")
    summaries = [json.loads((d / "summary.json").read_text()) for d in (source, candidate)]
    for k in ("key", "split", "samples_averaged", "n_anchors", "episodes"):
        if summaries[0][k] != summaries[1][k]:
            raise ValueError(f"Evaluation protocol mismatch: {k}")
    for k in ("fractions", "sampler_seed", "num_flow_steps"):   # absent in older summaries; checked when both record it
        if k in summaries[0] and k in summaries[1] and summaries[0][k] != summaries[1][k]:
            raise ValueError(f"Evaluation protocol mismatch: {k}")
    if summaries[0]["split"] != "validation" or any(s.get("task_success_evaluated") is not False for s in summaries):
        raise ValueError("This gate is only for isolated offline validation predictions")
    report = {"preregistered": PREREGISTERED}
    report.update(compare(*(json.loads((d / "rows.json").read_text()) for d in (source, candidate))))
    report["inputs"] = {str(d): {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("rows.json", "summary.json")}
                        for d in (source, candidate)}
    report["checkpoints"] = [s["checkpoint"] for s in summaries]
    report["checkpoint_details"] = {role: {k: s.get(k) for k in ("checkpoint", "checkpoint_step", "checkpoint_sha256",
                                                                  "views", "multiview", "action_repr", "samples_averaged")}
                                    for role, s in zip(("source", "candidate"), summaries)}
    report["verdict"] = "PASS" if report["gate"]["passed"] else "FAIL"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_fraction", "preregistered")}, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", type=Path); ap.add_argument("candidate", nargs="?", type=Path)
    ap.add_argument("--out", type=Path); ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.source or not args.candidate or not args.out:
        ap.error("source, candidate and --out are required")
    if args.out.exists():
        ap.error("Refusing to overwrite an existing comparison")
    admit(args.source, args.candidate, args.out)


if __name__ == "__main__":
    main()
