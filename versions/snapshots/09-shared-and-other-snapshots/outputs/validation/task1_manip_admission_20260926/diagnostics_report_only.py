"""Report-only diagnostics on admission rows (does not change the pre-registered verdict)."""
import json, sys
import numpy as np
D = "/home/user/Robotic/project2/outputs/validation/task1_manip_admission_20260926"
MOVE = 0.05
out = {}
for tag in sys.argv[1:]:
    rows = json.load(open(f"{D}/{tag}/rows.json"))
    res = {}
    for key, sets in rows.items():
        fx, ev = sets["fixed"], sets["event"]
        g_out = {}
        for g in ("trunk", "left_arm", "right_arm"):
            mv = [r for r in fx if r["groups"][g]["intent_norm_demo"] >= MOVE]
            if not mv:
                continue
            nd = np.mean([r["groups"][g]["intent_norm_demo"] for r in mv])
            g_out[g] = {"moving": len(mv),
                        "ratio_kmean": round(float(np.mean([r["groups"][g]["intent_norm_pred"] for r in mv]) / nd), 3),
                        "ratio_single_sample": round(float(np.mean([r["groups"][g]["intent_norm_k1"] for r in mv]) / nd), 3),
                        "cos_single_sample": round(float(np.nanmean([r["groups"][g]["intent_cos_k1"] for r in mv])), 3),
                        "ratio_kmean_first_half_fracs": round(float(np.mean([r["groups"][g]["intent_norm_pred"] for r in mv if r["fraction"] <= 0.4]) /
                                                                     max(np.mean([r["groups"][g]["intent_norm_demo"] for r in mv if r["fraction"] <= 0.4] or [np.nan]), 1e-9)), 3)
                                                          if any(r["fraction"] <= 0.4 for r in mv) else None}
        e_out = {}
        for hand in ("left", "right"):
            es = [e for e in ev if e["hand"] == hand and not e["event"]["double_flip"]]
            if not es:
                continue
            te = [e["event"]["timing_error_steps"] for e in es if e["event"]["timing_error_steps"] is not None]
            e_out[hand] = {"n": len(es), "direction": sorted({e["direction"] for e in es}),
                           "start_ok": round(float(np.mean([e["event"]["start_ok"] for e in es])), 2),
                           "end_ok": round(float(np.mean([e["event"]["end_ok"] for e in es])), 2),
                           "switch_within_window": round(float(np.mean([e["event"]["both_endpoints"] for e in es])), 2),
                           "early_already_switched_at_step0": round(float(np.mean([(not e["event"]["start_ok"]) and e["event"]["end_ok"] for e in es])), 2),
                           "late_not_switched_by_step31": round(float(np.mean([e["event"]["start_ok"] and not e["event"]["end_ok"] for e in es])), 2),
                           "timing_error_steps_median": float(np.median(te)) if te else None,
                           "single_sample_sign_agree_steady": None}
        res[key] = {"upper_body": g_out, "gripper_events": e_out}
    out[tag] = res
print(json.dumps(out, indent=1))
