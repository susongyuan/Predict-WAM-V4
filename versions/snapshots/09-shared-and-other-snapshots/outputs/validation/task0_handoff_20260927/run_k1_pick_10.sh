#!/bin/bash
set -euo pipefail
cd /home/user/Robotic/project2

root=outputs/validation/task0_handoff_20260927
ckpt=outputs/training/task0_manip_v2_20260928/checkpoint_es_best.pt
tag=${TAG:-r2}
reserve=${RESERVE:-1800}
for seed in $(seq 0 9); do
  name="handoff_pick_ep61_N650_v2esbest_k1_${tag}_seed${seed}"
  echo "[k1-10] start seed=$seed $(date -Is)"
  CKPT="$ckpt" NAME="$name" SAMPLES=1 FIXED_NOISE_SEED="$seed" RESERVE="$reserve" TIMEOUT=1800 \
    NO_VIDEO="$([ "$seed" -eq 0 ] && echo 0 || echo 1)" \
    bash "$root/launch_handoff_pick.sh"
  echo "[k1-10] finish seed=$seed $(date -Is)"
done

.conda-envs/lilawam/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/validation/task0_handoff_20260927")
tag = "r2"
rows = []
for seed in range(10):
    run = root / f"handoff_pick_ep61_N650_v2esbest_k1_{tag}_seed{seed}"
    summary = json.loads((run / "analysis/summary.json").read_text())
    h = summary["handoff"]
    rows.append({
        "seed": seed,
        "criterion_met": bool(summary["verdict"]["criterion_met"]),
        "lift_max_m": float(summary["verdict"]["b_radio_lift_max_m"]),
        "lift_end_m": float(h["radio_lowest_point_lift_at_end_m"]),
        "right_first_close_cmd_step": h["gripper"]["right"]["first_close_cmd_step"],
        "right_wam_closed_fraction": h["gripper"]["right"]["wam_closed_fraction"],
        "q_score": float(summary["verdict"]["a_q_score"]),
        "run": str(run),
    })
out = {
    "experiment": "Task0 fixed physical pick handoff; K=1; fixed flow-noise seeds 0..9",
    "checkpoint": "outputs/training/task0_manip_v2_20260928/checkpoint_es_best.pt",
    "episode": 61,
    "instance": 65,
    "handoff_step": 650,
    "budget_steps": 1396,
    "success_rule": "radio lift >=0.10m sometime and >=0.05m at budget end",
    "successes": sum(r["criterion_met"] for r in rows),
    "trials": len(rows),
    "success_rate": sum(r["criterion_met"] for r in rows) / len(rows),
    "rows": rows,
}
(root / f"k1_pick_10_{tag}_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
