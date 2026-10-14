#!/usr/bin/env bash
# Task 0 (turning_on_radio) closed-loop run of the full system: RGB-D radio perception + docking controller execute
# move to(radio) (stop at the TRAIN grasp-time relative pose, dock_target_at=grasp, height-aware coffee-table shield);
# the learned three-view WAM executes pick up from(radio,coffee_table) / press(radio) / place on(radio,coffee_table)
# with the base held (right hand picks and holds the radio, left closed fist presses); proprio postconditions
# (harness/task0_predicates.py) complete contracts and trigger recovery (missed grasp -> re-approach + retry).
# press is an onboard-UNOBSERVABLE contact candidate: it advances the program unverified and is never re-pressed
# unless the failed attempt provably had no contact (retry_safe). Program completion or a press candidate is never
# success; only the evaluator's Q/BDDL decides. Evaluator-side object poses/toggled_on are logged for audit only.
# Head 720 RGB-D + wrists 224 RGB (harness/head_rgbd_wrapper.py, VRAM). Takes the GPU lock.
# Episode timeout: the evaluator's default 1.5 x mean demo length (2149.64 frames -> 3224 steps) unless MAXSTEPS.
# Usage: NAME=<run_name> CKPT=<checkpoint.pt> [INST=10] [PROFILE=development] [SAMPLES=1] [EXEC=8] [MAXSTEPS=]
#        [P2_HEAD_RES=720] [VIDEO_VIEW=viewer|robot] [VIDEO_W=1280 VIDEO_H=720] [MAX_ATTEMPTS=3]
#        [ORCH_EXTRA='"answer":"none"'] [DRY_RUN=1] scripts/run_task0_closed_loop.sh
#   DRY_RUN=1 prints the orchestrate JSON and the launcher argv, runs the CPU preflight, and exits (no GPU, no sim).
# Stop: systemctl --user stop <the behavior-run-* / wam-policy-* units listed in the run dir resource_summary.json>
#       (and the launcher's own unit if it was started with systemd-run).
set -euo pipefail
cd /home/user/Robotic/project2
name=${NAME:?set NAME}
ckpt=${CKPT:?set CKPT}
inst=${INST:-10}
profile=${PROFILE:-development}
samples=${SAMPLES:-1}   # User protocol: K=1, independent of exec horizon; explicit legacy override is recorded.
exec_h=${EXEC:-8}
vw=${VIDEO_W:-1280}
vh=${VIDEO_H:-720}
video_view=${VIDEO_VIEW:-viewer}
extra=${ORCH_EXTRA:-}
attempts=${MAX_ATTEMPTS:-3}
program=configs/task000_fixed_program.json
targets=data/skill_vtt_task000/contract_targets.json
# Task 0 manipulation is the WAM's (IK primitives are Task-1 diagnostics; policy_server rejects them for Task 0).
executor=wam
# The base is held during manipulation, so dock at the TRAIN grasp-time relative pose: the WAM is not asked to
# reproduce the sparse base pulses its K-mean prediction averaged to zero in the Task 1 physical handoff.
dock_at=grasp
# Contract budgets = the program's budget_frames (conservative fallbacks above the TRAIN medians); the orchestrator
# uses max(1.5 x median, this minimum).
budgets=$(python3 -c "import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps({s['skill']: int(s['budget_frames']) for s in p['program']}))" "$program")
orch=$(cat <<EOF
{"vlm": "none", "program": "$program", "frozen": true,
 "train_targets": "$targets",
 "navigation": "docking", "postconditions": "task0_physical", "precondition": false,
 "reapproach_on_retry": true, "max_attempts": $attempts,
 "visible_objects": ["radio", "coffee_table"], "perceive_every": 5,
 "min_budget_steps": $budgets,
 "navigation_settle_steps": 10, "manip_snapshot_every": 30,
 "manip_executor": "$executor", "dock_target_at": "$dock_at", "base_control": "hold_during_manipulation"${extra:+, $extra}}
EOF
)
orch=$(python3 -c "import json,sys; print(json.dumps(json.loads(sys.argv[1])))" "$orch")   # validate + one line
args=(--profile "$profile" --run-name "$name" --checkpoint "$ckpt" --orchestrate "$orch"
      --exec-horizon "$exec_h" --samples "$samples" --seed 0 --timeout 3600
      --write-video --video-view "$video_view" --video-width "$vw" --video-height "$vh"
      --log-object-poses --env-wrapper harness.head_rgbd_wrapper.HeadRGBDWrapper)
if [ "$profile" != "report" ]; then args+=(--instances "$inst"); fi
if [ -n "${MAXSTEPS:-}" ]; then args+=(--max-steps "$MAXSTEPS"); fi

# CPU preflight (before the GPU lock): detector module + interface, checkpoint config vs runtime condition vectors.
CUDA_VISIBLE_DEVICES= P2_TASK_ID=0 .conda-envs/lilawam/bin/python -B - "$ckpt" "$targets" <<'PY'
import hashlib, inspect, sys
import harness.radio_perception as rp
assert callable(rp.detect_from_observation) and callable(rp.obstacle_points), "radio_perception interface"
assert "z_band" in inspect.signature(rp.obstacle_points).parameters, "obstacle_points needs z_band (height bands)"
import torch
cfg = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["config"]
want = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
got = cfg["dataset"].get("contract_targets_sha256")
assert got == want, f"checkpoint trained on contract targets {got}, runtime file is {want}"
views = list(cfg["model"].get("views") or [])
assert len(views) == 3, f"expected the three-view WAM, got views={views}"
print(f"preflight OK: radio_perception interface, checkpoint targets sha {want[:12]}, views {len(views)}")
PY

if [ -n "${DRY_RUN:-}" ]; then
  echo "ORCHESTRATE=$orch"
  printf 'LAUNCH: env P2_TASK_ID=0 python3 scripts/behavior_closed_loop.py'; printf ' %q' "${args[@]}"; echo
  exit 0
fi
exec python3 scripts/gpu_lock.py run --owner "task0-closed-loop-$name" -- \
  env P2_TASK_ID=0 python3 scripts/behavior_closed_loop.py "${args[@]}"
