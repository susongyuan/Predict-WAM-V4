#!/usr/bin/env bash
# Task 1 (picking_up_trash) closed-loop run of the full system: perception docking for move-to, manipulation by
# EXECUTOR (wam = the learned three-view WAM nominal manipulation policy; primitive = perception+IK fallback),
# calibrated proprio postconditions, recovery (re-approach / rollback /
# skip). Head 720 RGB-D + wrists 224 RGB (harness/head_rgbd_wrapper.py, VRAM). Takes the GPU lock.
# Usage: NAME=<run_name> CKPT=<checkpoint.pt> [INST=10] [PROFILE=development] [SAMPLES=8] [EXEC=8] [MAXSTEPS=]
#        [EXECUTOR=primitive|wam]
#        [P2_HEAD_RES=720] [VIDEO_VIEW=viewer|robot] [VIDEO_W=1280 VIDEO_H=720]
#        [ORCH_EXTRA='"answer":"none"'] scripts/run_task1_closed_loop.sh
# Stop: systemctl --user stop <the behavior-run-* / wam-policy-* units listed in the run dir resource_summary.json>
set -euo pipefail
cd /home/user/Robotic/project2
name=${NAME:?set NAME}
ckpt=${CKPT:?set CKPT}
inst=${INST:-10}
profile=${PROFILE:-development}
samples=${SAMPLES:-8}   # = K of the offline admission gate (K-mean estimator must match what was admitted)
exec_h=${EXEC:-8}
vw=${VIDEO_W:-1280}
vh=${VIDEO_H:-720}
video_view=${VIDEO_VIEW:-viewer}
extra=${ORCH_EXTRA:-}
executor=${EXECUTOR:-wam}
# Manipulation holds the base for both executors. Dock at the train grasp-time relative pose so the WAM is not asked
# to reproduce the sparse 6--21 cm base pulses that its K-mean prediction averaged to zero in the physical handoff.
dock_at=grasp
orch=$(cat <<EOF
{"vlm": "none", "program": "configs/task001_fixed_program.json", "frozen": true,
 "train_targets": "data/skill_vtt_task001/contract_targets.json",
 "navigation": "docking", "postconditions": "task1_physical", "precondition": false,
 "rollback_on_release_failure": true, "skip_failed_objects": true, "reapproach_on_retry": true, "max_attempts": 3,
 "visible_objects": ["trash_can", "can_of_soda", "floors"], "perceive_every": 5,
 "min_budget_steps": {"move to": 2000, "pick up from": 700, "place in": 450},
 "navigation_settle_steps": 10, "manip_snapshot_every": 30,
 "manip_executor": "$executor", "dock_target_at": "$dock_at", "base_control": "hold_during_manipulation"${extra:+, $extra}}
EOF
)
args=(--profile "$profile" --run-name "$name" --checkpoint "$ckpt" --orchestrate "$orch"
      --exec-horizon "$exec_h" --samples "$samples" --seed 0 --timeout 4200
      --write-video --video-view "$video_view" --video-width "$vw" --video-height "$vh"
      --log-object-poses --env-wrapper harness.head_rgbd_wrapper.HeadRGBDWrapper)
if [ "$profile" != "report" ]; then args+=(--instances "$inst"); fi
if [ -n "${MAXSTEPS:-}" ]; then args+=(--max-steps "$MAXSTEPS"); fi
exec python3 scripts/gpu_lock.py run --owner "task1-closed-loop-$name" -- \
  env P2_TASK_ID=1 python3 scripts/behavior_closed_loop.py "${args[@]}"
