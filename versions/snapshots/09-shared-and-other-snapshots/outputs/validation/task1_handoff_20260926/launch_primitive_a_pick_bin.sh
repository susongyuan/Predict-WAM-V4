#!/bin/bash
# Task 1 demo-prefix -> perception+IK PRIMITIVE handoff diagnostic (a') (train instance; never a task/skill result):
# train demo ep389 on its train instance 290. Demo actions for t < 920 (bin pick-segment start), then
# harness/grasp_primitive.py executes 'pick up from(trash_can,floors)' with the right hand (rim grasp) (no WAM loaded).
# Target = trash_can_116 centre in the base frame at N from the deterministic reference replay's evaluator pose log
# (run_ep389_inst290_rgbd, step 920: 0.703, -0.126, 0.13) = ORACLE position: checks grasp geometry / IK / controller
# / gripper physics only (perception is tested separately). ALIGN=1 (default) first servoes the base on odometry to
# the grasp-time dock pose. Same pre-registered criterion as (a): bin lowest point >= 0.10 m above step 0, >= 0.05 m
# at the end (evaluator pose log only).
# Start: systemd-run --user --unit=p2-t1prim-a --collect bash outputs/validation/task1_handoff_20260926/launch_primitive_a_pick_bin.sh
# PRIM=pick_can,place_in chains the place into the held bin (inside reported from the pose-log goal literals).
# Stop:  systemctl --user stop p2-t1prim-a; then behavior-run-<pid> / wam-handoff-<pid> from the run manifest.
set -euo pipefail
cd /home/user/Robotic/project2
export P2_TASK_ID=1
name=${NAME:-prim_a_pickbin_ep389_N920_oracle}
align=(--primitive-align)
[ "${ALIGN:-1}" = "0" ] && align=()
extra=()
[ "${DRY_RUN:-0}" = "1" ] && extra=(--dry-run)
args=(run --episode 389 --instance 290 --run-name "$name" --handoff-step 920
      --wam-contract 'pick up from(trash_can,floors)' --wam-checkpoint outputs/training/task1_manip_v2_20260926/checkpoint_final.pt
      --primitive "${PRIM:-pick_bin}" --primitive-target "${TARGET:-0.703,-0.126,0.13}" "${align[@]}"
      --env-wrapper headrgbd --video-width 1280 --video-height 720 --wam-reserve-mib 300
      ${EARLY:+--early-stop-steps $EARLY} --seed 0 --timeout 5400 "${extra[@]}")
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec .conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py "${args[@]}"
fi
exec .conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py "${args[@]}" \
  >> "outputs/validation/task1_handoff_20260926/launcher_${name}.log" 2>&1
