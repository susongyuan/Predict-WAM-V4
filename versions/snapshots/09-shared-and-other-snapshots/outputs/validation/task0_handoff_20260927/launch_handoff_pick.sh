#!/bin/bash
# Task 0 demo-prefix -> WAM handoff diagnostic (pick) (demo_prefix_handoff; TRAIN instance; never a task/skill result):
# TRAIN demo ep61 on its train instance 65. Demo actions for t < 650 (= start of the demo 'pick up from(radio,coffee_table)' segment), then
# the 3-view WAM alone executes 'pick up from(radio,coffee_table)' for 2 x (segment end - N) steps, K=8 (= admission K), exec horizon 8,
# frozen contract vector, base held during manipulation (Task 0 default). Physical criterion pre-registered in
# replay_demo_actions_feasibility.t0_handoff_spec and copied into the run manifest (pick: radio lowest point >= 0.10 m
# above step 0 and >= 0.05 m at the budget end; press: evaluator ToggledOn goal literal true within the budget).
# Evaluator-side pose log only (never a policy input). The run command takes /tmp/project2_gpu.lock itself.
# Start: systemd-run --user --unit=p2-t0handoff-pick --collect --working-directory=$PWD bash $PWD/outputs/validation/task0_handoff_20260927/launch_handoff_pick.sh
# Stop:  systemctl --user stop p2-t0handoff-pick; then behavior-run-<pid> / wam-handoff-<pid> from the run manifest.
# Knobs: CKPT, NAME, SAMPLES, EXEC, FIXED_NOISE_SEED, WRAPPER, NO_VIDEO
# (headrgbd default: head 720 RGB-D + wrists 224 RGB, VRAM), RESERVE, DRY_RUN=1.
set -euo pipefail
cd /home/user/Robotic/project2
export P2_TASK_ID=0
ckpt=${CKPT:-outputs/training/task0_manip_v1_20260927/checkpoint_final.pt}
name=${NAME:-handoff_pick_ep61_N650_v1final}
[ -f "$ckpt" ] || { echo "missing checkpoint $ckpt" >&2; exit 2; }
extra=()
[ "${DRY_RUN:-0}" = "1" ] && extra=(--dry-run)
noise=()
[ -n "${FIXED_NOISE_SEED:-}" ] && noise=(--wam-fixed-noise-seed "$FIXED_NOISE_SEED")
video=()
[ "${NO_VIDEO:-0}" = "1" ] && video=(--no-video)
args=(run --episode 61 --instance 65 --run-name "$name" --handoff-step 650
      --wam-contract 'pick up from(radio,coffee_table)' --wam-checkpoint "$ckpt"
      --wam-samples "${SAMPLES:-8}" --wam-exec-horizon "${EXEC:-8}" --env-wrapper "${WRAPPER:-headrgbd}"
      --video-width 1280 --video-height 720 --wam-reserve-mib "${RESERVE:-2100}" --seed 0 --timeout "${TIMEOUT:-1800}"
      "${noise[@]}" "${video[@]}" "${extra[@]}")
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec .conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py "${args[@]}"
fi
exec .conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py "${args[@]}" \
  >> "outputs/validation/task0_handoff_20260927/launcher_${name}.log" 2>&1
