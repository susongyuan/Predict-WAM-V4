#!/bin/bash
# Task 1 demo-prefix -> WAM handoff diagnostic (a) (demo_prefix_handoff; train instance; never a task/skill result):
# train demo ep389 on its train instance 290. Demo actions for t < 920, then the 3-view WAM alone executes
# 'pick up from(trash_can,floors)' (right hand) for 2 x (segment end - N) steps. Physical criterion pre-registered in
# the run manifest (handoff_criterion): bin lowest point >= 0.10 m above step 0, still >= 0.05 m at the end.
# The run command itself waits for / takes /tmp/project2_gpu.lock (acquire_gpu_lock) and releases it after the
# simulator unit is gone.
# Start: systemd-run --user --unit=p2-t1handoff-a --collect bash outputs/validation/task1_handoff_20260926/launch_handoff_a_pick_bin.sh
# Stop:  systemctl --user stop p2-t1handoff-a; then stop behavior-run-<pid> / wam-handoff-<pid> if still active
#        (names in <run dir>/manifest.json evaluator_unit / server_unit); rm /tmp/project2_gpu.lock only if it holds NAME.
# Knobs: CKPT, NAME, VIDEO=0 (no 1280x720 viewer video), WRAPPER=default (224 RGB; default rgbd = RGBDFullResWrapper
#        head 720 / wrists 480; headrgbd = head 720 RGB-D + wrists 224 RGB), RESERVE (policy VRAM reserve, measured
#        3-view K8 peak 1964 MiB on 2026-09-27), SAMPLES, EXEC, DRY_RUN=1 (CPU check, no GPU, prints argv).
set -euo pipefail
cd /home/user/Robotic/project2
export P2_TASK_ID=1
ckpt=${CKPT:-outputs/training/task1_manip_v2_20260926/checkpoint_final.pt}
name=${NAME:-handoff_a_pickbin_ep389_N920_v2final}
[ -f "$ckpt" ] || { echo "missing checkpoint $ckpt" >&2; exit 2; }
video=(--video-width 1280 --video-height 720)
[ "${VIDEO:-1}" = "0" ] && video=(--no-video)
extra=()
[ "${DRY_RUN:-0}" = "1" ] && extra=(--dry-run)
args=(run --episode 389 --instance 290 --run-name "$name" --handoff-step 920
      --wam-contract 'pick up from(trash_can,floors)' --wam-checkpoint "$ckpt"
      --wam-samples "${SAMPLES:-8}" --wam-exec-horizon "${EXEC:-8}" --env-wrapper "${WRAPPER:-rgbd}" "${video[@]}"
      --wam-reserve-mib "${RESERVE:-2300}" ${BASE_DEMO:+--wam-base-from-demo} --seed 0 --timeout 5400 "${extra[@]}")
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec .conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py "${args[@]}"
fi
exec .conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py "${args[@]}" \
  >> "outputs/validation/task1_handoff_20260926/launcher_${name}.log" 2>&1
