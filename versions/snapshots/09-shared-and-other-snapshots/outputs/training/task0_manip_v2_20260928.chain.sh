#!/bin/bash
# Chain launcher (2026-09-28 01:20): wait for the two Task-0 v1 physical handoff diagnostics (units p2-t0handoff-pick,
# p2-t0handoff-press) to finish, take /tmp/project2_gpu.lock as wam-train-task0_manip_v2_20260928, start the v2
# training unit (scripts/run_train_unit.sh: CPU 0-15, MemoryMax 20G) and the lock releaser (frees the lock when the
# training unit stops). Config: configs/lilawam_task000_multiview_manip_v2_train.yaml; init = v1 final (SHA pinned in
# the config). Stop the chain before it launches: systemctl --user stop p2-chain-task0-v2.
# Stop the training (saves checkpoint_last incl. early-stopping state at a step boundary):
#   systemctl --user kill --kill-who=main --signal=SIGTERM wam-train-task0_manip_v2_20260928
set -euo pipefail
cd /home/user/Robotic/project2
name=task0_manip_v2_20260928
for u in p2-t0handoff-pick p2-t0handoff-press; do
  while systemctl --user is-active --quiet "$u"; do sleep 15; done
done
echo "[chain] handoff units done at $(date +%T)"
python3 scripts/gpu_lock.py acquire --owner "wam-train-$name"
P2_TASK_ID=0 NAME="$name" CONFIG=configs/lilawam_task000_multiview_manip_v2_train.yaml \
  INIT=outputs/training/task0_manip_v1_20260927/checkpoint_final.pt scripts/run_train_unit.sh
python3 scripts/gpu_lock.py hold-for-unit --owner "wam-train-$name" --unit "wam-train-$name"
echo "[chain] training launched at $(date +%T)"
