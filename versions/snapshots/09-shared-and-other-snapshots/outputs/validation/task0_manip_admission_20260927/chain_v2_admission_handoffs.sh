#!/bin/bash
# 2026-09-28 04:00 chain: (1) pause v2 training at 05:15 CST if early stopping has not fired (gate_v2 wall-clock clause);
# (2) after the training unit stops: registered admission of v2 checkpoint_es_best.pt (gate_v2.json, K=8, once);
# (3) physical demo-prefix handoffs with the same checkpoint on TRAIN instance 65 (pick N=650, press N=1348; diagnostics).
# Every GPU step takes /tmp/project2_gpu.lock. Stop: systemctl --user stop p2-chain-task0-v2-eval
set -uo pipefail
cd /home/user/Robotic/project2
unit=wam-train-task0_manip_v2_20260928
A=outputs/validation/task0_manip_admission_20260927
H=outputs/validation/task0_handoff_20260927
ck=outputs/training/task0_manip_v2_20260928/checkpoint_es_best.pt
while systemctl --user is-active --quiet "$unit"; do
  if [ "$(date +%H%M)" -ge 0515 ] && [ "$(date +%H%M)" -lt 0900 ]; then
    echo "[chain] 05:15 wall-clock pause: SIGTERM $unit at $(date +%T)"
    systemctl --user kill --kill-who=main --signal=SIGTERM "$unit"
    while systemctl --user is-active --quiet "$unit"; do sleep 10; done
    break
  fi
  sleep 20
done
echo "[chain] training unit stopped at $(date +%T)"
# wait for the lock releaser to free the GPU
python3 scripts/gpu_lock.py run --owner p2-admission-task0-v2 -- env P2_TASK_ID=0 .conda-envs/lilawam/bin/python \
  scripts/check_task0_manip_admission.py --gate $A/gate_v2.json --checkpoint $ck --out $A/v2_esbest --duty-cycle 0.8 \
  > $A/admission_v2_esbest.log 2>&1
echo "[chain] admission rc=$? at $(date +%T)"
CKPT=$ck NAME=handoff_pick_ep61_N650_v2esbest bash $H/launch_handoff_pick.sh
echo "[chain] pick handoff rc=$? at $(date +%T)"
CKPT=$ck NAME=handoff_press_ep61_N1348_v2esbest bash $H/launch_handoff_press.sh
echo "[chain] press handoff rc=$? at $(date +%T)"
