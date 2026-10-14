#!/usr/bin/env bash
# Launch one WAM (adaptation) training run as a transient user systemd unit with the same cgroup limits used on
# 2026-09-22 (CPU 0-15, MemoryMax=20G, no swap). GPU-exclusive: never start while a simulator run is active.
# Usage: [P2_TASK_ID=<id>] NAME=<run_name> CONFIG=<yaml> INIT=<checkpoint.pt> scripts/run_train_unit.sh
# systemd-run does not inherit the caller's environment: P2_TASK_ID (default 13 = Task 13 paths, unchanged) is passed
# into the unit explicitly. GPU lock protocol: `scripts/gpu_lock.py --owner wam-train-<run_name> acquire` before, then
# `scripts/gpu_lock.py --owner wam-train-<run_name> --unit wam-train-<run_name> hold-for-unit` after launching; this
# script refuses to start while /tmp/project2_gpu.lock is held by any other owner.
# Stop safely (saves checkpoint_last at the next step boundary):
#   systemctl --user kill --kill-who=main --signal=SIGTERM wam-train-<run_name>
set -euo pipefail
cd /home/user/Robotic/project2
name=${NAME:?set NAME}
config=${CONFIG:?set CONFIG}
init=${INIT:?set INIT}
task_id=${P2_TASK_ID:-13}
if ! [[ "$task_id" =~ ^[0-9]+$ ]]; then echo "P2_TASK_ID must be an integer, got '$task_id'" >&2; exit 2; fi
lock_owner=$(cat /tmp/project2_gpu.lock 2>/dev/null || true)
if [ -n "$lock_owner" ] && [ "$lock_owner" != "wam-train-$name" ]; then echo "GPU lock held by '$lock_owner'; refusing to start training" >&2; exit 3; fi
if [ -e "outputs/training/$name" ]; then echo "outputs/training/$name exists; choose a new NAME" >&2; exit 2; fi
if pgrep -f '^\S*python\S* -m omnigibson\.eval\.eval' >/dev/null; then echo "simulator running; refusing to share the GPU" >&2; exit 3; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 4000 ]; then echo "GPU already holds ${used} MiB; refusing to start training" >&2; exit 3; fi
mkdir -p "outputs/training/$name"
systemd-run --user --quiet --collect --unit="wam-train-$name" --working-directory="$PWD" \
  -p "CPUAffinity=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" -p MemoryMax=20G -p MemorySwapMax=0 \
  -p StandardOutput=append:"$PWD/outputs/training/$name/launch.log" -p StandardError=append:"$PWD/outputs/training/$name/launch.log" \
  /usr/bin/env P2_TASK_ID="$task_id" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  "$PWD/.conda-envs/lilawam/bin/python" "$PWD/scripts/train_lilawam_behavior.py" --config "$config" \
  --run-name "$name" --init-from "$init" --num-workers 4
sleep 2
systemctl --user show "wam-train-$name" -p MainPID -p ActiveState --no-pager
