#!/usr/bin/env bash
# usage: run_attempt.sh <n> <python> <script> [script-args...]   (env vars for the child are taken from ATTEMPT_ENV, space-separated K=V or -u K)
set -u
n="$1"; shift
ROOT=/home/user/Robotic/project2
LOG="$ROOT/outputs/behavior/isaac_probe/attempt_${n}.log"
UNIT="isaac-probe-${n}-$$"
{
  echo "### attempt $n  unit=$UNIT  $(date -u +%FT%TZ)"
  echo "### env: ${ATTEMPT_ENV:-<none>}"
  echo "### argv: $*"
  echo "### gpu before: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
} > "$LOG"
# shellcheck disable=SC2086
systemd-run --user --quiet --wait --pipe --collect --unit="$UNIT" --working-directory="$ROOT" \
  -p 'CPUAffinity=0 1 2 3' -p MemoryMax=12G -p MemorySwapMax=0 \
  env ${ATTEMPT_ENV:-} timeout 300 "$@" >> "$LOG" 2>&1
rc=$?
echo "### exit=$rc  $(date -u +%FT%TZ)  gpu after: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$LOG"
echo "attempt $n exit=$rc"
exit $rc
