#!/usr/bin/env bash
# One explicit orchestrator diagnostic run on a development instance with a chosen WAM checkpoint.
# Usage: NAME=<run> CKPT=<checkpoint.pt> [MANIP_CKPT=<manip checkpoint.pt>] [INSTANCE=10] [MAXSTEPS=6000] [TIMEOUT=1500] [EXEC=8] [ORCH='<json>'] [VIEW=viewer] [VW=1280 VH=720] [JOINTLOG=1] [SAMPLES=1] scripts/run_s1_contract.sh
# MANIP_CKPT (optional): non-navigation contracts sample from this checkpoint; navigation keeps CKPT.
# VIEW=viewer records a native third-person video; JOINTLOG=1 adds evaluator-side task-object joint angles (evidence only).
# No stale-log wait, automatic answer, deletion, or broad process kill. Never run while training holds the GPU.
set -euo pipefail
cd /home/user/Robotic/project2
python3 scripts/behavior_preflight.py --check >/dev/null
name=${NAME:?set NAME}
ckpt=${CKPT:?set CKPT}
orch=${ORCH:-'{"vlm":"none","answer":"none","ask_budget":2,"max_attempts":2,"frozen":true,"precondition":true,"train_targets":"data/skill_vtt_task013/contract_targets.json"}'}
python3 scripts/behavior_closed_loop.py --profile development --instances "${INSTANCE:-10}" \
  --run-name "$name" --max-steps "${MAXSTEPS:-6000}" --timeout "${TIMEOUT:-1500}" \
  --checkpoint "$ckpt" --orchestrate "$orch" --exec-horizon "${EXEC:-8}" --samples "${SAMPLES:-1}" --write-video \
  --video-view "${VIEW:-robot}" --video-width "${VW:-1280}" --video-height "${VH:-720}" ${JOINTLOG:+--log-object-joints} \
  ${MANIP_CKPT:+--manip-checkpoint "$MANIP_CKPT"} || true
python3 scripts/summarize_closed_loop.py "outputs/behavior/closed_loop/$name"
