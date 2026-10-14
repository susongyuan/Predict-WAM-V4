#!/usr/bin/env bash
# Explicit diagnostic run; no stale-log wait, automatic answer, deletion, or broad process kill.
set -euo pipefail
cd /home/user/Robotic/project2
python3 scripts/behavior_preflight.py --check >/dev/null
name=${NAME:-s1_orchestrated_$(date +%Y%m%d_%H%M%S)}
orch='{"vlm":"none","answer":"none","ask_budget":2,"max_attempts":2,"frozen":true,"precondition":true}'
python3 scripts/behavior_closed_loop.py --profile development --instances 10 \
  --run-name "$name" --max-steps "${MAXSTEPS:-300}" --timeout "${TIMEOUT:-1200}" \
  --orchestrate "$orch" --exec-horizon 8 --write-video
python3 scripts/summarize_closed_loop.py "outputs/behavior/closed_loop/$name"
