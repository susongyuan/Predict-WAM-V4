#!/usr/bin/env bash
# Legacy entry point: one feasibility run, no automatic comparison or training restart.
set -euo pipefail
cd /home/user/Robotic/project2
python3 scripts/behavior_preflight.py --check >/dev/null
name=${NAME:-s1_program_$(date +%Y%m%d_%H%M%S)}
python3 scripts/behavior_closed_loop.py --profile development --instances 10 \
  --run-name "$name" --max-steps "${MAXSTEPS:-300}" --timeout "${TIMEOUT:-1200}" \
  --checkpoint "${CKPT:-outputs/training/stage1_skillproto/checkpoint_best.pt}" \
  --program configs/task013_fixed_schedule.json --exec-horizon 8 --write-video
python3 scripts/summarize_closed_loop.py "outputs/behavior/closed_loop/$name"
