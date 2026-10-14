#!/bin/bash
cd /home/user/Robotic/project2
exec /home/user/Robotic/project2/.conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py run --episode 2787 --instance 276 \
  --run-name handoff_ep2787_N849_stage8final --handoff-step 849 \
  --wam-checkpoint outputs/training/stage8_motionw_manip_20260925/checkpoint_final.pt \
  --wam-samples 8 --budget 3084 --seed 0
