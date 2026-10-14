#!/bin/bash
# Task 1 demo-action replay diagnostic (demo_action_replay, policy_result=false): train demo ep389 on its train instance 290.
# Waits for /tmp/project2_gpu.lock (held by WAM training until ~07:05), then runs the guarded evaluator (--mode train,
# RGBDFullResWrapper, --log-object-poses, viewer video 1280x720) and the analysis.
cd /home/user/Robotic/project2
export P2_TASK_ID=1
exec /home/user/Robotic/project2/.conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py run \
  --episode 389 --instance 290 --run-name run_ep389_inst290_rgbd --env-wrapper rgbd --capture-every 10 \
  --seed 0 --timeout 7200 \
  >> outputs/validation/task1_demo_replay_20260926/launcher_run_ep389_inst290_rgbd.log 2>&1
