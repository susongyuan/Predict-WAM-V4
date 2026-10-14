#!/bin/bash
# Task 0 demo-action replay diagnostic (demo_action_replay, policy_result=false): TRAIN demo ep61 on its train instance 65
# (mapping by metadata task_instance_id, checked in /tmp/p2_t0_scratch/selftest_t0.json: radio 0.36 m, id-1 1.55 m,
# id+1 2.48 m). Purpose: (1) does the Isaac-6.1 port + evaluator turn the radio on (toggled_on, Q) when the recorded
# actions are replayed open loop -> evaluation validity for every later Task 0 result; (2) head 720 RGB-D + evaluator
# poses captured every 10 steps for the radio detector's 3-D evaluation (evaluator-side truth, never fed to a policy).
# The replay script takes /tmp/project2_gpu.lock itself. Stop: systemctl --user stop p2-t0replay-ep61, then the
# behavior-run-* unit named in the run manifest.
cd /home/user/Robotic/project2
export P2_TASK_ID=0
exec /home/user/Robotic/project2/.conda-envs/lilawam/bin/python3 scripts/replay_demo_actions_feasibility.py run \
  --episode 61 --instance 65 --run-name run_ep61_inst65_rgbd --env-wrapper rgbd --capture-every 10 \
  --seed 0 --timeout 5400 \
  >> outputs/validation/task0_handoff_20260927/launcher_run_ep61_inst65_rgbd.log 2>&1
