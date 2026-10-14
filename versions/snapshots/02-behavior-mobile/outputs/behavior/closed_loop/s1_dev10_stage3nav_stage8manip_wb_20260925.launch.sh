#!/usr/bin/env bash
cd /home/user/Robotic/project2
trap 'rm -f /tmp/project2_gpu.lock' EXIT
NAME=s1_dev10_stage3nav_stage8manip_wb_20260925 \
CKPT=outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt \
MANIP_CKPT=outputs/training/stage8_motionw_manip_20260925/checkpoint_final.pt \
INSTANCE=10 MAXSTEPS=6000 TIMEOUT=1800 EXEC=32 SAMPLES=8 VIEW=viewer VW=1280 VH=720 JOINTLOG=1 \
ORCH='{"vlm":"none","answer":"none","ask_budget":2,"max_attempts":2,"frozen":true,"precondition":true,"train_targets":"data/skill_vtt_task013/contract_targets.json","base_control":"wam_whole_body","reapproach_on_retry":false,"move_to_base_gain":[4.0,4.0,2.0],"arrival_gate":"patch","postcondition_gate":"patch","samples_navigation":1,"manip_snapshot_every":15,"offload_inactive":true}' \
scripts/run_s1_contract.sh
echo "launcher exit rc=$?"
