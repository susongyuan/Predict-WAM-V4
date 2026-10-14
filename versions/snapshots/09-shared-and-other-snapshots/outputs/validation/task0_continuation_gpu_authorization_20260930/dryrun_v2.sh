#!/usr/bin/env bash
# CPU preview only. This file deliberately has no execute mode.
set -euo pipefail
root=/home/user/Robotic/project2
out="$root/outputs/validation/task0_continuation_gpu_authorization_20260930"
exec env -u INIT -u RUN_NAME \
    DRY_RUN=1 \
    GPU_AUTHORIZATION="$root/outputs/validation/task0_gpu_peak_authorization_20260930/authorization.json" \
    NAME=task0_v3_cont20k_20260929_gpupeak_20260930_v2_s035000 MODE=long MAX_STEPS=35000 \
    CONFIG="$root/configs/lilawam_task000_multiview_manip_v3_micro16_30k_train.yaml" \
    CACHE_LEDGER="$root/data/dino_cache_task000/sha256_ledger_micro16_30k_v1.json" \
    APPROVAL="$out/approval_20k_gpu_peak_v2.json" \
    RESUME="$root/outputs/training/task0_v3_30k_20260929_s030000/checkpoint_last.pt" \
    RESUME_SHA256=3fe60483a4ee7087105b528ca787d84d790b4bf3f14d80e6ea5882b0bf56f75b \
    CONTINUATION_PLAN="$out/continuation_plan_v2.json" \
    bash "$root/scripts/run_wam_v3_unit.sh"
