#!/usr/bin/env bash
# Launch ONE explicitly requested V3 run. This script never chains tiny -> pilot -> long.
# Required: NAME (or RUN_NAME), CONFIG, CACHE_LEDGER. MODE defaults to tiny.
# Optional: MAX_STEPS (absolute target, defaults tiny=200 / pilot=2000 / long=100000),
#           RESUME or INIT (mutually exclusive), APPROVAL (mandatory for long),
#           RESUME_SHA256, LIMIT_NOFILE (default/minimum 8192).
# Continuation execution also requires GPU_AUTHORIZATION naming the pinned actual-user
# receipt. Numeric/boolean environment overrides cannot authorize GPU work. DRY_RUN
# only previews commands and resource metadata; it never grants training approval.
# CONFIG is deliberately explicit: B32 OOMed; the verified pilot uses the separate
# configs/lilawam_task000_multiview_manip_v3_micro16_train.yaml (microbatch16, accum2).
# CACHE_LEDGER must match that config; never substitute the original B32 receipt.
#
# Example ONLY (not executed by --help or bash -n):
# NAME=task0_v3_pilot_new MODE=pilot MAX_STEPS=2000 \
# CONFIG=configs/lilawam_task000_multiview_manip_v3_micro16_train.yaml \
# CACHE_LEDGER=data/dino_cache_task000/sha256_ledger_micro16_v1.json \
# RESUME=outputs/training/OLD_RUN/checkpoint_last.pt bash scripts/run_wam_v3_unit.sh
# Long mode additionally requires APPROVAL=path/to/reviewed_approval.json. The file
# is passed to the trainer for its own checks, never synthesized or bypassed here.
#
# SAFE STOP (SIGTERM to trainer main process, NOT SIGKILL / whole worker group):
#   systemctl --user kill --kill-who=main --signal=SIGTERM wam-train-NAME.service
# Then monitor its log/checkpoint_last and wait for MainPID=0 and GPU-lock release.
# Do NOT manually release the GPU lock while MainPID is alive. systemctl stop also
# uses SIGTERM (KillMode=mixed, 15-minute save timeout), but the command above avoids
# starting the stop timeout while a large checkpoint is being saved.
# Resume always uses a NEW NAME; a model-only best/periodic file is not resumable.
set -Eeuo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    sed -n '2,/^set -Eeuo pipefail/{ /^#/s/^# \{0,1\}//p; }' "${BASH_SOURCE[0]}"
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use environment variables, or --help; no positional arguments.' >&2; exit 2; }

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$ROOT"
PYTHON="$ROOT/.conda-envs/lilawam/bin/python"
LOCK_TOOL="$ROOT/scripts/gpu_lock.py"
LOCK_FILE=/tmp/project2_gpu.lock

die() { printf 'V3 launcher: %s\n' "$*" >&2; exit 2; }
[[ -z ${NAME:-} || -z ${RUN_NAME:-} || $NAME == "$RUN_NAME" ]] || die 'NAME and RUN_NAME disagree'
name=${NAME:-${RUN_NAME:-}}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || die 'set a safe NAME/RUN_NAME (1-80 characters; no path)'
[[ ${P2_TASK_ID:-0} == 0 ]] || die 'P2_TASK_ID must be 0; refusing a conflicting task environment'
mode=${MODE:-tiny}
case "$mode" in
    tiny)  default_steps=200;    cap=2000 ;;
    pilot) default_steps=2000;   cap=2000 ;;
    long)  default_steps=100000; cap=100000 ;;
    *) die 'MODE must be tiny, pilot or long' ;;
esac
steps=${MAX_STEPS:-$default_steps}
[[ $steps =~ ^[0-9]{1,6}$ ]] || die 'MAX_STEPS must be a positive integer'
steps=$((10#$steps))
(( steps > 0 && steps <= cap )) || die "MAX_STEPS must be 1..$cap for $mode (absolute target, not extra steps)"
config=${CONFIG:?set CONFIG explicitly; use the micro16 config for the verified pilot}
cache_ledger=${CACHE_LEDGER:?set CACHE_LEDGER explicitly to the config-matched receipt path}
[[ -f $config ]] || die "config not found: $config"
[[ ! -d $cache_ledger ]] || die 'CACHE_LEDGER must be a JSON file path, not a directory'
[[ -z ${RESUME:-} || -z ${INIT:-} ]] || die 'RESUME and INIT are mutually exclusive'
[[ -n ${RESUME:-} || -z ${RESUME_SHA256:-} ]] || die 'RESUME_SHA256 requires RESUME'
limit=${LIMIT_NOFILE:-8192}
[[ $limit =~ ^[0-9]{1,7}$ ]] || die 'LIMIT_NOFILE must be an integer'
limit=$((10#$limit))
(( limit >= 8192 )) || die 'LIMIT_NOFILE below 8192 would reintroduce the mmap FD exhaustion failure'
[[ -x $PYTHON && -f $LOCK_TOOL ]] || die 'lilawam Python or gpu_lock.py missing'
command -v systemd-run >/dev/null || die 'systemd-run unavailable'
command -v systemctl >/dev/null || die 'systemctl unavailable'

trainer=("$PYTHON" "$ROOT/scripts/train_wam_v3.py" --config "$config" --cache-ledger "$cache_ledger"
         --run-name "$name" --mode "$mode" --max-steps "$steps" --device cuda --num-workers 0)
if [[ -n ${RESUME:-} ]]; then
    [[ -f $RESUME ]] || die "resume checkpoint not found: $RESUME"
    trainer+=(--resume-source "$RESUME")
    if [[ -n ${RESUME_SHA256:-} ]]; then
        [[ $RESUME_SHA256 =~ ^[0-9a-f]{64}$ ]] || die 'RESUME_SHA256 must be 64 lowercase hex characters'
        trainer+=(--resume-source-sha256 "$RESUME_SHA256")
    else
        [[ -f ${RESUME}.sha256.json ]] || die 'resume SHA256 sidecar missing; provide a verified RESUME_SHA256 instead'
    fi
else
    init=${INIT:-outputs/training/task0_manip_v2_20260928/checkpoint_es_best.pt}
    [[ -f $init ]] || die "initialization checkpoint not found: $init"
    trainer+=(--init-from "$init")
fi
if [[ $mode == long ]]; then
    [[ -n ${APPROVAL:-} && -f $APPROVAL ]] || die 'long requires APPROVAL pointing to a reviewed approval JSON'
    trainer+=(--long-run-approval "$APPROVAL")
elif [[ -n ${APPROVAL:-} ]]; then
    die 'APPROVAL is only meaningful with MODE=long; refusing ambiguous invocation'
fi

if [[ -n ${CONTINUATION_PLAN:-} ]]; then
    [[ $mode == long && -n ${RESUME:-} && -f $CONTINUATION_PLAN ]] || die 'continuation requires long, RESUME and an existing plan'
    trainer+=(--continuation-plan "$CONTINUATION_PLAN")
fi
resource_json=
resource_env=()
if [[ -n ${GPU_AUTHORIZATION:-} || ( -n ${CONTINUATION_PLAN:-} && ${DRY_RUN:-0} != 1 ) ]]; then
    # Independent revalidation; inheriting a controller's environment is not consent.
    resource_json=$(/usr/bin/python3 -B "$ROOT/scripts/wam_v3_resource_authorization.py" \
        --authorization "${GPU_AUTHORIZATION:-}") || die 'GPU resource authorization rejected'
    resource_env=("GPU_AUTHORIZATION=$GPU_AUTHORIZATION"
        P2_GPU_AUTHORIZATION_SHA256=9bd06b2fbdda711378c8e31b2b7605da3fd3ed0453e6ebf2c797925ec5c69cad
        P2_GPU_UTILIZATION_ALLOWED=100 P2_TOTAL_VRAM_ABORT_MIB=15300
        P2_CPU_PERCENT_CEILING=85 P2_RAM_PERCENT_CEILING=85 P2_GPU_CONCURRENCY=exclusive)
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'GPU resource contract: %s\n' "${resource_json:-not authorized; CPU preview only}"
    printf 'Command:'; printf ' %q' "${trainer[@]}"; printf '\n'
    exit 0
fi
# Refuse an occupied lock before any systemd query or output creation. The atomic
# lock acquire below closes the race; neither check waits, steals or auto-queues.
[[ ! -e $LOCK_FILE && ! -L $LOCK_FILE ]] || die 'external GPU lock is already held; no waiting, stealing or auto-queuing'
if [[ -n ${CONTINUATION_PLAN:-} ]]; then
    /usr/bin/python3 -B - "$APPROVAL" <<'PY' || die 'training approval is not admitted; resource consent alone is insufficient'
import json, sys
if json.load(open(sys.argv[1])).get('approved') is not True:
    raise SystemExit(2)
PY
fi

owner="wam-train-$name"
unit="$owner.service"
releaser="p2-gpulock-$owner.service"  # NAME length keeps gpu_lock.py's 120-character limit unambiguous
out="$ROOT/outputs/training/$name"
log="$ROOT/outputs/training/unit_logs/$name.log"
[[ ! -e $out && ! -L $out ]] || die "run already exists: $out; use a NEW name, including for RESUME"
# Trainer creates its own run directory exclusively. Never mkdir it for launch.log.
load_state=$(systemctl --user show "$unit" -p LoadState --value 2>/dev/null || true)
[[ $load_state == not-found ]] || die "unit exists or user manager unavailable: $unit ($load_state)"
load_state=$(systemctl --user show "$releaser" -p LoadState --value 2>/dev/null || true)
[[ $load_state == not-found ]] || die "releaser unit already exists: $releaser ($load_state)"
[[ ! -e $LOCK_FILE && ! -L $LOCK_FILE ]] || die 'external GPU lock is already held; no waiting, stealing or auto-queuing'
mkdir -p -- "$(dirname -- "$log")"
( set -o noclobber; : > "$log" ) || die "launch log exists; preserve it and use a NEW name: $log"
if [[ -n $resource_json ]]; then
    printf 'GPU resource contract: %s\n' "$resource_json" | tee -a "$log"
    ( set -o noclobber; printf '%s\n' "$resource_json" > "${log%.log}.resources.json" ) \
        || die 'resource log already exists; preserve it and use a NEW name'
fi

acquired=0
launched=0
committed=0
live_unit() {
    local pid state loaded
    loaded=$(systemctl --user show "$unit" -p LoadState --value 2>/dev/null || true)
    [[ $loaded != not-found ]] || return 1
    # An unreachable manager is NOT proof that a GPU process has exited: fail closed.
    [[ $loaded == loaded ]] || return 0
    pid=$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null) || return 0
    state=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null) || return 0
    [[ $pid =~ ^[0-9]+$ && -n $state ]] || return 0
    [[ $pid =~ ^[1-9][0-9]*$ || $state == active || $state == activating || $state == deactivating ]]
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    if (( acquired && ! committed )); then
        [[ ! -f $log ]] || tail -n 25 -- "$log" >&2
        if (( launched )) && live_unit; then
            echo "Startup/handoff failed; stopping $unit before releasing GPU ownership." >&2
            systemctl --user kill --kill-who=main --signal=SIGTERM "$unit" 2>/dev/null
            systemctl --user stop "$unit" 2>/dev/null
        fi
        if (( launched )) && live_unit; then
            echo 'Unit still alive: KEEPING GPU lock for safety; do not release it manually.' >&2
        else
            PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$LOCK_TOOL" release --owner "$owner"
        fi
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# External project lock acquisition is atomic; an intervening owner is not overwritten.
# --max-wait 0 makes contention a failed launch, not an unattended future GPU job.
# release() itself verifies ownership; arm cleanup before acquire to cover a signal
# or stdout failure immediately after the helper has created our lock.
acquired=1
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$LOCK_TOOL" acquire --owner "$owner" --max-wait 0
[[ ! -e $out && ! -L $out ]] || die "run appeared during lock acquisition: $out"
printf 'Launching mode=%s target_step=%s unit=%s\nstdout=%s\n' "$mode" "$steps" "$unit" "$log"
printf 'Command:'; printf ' %q' "${trainer[@]}"; printf '\n'
# Mark attempted start so a late systemd-run error also checks/stops any created unit.
launched=1
systemd-run --user --quiet --collect --unit="$unit" --working-directory="$ROOT" \
    -p Type=exec -p Restart=no -p "LimitNOFILE=$limit" -p CPUAffinity=0-15 \
    -p MemoryMax=20G -p MemorySwapMax=0 -p KillSignal=SIGTERM -p KillMode=mixed -p TimeoutStopSec=900s \
    -p "StandardOutput=append:$log" -p "StandardError=append:$log" \
    /usr/bin/env "${resource_env[@]}" P2_TASK_ID=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "${trainer[@]}"

# The detached watcher releases only this owner's lock after this unit exits.
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$LOCK_TOOL" hold-for-unit --owner "$owner" --unit "$unit"
systemctl --user is-active --quiet "$releaser" || die 'detached GPU-lock releaser did not become active'
echo 'Immediate unit state:'
systemctl --user show "$unit" -p MainPID -p ActiveState -p SubState -p LimitNOFILE \
    -p CPUAffinity -p MemoryMax -p MemorySwapMax --no-pager
sleep 3
echo 'After 3 seconds:'
systemctl --user show "$unit" -p MainPID -p ActiveState -p SubState -p Result -p ExecMainStatus --no-pager
[[ ! -f $log ]] || tail -n 25 -- "$log"
[[ ! -f $out/log.jsonl ]] || tail -n 5 -- "$out/log.jsonl"
pid=$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)
state=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null || true)
[[ $state == active && $pid =~ ^[1-9][0-9]*$ ]] || die 'trainer did not remain active for the 3-second startup check'
[[ $(cat "$LOCK_FILE" 2>/dev/null || true) == "$owner" ]] || die 'GPU ownership changed during startup'
systemctl --user is-active --quiet "$releaser" || die 'GPU-lock releaser exited during startup'
committed=1
printf 'STARTED (not a training-success claim): PID=%s; no automatic next stage.\n' "$pid"
printf 'SIGTERM/save-last: systemctl --user kill --kill-who=main --signal=SIGTERM %q\n' "$unit"
printf 'Log: tail -f %q\n' "$log"
