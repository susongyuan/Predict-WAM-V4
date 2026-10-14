#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BEHAVIOR_ENV="$ROOT/.conda-envs/behavior"
BEHAVIOR_SOURCE="$ROOT/third_party/BEHAVIOR-1K"

case "${1:-plan}" in
  plan)
    cat <<'EOF'
Prepared environment: .conda-envs/behavior (Python 3.11)
bash scripts/setup_behavior.sh base             # Small CPU-only foundation; no license acceptance.
bash scripts/setup_behavior.sh sim-interactive  # User terminal only; official NVIDIA EULA prompt.
bash scripts/setup_behavior.sh sim-authorized   # Continue under the user's recorded EULA acceptance.
Assets are separate: see docs/benchmark_environment.md and configs/behavior_environment.json.
No simulator, training job, or asset download runs by default.
EOF
    exit 0 ;;
  base|sim-interactive|sim-authorized) ;;
  *) echo "Usage: $0 [plan|base|sim-interactive|sim-authorized]" >&2; exit 2 ;;
esac

test "$(git -C "$BEHAVIOR_SOURCE" rev-parse HEAD)" = b1979916ec1549b10a4e65e630bc6504a9af1b00
for directory in OmniGibson bddl3 joylo; do
  test -f "$BEHAVIOR_SOURCE/$directory/setup.py" || {
    echo "Complete the pinned source checkout first: git -C third_party/BEHAVIOR-1K sparse-checkout add --no-cone /OmniGibson/ /bddl3/ /joylo/" >&2
    exit 2
  }
done
# CPUAffinity works even when the user cgroup has no delegated CPU controller.
limited=(systemd-run --user --quiet --wait --pipe -p 'CPUAffinity=4 5' -p MemoryMax=4G -p MemorySwapMax=0)
"${limited[@]}" /usr/bin/true
if [[ ! -x "$BEHAVIOR_ENV/bin/python" ]]; then
  "${limited[@]}" env CONDA_REPODATA_THREADS=2 OMP_NUM_THREADS=2 \
    "$(command -v conda)" create --prefix "$BEHAVIOR_ENV" --override-channels -c conda-forge \
    --yes python=3.11 pip 'setuptools>=71,<81' wheel
fi
"$BEHAVIOR_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11)'
if [[ "$1" == base ]]; then
  "${limited[@]}" "$BEHAVIOR_ENV/bin/python" -m pip install \
    numpy==1.26.4 PyYAML==6.0.2 msgpack==1.1.2 websockets==15.0.1 omegaconf==2.3.0
  "$BEHAVIOR_ENV/bin/python" -m pip check
  exit 0
fi

mode=(--pty)
acceptance=()
if [[ "$1" == sim-authorized ]]; then
  "$BEHAVIOR_ENV/bin/python" -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["authorization"]["nvidia_eula"] and c["authorization"]["behavior_asset_license"]' "$ROOT/configs/behavior_environment.json"
  mode=(--pipe)
  acceptance=(--accept-nvidia-eula)
else
  [[ -t 0 && -t 1 ]] || { echo 'Use your terminal for sim-interactive.' >&2; exit 2; }
fi
echo 'Installing official Isaac Sim / CUDA dependencies in the isolated behavior environment.'
cd "$BEHAVIOR_SOURCE"
exec systemd-run --user --quiet --wait "${mode[@]}" --working-directory="$BEHAVIOR_SOURCE" \
  -p 'CPUAffinity=4 5' -p MemoryMax=4G -p MemorySwapMax=0 \
  env -u EXP_PATH -u CARB_APP_PATH -u ISAAC_PATH -u OMNI_KIT_ACCEPT_EULA \
  CONDA_PREFIX="$BEHAVIOR_ENV" PATH="$BEHAVIOR_ENV/bin:$PATH" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2 \
  bash ./setup.sh --omnigibson --bddl --joylo --eval --cuda-version 12.8 "${acceptance[@]}"
