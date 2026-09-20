#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=${1:?Usage: crossval.sh FEATURES_PT NEW_RUN_DIR NEW_SPLIT_DIR [low|medium|high] [PROFILES_JSON] [cpu|cu128]}
RUN_DIR=${2:?A new run directory is required}
SPLIT_DIR=${3:?A new split directory is required}
PROFILE=${4:-low}
BACKEND=${6:-cu128}
case "$PROFILE" in low|medium|high) ;; *) exit 1;; esac
case "$BACKEND" in cpu) DEVICE=cpu;; cu128) DEVICE=cuda:0;; *) exit 1;; esac
[[ -f "$FEATURES" && -f uv.lock && ! -e "$RUN_DIR" && ! -e "$SPLIT_DIR" ]]
EXTRA=()
if [[ -n "${5:-}" ]]; then
  [[ -f "$5" ]]
  EXTRA=(--profiles "$5")
fi
uv run --locked --extra "$BACKEND" python -m surgenet_dino crossval-head \
  --features "$FEATURES" --output-dir "$RUN_DIR" --split-output-dir "$SPLIT_DIR" --folds 4 \
  --gpu-profile "$PROFILE" --device "$DEVICE" "${EXTRA[@]}"
exec uv run --isolated --no-project --with matplotlib==3.11.2 python -m scripts.plot_crossval "$RUN_DIR"
