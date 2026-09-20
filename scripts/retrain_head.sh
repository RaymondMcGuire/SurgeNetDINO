#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=${1:?Usage: retrain_head.sh FEATURES_PT [low|medium|high] [NEW_RUN_DIR] [cpu|cu128]}
PROFILE=${2:-low}
RUN_DIR=${3:-outputs/head_$(date +%Y%m%d_%H%M%S)}
BACKEND=${4:-cu128}
case "$PROFILE" in low|medium|high) ;; *) echo 'Invalid GPU profile' >&2; exit 1;; esac
case "$BACKEND" in cpu) DEVICE=cpu;; cu128) DEVICE=cuda:0;; *) echo 'Invalid backend' >&2; exit 1;; esac
[[ -f "$FEATURES" && -f uv.lock && ! -e "$RUN_DIR" ]]
exec uv run --locked --extra "$BACKEND" python -m surgenet_dino train-head \
  --features "$FEATURES" --output-dir "$RUN_DIR" --gpu-profile "$PROFILE" --device "$DEVICE"
