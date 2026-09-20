#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT=${1:?Usage: train.sh SPLIT_JSON NEW_RUN_DIR [PROFILES_JSON] [low|medium|high]}
RUN_DIR=${2:?A new output directory is required}
PROFILE=${4:-low}
case "$PROFILE" in low|medium|high) ;; *) echo 'Invalid GPU profile' >&2; exit 1;; esac
[[ -f "$SPLIT" && -f uv.lock && ! -e "$RUN_DIR" ]]
EXTRA=()
if [[ -n "${3:-}" ]]; then
  [[ -f "$3" ]]
  EXTRA=(--profiles "$3")
fi
uv run --locked --extra cu128 python -m surgenet_dino cache-features \
  --split "$SPLIT" --output-dir "$RUN_DIR/cache" --gpu-profile "$PROFILE" "${EXTRA[@]}"
exec uv run --locked --extra cu128 python -m surgenet_dino train-head \
  --features "$RUN_DIR/cache/features.pt" --output-dir "$RUN_DIR/train" --gpu-profile "$PROFILE" "${EXTRA[@]}"
