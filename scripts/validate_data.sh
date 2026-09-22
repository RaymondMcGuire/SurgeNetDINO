#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
dataset=${1:?Usage: validate_data.sh DATASET_DIR [NEW_REPORT_DIR] [cpu|cu128]}
report=${2:-$dataset/validation/$(date +%Y%m%d_%H%M%S)}
backend=${3:-cu128}
case "$backend" in cpu|cu128) ;; *) echo 'Invalid backend' >&2; exit 1;; esac
[[ -f uv.lock && -f "$dataset/dataset.json" ]]
exec uv run --locked --extra "$backend" python -m surgenet_dino validate-data \
  --dataset "$dataset" --output-dir "$report"
