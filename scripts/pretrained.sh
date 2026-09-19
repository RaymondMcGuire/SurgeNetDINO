#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
backend="${1:-cu128}"
model="${2:-dinov2_vitb}"
case "$backend" in
  cu128) device=cuda ;;
  cpu) device=cpu ;;
  *) echo "ERROR: backend must be cu128 or cpu." >&2; exit 1 ;;
esac
[[ -f uv.lock ]] || { echo "ERROR: run bash scripts/init.sh first." >&2; exit 1; }
echo "Official surgical weights: compatibility/forward check, NOT classification accuracy."
uv run --locked --extra "$backend" python -m surgenet_dino smoke \
  --model "$model" --weights surgenet --device "$device" \
  --report "outputs/$model/pretrained_smoke.json" --overwrite
