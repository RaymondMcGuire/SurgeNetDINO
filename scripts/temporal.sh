#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${GSF827_BACKEND:-cu128}"
if [[ "$BACKEND" != "cpu" && "$BACKEND" != "cu128" ]]; then
  echo "GSF827_BACKEND must be cpu or cu128" >&2; exit 2
fi
exec uv run --locked --extra "$BACKEND" python -m surgenet_dino.temporal "$@"
