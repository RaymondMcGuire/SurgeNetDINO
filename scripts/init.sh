#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo "ERROR: install uv first (see README)." >&2; exit 1; }
backend="${1:-cu128}"
case "$backend" in
  cu128|cpu) ;;
  *) echo "ERROR: backend must be cu128 or cpu." >&2; exit 1 ;;
esac
[[ -f uv.lock ]] || { echo "ERROR: restore the committed uv.lock before initialization." >&2; exit 1; }
uv python install 3.12
uv sync --locked --extra "$backend" --group dev
if [[ "$backend" == cu128 ]]; then
  uv run --locked --extra "$backend" python -m surgenet_dino doctor --require-cuda
else
  uv run --locked --extra "$backend" python -m surgenet_dino doctor
fi
