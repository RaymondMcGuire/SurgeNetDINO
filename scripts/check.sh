#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo "ERROR: install uv first (see README)." >&2; exit 1; }
backend="${1:-cu128}"
case "$backend" in
  cu128|cpu) ;;
  *) echo "ERROR: backend must be cu128 or cpu." >&2; exit 1 ;;
esac
[[ -f uv.lock ]] || { echo "ERROR: run bash scripts/init.sh first." >&2; exit 1; }
uv run --locked --extra "$backend" python -m surgenet_dino doctor
uv run --locked --extra "$backend" ruff check surgenet_dino scripts tests examples load_models.py
uv run --locked --extra "$backend" python -m pytest -q
