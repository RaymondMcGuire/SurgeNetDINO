#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo "ERROR: install uv first (see README)." >&2; exit 1; }
source_root=${1:?Usage: prepare_data.sh SOURCE_ROOT [NEW_PREPROCESSED_ROOT] [EXPERIMENTS_ROOT]}
data_root=${2:-$(dirname -- "$source_root")/JSCAS_preprocessed}
experiments_root=${3:-$(dirname -- "$source_root")/JSCAS_experiments}
echo "Dataset: $data_root/labeled_1fps_v1"
echo "Split: $experiments_root/splits/video_holdout_9_3_v1/split.json"
echo "Extract and audit all 12 labeled videos, including 01. Allow at least 120 GiB free space."
exec uv run --no-project --python 3.12 --script tools/prepare_labeled.py prepare \
  --source "$source_root" --output "$data_root/labeled_1fps_v1" \
  --split-output "$experiments_root/splits/video_holdout_9_3_v1"
