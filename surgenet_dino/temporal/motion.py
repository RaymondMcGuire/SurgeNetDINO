"""Optional per-second motion features (tools/motion_features.py) as extra temporal-network inputs.

Label-free like the tokens: frame-difference and optical-flow magnitudes around each catalogued
frame, log-compressed here and standardised per video by the caller. They never enter the frame
heads or the PCA, so the emissions stay identical to a run without them.
"""

import csv
from pathlib import Path

import torch

from ..training import file_sha256

KEYS = ("video_id", "frame_index")


def load_motion(path, records):
    """([N,M] float32 log1p features aligned to `records`, provenance). Every record needs one row."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [c for c in reader.fieldnames or [] if c not in KEYS]
        if not columns or set(reader.fieldnames) != set(columns) | set(KEYS):
            raise ValueError(f"{path}: expected video_id, frame_index and feature columns")
        rows = {}
        for row in reader:
            key = (row["video_id"], int(row["frame_index"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate motion row {key}")
            rows[key] = [float(row[c]) for c in columns]
    try:
        values = torch.tensor([rows[(r["video_id"], r["frame_index"])] for r in records], dtype=torch.float32)
    except KeyError as error:
        raise ValueError(f"{path}: no motion row for catalogued frame {error}") from error
    if values.ndim != 2 or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: motion features must be finite and non-negative")
    return values.log1p(), {"motion_csv": str(path.resolve()), "motion_sha256": file_sha256(path),
                            "motion_columns": columns}
