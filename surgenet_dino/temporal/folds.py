"""Video-grouped folds with explicit leak checks. A video is never split between two sides."""

import json
from pathlib import Path

from ..experiments import grouped_folds
from ..training import LABELS


def check_split(train, validation):
    """No shared video, file, path or image hash; the training side must contain every phase."""
    for key in ("video_id", "filename", "image_path", "sha256"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f"Training/validation leakage: {key}")
    if {r["phase_id"] for r in train} != set(LABELS):
        raise ValueError("Training fold does not cover all classes")


def check_groups(records, groups):
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("At least two folds are required")
    if any(not g or any(not isinstance(v, str) for v in g) for g in groups):
        raise ValueError("Fold video IDs must be nonempty string groups (preserve leading zeros)")
    if sorted(v for g in groups for v in g) != sorted({r["video_id"] for r in records}):
        raise ValueError("Folds must cover every video exactly once, with no overlap")
    for held in groups:
        check_split([r for r in records if r["video_id"] not in held],
                    [r for r in records if r["video_id"] in held])
    return groups


def load_outer_folds(records, path):
    """Fixed outer folds from a JSON file with a `validation_videos` list of video-ID groups."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return check_groups(records, payload["validation_videos"])


def make_inner_groups(records, folds, seed):
    """Label-count-balanced grouped folds. Call it with TRAINING records only."""
    return check_groups(records, grouped_folds(records, folds, seed, LABELS))
