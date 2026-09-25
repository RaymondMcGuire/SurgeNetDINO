"""Motion features: CSV alignment and the extra input block of the temporal network."""

import pytest
import torch

from surgenet_dino.temporal import tcn
from surgenet_dino.temporal.motion import load_motion
from surgenet_dino.temporal.timeline import build_timelines
from surgenet_dino.training import LABELS


def records(video, seconds):
    return [{"video_id": video, "filename": f"{video}_{30 * s:06d}.png", "frame_index": 30 * s} for s in seconds]


def test_load_motion_aligns_rows_and_compresses(tmp_path):
    path = tmp_path / "motion.csv"
    path.write_text("video_id,frame_index,a,b\n01,60,1.0,0.0\n01,30,0.0,3.0\n02,30,7.0,7.0\n", encoding="utf-8")
    values, provenance = load_motion(path, records("01", [1, 2]) + records("02", [1]))
    assert torch.allclose(values, torch.tensor([[0.0, 3.0], [1.0, 0.0], [7.0, 7.0]]).log1p())
    assert provenance["motion_columns"] == ["a", "b"] and len(provenance["motion_sha256"]) == 64
    with pytest.raises(ValueError, match="no motion row"):
        load_motion(path, records("01", [1, 3]))
    path.write_text("video_id,frame_index,a\n01,30,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_motion(path, records("01", [1]))


def test_build_inputs_appends_extra_block_only_where_frames_exist():
    rows = records("01", [0, 1, 3])  # second 2 is a gap
    timelines = build_timelines(rows, long_gap_seconds=300)
    x, logits = torch.randn(3, 8), torch.randn(3, len(LABELS))
    pca = tcn.fit_pca(x, 2)
    extra = torch.tensor([[1.0, -9.0], [2.0, 0.5], [3.0, 0.0]])
    plain = tcn.build_inputs(timelines["01"], logits, x, pca)
    with_extra = tcn.build_inputs(timelines["01"], logits, x, pca, extra)
    assert with_extra.shape == (plain.shape[0] + 2, 4) and torch.equal(with_extra[:plain.shape[0]], plain)
    assert torch.equal(with_extra[-2:], torch.tensor([[1.0, 2.0, 0.0, 3.0], [-6.0, 0.5, 0.0, 0.0]]))
