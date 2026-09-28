"""Motion features: CSV alignment and the extra input block of the temporal network."""

import pytest
import torch

from surgenet_dino.temporal import tcn
from surgenet_dino.temporal.motion import load_motion
from surgenet_dino.temporal.options import TemporalOptions
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


def test_build_inputs_inserts_extra_block_before_the_flags_only_where_frames_exist():
    rows = records("01", [0, 1, 3])  # second 2 is a gap
    timelines = build_timelines(rows, long_gap_seconds=300)
    x, logits = torch.randn(3, 8), torch.randn(3, len(LABELS))
    pca = tcn.fit_pca(x, 2)
    extra = torch.tensor([[1.0, -9.0], [2.0, 0.5], [3.0, 0.0]])
    plain = tcn.build_inputs(timelines["01"], logits, x, pca)
    with_extra = tcn.build_inputs(timelines["01"], logits, x, pca, extra)
    split = plain.shape[0] - tcn.FLAGS
    assert with_extra.shape == (plain.shape[0] + 2, 4) and torch.equal(with_extra[:split], plain[:split])
    assert torch.equal(with_extra[-tcn.FLAGS:], plain[-tcn.FLAGS:])  # flags stay last
    assert torch.equal(with_extra[split:split + 2], torch.tensor([[1.0, 2.0, 0.0, 3.0], [-6.0, 0.5, 0.0, 0.0]]))


def test_augmentation_keeps_emissions_of_seconds_with_a_frame_when_extra_inputs_exist():
    rows = records("01", range(2000))  # longer than the masked spans (two of at most 300 s)
    timeline = build_timelines(rows, long_gap_seconds=300)["01"]
    x, logits = torch.randn(2000, 8), torch.randn(2000, len(LABELS))
    inputs = tcn.build_inputs(timeline, logits, x, tcn.fit_pca(x, 2), torch.randn(2000, 16))
    confusion = (torch.ones(len(LABELS), len(LABELS)) - torch.eye(len(LABELS))) / (len(LABELS) - 1)
    labels = torch.zeros(2000, dtype=torch.long)
    for seed in range(5):
        out, _ = tcn.augment(inputs, labels, TemporalOptions(), torch.Generator().manual_seed(seed), confusion)
        framed = out[-tcn.FLAGS] == 0  # the gap flag; masked spans set it to 1
        assert framed.any() and (out[:len(LABELS), framed] != 0).all(0).float().mean() > 0.9
