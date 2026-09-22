"""Timelines, grouped folds, label-free standardisation, cross-fitted heads and the token cache."""

import json

import pytest
import torch
from torch import nn

from surgenet_dino.gpu import CPUControl, Resources
from surgenet_dino.head_options import HeadOptions
from surgenet_dino.temporal import emissions, folds, tokens
from surgenet_dino.temporal.timeline import build_timelines, label_track, segments
from surgenet_dino.training import LABELS

CLASSES = len(LABELS)


def synthetic(videos=8, per_phase=12, gap=320, dimension=24, seed=0):
    """Every video: phases 0..9 in order, a long out-of-body gap, then phases 10..12. Noisy features."""
    generator = torch.Generator().manual_seed(seed)
    centres = torch.randn(CLASSES, dimension, generator=generator) * 1.2
    records, features = [], []
    for v in range(1, videos + 1):
        video, second = f"{v:02d}", 5 * v
        offset = torch.randn(dimension, generator=generator) * 3  # video-specific nuisance shift
        for target in range(CLASSES):
            if target == 10:
                second += gap
            for _ in range(per_phase + (v + target) % 4):
                name = f"{video}_{second * 30:06d}.png"
                records.append({"video_id": video, "filename": name, "frame_index": second * 30,
                                "image_path": f"/synthetic/{name}", "sha256": f"{video}{second:08d}",
                                "phase_id": LABELS[target], "target": target})
                features.append(centres[target] + offset + torch.randn(dimension, generator=generator) * 2.5)
                second += 1
    return records, torch.stack(features).float(), torch.tensor([r["target"] for r in records])


def guard():
    return CPUControl(torch.device("cpu"), Resources(cpu_threads=2))


def records_for(video, present_seconds, labels):
    return [{"video_id": video, "filename": f"{video}_{s * 30:06d}.png", "frame_index": s * 30,
             "target": y} for s, y in zip(present_seconds, labels)]


def test_timeline_gaps_anchor_and_fill_forward():
    seconds = [s for s in range(0, 40) if s not in (10, 11)] + list(range(350, 380))
    records = records_for("a", seconds, [0] * 18 + [1] * 20 + [2] * 30)
    targets = torch.tensor([r["target"] for r in records])
    timeline = build_timelines(records, 300)["a"]
    assert len(timeline) == 380 and timeline.long_gap == (40, 350) and timeline.extra_long_gaps == 0
    assert [c[0] for c in timeline.chains()] == ["pre", "post"]
    track = label_track(timeline, targets)
    assert track[10].item() == 0 and track[11].item() == 0  # short gap filled: both sides agree
    assert (track[40:350] == -1).all()  # the long out-of-body gap carries no label
    assert [s[0] for s in segments(track)] == [0, 1, 2]
    with pytest.raises(ValueError, match="whole 1 fps second"):
        build_timelines([{"video_id": "z", "filename": "z", "frame_index": 31}], 300)
    with pytest.raises(ValueError, match="Duplicate second"):
        build_timelines(records_for("z", [1, 1], [0, 0]), 300)
    flat = build_timelines(records_for("z", range(50), [0] * 50), 300)["z"]
    assert flat.long_gap is None and flat.chains() == [("whole", 0, 50)]


def test_outer_and_inner_folds_cover_every_video_once_and_reject_leaks(tmp_path):
    records, _, _ = synthetic()
    path = tmp_path / "folds.json"
    path.write_text(json.dumps({"validation_videos": [["01", "02"], ["03", "04"], ["05", "06"], ["07", "08"]]}))
    assert len(folds.load_outer_folds(records, path)) == 4
    inner = folds.make_inner_groups([r for r in records if r["video_id"] not in ("07", "08")], 3, 5)
    assert sorted(v for g in inner for v in g) == ["01", "02", "03", "04", "05", "06"]
    for bad in ([["01", "02"], ["03"]], [["01", "02"], ["02", "03", "04", "05", "06", "07", "08"]], [["01"]],
                [[1, 2], [3, 4, 5, 6, 7, 8]]):
        path.write_text(json.dumps({"validation_videos": bad}))
        with pytest.raises(ValueError):
            folds.load_outer_folds(records, path)
    leaked = [dict(r) for r in records]
    leaked[0]["sha256"] = leaked[-1]["sha256"]  # the same image on both sides of a split
    with pytest.raises(ValueError, match="leakage: sha256"):
        folds.check_split(leaked[:1], leaked[-1:])
    with pytest.raises(ValueError, match="cover all classes"):
        folds.check_split([r for r in records if r["target"] != 3 and r["video_id"] == "01"],
                          [r for r in records if r["video_id"] == "02"])


def test_zscore_is_label_free_and_local_to_each_video():
    records, x, _ = synthetic()
    z = emissions.per_video_zscore(x, records)
    rows = lambda video: torch.tensor([i for i, r in enumerate(records) if r["video_id"] == video])  # noqa: E731
    first = rows("01")
    torch.testing.assert_close(z[first].mean(0), torch.zeros(x.shape[1]), atol=1e-5, rtol=0)
    torch.testing.assert_close(z[first].std(0), torch.ones(x.shape[1]), atol=1e-4, rtol=0)
    changed = x.clone()
    changed[rows("02")] += 100.0  # another video changes; video 01 must be bit-identical
    assert torch.equal(emissions.per_video_zscore(changed, records)[first], z[first])
    shifted = x.clone()
    shifted[first] = shifted[first] * 3.0 + 7.0  # per-video affine nuisance is removed
    torch.testing.assert_close(emissions.per_video_zscore(shifted, records)[first], z[first], atol=1e-4, rtol=0)
    assert "targets" not in emissions.per_video_zscore.__code__.co_varnames
    with pytest.raises(ValueError, match="Too few frames"):
        emissions.per_video_zscore(x[:50], records[:50])


def test_fixed_budget_head_matches_the_shared_training_loop(tmp_path):
    from surgenet_dino.head_training import fit
    records, x, y = synthetic(videos=4)
    train = [i for i, r in enumerate(records) if r["video_id"] != "04"]
    held = [i for i, r in enumerate(records) if r["video_id"] == "04"]
    head_options = HeadOptions(epochs=2, patience=0, evals_per_epoch=1)
    with guard() as control:
        fixed = emissions.fit_head_fixed(x[train], y[train], head_options, 42, 2, control)
    output = tmp_path / "reference"
    output.mkdir()
    with guard() as control:
        fit(x[train], y[train], x[held], y[held], [records[i] for i in train], [records[i] for i in held],
            head_options, 42, control, output, {})
    last = torch.load(output / "head_last.pt", weights_only=True)["state_dict"]
    torch.testing.assert_close(fixed.linear.weight.detach(), last["linear.weight"], atol=1e-6, rtol=0)
    torch.testing.assert_close(fixed.linear.bias.detach(), last["linear.bias"], atol=1e-6, rtol=0)
    assert "validation" not in " ".join(emissions.fit_head_fixed.__code__.co_varnames)


def test_every_video_is_scored_by_a_head_that_never_saw_it(monkeypatch):
    records, x, y = synthetic()
    fit_videos, apply_videos = ["01", "02", "03", "04", "05", "06"], ["07", "08"]
    seen = []
    original = emissions.fit_head_fixed

    def spy(features, targets, *args):
        seen.append(len(features))
        return original(features, targets, *args)

    monkeypatch.setattr(emissions, "fit_head_fixed", spy)
    with guard() as control:
        logits, ledger = emissions.stacked_emissions(x, y, records, fit_videos, apply_videos, HeadOptions(),
                                                     2, 1, control)
    assert len(seen) == 6 + 1  # one leave-one-video-out head per fit video, plus the full head
    assert sorted(e["video_id"] for e in ledger) == fit_videos + apply_videos
    for entry in ledger:
        assert entry["video_id"] not in entry["trained_on"] and set(entry["trained_on"]) <= set(fit_videos)
        assert entry["role"] == ("apply" if entry["video_id"] in apply_videos else "crossfit")
    assert torch.isfinite(logits).all()  # every row of every fit/apply video predicted exactly once
    with guard() as control, pytest.raises(ValueError, match="disjoint"):
        emissions.stacked_emissions(x, y, records, fit_videos, ["01"], HeadOptions(), 1, 1, control)
    bad = [emissions.LedgerEntry("01", "crossfit", ("01", "02"), 1, 1, "x")]
    with pytest.raises(ValueError, match="Leaking"):
        emissions.check_ledger(bad, ["01"], [])


class StubEncoder(nn.Module):
    """Mimics the timm interface used by frame_tokens with index-coded tokens."""

    num_prefix_tokens, num_reg_tokens, has_class_token, global_pool = 1, 0, True, "token"

    def forward_features(self, images):
        cls = torch.full((len(images), 1, 3), 100.0)
        patches = torch.arange(64.0).reshape(1, 64, 1).expand(len(images), 64, 3)
        return torch.cat([cls, patches], 1)


def test_frame_tokens_are_cls_then_mean_patch_and_registers_are_rejected():
    out = tokens.frame_tokens(StubEncoder(), torch.zeros(2, 3, 4, 4))
    assert out.shape == (2, 6) and out[0].tolist() == [100.0] * 3 + [31.5] * 3
    registers = StubEncoder()
    registers.num_reg_tokens = 4
    with pytest.raises(ValueError, match="register"):
        tokens.check_architecture(registers)


def test_loading_a_token_cache_rechecks_alignment_with_the_base_cache(tmp_path):
    records, x, y = synthetic(videos=2)
    base = {"features": x, "targets": y, "feature_identity": {"id": "a"},
            "split": {"train": {"records": records}, "validation": {"records": []}}}
    base_path = tmp_path / "features.pt"
    base_path.write_bytes(b"base cache bytes")
    values = torch.cat([x, torch.randn_like(x)], 1)

    def write(directory, **changes):
        directory.mkdir()
        payload = {"format": tokens.FORMAT, "tokens": values, "targets": y.clone(),
                   "base_features_sha256": tokens.file_sha256(base_path),
                   "records_sha256": tokens.record_keys(records), "feature_identity": {"id": "a"}}
        torch.save({**payload, **changes}, directory / "tokens.pt")
        return directory

    loaded, check = tokens.load_tokens(write(tmp_path / "good"), base, base_path)
    assert torch.equal(loaded[:, :x.shape[1]], x) and check["max_abs_cls_difference"] == 0
    shuffled = values[torch.randperm(len(values), generator=torch.Generator().manual_seed(0))]
    for name, changes, message in (("rows", {"tokens": shuffled}, "differs from the base"),
                                   ("sha", {"base_features_sha256": "0" * 64}, "does not belong"),
                                   ("records", {"records_sha256": "0" * 64}, "does not belong"),
                                   ("identity", {"feature_identity": {"id": "b"}}, "does not belong"),
                                   ("targets", {"targets": (y + 1) % CLASSES}, "misaligned"),
                                   ("dtype", {"tokens": values.half()}, "misaligned"),
                                   ("shape", {"tokens": values[:, :-1]}, "misaligned"),
                                   ("nan", {"tokens": torch.full_like(values, float("nan"))}, "misaligned")):
        with pytest.raises(ValueError, match=message):
            tokens.load_tokens(write(tmp_path / name, **changes), base, base_path)
