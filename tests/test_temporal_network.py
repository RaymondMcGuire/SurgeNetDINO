"""Temporal network, nested cross-validation workflow, diagnostics, options, CLI and launchers."""

import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from surgenet_dino.config import ModelConfig
from surgenet_dino.gpu import CPUControl, Resources
from surgenet_dino.head_options import HeadOptions
from surgenet_dino.temporal import advanced, constraints, curves, emissions, tcn, workflows
from surgenet_dino.temporal import tokens as token_cache
from surgenet_dino.temporal.options import TemporalOptions, load_options
from surgenet_dino.temporal.timeline import build_timelines, label_track
from surgenet_dino.training import LABELS
from tests.test_temporal_data import synthetic

CLASSES = len(LABELS)
GROUPS = [["01", "02"], ["03", "04"], ["05", "06"], ["07", "08"]]


def small(**changes):
    base = TemporalOptions(channels=16, refine_channels=12, layers=4, pca=6, epochs=4, epoch_checks=2, seeds=2,
                           head_epochs=2, attention_factor=4)
    return replace(base, **changes)


def fast():
    """Enough budget for the tiny network to learn the synthetic task inside the end-to-end tests."""
    return small(epochs=12, epoch_checks=2, learning_rate=3e-3, inner_folds=2)


def guard():
    return CPUControl(torch.device("cpu"), Resources(cpu_threads=2))


def test_network_is_small_handles_any_length_and_has_no_batch_statistics():
    inputs = CLASSES + 32 + tcn.FLAGS
    model = tcn.PhaseTCN(inputs, TemporalOptions()).eval()
    assert sum(p.numel() for p in model.parameters()) <= 300_000
    assert any("predict.context" in key for key in model.state_dict())
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)) for m in model.modules())
    torch.manual_seed(0)
    for length in (1, 15, 16, 17, 1000):
        assert model(torch.randn(1, inputs, length))[-1].shape == (1, CLASSES, length)
    x = torch.randn(1, inputs, 400)
    both = model(torch.cat([x, torch.randn_like(x) * 9]))[-1]
    torch.testing.assert_close(both[:1], model(x)[-1], atol=1e-5, rtol=0)  # no mixing across the batch
    changed = x.clone()
    changed[:, :, 350:] += 5.0  # attention is global: a late change reaches the early seconds
    assert not torch.allclose(model(changed)[-1][:, :, :20], model(x)[-1][:, :, :20])
    local = tcn.PhaseTCN(inputs, replace(TemporalOptions(), attention_layers=0, layers=3)).eval()
    assert torch.allclose(local(changed)[-1][:, :, :20], local(x)[-1][:, :, :20])  # without it, context is local


def test_relative_bias_depends_on_the_signed_offset_only():
    bias = tcn.RelativeBias(4)
    nn.init.normal_(bias.table.weight)
    small_bias, large = bias(10, torch.device("cpu")), bias(20, torch.device("cpu"))
    assert small_bias.shape == (1, 4, 10, 10) and torch.equal(small_bias, large[:, :, :10, :10])
    assert torch.equal(large[0, :, 3, 9], large[0, :, 11, 17])
    assert not torch.equal(large[0, :, 3, 9], large[0, :, 9, 3])


def test_inputs_use_no_labels_and_mark_gaps():
    records, x, _ = synthetic(videos=2)
    timeline = build_timelines(records, 300)["01"]
    pca = tcn.fit_pca(x, 6)
    inputs = tcn.build_inputs(timeline, torch.randn(len(records), CLASSES), x, pca)
    assert inputs.shape == (CLASSES + 6 + tcn.FLAGS, len(timeline))
    start, stop = timeline.long_gap
    flags = inputs[CLASSES + 6:]
    assert (flags[0, start:stop] == 1).all() and (flags[1, start:stop] == 1).all() and (flags[1, :start] == 0).all()
    assert (flags[2, stop:] == 1).all() and (flags[2, :stop] == 0).all()
    assert (inputs[:CLASSES + 6, start:stop] == 0).all()
    assert flags[3, 0] == 0 and flags[3, -1] == 1 and flags[5, stop] == 0 and flags[5, -1] > 0
    assert "targets" not in tcn.build_inputs.__code__.co_varnames
    projected = (x.double() - pca["mean"]) @ pca["basis"] / pca["scale"]
    torch.testing.assert_close(projected.std(0), torch.ones(6, dtype=torch.float64), atol=1e-6, rtol=0)


def test_augmentation_keeps_labels_aligned_with_inputs():
    total = 900
    inputs = torch.zeros(CLASSES + 4 + tcn.FLAGS, total)
    labels = (torch.arange(total) // 70).clamp(max=CLASSES - 1)
    progress = CLASSES + 4 + 3
    inputs[progress] = torch.arange(total) / (total - 1)  # identifies the source second of every column
    confusion = torch.full((CLASSES, CLASSES), 1.0 / (CLASSES - 1), dtype=torch.float64).fill_diagonal_(0)
    generator = torch.Generator().manual_seed(4)
    lengths, untouched = set(), inputs.clone()
    for _ in range(20):
        a, b = tcn.augment(inputs, labels, TemporalOptions(), generator, confusion)
        source = (a[progress] * (total - 1)).round().long()
        assert torch.equal(b, labels[source]) and a.shape[1] == len(b)
        assert (source[1:] >= source[:-1]).all()  # time order is preserved
        lengths.add(len(b))
        masked = a[CLASSES + 4] == 1
        assert masked.any() and (a[:CLASSES + 4, masked] == 0).all()
    assert len(lengths) > 5 and min(lengths) >= 0.3 * total
    assert torch.equal(inputs, untouched)  # the source sequence is never modified


def test_final_refit_cannot_receive_validation_and_inner_validation_must_be_disjoint():
    records, x, y = synthetic(videos=4)
    timelines = build_timelines(records, 300)
    pca = tcn.fit_pca(x, 6)
    logits = torch.nn.functional.one_hot(y, CLASSES).float() * 2 + torch.randn(len(y), CLASSES)
    confusion = tcn.confusion_rows(logits, y, torch.arange(len(y)))
    sequences = tcn.make_sequences(timelines, ["01", "02"], logits, x, y, pca)
    held = tcn.make_sequences(timelines, ["03"], logits, x, y, pca)
    with guard() as control:
        with pytest.raises(ValueError, match="require inner validation"):
            tcn.fit_tcn(sequences, small(), 1, control, 2, confusion, checkpoints=(1,))
        with pytest.raises(ValueError, match="require inner validation"):
            tcn.fit_tcn(sequences, small(), 1, control, 2, confusion, validation=held)
        with pytest.raises(ValueError, match="overlap"):
            tcn.fit_tcn(sequences, small(), 1, control, 2, confusion, (1,), sequences, timelines, len(records))
        _, snapshots = tcn.fit_tcn(sequences, small(), 1, control, 2, confusion, (1, 2), held, timelines,
                                   len(records))
    rows = timelines["03"].rows[timelines["03"].present]
    assert sorted(snapshots) == [1, 2] and torch.isfinite(snapshots[2][rows]).all()
    others = torch.ones(len(records), dtype=torch.bool)
    others[rows] = False
    assert torch.isnan(snapshots[2][others]).all()
    torch.testing.assert_close(snapshots[2][rows].exp().sum(1), torch.ones(len(rows), dtype=torch.float64))


def test_network_learns_a_clean_toy_problem_quickly():
    records, x, y = synthetic(videos=3, per_phase=10)
    timelines = build_timelines(records, 300)
    pca = tcn.fit_pca(x, 6)
    logits = torch.nn.functional.one_hot(y, CLASSES).float() * 6
    confusion = tcn.confusion_rows(logits, y, torch.arange(len(y)))
    sequences = tcn.make_sequences(timelines, ["01", "02", "03"], logits, x, y, pca)
    with guard() as control:
        quick = small(augment=False, dropout=0.0, learning_rate=2e-3)
        model, _ = tcn.fit_tcn(sequences, quick, 3, control, 60, confusion)
        predicted = tcn.predict_tcn([model], sequences, timelines, control, len(records))
    assert (predicted.argmax(1) == y).double().mean().item() > 0.9
    assert torch.equal(label_track(timelines["01"], y), sequences["01"][1])


def test_inner_folds_never_show_a_video_to_the_models_that_score_it(monkeypatch):
    records, x, y = synthetic(videos=8, per_phase=8)
    timelines = build_timelines(records, 300)
    fit_videos, apply_videos = ["01", "02", "03", "04", "05", "06"], ["07", "08"]
    poisoned = x.clone()  # inner folds must not even READ the outer held-out videos
    poisoned[torch.tensor([i for i, r in enumerate(records) if r["video_id"] in apply_videos])] = float("nan")
    trained, heads = [], []
    fit_original, stack_original = tcn.fit_tcn, emissions.stacked_emissions

    def fit_spy(sequences, options, seed, guard_, epochs, confusion, checkpoints=(), validation=None, *args):
        trained.append((sorted(sequences), sorted(validation or []), epochs))
        return fit_original(sequences, options, seed, guard_, epochs, confusion, checkpoints, validation, *args)

    def stack_spy(features, targets, rows, fit, apply, *args):
        heads.append((sorted(fit), sorted(apply)))
        return stack_original(features, targets, rows, fit, apply, *args)

    monkeypatch.setattr(tcn, "fit_tcn", fit_spy)
    monkeypatch.setattr(emissions, "stacked_emissions", stack_spy)
    with guard() as control:
        inner = tcn.inner_cv(poisoned, y, records, timelines, fit_videos, HeadOptions(), small(), 1, control)
    assert len(trained) == 6 and len(heads) == 3
    for training, validation, epochs in trained:
        assert len(training) == 4 and len(validation) == 2 and epochs == 4
        assert not set(training) & set(validation) and not (set(training) | set(validation)) & set(apply_videos)
    assert sorted(v for _, val, _ in trained[::2] for v in val) == fit_videos  # each fit video held out once
    for fit, apply in heads:
        assert not set(fit) & set(apply) and not (set(fit) | set(apply)) & set(apply_videos)
    rows = torch.cat([inner["rows"][v] for v in fit_videos])
    assert torch.isfinite(inner["per_seed"][4][:, rows]).all()
    chosen, accuracy = tcn.common_epoch(inner, y, fit_videos)
    scrambled = y.clone()
    held = torch.cat([inner["rows"][v] for v in apply_videos])
    scrambled[held] = (scrambled[held] + 5) % CLASSES  # outer held-out labels cannot move the choice
    assert tcn.common_epoch(inner, scrambled, fit_videos) == (chosen, accuracy)


def test_segment_metrics_and_error_anatomy():
    truth = [(0, 0, 10), (1, 10, 20)]
    assert workflows.segmental_f1(truth, truth, 0.75) == 1.0 and workflows.edit_score(truth, truth) == 1.0
    flicker = [(0, 0, 4), (1, 4, 5), (0, 5, 10), (1, 10, 20)]
    assert workflows.segmental_f1(flicker, truth, 0.75) == pytest.approx(2 / 6)  # one match, 3 FP, 1 FN
    assert workflows.edit_score(flicker, truth) == pytest.approx(0.5)
    labels = torch.tensor([0] * 300 + [1] * 300)
    guess = labels.clone()
    guess[100:200] = 2  # one 100 s run far from the boundary
    guess[295:300] = 1  # 5 s of boundary jitter
    records = [{"video_id": "a", "filename": f"a_{s * 30:06d}.png", "frame_index": s * 30} for s in range(600)]
    result = workflows.error_anatomy(guess, labels, build_timelines(records, 300))
    assert result["error_frames"] == 105
    assert result["share_in_long_runs"] == pytest.approx(100 / 105)
    assert result["share_near_boundary"] == pytest.approx(5 / 105)


def test_advanced_score_counts_out_of_body_segments_at_30_fps(tmp_path):
    (tmp_path / "a.csv").write_text("start-frame,phase-id\n0,x\n60,0\n135,1\n240,x\n300,1\n390,e\n")
    truth = advanced.read_annotation(tmp_path / "a.csv")
    assert len(truth) == 390 and truth[59] == advanced.OUT_OF_BODY and truth[134] == 0 and truth[135] == 1
    records = [{"video_id": "a", "frame_index": f} for f in (60, 90, 120, 150, 180, 210, 300, 330, 360)]
    right = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert advanced.advanced_counts(right, records, tmp_path) == {"a": (5, 0, 0)}  # 30 fps edges at the midpoints
    # All 0: both x segments match; 0 [60,240) has IoU 75/180 with the true 0; the last 0 matches nothing.
    wrong = torch.zeros(9, dtype=torch.long)
    assert advanced.advanced_counts(wrong, records, tmp_path) == {"a": (2, 2, 3)}
    # FAQ 2026-09-27 Q7: counts are summed over videos first, 14/19, not the mean of F1 1 and 4/9.
    assert advanced.pooled_f1({"a": (5, 0, 0), "b": (2, 2, 3)}) == pytest.approx(14 / 19)


def write_annotations(records, directory):
    """Official-format CSVs of synthetic videos: x before the first frame and in every gap, 'e' at the end."""
    directory.mkdir()
    by_video = {}
    for r in records:
        by_video.setdefault(r["video_id"], []).append((r["frame_index"], r["phase_id"]))
    for video, items in by_video.items():
        rows, previous = [(0, "x")], None
        for frame, phase in sorted(items):
            if previous is not None and frame > previous + 30:
                rows.append((previous + 30, "x"))
            if rows[-1][1] != phase:
                rows.append((frame, phase))
            previous = frame
        rows.append((previous + 30, "e"))
        (directory / f"{video}.csv").write_text("start-frame,phase-id\n" + "".join(f"{f},{p}\n" for f, p in rows))


@pytest.fixture
def injected(tmp_path, monkeypatch):
    """A synthetic base cache and token cache injected in place of the files on disk."""
    import surgenet_dino.training as training
    records, x, y = synthetic(per_phase=10)
    cache = {"features": x, "targets": y, "labels": LABELS, "split": {"train": {"records": records},
             "validation": {"records": []}}, "feature_identity": {"contract": {"model": "dinov2_vitb"}}}
    monkeypatch.setattr(training, "load_features", lambda path: cache)
    monkeypatch.setattr(token_cache, "load_tokens",
                        lambda directory, base, features: (torch.cat([x, x.roll(1, 1)], 1), {"tokens_sha256": "stub"}))
    features = tmp_path / "features.pt"
    features.write_bytes(b"stub: the cache is injected")
    split = tmp_path / "folds.json"
    split.write_text(json.dumps({"validation_videos": GROUPS}))
    config = ModelConfig(name="dinov2_vitb", weights="none", device="cpu", seed=5)
    return records, y, features, split, config


def test_cv_end_to_end_never_shows_held_out_videos_to_fitting_or_selection(tmp_path, monkeypatch, injected):
    records, y, features, split, config = injected
    fitted, inner_calls, rule_calls = [], [], []
    stack_original, inner_original, rules_original = emissions.stacked_emissions, tcn.inner_cv, constraints.training_rules

    def stack_spy(features_, targets, rows, fit_videos, apply_videos, *args):
        fitted.append((sorted(fit_videos), sorted(apply_videos)))
        return stack_original(features_, targets, rows, fit_videos, apply_videos, *args)

    def inner_spy(x, targets, rows, timelines, fit_videos, *args, **kwargs):
        inner_calls.append(sorted(fit_videos))
        return inner_original(x, targets, rows, timelines, fit_videos, *args, **kwargs)

    def rules_spy(targets, timelines, videos):
        rule_calls.append(sorted(videos))
        return rules_original(targets, timelines, videos)

    monkeypatch.setattr(emissions, "stacked_emissions", stack_spy)
    monkeypatch.setattr(tcn, "inner_cv", inner_spy)
    monkeypatch.setattr(constraints, "training_rules", rules_spy)
    baseline = tmp_path / "baseline.csv"
    workflows.write_fold_predictions(baseline, torch.randn(len(y), CLASSES), y, records, [1] * len(y))
    write_annotations(records, tmp_path / "annotations")
    result = workflows.run_cv(features, tmp_path / "tokens", split, fast(), HeadOptions(), config,
                              Resources(cpu_threads=2), tmp_path / "cv", baseline=baseline,
                              annotations=tmp_path / "annotations")
    assert result["coverage_exactly_once"] and result["samples"] == len(records)
    assert result["held_out_used_for_selection"] is False
    rows = {r["variant"]: r for r in result["summary"]}
    assert set(rows) == {"frames", "network", "ordered", "constrained"}
    assert all(0 <= r["advanced_f1_at_75"] <= 1 for r in rows.values())
    assert rows["constrained"]["advanced_f1_at_75"] > rows["frames"]["advanced_f1_at_75"]
    assert rows["network"]["accuracy"] > rows["frames"]["accuracy"]
    assert rows["network"]["mean_predicted_segments"] < rows["frames"]["mean_predicted_segments"]
    assert result["pooled"]["accuracy"] == rows["constrained"]["accuracy"]  # the decoded labels are the output
    assert result["paired_comparison_vs_baseline"]["accuracy_delta"] > 0.2
    for held, fit_videos, rule_videos in zip(GROUPS, inner_calls, rule_calls, strict=True):
        assert not set(fit_videos) & set(held) and len(fit_videos) == 6
        assert rule_videos == fit_videos  # the order rules come from the training videos only
    for fit_videos, apply_videos in fitted:  # outer heads and every inner head
        assert not set(fit_videos) & set(apply_videos)
    for fold_number, held in enumerate(GROUPS, 1):
        selection = result["selections"][fold_number - 1]
        assert selection["validation_videos"] == held and not set(selection["fit_videos"]) & set(held)
        assert all(e["video_id"] not in e["trained_on"] for e in selection["emission_ledger"])
        assert len(selection["heldout_seed_accuracy"]) == 2 and selection["common_epoch"] in (6, 12)
        assert selection["order_rules"]["precedence_rules"] > 0 and selection["order_rules"]["changed_seconds"] >= 0
    with (tmp_path / "cv" / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == len(records) and {r["fold"] for r in written} == {"1", "2", "3", "4"}
    for name in ("summary.csv", "fold_metrics.csv", "video_metrics.csv", "segment_metrics.csv", "result.json",
                 "network_predictions.csv"):
        assert (tmp_path / "cv" / name).is_file()
    with pytest.raises(FileExistsError):
        workflows.run_cv(features, tmp_path / "tokens", split, small(), HeadOptions(), config,
                         Resources(cpu_threads=2), tmp_path / "cv")
    pilot = workflows.run_cv(features, tmp_path / "tokens", split, fast(), HeadOptions(), config,
                             Resources(cpu_threads=2), tmp_path / "pilot", only_fold=2)
    assert not pilot["coverage_exactly_once"] and [f["fold"] for f in pilot["folds"]] == [2]
    assert pilot["samples"] == sum(r["video_id"] in GROUPS[1] for r in records)


def test_curves_are_diagnostic_only_and_cover_both_trained_stages(tmp_path, injected):
    _, _, features, split, config = injected
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"selections": [{"fold": k, "common_epoch": 2} for k in (1, 2, 3, 4)]}))
    result = curves.run_curves(features, tmp_path / "tokens", split, small(), HeadOptions(), config,
                               Resources(cpu_threads=2), tmp_path / "curves", head_epochs=3, every=2, seeds=2,
                               reference=reference)
    assert result["diagnostic_only"] and [f["network_common_epoch"] for f in result["folds"]] == [2, 2, 2, 2]
    with (tmp_path / "curves" / "curves.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    head = [r for r in rows if r["stage"] == "head" and r["fold"] == "1"]
    network = [r for r in rows if r["stage"] == "network" and r["fold"] == "1"]
    assert [r["epoch"] for r in head] == ["1", "2", "3"]
    assert sorted((r["seed"], r["epoch"]) for r in network) == [("0", "2"), ("0", "4"), ("1", "2"), ("1", "4")]
    assert all(0 <= float(r["heldout_accuracy"]) <= 1 and float(r["train_loss"]) > 0 for r in rows)
    assert float(head[-1]["train_loss"]) < float(head[0]["train_loss"])
    assert not (tmp_path / "curves" / "validation_predictions.csv").exists()  # it produces no model or prediction


def test_options_are_strict(tmp_path):
    path = tmp_path / "t.toml"
    path.write_text("[temporal]\nepochs = 30\nepoch_checks = 3\n")
    assert load_options(path).epochs == 30
    path.write_text("[temporal]\nepoch = 30\n")
    with pytest.raises(ValueError, match="Unknown temporal options"):
        load_options(path)
    for bad in ({"attention_heads": 5}, {"dropout": 1.0}, {"inner_folds": 1}, {"epoch_checks": 99},
                {"learning_rate": 0.0}, {"augment": 1}, {"seeds": 0}, {"attention_layers": -1}):
        with pytest.raises(ValueError):
            replace(TemporalOptions(), **bad)
    shipped = load_options(Path(__file__).resolve().parents[1] / "configs" / "temporal.toml")
    assert shipped == TemporalOptions()  # the shipped config states the defaults explicitly


def test_cli_help_and_existing_output_fast_failure(tmp_path):
    base = [sys.executable, "-m", "surgenet_dino.temporal"]
    shown = subprocess.run([*base, "--help"], capture_output=True, text=True)
    assert shown.returncode == 0 and all(name in shown.stdout for name in ("cache-tokens", "cv", "curves"))
    failed = subprocess.run([*base, "cv", "--features", "x.pt", "--tokens", "t", "--output-dir", str(tmp_path)],
                            capture_output=True, text=True)
    assert failed.returncode != 0 and "Output exists" in failed.stderr


def test_posix_scripts_have_lf_endings_and_a_batch_sibling():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    shells = sorted(scripts.glob("*.sh"))
    assert scripts / "temporal.sh" in shells
    for shell in shells:
        assert b"\r" not in shell.read_bytes(), f"CRLF breaks `set -o pipefail`: {shell.name}"
        assert shell.with_suffix(".bat").is_file(), f"missing Windows launcher for {shell.name}"


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell test; Windows script requires Windows host")
@pytest.mark.parametrize("returncode", [0, 7])
def test_temporal_shell_forwards_paths_and_exitcodes(tmp_path, returncode):
    binary, log = tmp_path / "uv", tmp_path / "arguments.txt"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_LOG"\nexit "$FAKE_EXIT"\n')
    binary.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "ARGS_LOG": str(log),
           "FAKE_EXIT": str(returncode), "GSF827_BACKEND": "cpu"}
    out = subprocess.run(["bash", "scripts/temporal.sh", "cv", "--features", "some spaced path.pt",
                          "--output-dir", "new output"], env=env, capture_output=True, text=True)
    assert out.returncode == returncode, out.stderr
    arguments = log.read_text().splitlines()
    assert "some spaced path.pt" in arguments and "new output" in arguments
    assert arguments[:7] == ["run", "--locked", "--extra", "cpu", "python", "-m", "surgenet_dino.temporal"]
