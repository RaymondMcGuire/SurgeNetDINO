"""End-to-end public CLI tests with tiny PNG manifests and SYNTHETIC cached features."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from PIL import Image

from surgenet_dino.cli import main
from surgenet_dino.config import load_config
from surgenet_dino.experiments import grouped_folds
from surgenet_dino.training import LABELS, MANIFEST_FIELDS, read_split
from surgenet_dino.weights import file_sha256


def make_cache(root, train_videos=("01", "02", "03"), val_videos=("04",)):
    root.mkdir()
    config = load_config(Path(__file__).resolve().parents[1] / "configs/frozen_dinov2.toml")
    definition = {"status": "passed", "label_order": LABELS, "group_key": "video_id"}
    all_rows = []
    for role, videos in (("train", train_videos), ("validation", val_videos)):
        rows = []
        for video in videos:
            directory = root / video
            directory.mkdir()
            for label_index, label in enumerate(LABELS):
                for j in range(2):
                    frame = (label_index * 2 + j) * 30
                    name = f"{video}_{frame:06d}"
                    path = directory / (name + ".png")
                    Image.new("RGB", (8, 8), (int(video) * 15, label_index * 14, 100 + j * 5)).save(path)
                    rows.append({"video_id": video, "filename": name, "frame_index": frame,
                                 "time_seconds": frame / 30, "phase-id": label,
                                 "image_path": str(path.resolve()), "image_sha256": file_sha256(path)})
        with (root / f"{role}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        definition.update({f"{role}_video_ids": list(videos), f"{role}_samples": len(rows),
                           f"{role}_csv_sha256": file_sha256(root / f"{role}.csv")})
        all_rows.extend(rows)
    with (root / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    catalog_hash = file_sha256(root / "samples.csv")
    dataset_path = root / "dataset.json"
    dataset_path.write_text(json.dumps({"status": "passed", "samples_csv_sha256": catalog_hash}))
    definition.update(dataset_json=str(dataset_path), dataset_json_sha256=file_sha256(dataset_path),
                      samples_csv_sha256=catalog_hash)
    split_path = root / "split.json"
    split_path.write_text(json.dumps(definition))
    split = read_split(split_path)
    records = split["train"]["records"] + split["validation"]["records"]
    y = torch.tensor([r["target"] for r in records])
    x = torch.nn.functional.one_hot(y, 13).float()
    cache_path = root / "features.pt"
    torch.save({"features": x, "targets": y, "labels": LABELS, "split": split,
                "model_config": asdict(config), "feature_identity": {"contract": {
                    "model": config.name, "checkpoint_sha256": config.sha256, "output_dimension": 13}},
                "test_only": "These are synthetic features, not features extracted with DINO"}, cache_path)
    return cache_path






def test_real_cli_cpu_cache_training_and_no_output_overwrite(tmp_path, capsys):
    path = make_cache(tmp_path / "data")
    output = tmp_path / "run"
    command = ["train-head", "--features", str(path), "--output-dir", str(output),
               "--device", "cpu", "--epochs", "2", "--head-batch-size", "16"]
    assert main(command) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["gpu"]["gpu_used"] is False and result["reload_identical"]
    assert result["options"]["label_smoothing"] == .05
    assert result["train"]["per_video"].keys() == {"01", "02", "03"}
    assert result["validation"]["per_video"].keys() == {"04"}
    assert main(command) == 2  # no accidental overwrite
    assert "ERROR" in capsys.readouterr().err


def test_audit_needs_no_cuda_and_checks_source_hashes(tmp_path):
    path = make_cache(tmp_path / "data")
    assert main(["audit-features", "--features", str(path), "--output-dir", str(tmp_path / "audit")]) == 0
    report = json.loads((tmp_path / "audit/dataset_audit.json").read_text())
    assert report["train"]["samples"] == 78
    image = tmp_path / "data/01/01_000000.png"
    Image.new("RGB", (8, 8), "red").save(image)
    assert main(["audit-features", "--features", str(path), "--output-dir", str(tmp_path / "bad")]) == 2
    assert (tmp_path / "bad/failure.json").is_file()






@pytest.mark.parametrize("flag,value", [("--dropout", "1"), ("--patience", "-1"),
                                       ("--weight-decay", "nan"), ("--evals-per-epoch", "0")])
def test_invalid_head_flags_fail_before_output(tmp_path, flag, value):
    output = tmp_path / "run"
    assert main(["train-head", "--features", "missing.pt", "--output-dir", str(output),
                 "--device", "cpu", flag, value]) == 2
    assert not output.exists()


def test_cli_still_rejects_silent_cpu_extraction(tmp_path):
    assert main(["cache-features", "--split", "missing.json", "--output-dir", str(tmp_path / "run"),
                 "--device", "cpu"]) == 2
    assert not (tmp_path / "run").exists()


def test_four_fold_cv_coverage_fresh_initialization_and_single_cache_check(tmp_path, monkeypatch):
    import surgenet_dino.training as training
    from surgenet_dino import head_training
    path = make_cache(tmp_path / "data")
    cache = torch.load(path, weights_only=True)
    records = cache["split"]["train"]["records"] + cache["split"]["validation"]["records"]
    groups = grouped_folds(records, 4, 42, LABELS)
    assert groups == grouped_folds(records, 4, 42, LABELS)
    assert sorted(v for g in groups for v in g) == ["01", "02", "03", "04"]
    calls, held = [], []
    original_load, original_fit = training.load_features, head_training.fit

    def counted(p):
        calls.append(p)
        return original_load(p)

    def guarded(x, y, vx, vy, train_rows, val_rows, *args, **kwargs):
        assert not {r["video_id"] for r in train_rows} & {r["video_id"] for r in val_rows}
        held.extend(r["filename"] for r in val_rows)
        return original_fit(x, y, vx, vy, train_rows, val_rows, *args, **kwargs)

    monkeypatch.setattr(training, "load_features", counted)
    monkeypatch.setattr(head_training, "fit", guarded)
    output, splits = tmp_path / "cv", tmp_path / "splits"
    assert main(["crossval-head", "--features", str(path), "--output-dir", str(output),
                 "--split-output-dir", str(splits), "--device", "cpu", "--epochs", "1"]) == 0
    assert len(calls) == 1
    assert len(held) == len(set(held)) == len(records)
    results = [json.loads((output / f"fold_{k}/result.json").read_text()) for k in range(1, 5)]
    assert len({r["initial_head_sha256"] for r in results}) == 1
    summary = json.loads((output / "result.json").read_text())
    assert summary["coverage_exactly_once"] and summary["samples"] == len(records)
    assert set(summary["pooled"]["per_video"]) == {"01", "02", "03", "04"}
    for k in range(1, 5):
        split = read_split(splits / f"fold_{k}/split.json")
        assert len(split["validation"]["records"]) == 26
    with (output / "validation_predictions.csv").open() as f:
        assert len(list(csv.DictReader(f))) == len(records)


def test_cv_failure_does_not_continue_or_publish_summary(tmp_path, monkeypatch):
    from surgenet_dino import head_training
    path = make_cache(tmp_path / "data")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("simulated fold failure")

    monkeypatch.setattr(head_training, "fit", fail)
    output = tmp_path / "failed"
    assert main(["crossval-head", "--features", str(path), "--output-dir", str(output),
                 "--split-output-dir", str(tmp_path / "splits"), "--device", "cpu"]) == 2
    assert len(calls) == 1 and (output / "fold_1/failure.json").exists()
    assert not (output / "result.json").exists() and not (output / "fold_2").exists()
