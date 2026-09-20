import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from surgenet_dino.cli import main, make_parser, _configuration
from surgenet_dino.gpu import GPUControl, Resources, check_sample, sleep_seconds
from surgenet_dino.training import (
    LABELS, assert_identity, load_features, read_manifest, read_split, MANIFEST_FIELDS, resources_for, select_profile, subset,
)


def manifest(tmp_path, video="01"):
    from surgenet_dino.weights import file_sha256
    images = tmp_path / f"images_{video}"
    images.mkdir()
    rows = []
    for i, label in enumerate(LABELS):
        for j in range(2):
            frame = (i * 2 + j) * 30
            name = f"{video}_{frame:06d}"
            image = images / f"{name}.png"
            Image.new("RGB", (8, 8), (10 + i, int(video), 90 + j)).save(image)
            rows.append([video, name, frame, frame / 30, label, str(image), file_sha256(image)])
    path = tmp_path / f"{video}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(MANIFEST_FIELDS)
        writer.writerows(rows)
    return path


def split_fixture(tmp_path):
    from surgenet_dino.weights import file_sha256
    train, val = manifest(tmp_path), manifest(tmp_path, "04")
    (tmp_path / "train.csv").write_bytes(train.read_bytes())
    (tmp_path / "validation.csv").write_bytes(val.read_bytes())
    catalog = tmp_path / "samples.csv"
    catalog.write_text(train.read_text() + "\n".join(val.read_text().splitlines()[1:]) + "\n")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"status": "passed", "samples_csv_sha256": file_sha256(catalog)}))
    definition = {"status": "passed", "label_order": LABELS, "group_key": "video_id",
                  "dataset_json": str(dataset), "dataset_json_sha256": file_sha256(dataset),
                  "samples_csv_sha256": file_sha256(catalog)}
    for role, video in [("train", "01"), ("validation", "04")]:
        definition.update({f"{role}_csv_sha256": file_sha256(tmp_path / f"{role}.csv"),
                           f"{role}_samples": 26, f"{role}_video_ids": [video]})
    path = tmp_path / "split.json"
    path.write_text(json.dumps(definition))
    return path


def test_manifest_cache_alignment_and_source_changes(tmp_path):
    split = read_split(split_fixture(tmp_path))
    data = split["train"]
    assert len(data["records"]) == 26
    assert subset(data["records"], 2, 42) == subset(data["records"], 2, 42)
    cache = tmp_path / "features.pt"
    rows = data["records"] + split["validation"]["records"]
    payload = {"features": torch.ones(52, 3), "targets": torch.tensor([r["target"] for r in rows]),
               "split": split, "labels": LABELS,
               "feature_identity": {"contract": {"output_dimension": 3}}}
    torch.save(payload, cache)
    assert load_features(cache)["features"].shape == (52, 3)
    payload["targets"][0] = 1
    torch.save(payload, cache)
    with pytest.raises(ValueError, match="alignment"):
        load_features(cache)
    payload["targets"][0] = 0
    torch.save(payload, cache)
    Image.new("RGB", (8, 8), "red").save(tmp_path / "images_01/01_000000.png")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_features(cache)


@pytest.mark.parametrize("problem", ["duplicate", "black", "unknown", "missing", "path", "corrupt"])
def test_bad_manifest_is_rejected(tmp_path, problem):
    path = manifest(tmp_path)
    content = path.read_text()
    image = tmp_path / "images_01/01_000000.png"
    if problem == "duplicate":
        path.write_text(content + content.splitlines()[1] + "\n")
    elif problem == "black":
        Image.new("RGB", (8, 8), (1, 1, 1)).save(image)
    elif problem == "unknown":
        path.write_text(content.replace(",0.0,0,", ",0.0,x,"))
    elif problem == "path":
        path.write_text(content.replace("01_000000", "../outside"))
    elif problem == "corrupt":
        image.write_bytes(b"broken PNG")
    else:
        image.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        read_manifest(path)


def test_split_rejects_video_leakage(tmp_path):
    from surgenet_dino.weights import file_sha256
    path = split_fixture(tmp_path)
    definition = json.loads(path.read_text())
    (tmp_path / "validation.csv").write_bytes((tmp_path / "train.csv").read_bytes())
    definition["validation_csv_sha256"] = file_sha256(tmp_path / "validation.csv")
    definition["validation_video_ids"] = ["01"]
    path.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match="leakage"):
        read_split(path)


def test_training_uses_only_training_rows_and_selects_validation_best(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from surgenet_dino import training as module
    from surgenet_dino.config import ModelConfig
    split = read_split(split_fixture(tmp_path))
    targets = torch.tensor([r["target"] for r in split["train"]["records"]])
    positive = torch.nn.functional.one_hot(targets, 13).float()
    config = ModelConfig(device="cpu", seed=42)
    cache = tmp_path / "features.pt"
    torch.save({"features": torch.cat([positive, -positive]), "targets": targets.repeat(2),
                "split": split, "labels": LABELS, "model_config": asdict(config),
                "feature_identity": {"contract": {"output_dimension": 13, "model": config.name}}}, cache)

    class Guard:
        device = torch.device("cpu")

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def batch(self, *args):
            return nullcontext()

        def mark(self):
            return 0

        def summary(self, *args):
            return {}

    original = module.head_step
    updates = []

    def step(head, optimizer, x, y, guard):
        assert (x >= 0).all(), "Validation features reached the optimizer"
        updates.append(len(x))
        return original(head, optimizer, x, y, guard)

    monkeypatch.setattr(module, "GPUControl", Guard)
    monkeypatch.setattr(module, "environment", lambda *args: {})
    monkeypatch.setattr(module, "head_step", step)
    args = make_parser().parse_args(["train-head", "--features", str(cache), "--output-dir",
                                    str(tmp_path / "train"), "--epochs", "3", "--learning-rate", "0.1"])
    args.output_dir.mkdir()
    result = module.train_head(args, config, args.output_dir)
    assert len(updates) == 3 and result["parameters_changed"] and result["train_loss_decreased"]
    assert result["reload_identical"]
    assert result["best_epoch"] == min(result["history"][1:], key=lambda r: r["validation_loss"])["epoch"]
    assert set(result["validation"]["per_video"]) == {"04"}
    assert (args.output_dir / "head_last.pt").exists()


def test_pacing_counts_existing_idle_and_stops_on_bad_telemetry():
    assert sleep_seconds(.03, .04, .3) == pytest.approx(.06)
    assert sleep_seconds(.03, .15, .3) == 0
    resource = Resources()
    sample = dict(monotonic=10, temperature=60, used_mib=1000, total_mib=32000)
    check_sample(sample, resource, 11)
    for replacement, now in [({"temperature": 75}, 11), ({"used_mib": 6400}, 11), ({}, 16)]:
        with pytest.raises(RuntimeError):
            check_sample({**sample, **replacement}, resource, now)
    guard = object.__new__(GPUControl)
    guard.error = "telemetry failed"
    with pytest.raises(RuntimeError, match="telemetry failed"):
        guard.check()
    for overrides in [{"duty_cycle": 0}, {"memory_fraction": 2}, {"duty_cycle": float("nan")},
                      {"cpu_threads": 0}, {"max_temperature": float("inf")}]:
        with pytest.raises(ValueError):
            Resources(**overrides)
    args = SimpleNamespace(gpu_profile="low", gpu_memory_fraction=0, gpu_duty_cycle=None,
                           max_gpu_temperature=75, cpu_threads=2)
    with pytest.raises(ValueError):
        resources_for(args)


def test_profile_selection_and_identity():
    def trial(batch, speed):
        return {"status": "passed", "score": speed, "batch_size": batch}
    assert select_profile([trial(4, 96), trial(8, 100), trial(1, 70)])["batch_size"] == 4
    with pytest.raises(RuntimeError):
        select_profile([])
    with pytest.raises(ValueError, match="identity mismatch"):
        assert_identity({"gpu": "5090"}, {"gpu": "6000 Ada"}, "Calibration")


def test_saved_profile_defaults_and_explicit_override(tmp_path):
    path = tmp_path / "profiles.json"
    saved = Resources("low", .15, .25, 70, 3)
    path.write_text(json.dumps({"profiles": {"low": {"resources": asdict(saved)}}}))
    args = SimpleNamespace(gpu_profile="low", profiles=path, gpu_memory_fraction=None,
                           gpu_duty_cycle=None, max_gpu_temperature=None, cpu_threads=None)
    assert resources_for(args) == saved
    args.gpu_duty_cycle = .2
    assert resources_for(args).duty_cycle == .2


def test_phase_cli_defaults_and_invalid_settings(tmp_path, capsys):
    args = make_parser().parse_args(["cache-features", "--split", "missing.json",
                                     "--output-dir", str(tmp_path / "new")])
    config = _configuration(args)
    assert config.offline and config.seed == 42 and config.device == "cuda:0" and config.sha256
    assert main(["cache-features", "--split", "missing.json", "--output-dir", str(tmp_path / "new"),
                 "--gpu-duty-cycle", "0"]) == 2
    assert not (tmp_path / "new").exists()
    assert "duty_cycle" in capsys.readouterr().err


@pytest.mark.skipif(__import__("os").name != "nt", reason="cmd.exe")
def test_train_launcher_stops_and_pauses(tmp_path):
    import re
    import subprocess

    source = (Path(__file__).resolve().parents[1] / "scripts/train.bat").read_text()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "split.json").write_text("{}")
    (tmp_path / "profiles.json").write_text("{}")
    for code in (0, 7):
        stub = re.sub(r"(?m)^uv [^\n]+", f"echo TRAIN_COMMAND\ncmd /d /c exit {code}", source)
        stub = stub.replace("where uv >nul 2>&1", "cmd /d /c exit 0")
        target = scripts / "train.bat"
        target.write_text(stub)
        result = subprocess.run(["cmd", "/d", "/c", str(target), "low", str(tmp_path / f"run{code}"),
                                 str(tmp_path / "split.json"), str(tmp_path / "profiles.json")],
                                input=b"\n", capture_output=True, timeout=10)
        assert result.returncode == code, result.stdout + result.stderr
        assert result.stdout.count(b"TRAIN_COMMAND") == (2 if code == 0 else 1)
        marker = f"Script finished with exit code {code} (0 = success).".encode()
        assert result.stdout.count(marker) == 1
        assert result.stdout.split(marker)[1].strip()
