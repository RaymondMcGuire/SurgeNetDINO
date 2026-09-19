import importlib.metadata
import json
import runpy
from pathlib import Path

import pytest
import torch
from PIL import Image

from surgenet_dino.cli import main
from surgenet_dino.config import ModelConfig
from surgenet_dino.model import build_model, set_trainable
from surgenet_dino.registry import MODELS, get_spec
from surgenet_dino.runtime import preprocess_image, resolve_device


@pytest.mark.parametrize("name", MODELS)
def test_factory_creates_exactly_one_original_architecture(fake_timm, name):
    model, report = build_model(ModelConfig(name=name, weights="none"))
    spec = get_spec(name)
    assert len(fake_timm) == 1
    assert fake_timm[0] == (spec.timm_name, {
        "pretrained": False, "img_size": (spec.image_size, spec.image_size),
        "patch_size": spec.patch_size, "num_classes": 0,
    })
    assert model.head.__class__ is torch.nn.Identity
    assert report["backbone_pretrained"] is False


def test_factory_loads_only_selected_local_weights(fake_timm, tmp_path):
    source, _ = build_model(ModelConfig(weights="none"))
    checkpoint = tmp_path / "fixture.pth"
    torch.save(source.state_dict(), checkpoint)
    target, report = build_model(ModelConfig(weights="local", checkpoint=str(checkpoint)))
    assert report["checkpoint_sha256"]
    assert report["backbone_pretrained"] is True
    x = torch.randn(1, 3)
    torch.testing.assert_close(source(x), target(x), rtol=0, atol=0)


def test_head_training_flags(fake_timm):
    model, report = build_model(ModelConfig(weights="none", num_classes=13))
    assert report["task_head_restored"] is False
    assert set_trainable(model, "head") == sum(p.numel() for p in model.head.parameters())
    assert all(p.requires_grad == name.startswith("head.") for name, p in model.named_parameters())
    assert set_trainable(model, "all") == sum(p.numel() for p in model.parameters())
    assert set_trainable(model, "none") == 0
    with pytest.raises(ValueError):
        set_trainable(model, "mystery")
    features, _ = build_model(ModelConfig(weights="none"))
    with pytest.raises(ValueError, match="No trainable task head"):
        set_trainable(features, "head")


def test_cuda_is_not_silently_downgraded(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"


def test_image_preprocessing_handles_grayscale(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (23, 41), color=128).save(path)
    result = preprocess_image(path, 224)
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32
    expected = torch.tensor([(128 / 255 - m) / s
                             for m, s in zip((.485, .456, .406), (.229, .224, .225))])
    torch.testing.assert_close(result[:, 0, 0], expected)


def test_list_does_not_construct_any_model(fake_timm, capsys):
    assert main(["list", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 8
    assert fake_timm == []


def test_compatibility_module_import_has_no_side_effect(fake_timm):
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "load_models.py"), run_name="not_main")
    assert fake_timm == []


def _fake_timm_version(monkeypatch):
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "1.0.24" if name == "timm" else original(name))


@pytest.mark.parametrize("backward", [False, True])
def test_smoke_cli_synthetic_check(fake_timm, monkeypatch, capsys, backward):
    _fake_timm_version(monkeypatch)
    args = ["smoke", "--weights", "none", "--device", "cpu", "--model", "dinov2_vits"]
    if backward:
        args += ["--backward"]
    assert main(args) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["output_shape"] == [1, 384]
    assert info["challenge_accuracy_measured"] is False
    assert info["loading"]["weights_source"] == "none"
    if backward:
        assert info["parameters_with_grad"] > 0


def test_cli_bad_batch_fails_before_build(fake_timm, capsys):
    assert main(["smoke", "--batch-size", "0"]) == 2
    assert "batch-size" in capsys.readouterr().err
    assert fake_timm == []


def test_encode_local_fixture_and_metadata(fake_timm, monkeypatch, tmp_path, capsys):
    _fake_timm_version(monkeypatch)
    image = tmp_path / "test.png"
    Image.new("RGB", (25, 31), color=(10, 20, 30)).save(image)
    source, _ = build_model(ModelConfig(name="dinov2_vits", weights="none"))
    checkpoint = tmp_path / "fixture.pth"
    torch.save(source.state_dict(), checkpoint)
    output = tmp_path / "features.pt"
    args = ["encode", "--model", "dinov2_vits", "--weights", "local", "--checkpoint", str(checkpoint),
            "--device", "cpu", "--image", str(image), "--output", str(output)]
    assert main(args) == 0
    saved = torch.load(output, weights_only=True)
    assert saved["features"].shape == (1, 384)
    assert saved["metadata"]["loading"]["checkpoint_sha256"]
    assert saved["metadata"]["preprocessing"]["color"] == "RGB"
    assert main(args) == 2
    assert "Output exists" in capsys.readouterr().err


def test_encode_refuses_random_features(fake_timm, capsys):
    assert main(["encode", "--weights", "none", "--image", "fake.png", "--output", "fake.pt"]) == 2
    assert "refuses random features" in capsys.readouterr().err
    assert fake_timm == []
