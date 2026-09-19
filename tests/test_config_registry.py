import dataclasses
import tomllib
from pathlib import Path

import pytest

from surgenet_dino.config import ModelConfig, load_config
from surgenet_dino.registry import MODELS, get_spec


EXPECTED = {
    "dinov1_vits": ("vit_small_patch16_224.dino", 224, 16, 384),
    "dinov1_vitb": ("vit_base_patch16_224.dino", 224, 16, 768),
    "dinov2_vits": ("vit_small_patch14_dinov2", 336, 14, 384),
    "dinov2_vitb": ("vit_base_patch14_dinov2", 336, 14, 768),
    "dinov2_vitl": ("vit_large_patch14_dinov2", 336, 14, 1024),
    "dinov3_vits": ("vit_small_patch16_dinov3.lvd1689m", 336, 16, 384),
    "dinov3_vitb": ("vit_base_patch16_dinov3.lvd1689m", 336, 16, 768),
    "dinov3_vitl": ("vit_large_patch16_dinov3.lvd1689m", 336, 16, 1024),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_registry_preserves_original_arguments(name):
    spec = get_spec(name)
    assert (spec.timm_name, spec.image_size, spec.patch_size, spec.feature_dim) == EXPECTED[name]
    assert spec.filename.endswith("_SurgeNetXL.pth")
    assert spec.image_size % spec.patch_size == 0


def test_all_variants_and_immutability():
    assert set(MODELS) == set(EXPECTED)
    with pytest.raises(TypeError):
        MODELS["bad"] = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_spec("dinov2_vitb").name = "changed"
    with pytest.raises(ValueError, match="Unknown model"):
        get_spec("vit_magic")


@pytest.mark.parametrize("change", [
    {"num_classes": -1}, {"num_classes": True}, {"weights": "invalid"},
    {"name": "invalid"}, {"device": "cuda:-1"}, {"device": "mps"},
    {"seed": -1}, {"seed": True}, {"seed": 2**32}, {"offline": "false"},
    {"weights": "local"}, {"checkpoint": "file.pth"}, {"weights_scope": "full"},
    {"sha256": "abc"}, {"weights": "none", "sha256": "a" * 64},
    {"cache_dir": ""}, {"checkpoint_key": ""}, {"name": 42},
])
def test_invalid_config_fails_early(change):
    with pytest.raises(ValueError):
        ModelConfig(**change)


def test_defaults_are_features_not_ten_class_predictions():
    config = ModelConfig()
    assert config.num_classes == 0
    assert config.name == "dinov2_vitb"
    assert config.device == "cuda"


def test_config_loading_and_unknown_keys(tmp_path):
    path = tmp_path / "model.toml"
    path.write_text('[model]\nweights="none"\ndevice="cpu"\n', encoding="utf-8")
    assert load_config(path).weights == "none"
    path.write_text('[model]\npretend_option=1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)
    path.write_text('[wrong]\nname="dinov2_vitb"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="one \\[model\\]"):
        load_config(path)


def test_project_manifest_backend_pins():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    assert data["tool"]["uv"]["package"] is False
    assert data["project"]["optional-dependencies"]["cu128"] == ["torch==2.10.0", "torchvision==0.25.0"]
    assert data["tool"]["uv"]["conflicts"] == [[{"extra": "cpu"}, {"extra": "cu128"}]]
    assert all(index["explicit"] for index in data["tool"]["uv"]["index"])
    assert not any("torchaudio" in dep for dep in data["project"]["dependencies"])
