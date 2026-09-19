"""Real architecture equivalence checks. No pretrained weights or network access.

These tests are SKIPPED when timm is unavailable; a skip is not a successful model test.
"""

import pytest
import torch

from surgenet_dino.config import ModelConfig
from surgenet_dino.model import build_model


@pytest.mark.integration
@pytest.mark.parametrize("name,timm_name,size,patch", [
    ("dinov1_vits", "vit_small_patch16_224.dino", 224, 16),
    ("dinov2_vits", "vit_small_patch14_dinov2", 336, 14),
    ("dinov3_vits", "vit_small_patch16_dinov3.lvd1689m", 336, 16),
])
def test_real_timm_matches_direct_original_construction(name, timm_name, size, patch):
    timm = pytest.importorskip("timm", reason="real timm package is not installed")
    torch.manual_seed(123)
    original = timm.create_model(timm_name, img_size=(size, size), patch_size=patch, num_classes=10)
    torch.manual_seed(123)
    refactored, _ = build_model(ModelConfig(name=name, weights="none", num_classes=10, device="cpu"))
    original.eval()
    refactored.eval()
    sample = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        expected = original(sample)
        actual = refactored(sample)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.integration
@pytest.mark.parametrize("name", [
    "dinov1_vits", "dinov1_vitb", "dinov2_vits", "dinov2_vitb", "dinov2_vitl",
    "dinov3_vits", "dinov3_vitb", "dinov3_vitl",
])
def test_all_real_timm_model_names_are_registered(name):
    timm = pytest.importorskip("timm", reason="real timm package is not installed")
    from surgenet_dino.registry import get_spec

    spec = get_spec(name)
    assert timm.is_model(spec.timm_name)
