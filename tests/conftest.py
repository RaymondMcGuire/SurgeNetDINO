"""Small test doubles exercise the loader, not the DINO architecture or trained weights."""

import sys
import types

import pytest
import torch
from torch import nn


class TinyModel(nn.Module):
    def __init__(self, feature_dim=4, num_classes=0):
        super().__init__()
        self.patch_embed = nn.Linear(3, feature_dim)
        self.blocks = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Linear(feature_dim, num_classes) if num_classes else nn.Identity()

    def forward(self, x):
        if x.ndim == 4:
            x = x.mean(dim=(-1, -2))
        return self.head(self.norm(self.blocks(self.patch_embed(x))))


@pytest.fixture
def fake_timm(monkeypatch):
    calls = []

    def create_model(name, **kwargs):
        from surgenet_dino.registry import MODELS

        spec = next(spec for spec in MODELS.values() if spec.timm_name == name)
        calls.append((name, kwargs))
        return TinyModel(spec.feature_dim, kwargs["num_classes"])

    module = types.ModuleType("timm")
    module.create_model = create_model
    module.__version__ = "1.0.24"
    monkeypatch.setitem(sys.modules, "timm", module)
    return calls


@pytest.fixture
def tiny_pair():
    torch.manual_seed(11)
    source = TinyModel()
    torch.manual_seed(99)
    target = TinyModel()
    return source, target
