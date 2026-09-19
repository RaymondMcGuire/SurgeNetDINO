"""Model selection and feature identities; TinyModel fixtures do not validate a real ViT."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from surgenet_dino.cli import main
from surgenet_dino.config import ModelConfig, load_config
from surgenet_dino.provenance import feature_identity
from surgenet_dino.registry import HF_REPO_ID, MODELS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", [*MODELS, "offline_smoke"])
def test_presets(name):
    cfg = load_config(ROOT / "configs" / f"{name}.toml")
    assert cfg.name == ("dinov2_vitb" if name == "offline_smoke" else name)
    assert cfg.weights == ("none" if name == "offline_smoke" else "surgenet")
    assert cfg.num_classes == 0


@pytest.mark.parametrize("name", MODELS)
def test_plan_resolves_correct_weight_without_constructing_model(fake_timm, capsys, name):
    assert main(["plan", "--model", name]) == 0
    plan = json.loads(capsys.readouterr().out)
    spec = MODELS[name]
    assert plan["checkpoint_source"] == {"repo_id": HF_REPO_ID, "filename": spec.filename}
    assert plan["expected_output_dimension"] == spec.feature_dim
    assert not plan["network_constructed"]
    assert not plan["downloads_performed"]
    assert fake_timm == []


def test_plan_and_list_have_no_torch_import_in_fresh_process():
    code = '''import sys
from surgenet_dino.cli import main
assert main(["plan", "--model", "dinov3_vitb"]) == 0
assert "torch" not in sys.modules
assert "timm" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


@pytest.mark.parametrize("field,value", [
    ("sha256", "a" * 64),
    ("checkpoint_key", "teacher"),
])
def test_switch_does_not_reuse_pinned_weight_metadata(tmp_path, capsys, field, value):
    path = tmp_path / "pin.toml"
    path.write_text(f'[model]\nname="dinov2_vitb"\n{field}="{value}"\n')
    assert main(["plan", "--config", str(path), "--model", "dinov3_vitb"]) == 2
    assert "model-specific config" in capsys.readouterr().err


def test_switch_does_not_reuse_local_checkpoint(tmp_path, capsys):
    path = tmp_path / "local.toml"
    path.write_text('[model]\nname="dinov2_vitb"\nweights="local"\ncheckpoint="v2.pth"\n')
    assert main(["plan", "--config", str(path), "--model", "dinov3_vitb"]) == 2
    assert "pinned checkpoint" in capsys.readouterr().err


def test_feature_identity_separates_v2_v3_equal_dimension():
    report = {"checkpoint_sha256": "a" * 64}
    v2 = feature_identity(ModelConfig(name="dinov2_vitb"), report, "1.0.24")
    v3 = feature_identity(ModelConfig(name="dinov3_vitb"), report, "1.0.24")
    assert v2["id"] != v3["id"]
    assert v2["contract"]["output_dimension"] == v3["contract"]["output_dimension"]


def test_feature_identity_separates_checkpoints_and_library_versions():
    cfg = ModelConfig()
    a = feature_identity(cfg, {"checkpoint_sha256": "a" * 64}, "1.0.24")
    b = feature_identity(cfg, {"checkpoint_sha256": "b" * 64}, "1.0.24")
    c = feature_identity(cfg, {"checkpoint_sha256": "a" * 64}, "1.0.25")
    assert len({a["id"], b["id"], c["id"]}) == 3
    assert a == feature_identity(cfg, {"checkpoint_sha256": "a" * 64}, "1.0.24")


def test_legacy_entrypoint_understands_plan():
    result = subprocess.run(
        [sys.executable, "load_models.py", "plan", "--model", "dinov3_vitb"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout)["spec"]["name"] == "dinov3_vitb"
