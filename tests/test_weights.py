import copy

import pytest
import torch

from surgenet_dino.config import ModelConfig
from surgenet_dino.weights import (
    CheckpointError,
    extract_state_dict,
    file_sha256,
    load_checked_state_dict,
    normalize_keys,
    read_checkpoint,
    resolve_checkpoint,
)


def test_backbone_load_preserves_outputs_exactly(tiny_pair):
    source, target = tiny_pair
    report = load_checked_state_dict(target, source.state_dict())
    x = torch.randn(3, 3)
    torch.testing.assert_close(source(x), target(x), rtol=0, atol=0)
    assert report["missing_keys"] == []


def test_missing_backbone_is_rejected_before_mutation(tiny_pair):
    source, target = tiny_pair
    before = copy.deepcopy(target.state_dict())
    state = dict(source.state_dict())
    del state["patch_embed.weight"]
    with pytest.raises(CheckpointError, match="refusing partial"):
        load_checked_state_dict(target, state)
    for name, tensor in before.items():
        torch.testing.assert_close(tensor, target.state_dict()[name], rtol=0, atol=0)


def test_shape_mismatch_is_rejected(tiny_pair):
    source, target = tiny_pair
    state = dict(source.state_dict())
    state["patch_embed.weight"] = torch.zeros(7, 3)
    with pytest.raises(CheckpointError, match="shape mismatches"):
        load_checked_state_dict(target, state)


def test_unused_pretraining_key_is_reported(tiny_pair):
    source, target = tiny_pair
    state = dict(source.state_dict())
    state["mask_token"] = torch.zeros(1, 1, 4)
    report = load_checked_state_dict(target, state)
    assert report["ignored_pretraining_keys"] == ["mask_token"]


def test_new_head_is_allowed_and_old_head_is_explicitly_discarded(tiny_pair):
    source, target = tiny_pair
    source.head = torch.nn.Linear(4, 20)
    target.head = torch.nn.Linear(4, 13)
    original_head = target.head.weight.detach().clone()
    report = load_checked_state_dict(target, source.state_dict(), "backbone")
    assert report["missing_keys"] == ["head.bias", "head.weight"]
    assert report["discarded_head_keys"] == ["head.bias", "head.weight"]
    torch.testing.assert_close(original_head, target.head.weight, rtol=0, atol=0)


def test_full_task_checkpoint_restores_head(tiny_pair):
    source, target = tiny_pair
    source.head = torch.nn.Linear(4, 13)
    target.head = torch.nn.Linear(4, 13)
    report = load_checked_state_dict(target, source.state_dict(), "full")
    assert report["missing_keys"] == []
    x = torch.randn(2, 3)
    torch.testing.assert_close(source(x), target(x), rtol=0, atol=0)


def test_full_checkpoint_rejects_missing_head_or_extra_key(tiny_pair):
    source, target = tiny_pair
    target.head = torch.nn.Linear(4, 13)
    with pytest.raises(CheckpointError):
        load_checked_state_dict(target, source.state_dict(), "full")
    source.head = torch.nn.Linear(4, 13)
    state = dict(source.state_dict())
    state["unused"] = torch.ones(1)
    with pytest.raises(CheckpointError, match="unexpected"):
        load_checked_state_dict(target, state, "full")


def test_wrapped_teacher_and_prefixes(tiny_pair):
    source, _ = tiny_pair
    state = {"module.teacher.backbone." + k: v for k, v in source.state_dict().items()}
    raw = {"teacher": state, "student": {"different": torch.zeros(1)}, "epoch": 4}
    normalized = normalize_keys(extract_state_dict(raw))
    assert set(normalized) == set(source.state_dict())


def test_explicit_checkpoint_key():
    data = {"foo": {"model": {"tensor": torch.zeros(2)}}}
    assert "tensor" in extract_state_dict(data, "foo.model")
    with pytest.raises(CheckpointError):
        extract_state_dict(data, "missing")
    with pytest.raises(CheckpointError):
        extract_state_dict({"state_dict": {"x": torch.zeros(1)}, "model": {"y": torch.zeros(1)}})


def test_key_collision_and_invalid_container_rejected():
    with pytest.raises(CheckpointError, match="collision"):
        normalize_keys({"module.x": torch.ones(1), "x": torch.zeros(1)})
    for value in (None, {}, {"bad": "not a tensor"}, [torch.ones(1)]):
        with pytest.raises(CheckpointError):
            extract_state_dict(value)


def test_safe_checkpoint_read_and_sha256(tmp_path, tiny_pair):
    source, _ = tiny_pair
    path = tmp_path / "fixture.pth"
    torch.save(source.state_dict(), path)
    data, checksum = read_checkpoint(path)
    assert checksum == file_sha256(path)
    assert set(data) == set(source.state_dict())
    read_checkpoint(path, checksum.upper())
    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        read_checkpoint(path, "0" * 64)
    config = ModelConfig(weights="local", checkpoint=str(path))
    assert resolve_checkpoint(config) == (path.resolve(), None)


def test_missing_local_file_never_downloads(tmp_path):
    config = ModelConfig(weights="local", checkpoint=str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(config)


def test_download_failure_never_becomes_random_weights(monkeypatch):
    import huggingface_hub

    def fail(**kwargs):
        raise OSError("test: network unavailable")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    with pytest.raises(CheckpointError, match="No random-weight fallback"):
        resolve_checkpoint(ModelConfig())


def test_offline_and_revision_are_passed_to_cache(monkeypatch, tmp_path):
    import huggingface_hub

    path = tmp_path / "snapshots" / ("a" * 40) / "weights.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture")
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    resolved, commit = resolve_checkpoint(ModelConfig(offline=True, revision="a" * 40))
    assert resolved == path
    assert commit == "a" * 40
    assert seen["local_files_only"] is True
    assert seen["revision"] == "a" * 40
