"""A single model factory; no changes to timm's forward, attention, or loss code."""

from typing import Any

from torch import nn

from .config import ModelConfig
from .registry import get_spec
from .weights import extract_state_dict, load_checked_state_dict, read_checkpoint, resolve_checkpoint


def build_model(config: ModelConfig) -> tuple[nn.Module, dict[str, Any]]:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is missing. Run scripts/init.bat (or bash scripts/init.sh).") from exc
    spec = get_spec(config.name)
    model = timm.create_model(
        spec.timm_name,
        pretrained=False,  # only the explicitly selected source is allowed to supply weights
        img_size=(spec.image_size, spec.image_size),
        patch_size=spec.patch_size,
        num_classes=config.num_classes,
    )
    report: dict[str, Any] = {
        "model": spec.name,
        "architecture": spec.timm_name,
        "image_size": spec.image_size,
        "patch_size": spec.patch_size,
        "feature_dim": spec.feature_dim,
        "num_classes": config.num_classes,
        "weights_source": config.weights,
        "weights_scope": config.weights_scope,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "resolved_hf_commit": None,
        "backbone_pretrained": config.weights != "none",
        "task_head_restored": config.weights_scope == "full" and config.num_classes > 0,
        "missing_keys": [],
        "ignored_pretraining_keys": [],
        "discarded_head_keys": [],
    }
    if config.weights != "none":
        path, commit = resolve_checkpoint(config)
        data, checksum = read_checkpoint(path, config.sha256)
        state = extract_state_dict(data, config.checkpoint_key)
        report.update(load_checked_state_dict(model, state, config.weights_scope))
        report.update(checkpoint_path=str(path), checkpoint_sha256=checksum,
                      resolved_hf_commit=commit)
    if config.num_classes > 0 and not report["task_head_restored"]:
        report["warning"] = "The task head is randomly initialized, NOT a trained Challenge classifier."
    if config.weights == "none":
        report["warning"] = "Entire model is random; this is only an architecture/environment test."
    return model, report


def set_trainable(model: nn.Module, mode: str) -> int:
    """Set gradient flags only. Training/eval mode remains the caller's responsibility."""
    if mode not in {"all", "head", "none"}:
        raise ValueError("mode must be all, head, or none")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(mode == "all" or (mode == "head" and name.startswith("head.")))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if mode == "head" and count == 0:
        raise ValueError("No trainable task head. Set num_classes to the verified number of classes.")
    return count
