"""Safe checkpoint deserialization and fail-closed backbone loading.

Architecture math and parameter names are not rewritten. Only wrapper prefixes
are removed, and unused pretraining keys are reported rather than silently hidden.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig
from .registry import HF_REPO_ID, get_spec


class CheckpointError(RuntimeError):
    """A checkpoint cannot be proven to cover the requested network."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(config: ModelConfig) -> tuple[Path, str | None]:
    if config.weights == "local":
        path = Path(config.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path, None
    if config.weights != "surgenet":
        raise ValueError("No checkpoint exists for weights=none")
    from huggingface_hub import hf_hub_download

    spec = get_spec(config.name)
    try:
        path = Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=spec.filename,
            revision=config.revision,
            cache_dir=config.cache_dir,
            local_files_only=config.offline,
        )).absolute()
    except Exception as exc:
        raise CheckpointError(
            f"Cannot obtain {spec.filename}. No random-weight fallback was used. "
            "Check network/cache/revision, or use --weights local --checkpoint PATH."
        ) from exc
    parts = path.parts
    commit = None
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            commit = parts[index + 1]
    return path, commit


def read_checkpoint(path: Path, expected_sha256: str | None = None) -> tuple[Any, str]:
    actual = file_sha256(path)
    if expected_sha256 is not None and actual.lower() != expected_sha256.lower():
        raise CheckpointError(f"SHA-256 mismatch for {path}; expected {expected_sha256}, got {actual}")
    # Do not fall back to unsafe pickle loading for an arbitrary downloaded file.
    return torch.load(path, map_location="cpu", weights_only=True), actual


def extract_state_dict(data: Any, key: str | None = None) -> dict[str, torch.Tensor]:
    if key is not None:
        for part in key.split("."):
            if not isinstance(data, Mapping) or part not in data:
                raise CheckpointError(f"Checkpoint key not found: {key!r}")
            data = data[part]
    for _ in range(5):
        if isinstance(data, Mapping) and data and all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in data.items()
        ):
            return dict(data)
        if not isinstance(data, Mapping):
            break
        # Prefer the teacher for published self-supervised checkpoints, explicitly.
        if isinstance(data.get("teacher"), Mapping):
            data = data["teacher"]
            continue
        candidates = [k for k in ("state_dict", "model", "backbone")
                      if isinstance(data.get(k), Mapping)]
        if len(candidates) == 1:
            data = data[candidates[0]]
        else:
            break
    raise CheckpointError(
        "Expected a non-empty tensor state_dict. For other containers, select "
        "--checkpoint-key explicitly (for example student or state_dict)."
    )


def normalize_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for original, tensor in state.items():
        key = original
        while True:
            prefix = next((p for p in ("module.", "teacher.", "backbone.")
                           if key.startswith(p)), None)
            if prefix is None:
                break
            key = key[len(prefix):]
        if key in result:
            raise CheckpointError(f"Key collision after wrapper removal: {original!r} -> {key!r}")
        result[key] = tensor
    return result


def _is_head(key: str) -> bool:
    return key.startswith("head.")


def load_checked_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    scope: str = "backbone",
) -> dict[str, list[str]]:
    if scope not in {"backbone", "full"}:
        raise ValueError("scope must be backbone or full")
    state = normalize_keys(state)
    dropped_head = sorted(k for k in state if scope == "backbone" and _is_head(k))
    candidate = {k: v for k, v in state.items() if k not in dropped_head}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(candidate))
    unexpected = sorted(set(candidate) - set(expected))
    fatal_missing = [k for k in missing if scope == "full" or not _is_head(k)]
    mismatches = [
        f"{k}: checkpoint {tuple(v.shape)} != model {tuple(expected[k].shape)}"
        for k, v in candidate.items() if k in expected and v.shape != expected[k].shape
    ]
    # Validate before mutating the model, so a failed load cannot partially alter it.
    if fatal_missing or mismatches:
        raise CheckpointError(
            "Checkpoint is incompatible; refusing partial/random backbone loading. "
            f"Missing: {fatal_missing[:20]}; shape mismatches: {mismatches[:10]}. "
            "Check the selected architecture, checkpoint and timm version."
        )
    if scope == "full" and unexpected:
        raise CheckpointError(f"Full task checkpoint contains unexpected keys: {unexpected[:20]}")
    model.load_state_dict({k: v for k, v in candidate.items() if k in expected}, strict=False)
    return {
        "missing_keys": missing,
        "ignored_pretraining_keys": unexpected,
        "discarded_head_keys": dropped_head,
    }
