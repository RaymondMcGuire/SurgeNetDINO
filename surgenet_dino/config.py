"""Typed, validated TOML configuration. Relative paths use the project working directory."""

import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .registry import get_spec


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "dinov2_vitb"
    weights: str = "surgenet"  # surgenet | local | none
    num_classes: int = 0      # 0 returns features; positive creates a task head
    device: str = "cuda"      # no silent CPU fallback when CUDA is requested
    checkpoint: str | None = None
    checkpoint_key: str | None = None
    weights_scope: str = "backbone"  # backbone resets the task head; full restores it
    cache_dir: str = ".cache/huggingface"
    revision: str = "main"    # pin the resolved HF commit for a formal experiment
    offline: bool = False
    sha256: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        for key in ("name", "weights", "device", "weights_scope", "cache_dir", "revision"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a non-empty string")
        get_spec(self.name)
        if type(self.num_classes) is not int or self.num_classes < 0:
            raise ValueError("num_classes must be an integer >= 0")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.offline) is not bool:
            raise ValueError("offline must be a boolean")
        if self.weights not in {"surgenet", "local", "none"}:
            raise ValueError("weights must be surgenet, local, or none")
        if self.weights_scope not in {"backbone", "full"}:
            raise ValueError("weights_scope must be backbone or full")
        if re.fullmatch(r"cpu|auto|cuda(?::[0-9]+)?", self.device) is None:
            raise ValueError("device must be cpu, auto, cuda, or cuda:N")
        for key in ("checkpoint", "checkpoint_key", "sha256"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{key} must be a non-empty string when supplied")
        if self.weights == "local" and not self.checkpoint:
            raise ValueError("weights=local requires checkpoint")
        if self.weights != "local" and self.checkpoint is not None:
            raise ValueError("checkpoint may only be used with weights=local")
        if self.weights_scope == "full" and self.weights != "local":
            raise ValueError("Full task restoration requires weights=local and a task checkpoint")
        if self.weights == "none" and (self.sha256 or self.checkpoint_key):
            raise ValueError("Random initialization has no checkpoint or checksum")
        if self.sha256 is not None and re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256) is None:
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")


def load_config(path: str | Path) -> ModelConfig:
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"model"} or not isinstance(data["model"], dict):
        raise ValueError("Configuration must contain exactly one [model] table")
    allowed = {field.name for field in fields(ModelConfig)}
    unknown = set(data["model"]) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return ModelConfig(**data["model"])
