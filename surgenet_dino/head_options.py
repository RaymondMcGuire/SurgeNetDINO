"""Current regularized head settings: CLI > [training] TOML > defaults."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class HeadOptions:
    epochs: int = 20
    head_batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    dropout: float = 0.10
    evals_per_epoch: int = 4
    patience: int = 8  # validation checks, not epochs; 0 disables early stopping
    min_delta: float = 1e-4
    min_epochs: float = 1.0

    def __post_init__(self):
        for key in ("epochs", "head_batch_size", "evals_per_epoch"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("patience must be an integer >= 0")
        for key in ("learning_rate", "weight_decay", "label_smoothing", "dropout",
                    "min_delta", "min_epochs"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate == 0 or self.learning_rate * self.weight_decay >= 1:
            raise ValueError("learning_rate must be positive and learning_rate * weight_decay < 1")
        if not 0 <= self.dropout < 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError("dropout and label_smoothing must be in [0, 1)")

def resolve_options(args) -> HeadOptions:
    options = asdict(HeadOptions())
    path = getattr(args, "head_config", None)
    if path:
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        if set(data) != {"training"} or not isinstance(data["training"], dict):
            raise ValueError("Head configuration must contain exactly one [training] table")
        unknown = set(data["training"]) - set(options)
        if unknown:
            raise ValueError(f"Unknown training options: {sorted(unknown)}")
        options.update(data["training"])
    for field in fields(HeadOptions):
        value = getattr(args, field.name, None)
        if value is not None:
            options[field.name] = value
    return HeadOptions(**options)


def add_head_arguments(parser):
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--head-config", type=Path, help="Optional TOML with one [training] table")
    for key in ("epochs", "head_batch_size", "evals_per_epoch", "patience"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("learning_rate", "weight_decay", "label_smoothing", "dropout",
                "min_delta", "min_epochs"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
