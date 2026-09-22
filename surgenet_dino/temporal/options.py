"""Strict, predeclared options of the temporal phase network: exactly one [temporal] table."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TemporalOptions:
    long_gap_seconds: int = 300  # one >= 880 s out-of-body gap per labeled video; next largest is 104 s
    head_epochs: int = 5  # fixed budget of the cross-fitted frame heads; held-out data cannot select it
    inner_folds: int = 3  # inner grouped folds that choose ONE common epoch for the temporal network
    channels: int = 64
    refine_channels: int = 48
    layers: int = 10  # dilations 1..2**(layers-1) seconds
    dropout: float = 0.3
    pca: int = 32  # unsupervised projection of the standardised tokens, fitted on training videos only
    attention_layers: int = 1  # self-attention over the whole video at 1/attention_factor resolution
    attention_heads: int = 4
    attention_factor: int = 16
    epochs: int = 60
    epoch_checks: int = 6  # inner-CV checkpoints evaluated to pick the common epoch
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    smoothing_weight: float = 0.15
    label_smoothing: float = 0.05
    seeds: int = 5
    augment: bool = True
    noise: float = 0.5

    def __post_init__(self):
        for key in ("long_gap_seconds", "head_epochs", "channels", "refine_channels", "layers", "pca",
                    "attention_heads", "attention_factor", "epochs", "epoch_checks", "seeds"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.attention_layers) is not int or self.attention_layers < 0:
            raise ValueError("attention_layers must be a nonnegative integer")
        if type(self.inner_folds) is not int or self.inner_folds < 2:
            raise ValueError("inner_folds must be an integer >= 2")
        for key in ("dropout", "learning_rate", "weight_decay", "smoothing_weight", "label_smoothing", "noise"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate <= 0 or not 0 <= self.dropout < 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError("learning_rate must be positive; dropout and label_smoothing must be in [0, 1)")
        if type(self.augment) is not bool:
            raise ValueError("augment must be a TOML boolean")
        if self.channels % self.attention_heads or self.epoch_checks > self.epochs:
            raise ValueError("channels must be divisible by attention_heads, and epoch_checks <= epochs")


def load_options(path: str | Path | None, **overrides) -> TemporalOptions:
    values = asdict(TemporalOptions())
    if path is not None:
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        if set(data) != {"temporal"} or not isinstance(data["temporal"], dict):
            raise ValueError("Temporal config must contain exactly one [temporal] table")
        unknown = set(data["temporal"]) - set(values)
        if unknown:
            raise ValueError(f"Unknown temporal options: {sorted(unknown)}")
        values.update(data["temporal"])
    for key, value in overrides.items():
        if key not in values:
            raise ValueError(f"Unknown override: {key}")
        if value is not None:
            values[key] = value
    return TemporalOptions(**values)
