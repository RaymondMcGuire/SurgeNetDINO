"""Small frozen-feature classifier, versioned inference, and early stopping."""

import math
from dataclasses import asdict, fields

import torch
from torch import nn
from torch.nn import functional as F

from .head_options import HeadOptions


class RegularizedLinearHead(nn.Module):
    """Dropout followed by a linear classifier; no hidden layers."""

    def __init__(self, dimension: int, classes: int, options: HeadOptions):
        super().__init__()
        if dimension < 1 or classes < 2:
            raise ValueError("Invalid head dimensions")
        self.options = options
        self.dimension, self.classes = dimension, classes
        self.dropout = nn.Dropout(options.dropout)
        self.linear = nn.Linear(dimension, classes)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"Expected [N,{self.dimension}] features, got {tuple(x.shape)}")
        return self.linear(self.dropout(x))

    def training_loss(self, logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=self.options.label_smoothing)

    def specification(self):
        return {"schema_version": 3, "dimension": self.dimension, "classes": self.classes,
                "options": asdict(self.options)}


def make_head(x: torch.Tensor, classes: int, device, seed: int, options: HeadOptions):
    torch.manual_seed(seed)
    head = RegularizedLinearHead(x.shape[1], classes, options)
    return head.to(device)


def head_from_checkpoint(checkpoint: dict, dimension: int, classes: int, device):
    """Restore original Linear, historical unnormalized v2 heads, and current heads."""
    spec = checkpoint.get("head_spec")
    state = dict(checkpoint["state_dict"])
    if spec is None:
        head = nn.Linear(dimension, classes)
    else:
        version = spec.get("schema_version")
        if (version not in (2, 3) or spec.get("dimension") != dimension
                or spec.get("classes") != classes):
            raise ValueError("Head architecture/dimension mismatch")
        saved = spec["options"]
        if version == 2:
            if saved.get("feature_norm", "none") != "none":
                raise ValueError("Unsupported historical feature normalization; use the original implementation")
            for key, expected in (("feature_mean", 0), ("feature_scale", 1)):
                value = state.pop(key)
                if value.shape != (dimension,) or not torch.all(value == expected):
                    raise ValueError("Invalid historical normalization buffers")
            saved = {f.name: saved[f.name] for f in fields(HeadOptions) if f.name in saved}
        head = RegularizedLinearHead(dimension, classes, HeadOptions(**saved))
    head.load_state_dict(state, strict=True)
    if any(not torch.isfinite(t).all() for t in head.state_dict().values()):
        raise ValueError("Non-finite checkpoint parameters")
    return head.to(device).eval()


def make_optimizer(head: RegularizedLinearHead):
    params = [{"params": [head.linear.weight]},
              {"params": [head.linear.bias], "weight_decay": 0.0}]
    return torch.optim.AdamW(params, lr=head.options.learning_rate, weight_decay=head.options.weight_decay)


def evaluation_steps(total_steps: int, checks: int) -> set[int]:
    if total_steps < 1 or checks < 1:
        raise ValueError("Evaluation schedule must be positive")
    checks = min(checks, total_steps)
    return {math.ceil(total_steps * k / checks) for k in range(1, checks + 1)}


class EarlyStopping:
    """Exact best checkpoint selection is separate from min_delta/patience."""

    def __init__(self, options: HeadOptions):
        self.options = options
        self.reference = None
        self.bad_checks = 0

    def update(self, score: float, epoch: float) -> bool:
        if not math.isfinite(score):
            raise ValueError("Non-finite selection score")
        if self.reference is None or score < self.reference - self.options.min_delta:
            self.reference = score
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return bool(self.options.patience and self.bad_checks >= self.options.patience
                    and epoch >= self.options.min_epochs)
