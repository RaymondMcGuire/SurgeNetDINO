"""Label-free per-video standardisation and cross-fitted frame emissions.

The temporal network must see frame-head outputs with HELD-OUT statistics. Heads reach ~96% on
videos they trained on and ~76% on new ones, so every video receives emissions from a head that
never saw it. Budgets are fixed: no validation data can select anything here.
"""

from dataclasses import asdict, dataclass

import torch

from ..head import make_head, make_optimizer
from ..training import LABELS, state_digest


def per_video_zscore(x, records, min_frames=100, floor=1e-6):
    """z = (f - mean_v) / std_v from the frames of the SAME video. No labels enter this function."""
    if x.ndim != 2 or len(x) != len(records):
        raise ValueError("Features/records mismatch")
    rows = {}
    for index, record in enumerate(records):
        rows.setdefault(record["video_id"], []).append(index)
    result = torch.empty_like(x)
    for video, indices in rows.items():
        if len(indices) < min_frames:
            raise ValueError(f"Too few frames for per-video statistics: {video}")
        ids = torch.tensor(indices)
        values = x[ids].double()
        result[ids] = ((values - values.mean(0)) / values.std(0).clamp(min=floor)).to(x.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite standardised features")
    return result


def fit_head_fixed(x, y, head_options, seed, epochs, guard):
    """Same head, optimizer, seeding and batch order as head_training.fit, for a FIXED epoch count.

    There is deliberately no validation argument: cross-fitted emissions must not be tuned on
    the videos they are produced for.
    """
    if epochs < 1 or len(x) != len(y) or not len(x):
        raise ValueError("Invalid fixed-budget head inputs")
    head = make_head(x, len(LABELS), guard.device, seed, head_options)
    optimizer = make_optimizer(head)
    generator = torch.Generator().manual_seed(seed)
    device_x, device_y = x.to(guard.device), y.to(guard.device)
    head.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        with guard.batch(len(x)):
            for indices in order.split(head_options.head_batch_size):
                indices = indices.to(guard.device)
                optimizer.zero_grad(set_to_none=True)
                loss = head.training_loss(head(device_x[indices]), device_y[indices])
                loss.backward()
                optimizer.step()
        if not torch.isfinite(loss).item() or any(not torch.isfinite(p).all().item() for p in head.parameters()):
            raise RuntimeError("Non-finite head loss/parameter")
    return head.eval()


def head_logits(head, x, guard, batch=8192):
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            with guard.batch(len(x[start:start + batch])):
                parts.append(head(x[start:start + batch].to(guard.device)).float().cpu())
    result = torch.cat(parts)
    if not torch.isfinite(result).all():
        raise RuntimeError("Non-finite head logits")
    return result


@dataclass(frozen=True)
class LedgerEntry:
    video_id: str
    role: str  # crossfit | apply
    trained_on: tuple
    seed: int
    epochs: int
    head_sha256: str


def check_ledger(ledger, fit_videos, apply_videos):
    """Every emission comes from exactly one head that never saw the video it scores."""
    seen = [entry.video_id for entry in ledger]
    if sorted(seen) != sorted(list(fit_videos) + list(apply_videos)) or len(set(seen)) != len(seen):
        raise ValueError("Each video needs exactly one emission source")
    for entry in ledger:
        if entry.video_id in entry.trained_on or set(entry.trained_on) - set(fit_videos):
            raise ValueError(f"Leaking emission head for video {entry.video_id}")
        if (entry.role == "crossfit") != (entry.video_id in fit_videos):
            raise ValueError("Ledger role mismatch")


def stacked_emissions(x, y, records, fit_videos, apply_videos, head_options, epochs, seed, guard):
    """Frame logits [N,K]; NaN outside the fit/apply videos.

    fit videos: leave-one-video-out logits, from a head trained on the OTHER fit videos.
    apply videos: logits from one head trained on all fit videos.
    """
    fit_videos, apply_videos = sorted(fit_videos), sorted(apply_videos)
    if set(fit_videos) & set(apply_videos) or len(fit_videos) < 2:
        raise ValueError("fit/apply videos must be disjoint, with at least two fit videos")
    rows = {}
    for index, record in enumerate(records):
        rows.setdefault(record["video_id"], []).append(index)
    rows = {v: torch.tensor(i) for v, i in rows.items()}
    logits = torch.full((len(records), len(LABELS)), float("nan"))
    ledger = []

    def train(videos):
        ids = torch.cat([rows[v] for v in videos])
        if len(torch.unique(y[ids])) != len(LABELS):
            raise ValueError(f"Emission training videos do not cover all classes: {videos}")
        return fit_head_fixed(x[ids], y[ids], head_options, seed, epochs, guard)

    for held in fit_videos:
        training = [v for v in fit_videos if v != held]
        head = train(training)
        logits[rows[held]] = head_logits(head, x[rows[held]], guard)
        ledger.append(LedgerEntry(held, "crossfit", tuple(training), seed, epochs, state_digest(head)))
    if apply_videos:
        head = train(fit_videos)
        digest = state_digest(head)
        for video in apply_videos:
            logits[rows[video]] = head_logits(head, x[rows[video]], guard)
            ledger.append(LedgerEntry(video, "apply", tuple(fit_videos), seed, epochs, digest))
    check_ledger(ledger, fit_videos, apply_videos)
    return logits, [asdict(entry) for entry in ledger]
