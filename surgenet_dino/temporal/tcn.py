"""Offline temporal phase network, trained ONLY on cross-fitted (out-of-video) frame emissions.

A second stage that sees in-sample emissions learns to trust them (heads reach ~96% on their own
training videos, ~76% on new ones) and then fails on new videos. Therefore every input sequence,
in training and in epoch selection, carries emissions from a head that never saw that video.
Nine training sequences are few: the network is small, has no BatchNorm, and is regularised by
sequence-level augmentation and a fixed epoch budget chosen by inner folds. The network itself uses
no transition matrix and no phase order; its log-probabilities are decoded afterwards under the
order rules of the training videos (constraints.py).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..training import LABELS
from .folds import make_inner_groups
from .timeline import label_track

CLASSES = len(LABELS)
LOG_FLOOR, LOG_SCALE = -10.0, 5.0
FLAGS = 6  # is_gap, is_long_gap, post_stage, progress, elapsed hours, hours since the long gap ended


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each second: independent of sequence length, padding and batch."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = ChannelNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mix(F.gelu(self.conv(self.norm(x)))))


class RelativeBias(nn.Module):
    """Signed, log-bucketed relative-position bias [1,H,L,L]. No absolute position is encoded."""

    def __init__(self, heads, buckets=32, max_distance=512):
        super().__init__()
        self.buckets, self.max_distance = buckets, max_distance
        self.table = nn.Embedding(buckets, heads)
        nn.init.zeros_(self.table.weight)

    def forward(self, length, device):
        position = torch.arange(length, device=device)
        relative = position[None, :] - position[:, None]
        half = self.buckets // 2
        exact = half // 2
        distance = relative.abs()
        scaled = torch.log(distance.float().clamp(min=1) / exact) / math.log(self.max_distance / exact)
        large = (exact + scaled * (half - exact)).long().clamp(max=half - 1)
        bucket = (relative > 0).long() * half + torch.where(distance < exact, distance, large)
        return self.table(bucket).permute(2, 0, 1)[None]


class SelfAttention(nn.Module):
    """Pre-norm multi-head self-attention plus a small MLP, on [B,C,L]."""

    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm_attention, self.norm_mlp = ChannelNorm(channels), ChannelNorm(channels)
        self.qkv, self.project = nn.Linear(channels, 3 * channels), nn.Linear(channels, channels)
        self.bias = RelativeBias(heads)
        self.mlp = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * channels, channels))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, channels, length = x.shape
        tokens = self.norm_attention(x).transpose(1, 2)
        q, k, v = self.qkv(tokens).reshape(batch, length, 3, self.heads, channels // self.heads).unbind(2)
        mixed = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               attn_mask=self.bias(length, x.device).to(q.dtype),
                                               dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.project(mixed.transpose(1, 2).reshape(batch, length, channels))).transpose(1, 2)
        return x + self.drop(self.mlp(self.norm_mlp(x).transpose(1, 2))).transpose(1, 2)


class GlobalContext(nn.Module):
    """Self-attention over the WHOLE video at 1/factor resolution, added back as a residual.

    Every moment can consult any other moment of the same operation by content. The weights are
    similarities inside one video, which a per-video appearance shift leaves almost unchanged.
    """

    def __init__(self, channels, options):
        super().__init__()
        self.factor = options.attention_factor
        self.layers = nn.Sequential(*(SelfAttention(channels, options.attention_heads, options.dropout / 2)
                                      for _ in range(options.attention_layers)))

    def forward(self, x):
        coarse = self.layers(F.avg_pool1d(x, self.factor, ceil_mode=True))
        return F.interpolate(coarse, size=x.shape[-1], mode="linear", align_corners=False)


class Stage(nn.Module):
    def __init__(self, inputs, channels, layers, dropout, context=None):
        super().__init__()
        self.enter = nn.Conv1d(inputs, channels, 1)
        self.blocks = nn.ModuleList(DilatedBlock(channels, 2 ** i, dropout) for i in range(layers))
        self.leave = nn.Sequential(ChannelNorm(channels), nn.Conv1d(channels, CLASSES, 1))
        if context is not None:
            self.context = context

    def forward(self, x):
        x = self.enter(x)
        for block in self.blocks:
            x = block(x)
        if hasattr(self, "context"):
            x = x + self.context(x)
        return self.leave(x)


class PhaseTCN(nn.Module):
    """A prediction stage with global context, then a refinement stage over its probabilities."""

    def __init__(self, inputs, options):
        super().__init__()
        self.input_dropout = nn.Dropout(options.dropout / 2)
        context = GlobalContext(options.channels, options) if options.attention_layers else None
        self.predict = Stage(inputs, options.channels, options.layers, options.dropout, context)
        self.refine = Stage(CLASSES, options.refine_channels, options.layers, options.dropout)

    def forward(self, x):
        first = self.predict(self.input_dropout(x))
        return [first, self.refine(first.softmax(1))]


def fit_pca(x, components):
    """Unsupervised projection from TRAINING rows only (float64 eigh, sign-fixed, unit variance)."""
    values = x.double()
    mean = values.mean(0)
    centred = values - mean
    eigenvalues, vectors = torch.linalg.eigh(centred.T @ centred / (len(values) - 1))
    order = eigenvalues.argsort(descending=True)[:components]
    basis = vectors[:, order]
    basis = basis * torch.where(basis.abs().max(0).values == basis.max(0).values, 1.0, -1.0)
    return {"mean": mean, "basis": basis, "scale": eigenvalues[order].clamp(min=1e-12).sqrt()}


def build_inputs(timeline, logits, x, pca, extra=None):
    """[C,T] per-second inputs. Uses emissions, tokens and the observable timeline; never labels.

    `extra` [N,M]: optional label-free per-frame features (motion), already standardised per video.
    """
    total = len(timeline)
    present = timeline.present
    rows = timeline.rows[present]
    emission = torch.zeros((total, CLASSES), dtype=torch.float64)
    emission[present] = logits[rows].double().log_softmax(1).clamp(min=LOG_FLOOR) / LOG_SCALE + 1.0
    projected = torch.zeros((total, pca["basis"].shape[1]), dtype=torch.float64)
    projected[present] = ((x[rows].double() - pca["mean"]) @ pca["basis"] / pca["scale"]).clamp(-6, 6)
    flags = torch.zeros((total, FLAGS), dtype=torch.float64)
    seconds = torch.arange(total, dtype=torch.float64)
    flags[:, 0] = (~present).double()
    if timeline.long_gap is not None:
        start, stop = timeline.long_gap
        flags[start:stop, 1] = 1.0
        flags[stop:, 2] = 1.0
        flags[stop:, 5] = (seconds[stop:] - stop) / 3600.0
    flags[:, 3] = seconds / max(1, total - 1)
    flags[:, 4] = seconds / 3600.0
    parts = [emission, projected, flags]
    if extra is not None:
        motion = torch.zeros((total, extra.shape[1]), dtype=torch.float64)
        motion[present] = extra[rows].double().clamp(-6, 6)
        parts.append(motion)
    return torch.cat(parts, 1).T.float().contiguous()


def confusion_rows(logits, targets, rows):
    """Row-normalised off-diagonal confusion of CROSS-FITTED emissions: how heads err on unseen videos."""
    predicted = logits[rows].argmax(1)
    counts = torch.bincount(targets[rows] * CLASSES + predicted, minlength=CLASSES ** 2)
    counts = counts.reshape(CLASSES, CLASSES).double() + 1e-3
    counts[torch.arange(CLASSES), torch.arange(CLASSES)] = 0.0
    return counts / counts.sum(1, keepdim=True)


def augment(inputs, labels, options, generator, confusion):
    """Sequence-level regularisation. Labels follow exactly the same time indices as the inputs."""
    total = inputs.shape[1]
    uniform = lambda: torch.rand((), generator=generator).item()  # noqa: E731
    integer = lambda low, high: int(torch.randint(low, high + 1, (), generator=generator))  # noqa: E731
    # Piecewise speed warp, 0.7x to 1.4x per quarter.
    index, edges = [], torch.linspace(0, total, 5).long().tolist()
    for start, stop in zip(edges, edges[1:]):
        length = max(2, round((stop - start) * (0.7 + 0.7 * uniform())))
        index.append(torch.linspace(start, stop - 1, length).round().long())
    index = torch.cat(index)
    if uniform() < 0.5:  # crop to a window of at least half the sequence
        length = integer(len(index) // 2, len(index))
        start = integer(0, len(index) - length)
        window = index[start:start + length]
        if (labels[window] >= 0).sum() >= 0.25 * length:  # never a window made of the out-of-body gap
            index = window
    inputs, labels = inputs[:, index].clone(), labels[index].clone()
    total, features = inputs.shape[1], inputs.shape[0] - FLAGS
    inputs[:CLASSES] += torch.randn((CLASSES, total), generator=generator) * options.noise / LOG_SCALE
    # Systematic segment-level error: a span leans towards a phase that unseen-video heads confuse it with.
    for _ in range(integer(0, 3)):
        length = integer(10, 120)
        start = integer(0, max(0, total - length))
        known = labels[start:start + length]
        known = known[known >= 0]
        if len(known):
            true = int(known.mode().values)
            wrong = int(torch.multinomial(confusion[true], 1, generator=generator))
            inputs[wrong, start:start + length] += 3.0 / LOG_SCALE
            inputs[true, start:start + length] -= 3.0 / LOG_SCALE
    for _ in range(integer(1, 2)):  # masked spans look exactly like seconds without a frame
        length = integer(18, 300)
        start = integer(0, max(0, total - length))
        inputs[:features, start:start + length] = 0.0
        inputs[features, start:start + length] = 1.0
    if uniform() < 0.3:  # the network must not depend on the projected tokens alone
        inputs[CLASSES:features] = 0.0
    inputs[:CLASSES] *= (inputs[features] == 0).float()
    return inputs, labels


def sequence_loss(outputs, labels, options):
    """Masked cross-entropy plus the truncated log-probability smoothness term, on every stage."""
    loss = 0.0
    for logits in outputs:
        loss = loss + F.cross_entropy(logits, labels[None], ignore_index=-1, label_smoothing=options.label_smoothing)
        log_prob = logits.log_softmax(1)
        step = (log_prob[:, :, 1:] - log_prob.detach()[:, :, :-1]).square().clamp(max=16.0)
        loss = loss + options.smoothing_weight * step.mean()
    return loss


def make_sequences(timelines, videos, logits, x, targets, pca, extra=None):
    return {v: (build_inputs(timelines[v], logits, x, pca, extra), label_track(timelines[v], targets))
            for v in videos}


@torch.no_grad()
def predict_tcn(models, sequences, timelines, guard, total_rows):
    """Seed-averaged log-probabilities [N,K] for the catalogued frames of the given videos (NaN elsewhere)."""
    result = torch.full((total_rows, CLASSES), float("nan"), dtype=torch.float64)
    for video, (inputs, _) in sequences.items():
        with guard.batch(1):
            batch = inputs[None].to(guard.device)
            mean = torch.stack([m.eval()(batch)[-1][0].double().log_softmax(0) for m in models]).mean(0)
        timeline = timelines[video]
        result[timeline.rows[timeline.present]] = mean.T.cpu()[timeline.present]
    return result


def fit_tcn(sequences, options, seed, guard, epochs, confusion, checkpoints=(), validation=None,
            timelines=None, total_rows=0, on_epoch=None):
    """Train one network for a FIXED number of epochs.

    checkpoints/validation are for INNER folds only, to read held-out log-probabilities at the
    predeclared epochs. A final refit passes neither: it cannot look at any held-out video.
    on_epoch(epoch, model, mean_training_objective) is a read-only diagnostic hook (learning curves);
    nothing it observes feeds back into training or selection.
    """
    if bool(checkpoints) != (validation is not None):
        raise ValueError("Epoch checkpoints require inner validation sequences, and vice versa")
    if validation is not None and set(validation) & set(sequences):
        raise ValueError("Inner validation videos overlap the training videos")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = PhaseTCN(next(iter(sequences.values()))[0].shape[0], options).to(guard.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    videos, snapshots = sorted(sequences), {}
    for epoch in range(1, epochs + 1):
        model.train()
        objective = 0.0
        for position in torch.randperm(len(videos), generator=generator).tolist():
            inputs, labels = sequences[videos[position]]
            if options.augment:
                inputs, labels = augment(inputs, labels, options, generator, confusion)
            with guard.batch(1):
                optimizer.zero_grad(set_to_none=True)
                loss = sequence_loss(model(inputs[None].to(guard.device)), labels.to(guard.device), options)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite temporal-network loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                objective += loss.item() / len(videos)
        if on_epoch is not None:
            on_epoch(epoch, model, objective)
        if epoch in checkpoints:
            snapshots[epoch] = predict_tcn([model], validation, timelines, guard, total_rows)
    return model, snapshots


def checkpoint_epochs(options):
    return tuple(sorted({math.ceil(options.epochs * k / options.epoch_checks)
                         for k in range(1, options.epoch_checks + 1)}))


def video_rows(records):
    rows = {}
    for index, record in enumerate(records):
        rows.setdefault(record["video_id"], []).append(index)
    return {v: torch.tensor(i) for v, i in rows.items()}


def prepare(x, targets, records, timelines, training, apply, head_options, options, seed, guard, rows, logits=None,
            extra=None):
    """Sequences for `training` and `apply`. Everything fitted here (heads, projection) uses `training` only."""
    if logits is None:
        from .emissions import stacked_emissions
        logits, _ = stacked_emissions(x, targets, records, training, apply, head_options, options.head_epochs,
                                      seed, guard)
    training_rows = torch.cat([rows[v] for v in training])
    pca = fit_pca(x[training_rows], options.pca)
    confusion = confusion_rows(logits, targets, training_rows)
    return (make_sequences(timelines, training, logits, x, targets, pca, extra),
            make_sequences(timelines, apply, logits, x, targets, pca, extra), confusion)


def inner_cv(x, targets, records, timelines, fit_videos, head_options, options, seed, guard, inner_seed=0,
             extra=None):
    """Out-of-video log-probabilities of every fit video, per seed and per predeclared epoch.

    Inner folds are re-nested from scratch: their own cross-fitted heads, projection and networks.
    """
    fit_videos = sorted(fit_videos)
    fit_records = [r for r in records if r["video_id"] in set(fit_videos)]
    groups = make_inner_groups(fit_records, options.inner_folds, seed + inner_seed)
    rows, epochs = video_rows(records), checkpoint_epochs(options)
    per_seed = {e: torch.full((options.seeds, len(records), CLASSES), float("nan"), dtype=torch.float64)
                for e in epochs}
    for held in groups:
        training = [v for v in fit_videos if v not in held]
        sequences, validation, confusion = prepare(x, targets, records, timelines, training, sorted(held),
                                                   head_options, options, seed, guard, rows, extra=extra)
        held_rows = torch.cat([rows[v] for v in held])
        for s in range(options.seeds):
            _, snapshots = fit_tcn(sequences, options, seed + 1000 * (s + 1), guard, options.epochs,
                                   confusion, epochs, validation, timelines, len(records))
            for e in epochs:
                per_seed[e][s, held_rows] = snapshots[e][held_rows]
    return {"groups": groups, "epochs": epochs, "per_seed": per_seed, "rows": rows}


def common_epoch(inner, targets, fit_videos):
    """ONE epoch for every inner fold, from the seed-averaged out-of-video accuracy of the fit videos."""
    fit_rows = torch.cat([inner["rows"][v] for v in sorted(fit_videos)])
    accuracy = {e: (inner["per_seed"][e].mean(0)[fit_rows].argmax(1) == targets[fit_rows]).double().mean().item()
                for e in inner["epochs"]}
    return max(inner["epochs"], key=lambda e: (accuracy[e], -e)), accuracy


def refit(x, targets, records, timelines, fit_videos, apply_videos, logits, head_options, options, seed, guard,
          epochs, rows, extra=None):
    """Final networks on all fit videos for a FIXED epoch count; no held-out data can influence them."""
    sequences, held_out, confusion = prepare(x, targets, records, timelines, fit_videos, apply_videos,
                                             head_options, options, seed, guard, rows, logits, extra)
    models = [fit_tcn(sequences, options, seed + 1000 * (s + 1), guard, epochs, confusion)[0]
              for s in range(options.seeds)]
    ensemble = predict_tcn(models, held_out, timelines, guard, len(records))
    per_seed = torch.stack([predict_tcn([m], held_out, timelines, guard, len(records)) for m in models])
    info = {"parameters": sum(p.numel() for p in models[0].parameters()), "seeds": options.seeds,
            "inputs": sequences[fit_videos[0]][0].shape[0]}
    return models, ensemble, per_seed, info
