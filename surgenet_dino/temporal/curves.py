"""Learning-curve DIAGNOSTICS: training versus held-out loss and accuracy, per epoch.

This module selects nothing. Epoch budgets were fixed beforehand by inner folds. The held-out
curves exist only to show whether the two trained parts, the frame head and the temporal
network, overfit.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from ..experiments import write_csv
from ..head import make_head, make_optimizer
from ..training import LABELS, write_json
from .emissions import head_logits, per_video_zscore, stacked_emissions
from .tcn import confusion_rows, fit_pca, fit_tcn, make_sequences, predict_tcn
from .timeline import build_timelines


def measure(logits, targets):
    """(ordinary unsmoothed cross-entropy, accuracy); logits may already be log-probabilities."""
    return (F.cross_entropy(logits.float(), targets).item(),
            (logits.argmax(1) == targets).float().mean().item())


def head_curve(x, y, train_rows, held_rows, head_options, seed, guard, epochs):
    """Same recipe as fit_head_fixed, evaluated after every epoch with dropout disabled."""
    head = make_head(x[train_rows], len(LABELS), guard.device, seed, head_options)
    optimizer = make_optimizer(head)
    generator = torch.Generator().manual_seed(seed)
    features, targets = x[train_rows].to(guard.device), y[train_rows].to(guard.device)
    rows = []
    for epoch in range(1, epochs + 1):
        head.train()
        order, objective = torch.randperm(len(features), generator=generator), 0.0
        with guard.batch(len(features)):
            for indices in order.split(head_options.head_batch_size):
                indices = indices.to(guard.device)
                optimizer.zero_grad(set_to_none=True)
                loss = head.training_loss(head(features[indices]), targets[indices])
                loss.backward()
                optimizer.step()
                objective += loss.item() * len(indices) / len(features)
        head.eval()
        train_loss, train_accuracy = measure(head_logits(head, x[train_rows], guard), y[train_rows])
        held_loss, held_accuracy = measure(head_logits(head, x[held_rows], guard), y[held_rows])
        rows.append({"stage": "head", "epoch": epoch, "train_objective": objective, "train_loss": train_loss,
                     "train_accuracy": train_accuracy, "heldout_loss": held_loss,
                     "heldout_accuracy": held_accuracy})
    return rows


def network_curve(x, y, records, timelines, fit_videos, held, head_options, options, seed, guard, every, seeds):
    """Temporal network on cross-fitted inputs; evaluated WITHOUT augmentation and with dropout off."""
    logits, _ = stacked_emissions(x, y, records, fit_videos, held, head_options, options.head_epochs, seed, guard)
    index = {}
    for i, record in enumerate(records):
        index.setdefault(record["video_id"], []).append(i)
    fit_rows = torch.tensor([i for v in fit_videos for i in index[v]])
    held_rows = torch.tensor([i for v in held for i in index[v]])
    pca = fit_pca(x[fit_rows], options.pca)
    confusion = confusion_rows(logits, y, fit_rows)
    training = make_sequences(timelines, fit_videos, logits, x, y, pca)
    validation = make_sequences(timelines, held, logits, x, y, pca)
    rows = []
    for s in range(seeds):
        def record_epoch(epoch, model, objective, s=s):
            if epoch % every and epoch != options.epochs:
                return
            fitted = predict_tcn([model], training, timelines, guard, len(records))
            unseen = predict_tcn([model], validation, timelines, guard, len(records))
            train_loss, train_accuracy = measure(fitted[fit_rows], y[fit_rows])
            held_loss, held_accuracy = measure(unseen[held_rows], y[held_rows])
            rows.append({"stage": "network", "seed": s, "epoch": epoch, "train_objective": objective,
                         "train_loss": train_loss, "train_accuracy": train_accuracy,
                         "heldout_loss": held_loss, "heldout_accuracy": held_accuracy})
        fit_tcn(training, options, seed + 1000 * (s + 1), guard, options.epochs, confusion, on_epoch=record_epoch)
    frame = measure(logits[held_rows], y[held_rows])
    return rows, {"heldout_frame_loss": frame[0], "heldout_frame_accuracy": frame[1]}


def run_curves(features, tokens_directory, folds_json, options, head_options, config, resources, output, *,
               head_epochs=12, every=2, seeds=3, reference=None, network_epochs=None):
    from dataclasses import replace

    from ..gpu import CPUControl, GPUControl
    from ..runtime import resolve_device
    from .folds import load_outer_folds
    from .workflows import load_inputs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if network_epochs is not None:  # look PAST the configured budget; affects this diagnostic only
        options = replace(options, epochs=network_epochs)
    _, records, y, tokens, provenance = load_inputs(features, tokens_directory)
    groups = load_outer_folds(records, folds_json)
    x = per_video_zscore(tokens, records)
    timelines = build_timelines(records, options.long_gap_seconds)
    chosen = None
    if reference is not None:  # only to MARK the epochs that inner folds had already chosen
        payload = json.loads(Path(reference).read_text(encoding="utf-8"))
        chosen = {s["fold"]: s["common_epoch"] for s in payload["selections"]}
    device = resolve_device(config.device)
    control = GPUControl if device.type == "cuda" else CPUControl
    rows, folds = [], []
    with control(device, resources, output / "gpu.csv") as guard:
        for k, held in enumerate(groups, 1):
            fit_videos = sorted(v for v in timelines if v not in held)
            train_rows = torch.tensor([i for i, r in enumerate(records) if r["video_id"] not in held])
            held_rows = torch.tensor([i for i, r in enumerate(records) if r["video_id"] in held])
            print(f"Fold {k}/{len(groups)}: held-out {sorted(held)}", flush=True)
            part = head_curve(x, y, train_rows, held_rows, head_options, config.seed, guard, head_epochs)
            network, frame = network_curve(x, y, records, timelines, fit_videos, sorted(held), head_options,
                                           options, config.seed, guard, every, seeds)
            rows += [{"fold": k, "seed": 0, **row} for row in part + network]
            folds.append({"fold": k, "validation_videos": sorted(held), "head_fixed_epochs": options.head_epochs,
                          **frame, "network_common_epoch": None if chosen is None else chosen.get(k)})
    columns = ["fold", "stage", "seed", "epoch", "train_objective", "train_loss", "train_accuracy",
               "heldout_loss", "heldout_accuracy"]
    write_csv(output / "curves.csv", [{c: row[c] for c in columns} for row in rows])
    result = {"status": "passed", "folds": folds, "options": asdict(options), "head_options": asdict(head_options),
              "head_epochs_plotted": head_epochs, "network_seeds": seeds, "evaluated_every": every,
              "provenance": provenance, "validation_videos": groups, "diagnostic_only": True,
              "note": "Held-out curves are shown for inspection. Budgets were fixed beforehand by inner folds; "
                      "nothing in this run selects an epoch, a setting or a model. train_loss/heldout_loss are "
                      "ordinary unsmoothed cross-entropy with dropout and augmentation off; train_objective is "
                      "the augmented, smoothed training objective and is not comparable to them."}
    write_json(output / "result.json", result)
    return result
