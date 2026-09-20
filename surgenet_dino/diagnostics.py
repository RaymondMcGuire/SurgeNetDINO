"""Prediction diagnostics and counts. No fitting or tuning on validation data."""

import csv
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F


def probability_metrics(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> dict:
    if (logits.ndim != 2 or len(logits) == 0 or targets.shape != (len(logits),)
            or not torch.isfinite(logits).all() or bins < 1):
        raise ValueError("Malformed predictions for probability metrics")
    logp = logits.log_softmax(1)
    p = logp.exp()
    confidence, predictions = p.max(1)
    correct = predictions == targets
    index = torch.arange(len(targets))
    nll = -logp[index, targets]
    true_p = p[index, targets]
    membership = (confidence * bins).long().clamp(max=bins - 1)
    histogram, ece = [], 0.0
    for b in range(bins):
        mask = membership == b
        count = mask.sum().item()
        acc = correct[mask].float().mean().item() if count else None
        conf = confidence[mask].mean().item() if count else None
        if count:
            ece += count / len(targets) * abs(acc - conf)
        histogram.append({"lower": b / bins, "upper": (b + 1) / bins,
                          "count": count, "accuracy": acc, "confidence": conf})
    wrong = ~correct
    centered = logits - logits.mean(1, keepdim=True)
    return {"nll": nll.mean().item(), "ece": ece, "ece_bins": histogram,
            "brier": (p.square().sum(1) - 2 * true_p + 1).mean().item(),
            "mean_confidence": confidence.mean().item(),
            "wrong_confidence": confidence[wrong].mean().item() if wrong.any() else None,
            "wrong_nll": nll[wrong].mean().item() if wrong.any() else None,
            "correct_nll": nll[correct].mean().item() if correct.any() else None,
            "wrong_count": wrong.sum().item(), "mean_true_probability": true_p.mean().item(),
            "predictive_entropy": -(p * logp).sum(1).mean().item(),
            "centered_logit_rms": centered.square().mean().sqrt().item()}


def dataset_audit(train_rows, validation_rows, train_x, validation_x, labels) -> dict:
    result = {"labels": labels, "group_key": "video_id", "warnings": [],
              "note": "Counts are metadata, not an independent audit of clinical label correctness."}
    for role, rows, x in (("train", train_rows, train_x),
                          ("validation", validation_rows, validation_x)):
        videos = sorted({r["video_id"] for r in rows})
        by_video = {v: dict(Counter(r["phase_id"] for r in rows if r["video_id"] == v))
                    for v in videos}
        counts = Counter(r["phase_id"] for r in rows)
        coverage = {label: [v for v in videos if by_video[v].get(label, 0)] for label in labels}
        norms = x.norm(dim=1)
        result[role] = {"samples": len(rows), "video_ids": videos,
                        "class_counts": {k: counts[k] for k in labels},
                        "video_class_counts": by_video, "class_video_coverage": coverage,
                        "feature_norm": {"min": norms.min().item(), "mean": norms.mean().item(),
                                         "max": norms.max().item(),
                                         "zero_count": (norms == 0).sum().item()}}
        for label in labels:
            if not coverage[label]:
                result["warnings"].append(f"{role}: phase {label} is absent")
            elif role == "train" and len(coverage[label]) < 2:
                result["warnings"].append(f"train: phase {label} occurs in only one video")
    if set(result["train"]["video_ids"]) & set(result["validation"]["video_ids"]):
        raise ValueError("Training/validation video leakage")
    if len(result["validation"]["video_ids"]) < 5:
        result["warnings"].append("Few validation videos: frame count is not independent case count")
    return result


def write_predictions(path: Path, logits, targets, records, labels):
    logp = logits.log_softmax(1)
    probs = logp.exp()
    confidence, pred = probs.max(1)
    columns = ["video_id", "filename", "frame_index", "phase_id", "predicted_phase_id",
               "top_score", "true_probability", "nll", "correct"]
    columns += [f"logit_{k}" for k in labels] + [f"probability_{k}" for k in labels]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, row in enumerate(records):
            data = {"video_id": row["video_id"], "filename": row["filename"],
                    "frame_index": row["frame_index"], "phase_id": row["phase_id"],
                    "predicted_phase_id": labels[pred[i].item()], "top_score": confidence[i].item(),
                    "true_probability": probs[i, targets[i]].item(),
                    "nll": -logp[i, targets[i]].item(), "correct": int(pred[i] == targets[i])}
            data.update({f"logit_{k}": logits[i, j].item() for j, k in enumerate(labels)})
            data.update({f"probability_{k}": probs[i, j].item() for j, k in enumerate(labels)})
            writer.writerow(data)


def checkpoint_disagreement(best_logits, last_logits, targets) -> dict:
    best_pred, last_pred = best_logits.argmax(1), last_logits.argmax(1)
    shared_wrong = (best_pred != targets) & (last_pred != targets)
    best_loss = F.cross_entropy(best_logits, targets, reduction="none")
    last_loss = F.cross_entropy(last_logits, targets, reduction="none")
    return {"prediction_changes": (best_pred != last_pred).sum().item(),
            "correct_to_wrong": ((best_pred == targets) & (last_pred != targets)).sum().item(),
            "wrong_to_correct": ((best_pred != targets) & (last_pred == targets)).sum().item(),
            "shared_wrong_count": shared_wrong.sum().item(),
            "shared_wrong_nll_best": best_loss[shared_wrong].mean().item() if shared_wrong.any() else None,
            "shared_wrong_nll_last": last_loss[shared_wrong].mean().item() if shared_wrong.any() else None}
