"""Head-only engine shared by CLI, cross-validation and CPU regression tests.

Images and DINO weights are deliberately NOT read here: the caller must validate
its cache before entering. The feature tensors supplied to fit() are never mutated.
"""

import csv
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch

from .diagnostics import (checkpoint_disagreement, dataset_audit, probability_metrics,
                          write_predictions)
from .head import (EarlyStopping, evaluation_steps, head_from_checkpoint,
                   make_head, make_optimizer)
from .head_options import HeadOptions


def atomic_checkpoint(path: Path, payload: dict):
    # Only used inside a freshly created run directory. Replace this run's best,
    # never somebody else's experiment. An interrupted write leaves old best intact.
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def fit(x, y, vx, vy, train_rows, val_rows, options: HeadOptions, seed, guard,
        output: Path, metadata: dict, step_fn=None) -> dict:
    # Lazy import: keep the original public training API and calibration step usable.
    from .training import (LABELS, classification_metrics, evaluate, head_step, state_digest,
                           write_json)

    step_fn = head_step if step_fn is None else step_fn
    for values, targets, rows in ((x, y, train_rows), (vx, vy, val_rows)):
        if (values.ndim != 2 or values.dtype != torch.float32 or targets.dtype != torch.int64
                or targets.shape != (len(rows),) or len(values) != len(rows) or not len(rows)
                or not torch.isfinite(values).all() or targets.min() < 0
                or targets.max() >= len(LABELS)
                or targets.tolist() != [r["target"] for r in rows]
                or any(r["phase_id"] != LABELS[r["target"]] for r in rows)):
            raise ValueError("Invalid head-engine inputs/label alignment")
    if x.shape[1] != vx.shape[1]:
        raise ValueError("Training/validation feature dimension mismatch")
    for key in ("video_id", "filename"):
        if {r[key] for r in train_rows} & {r[key] for r in val_rows}:
            raise ValueError(f"Training/validation leakage in head engine: {key}")
    audit = dataset_audit(train_rows, val_rows, x, vx, LABELS)
    write_json(output / "dataset_audit.json", audit)
    write_json(output / "training_options.json", asdict(options))
    head = make_head(x, len(LABELS), guard.device, seed, options)
    optimizer = make_optimizer(head)
    generator = torch.Generator().manual_seed(seed)
    stop_rule = EarlyStopping(options)
    initial_digest = state_digest(head)
    start = guard.mark()
    history = []
    best_score, best_epoch, best_step, best_logits = float("inf"), None, None, None
    global_step, interval_sum, interval_samples = 0, 0.0, 0
    last_epoch, stopped_early = 0.0, False
    payload = {**metadata, "labels": LABELS, "head_spec": head.specification(),
               "training_options": asdict(options), "seed": seed,
               "selection": "minimum validation loss, trained checkpoints only",
               "checkpoint_format": 3}


    def snapshot(epoch):
        return {**payload, "epoch": epoch, "global_step": global_step,
                "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}

    columns = ["epoch", "global_step", "train_loss", "train_accuracy", "validation_loss",
               "validation_accuracy", "validation_macro_f1", "train_objective_interval",
               "learning_rate", "weight_norm", "validation_ece", "validation_brier",
               "validation_confidence", "validation_wrong_confidence", "validation_wrong_nll",
               "validation_centered_logit_rms", "early_stop_bad_checks"]
    with (output / "learning_curve.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        def check(epoch):
            nonlocal best_score, best_epoch, best_step, best_logits, interval_sum, interval_samples
            logits, loss = evaluate(head, x, y, guard, options.head_batch_size)
            val_logits, val_loss = evaluate(head, vx, vy, guard, options.head_batch_size)
            metrics = classification_metrics(val_logits, vy)
            probability = probability_metrics(val_logits, vy)
            score = val_loss
            stop = False
            if global_step:
                # Strict best selection: min_delta governs only stopping, not saved-best correctness.
                if score < best_score:
                    best_score, best_epoch, best_step = score, epoch, global_step
                    best_logits = val_logits.clone()
                    atomic_checkpoint(output / "head_best.pt", snapshot(epoch))
                stop = stop_rule.update(score, epoch)
            row = {"epoch": epoch, "global_step": global_step, "train_loss": loss,
                   "train_accuracy": (logits.argmax(1) == y).float().mean().item(),
                   "validation_loss": val_loss, "validation_accuracy": metrics["accuracy"],
                   "validation_macro_f1": metrics["macro_f1"],
                   "train_objective_interval": interval_sum / interval_samples if interval_samples else None,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "weight_norm": head.linear.weight.detach().norm().item(),
                   "validation_ece": probability["ece"], "validation_brier": probability["brier"],
                   "validation_confidence": probability["mean_confidence"],
                   "validation_wrong_confidence": probability["wrong_confidence"],
                   "validation_wrong_nll": probability["wrong_nll"],
                   "validation_centered_logit_rms": probability["centered_logit_rms"],
                   "early_stop_bad_checks": stop_rule.bad_checks}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            interval_sum, interval_samples = 0.0, 0
            print(f"Epoch {epoch:.4f}/{options.epochs}, step {global_step}: "
                  f"train CE={loss:.5f}, acc={row['train_accuracy']:.4f}; "
                  f"val CE={val_loss:.5f}, acc={metrics['accuracy']:.4f}, "
                  f"macro-F1={metrics['macro_f1']:.4f}", flush=True)
            head.train()  # IMPORTANT: evaluation must not disable dropout for subsequent subepochs.
            return stop

        check(0.0)
        for epoch in range(1, options.epochs + 1):
            order = torch.randperm(len(train_rows), generator=generator)
            total_steps = math.ceil(len(order) / options.head_batch_size)
            checks = evaluation_steps(total_steps, options.evals_per_epoch)
            for step, indices in enumerate(order.split(options.head_batch_size), start=1):
                head.train()
                value = step_fn(head, optimizer, x[indices], y[indices], guard)
                global_step += 1
                interval_sum += value * len(indices)
                interval_samples += len(indices)
                last_epoch = epoch - 1 + step / total_steps
                if step in checks and check(last_epoch):
                    stopped_early = True
                    break
            if stopped_early:
                print(f"Early stop after {stop_rule.bad_checks} checks without sufficient improvement; "
                      f"restoring epoch {best_epoch:.4f}, step {best_step}", flush=True)
                break

    if best_logits is None:
        raise RuntimeError("No trained checkpoint was evaluated")
    atomic_checkpoint(output / "head_last.pt", snapshot(last_epoch))
    changed = initial_digest != state_digest(head)
    last_logits, last_loss = evaluate(head, vx, vy, guard, options.head_batch_size)
    last_metrics = classification_metrics(last_logits, vy, val_rows)
    last_metrics.update(probability_metrics(last_logits, vy))
    restored = torch.load(output / "head_best.pt", weights_only=True, map_location="cpu")
    head = head_from_checkpoint(restored, x.shape[1], len(LABELS), guard.device)
    reloaded, best_loss = evaluate(head, vx, vy, guard, options.head_batch_size)
    torch.testing.assert_close(best_logits, reloaded, rtol=0, atol=0)
    best_train, train_loss = evaluate(head, x, y, guard, options.head_batch_size)
    train_metrics = classification_metrics(best_train, y, train_rows)
    train_metrics.update(probability_metrics(best_train, y))
    val_metrics = classification_metrics(reloaded, vy, val_rows)
    val_metrics.update(probability_metrics(reloaded, vy))
    write_predictions(output / "validation_predictions.csv", reloaded, vy, val_rows, LABELS)
    write_predictions(output / "validation_predictions_last.csv", last_logits, vy, val_rows, LABELS)
    result = {"status": "passed", "scope": metadata.get("scope", "video-disjoint internal validation"),
              "history": history, "best_epoch": best_epoch, "best_global_step": best_step,
              "last_epoch": last_epoch, "total_steps": global_step, "stopped_early": stopped_early,
              "selection": payload["selection"], "selection_metric": "loss",
              "train": train_metrics, "validation": val_metrics, "validation_last": last_metrics,
              "best_validation_loss": best_loss, "last_validation_loss": last_loss,
              "best_train_loss": train_loss,
              "best_vs_last": checkpoint_disagreement(reloaded, last_logits, vy),
              "split": metadata.get("split"), "parameters_changed": changed,
              "initial_head_sha256": initial_digest, "seed": seed,
              "train_loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
              "labels": LABELS, "reload_identical": True, "gpu": guard.summary(start),
              "environment": metadata.get("environment"), "options": asdict(options),
              "head_batch_size": options.head_batch_size,
              "head_parameters": sum(p.numel() for p in head.parameters()),
              "evaluation_loss": "ordinary unsmoothed unweighted cross-entropy, dropout disabled",
              "probabilities_calibrated": False,
              "metric_notes": {"macro_f1": "all 13 classes; absent classes have F1=0",
                               "macro_f1_present": "only classes with positive ground-truth support",
                               "validation_is_test_set": False}}
    write_json(output / "result.json", result)
    return result
