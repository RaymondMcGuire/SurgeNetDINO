"""Leak-free grouped cross-validation of the temporal phase network on the labeled videos.

Rule used everywhere: a video is predicted by frame heads, a projection, temporal networks, an
epoch budget and order rules that never saw that video. Scores are development estimates, not an
independent test.
"""

import csv
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ..diagnostics import probability_metrics, write_predictions
from ..experiments import write_csv
from ..training import LABELS, classification_metrics, file_sha256, write_json
from .timeline import build_timelines, segments

CLASSES = len(LABELS)
LOGIT_FLOOR = -50.0  # written log-probabilities stay finite for the shared prediction CSV schema


def edit_score(predicted, truth):
    a, b = [s[0] for s in predicted], [s[0] for s in truth]
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b), 1)


def segmental_f1(predicted, truth, threshold):
    return f1_score(*segment_counts(predicted, truth, threshold))


def f1_score(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def segment_counts(predicted, truth, threshold):
    """(TP, FP, FN) of Lea et al. segmental F1@threshold: a true segment can be matched once, by the same label."""
    hit, tp = [False] * len(truth), 0
    for label, start, stop in predicted:
        best, index = 0.0, -1
        for i, (other, a, b) in enumerate(truth):
            if other != label:
                continue
            iou = max(0, min(stop, b) - max(start, a)) / (max(stop, b) - min(start, a))
            if iou > best:
                best, index = iou, i
        if index >= 0 and best >= threshold and not hit[index]:
            hit[index], tp = True, tp + 1
    return tp, len(predicted) - tp, len(truth) - sum(hit)


def segment_metrics(predicted, targets, timelines):
    """Per video, on the time-ordered catalogued frames."""
    rows = []
    for video, timeline in sorted(timelines.items()):
        index = timeline.rows[timeline.present]
        found, truth = segments(predicted[index]), segments(targets[index])
        rows.append({"video_id": video, "true_segments": len(truth), "predicted_segments": len(found),
                     "edit": edit_score(found, truth),
                     **{f"f1_at_{int(t * 100)}": segmental_f1(found, truth, t) for t in (0.25, 0.5, 0.75)}})
    return rows


def error_anatomy(predicted, targets, timelines, long_run=60, near=15):
    """Where the errors are: close to a true boundary, or inside long uniformly mislabelled runs."""
    errors = close = in_long = 0
    for timeline in timelines.values():
        index = timeline.rows[timeline.present]
        truth, guess = targets[index].tolist(), predicted[index].tolist()
        boundaries = [i for i in range(1, len(truth)) if truth[i] != truth[i - 1]]
        position = 0
        while position < len(truth):
            if truth[position] == guess[position]:
                position += 1
                continue
            stop = position
            while stop < len(truth) and truth[stop] == truth[position] and guess[stop] == guess[position]:
                stop += 1
            length = stop - position
            errors += length
            in_long += length if length >= long_run else 0
            close += sum(1 for i in range(position, stop)
                         if boundaries and min(abs(i - b) for b in boundaries) <= near)
            position = stop
    return {"error_frames": errors, "share_near_boundary": close / max(1, errors),
            "share_in_long_runs": in_long / max(1, errors)}


def storable_logits(log_prob):
    """float32 log-probabilities whose argmax survives the CSV round trip and the softmax in the writer."""
    logits = log_prob.clamp(min=LOGIT_FLOOR).to(torch.float32)
    if not torch.isfinite(logits).all():
        raise ValueError("Missing/nonfinite predictions")
    top = logits.argmax(1)
    disagree = logits.log_softmax(1).exp().argmax(1) != top
    logits[disagree, top[disagree]] += 1e-4
    return logits


def write_fold_predictions(path, logits, targets, records, fold_ids):
    temporary = path.with_name("_" + path.name)
    write_predictions(temporary, logits, targets, records, LABELS)
    with temporary.open(encoding="utf-8", newline="") as source, path.open("x", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(out, fieldnames=["fold", *reader.fieldnames])
        writer.writeheader()
        for fold, row in zip(fold_ids, reader, strict=True):
            writer.writerow({"fold": fold, **row})
    temporary.unlink()


def summary_row(name, description, logits, targets, records, timelines, annotations=None):
    metrics = classification_metrics(logits, targets, records)
    predicted = logits.argmax(1)
    parts = segment_metrics(predicted, targets, timelines)
    mean = lambda key: statistics.mean(r[key] for r in parts)  # noqa: E731
    row = {"variant": name, "description": description, "accuracy": metrics["accuracy"],
           "macro_f1": metrics["macro_f1"], "video_mean_accuracy": metrics["video_mean_accuracy"],
           "min_video_accuracy": min(m["accuracy"] for m in metrics["per_video"].values()),
           "mean_true_segments": mean("true_segments"), "mean_predicted_segments": mean("predicted_segments"),
           "edit": mean("edit"), "f1_at_25": mean("f1_at_25"), "f1_at_50": mean("f1_at_50"),
           "f1_at_75": mean("f1_at_75"), **error_anatomy(predicted, targets, timelines)}
    if annotations is not None:
        from .advanced import advanced_counts, pooled_f1
        row["advanced_f1_at_75"] = pooled_f1(advanced_counts(predicted, records, annotations))
    return row


def read_baseline(path, records):
    """Frame logits of an existing out-of-fold CSV, aligned to `records` by (video, filename)."""
    rows = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows[(row["video_id"], row["filename"])] = [float(row[f"logit_{label}"]) for label in LABELS]
    try:
        return torch.tensor([rows[(r["video_id"], r["filename"])] for r in records])
    except KeyError as error:
        raise ValueError("Baseline predictions do not cover the evaluated frames") from error


def paired_video_bootstrap(records, targets, first, second, repetitions=2000, seed=42):
    """Descriptive video-clustered interval of accuracy(second) - accuracy(first)."""
    a, b = first.argmax(1) == targets, second.argmax(1) == targets
    videos = sorted({r["video_id"] for r in records})
    ids = {v: torch.tensor([i for i, r in enumerate(records) if r["video_id"] == v]) for v in videos}
    totals = torch.tensor([len(ids[v]) for v in videos], dtype=torch.float64)
    changes = torch.tensor([int(b[ids[v]].sum()) - int(a[ids[v]].sum()) for v in videos], dtype=torch.float64)
    draw = torch.randint(len(videos), (repetitions, len(videos)), generator=torch.Generator().manual_seed(seed))
    deltas = changes[draw].sum(1) / totals[draw].sum(1)
    interval = torch.quantile(deltas, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"accuracy_delta": float(b.float().mean() - a.float().mean()), "video_bootstrap_95_percentile": interval,
            "videos": len(videos), "repetitions": repetitions,
            "warning": "Cluster resampling over few videos with overlapping training sets: descriptive only."}


def load_inputs(features, tokens_directory):
    """(cache, records, targets, tokens, provenance). Tokens are verified against the base cache."""
    from ..training import load_features
    from .tokens import load_tokens
    cache = load_features(features)
    records = cache["split"]["train"]["records"] + cache["split"]["validation"]["records"]
    tokens, check = load_tokens(tokens_directory, cache, features)
    provenance = {"features": str(Path(features).resolve()), "features_sha256": file_sha256(Path(features)),
                  "tokens_directory": str(Path(tokens_directory).resolve()), **check,
                  "feature_identity": cache["feature_identity"]}
    return cache, records, cache["targets"], tokens, provenance


def predict_fold(x, targets, records, timelines, fit_videos, held, head_options, options, seed, guard, inner_seed,
                 extra=None):
    """Everything is learned from fit_videos; `held` is touched only by the final prediction."""
    from . import constraints, tcn
    from .emissions import stacked_emissions
    inner = tcn.inner_cv(x, targets, records, timelines, fit_videos, head_options, options, seed, guard, inner_seed,
                         extra)
    chosen, by_epoch = tcn.common_epoch(inner, targets, fit_videos)
    logits, ledger = stacked_emissions(x, targets, records, fit_videos, held, head_options, options.head_epochs,
                                       seed, guard)
    _, ensemble, per_seed, info = tcn.refit(x, targets, records, timelines, fit_videos, held, logits, head_options,
                                            options, seed, guard, chosen, inner["rows"], extra)
    rules = constraints.training_rules(targets, timelines, fit_videos)
    constrained, changed = constraints.apply_rules(ensemble, timelines, held, rules)
    ordered, _ = constraints.apply_rules(ensemble, timelines, held, replace(rules, minimum=()))
    held_rows = torch.cat([inner["rows"][v] for v in held])
    fit_rows = torch.cat([inner["rows"][v] for v in fit_videos])
    seeds = [(per_seed[s][held_rows].argmax(1) == targets[held_rows]).double().mean().item()
             for s in range(options.seeds)]
    return {"rows": held_rows.tolist(), "network": ensemble, "ordered": ordered, "constrained": constrained,
            "frames": logits.double().log_softmax(1),
            "selection": {"validation_videos": sorted(held), "fit_videos": sorted(fit_videos), "common_epoch": chosen,
                          "inner_accuracy_by_epoch": by_epoch, "inner_accuracy": max(by_epoch.values()),
                          "inner_groups": inner["groups"], "inner_frames": len(fit_rows),
                          "heldout_seed_accuracy": seeds, "emission_ledger": ledger,
                          "order_rules": {**rules.summary(), "changed_seconds": changed}, **info}}


def run_cv(features, tokens_directory, folds_json, options, head_options, config, resources, output, *,
           baseline=None, only_fold=None, motion=None, annotations=None):
    """Grouped outer folds; nothing is tuned on a held-out fold. The network's log-probabilities are
    decoded under the rules of each fold's training videos (order and minimum block length); the plain
    argmax and the order rules alone are reported too.
    `motion`: optional per-frame motion CSV (tools/motion_features.py), an extra network input only.
    `annotations`: optional directory of the official annotation CSVs, for the Advanced score (advanced.py)."""
    from ..gpu import CPUControl, GPUControl
    from ..runtime import resolve_device
    from .emissions import per_video_zscore
    from .folds import load_outer_folds
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache, records, targets, tokens, provenance = load_inputs(features, tokens_directory)
    contract = cache["feature_identity"]["contract"]
    if config.name != contract["model"] or (
            config.sha256 and config.sha256.lower() != contract.get("checkpoint_sha256")):
        raise ValueError("Requested model does not match the feature cache")
    groups = load_outer_folds(records, folds_json)
    if only_fold is not None and not 1 <= only_fold <= len(groups):
        raise ValueError("--only-fold is outside the fold range")
    x = per_video_zscore(tokens, records)
    extra = None
    if motion is not None:
        from .motion import load_motion
        values, motion_provenance = load_motion(motion, records)
        extra, provenance = per_video_zscore(values, records), {**provenance, **motion_provenance}
    timelines = build_timelines(records, options.long_gap_seconds)
    device = resolve_device(config.device)
    control = GPUControl if device.type == "cuda" else CPUControl
    network, ordered, constrained, frames, selections = {}, {}, {}, {}, []
    with control(device, resources, output / "gpu.csv") as guard:
        for k, held in enumerate(groups, 1):
            if only_fold is not None and k != only_fold:
                continue
            fit_videos = sorted(v for v in timelines if v not in held)
            print(f"Fold {k}/{len(groups)}: held-out {sorted(held)}", flush=True)
            part = predict_fold(x, targets, records, timelines, fit_videos, sorted(held), head_options, options,
                                config.seed, guard, inner_seed=k, extra=extra)
            for row in part["rows"]:
                network[row], frames[row] = part["network"][row], part["frames"][row]
                ordered[row], constrained[row] = part["ordered"][row], part["constrained"][row]
            selections.append({"fold": k, **part["selection"]})
            s = part["selection"]
            print(f"Fold {k}: common epoch {s['common_epoch']}, inner accuracy {s['inner_accuracy']:.4f}, "
                  f"order rules changed {s['order_rules']['changed_seconds']} seconds", flush=True)
        usage = guard.summary()
    keep = sorted(network)
    if only_fold is None and keep != list(range(len(records))):
        raise ValueError("Held-out predictions do not cover every frame exactly once")
    owner = {v: k for k, g in enumerate(groups, 1) for v in g}
    part_records = [records[i] for i in keep]
    part_targets = targets[keep]
    fold_ids = [owner[r["video_id"]] for r in part_records]
    part_timelines = build_timelines(part_records, options.long_gap_seconds)  # rebuilt on the reported rows
    variants = {"frames": ("cross-fitted frame head alone, argmax (no temporal context)",
                           storable_logits(torch.stack([frames[i] for i in keep]))),
                "network": ("temporal network, seed-averaged, argmax",
                            storable_logits(torch.stack([network[i] for i in keep]))),
                "ordered": ("temporal network decoded under the order rules of the training videos only",
                            storable_logits(torch.stack([ordered[i] for i in keep]))),
                "constrained": ("temporal network decoded under the order rules and minimum block lengths "
                                "of the training videos", storable_logits(torch.stack([constrained[i] for i in keep])))}
    main = variants["constrained"][1]
    plain = variants["network"][1]
    write_fold_predictions(output / "validation_predictions.csv", main, part_targets, part_records, fold_ids)
    write_fold_predictions(output / "network_predictions.csv", plain, part_targets, part_records, fold_ids)
    summary = [summary_row(name, text, logits, part_targets, part_records, part_timelines, annotations)
               for name, (text, logits) in variants.items()]
    write_csv(output / "summary.csv", summary)
    metrics = classification_metrics(main, part_targets, part_records)
    metrics.update(probability_metrics(main, part_targets))
    write_csv(output / "video_metrics.csv", [
        {"video_id": v, "accuracy": m["accuracy"], "macro_f1_present": m["macro_f1_present"],
         "samples": sum(c["support"] for c in m["per_class"])} for v, m in metrics["per_video"].items()])
    parts = segment_metrics(main.argmax(1), part_targets, part_timelines)
    if annotations is not None:
        from .advanced import advanced_counts
        counts = advanced_counts(main.argmax(1), part_records, annotations)
        parts = [{**r, "advanced_f1_at_75": f1_score(*counts[r["video_id"]])} for r in parts]
        provenance["annotations"] = str(Path(annotations).resolve())
    write_csv(output / "segment_metrics.csv", parts)
    folds = []
    for selection in selections:
        ids = [i for i, f in enumerate(fold_ids) if f == selection["fold"]]
        part = classification_metrics(main[ids], part_targets[ids])
        folds.append({"fold": selection["fold"], "validation_videos": ",".join(selection["validation_videos"]),
                      "validation_samples": len(ids), "accuracy": part["accuracy"], "macro_f1": part["macro_f1"],
                      "network_accuracy": classification_metrics(plain[ids], part_targets[ids])["accuracy"],
                      "ordered_accuracy": classification_metrics(variants["ordered"][1][ids],
                                                                 part_targets[ids])["accuracy"],
                      "common_epoch": selection["common_epoch"], "inner_accuracy": selection["inner_accuracy"],
                      "changed_seconds": selection["order_rules"]["changed_seconds"]})
    write_csv(output / "fold_metrics.csv", folds)
    result = {"status": "passed", "samples": len(part_records), "complete": only_fold is None,
              "coverage_exactly_once": only_fold is None, "summary": summary, "folds": folds, "pooled": metrics,
              "selections": selections, "options": asdict(options), "head_options": asdict(head_options),
              "provenance": {**provenance, "folds_json_sha256": file_sha256(Path(folds_json)),
                             "validation_videos": groups, "seed": config.seed, "resources": usage},
              "held_out_used_for_selection": False, "independent_final_test": False,
              "note": "Offline: the network reads the whole video, including future frames, and the per-video "
                      "standardisation uses all frames of that video. The epoch budget was chosen by inner folds and "
                      "the order rules and minimum block lengths were read from the training videos, all without "
                      "the held-out fold, but the method itself was designed on these same 12 videos: a development "
                      "estimate, not an unbiased test. The Advanced score takes phase x as exactly detected (its "
                      "frames are black)."}
    if baseline is not None and only_fold is None:
        result["paired_comparison_vs_baseline"] = paired_video_bootstrap(
            part_records, part_targets, read_baseline(baseline, part_records), main)
    write_json(output / "result.json", result)
    return result
