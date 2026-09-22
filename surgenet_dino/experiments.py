"""Fixed regularized-head cross-validation over all labeled cached videos."""

import csv
import random
import statistics
from collections import Counter
from dataclasses import asdict

import torch

from .diagnostics import dataset_audit, probability_metrics
from .head_options import resolve_options


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_folds(records: list, folds: int, seed: int, labels: list[str]) -> list[list[str]]:
    """Search label-count-balanced whole-video partitions; never use validation/model outputs."""
    videos = sorted({r["video_id"] for r in records})
    if type(folds) is not int or not 2 <= folds <= len(videos):
        raise ValueError("folds must be between 2 and the number of labeled videos")
    counts = {v: [sum(r["video_id"] == v and r["phase_id"] == label for r in records)
                  for label in labels] for v in videos}
    total = [sum(counts[v][i] for v in videos) for i in range(len(labels))]
    rare = [label for i, label in enumerate(labels) if sum(counts[v][i] > 0 for v in videos) < 2]
    if rare:
        raise ValueError(f"Cannot keep every class in each training fold: phases {rare} "
                         "appear in fewer than two labeled videos. Inspect dataset_audit.json.")
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(512):
        order = videos.copy()
        rng.shuffle(order)
        groups = [sorted(order[k::folds]) for k in range(folds)]
        held_counts = [[sum(counts[v][i] for v in group) for i in range(len(labels))]
                       for group in groups]
        if any(h[i] == 0 or total[i] == h[i] for h in held_counts for i in range(len(labels))):
            continue  # a training or validation fold would lack a class
        score = sum(abs(h[i] / total[i] - 1 / folds)
                    for h in held_counts for i in range(len(labels)))
        if score < best_score:
            best, best_score = groups, score
    if best is None:
        raise ValueError("No class-covered grouped partition found; use fewer folds")
    return best


def write_folds(cache, groups, directory, checksum, seed):
    from .training import LABELS, file_sha256, write_json

    records = cache["split"]["train"]["records"] + cache["split"]["validation"]["records"]
    videos = sorted({r["video_id"] for r in records})
    if sorted(v for group in groups for v in group) != videos:
        raise ValueError("Fold videos must cover the dataset exactly once")
    directory.mkdir(parents=True, exist_ok=False)
    partitions = []
    for k, held in enumerate(groups, 1):
        indices = {role: [i for i, r in enumerate(records) if (r["video_id"] in held) == is_val]
                   for role, is_val in (("train", False), ("validation", True))}
        train, val = ([records[i] for i in indices[role]] for role in ("train", "validation"))
        for key in ("video_id", "filename", "image_path", "sha256"):
            if {r[key] for r in train} & {r[key] for r in val}:
                raise ValueError(f"Fold {k} training/validation leakage: {key}")
        if any({r["phase_id"] for r in rows} != set(LABELS) for rows in (train, val)):
            raise ValueError(f"Fold {k} missing class coverage")
        child = directory / f"fold_{k}"
        child.mkdir()
        source = cache["split"]["definition"]
        definition = {key: source[key] for key in
                      ("dataset_json", "dataset_json_sha256", "samples_csv_sha256")}
        definition.update(status="passed", label_order=LABELS, group_key="video_id", seed=seed,
                          selection="whole-video label-count balance; no model predictions",
                          source_split_sha256=cache["split"]["sha256"],
                          feature_cache_sha256=checksum, class_counts={})
        for role in indices:
            rows = [records[i] for i in indices[role]]
            manifest = child / f"{role}.csv"
            write_csv(manifest, [{"video_id": r["video_id"], "filename": r["filename"][:-4],
                                 "frame_index": r["frame_index"], "time_seconds": r["frame_index"] / 30,
                                 "phase-id": r["phase_id"], "image_path": r["image_path"],
                                 "image_sha256": r["sha256"]} for r in rows])
            definition.update({f"{role}_video_ids": sorted({r["video_id"] for r in rows}),
                               f"{role}_samples": len(rows), f"{role}_csv_sha256": file_sha256(manifest)})
            definition["class_counts"][role] = dict(Counter(r["phase_id"] for r in rows))
        write_json(child / "split.json", definition)
        partitions.append(indices)
    write_json(directory / "folds.json", {"status": "passed", "seed": seed, "validation_videos": groups,
                                         "feature_cache_sha256": checksum, "samples": len(records)})
    return records, partitions


def summarize(output, records, partitions, results, options, split_directory):
    from .training import LABELS, classification_metrics, write_json

    seen = set()
    logits = torch.empty((len(records), len(LABELS)), dtype=torch.float32)
    summaries = []
    with (output / "validation_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = None
        for k, (partition, result) in enumerate(zip(partitions, results), 1):
            if result["status"] != "passed":
                raise ValueError("Cannot summarize incomplete folds")
            path = output / f"fold_{k}" / "validation_predictions.csv"
            with path.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=["fold", *reader.fieldnames])
                    writer.writeheader()
                count = 0
                for idx, row in zip(partition["validation"], reader, strict=True):
                    record = records[idx]
                    if (idx in seen or row["video_id"] != record["video_id"]
                            or row["filename"] != record["filename"] or row["phase_id"] != record["phase_id"]
                            or int(row["frame_index"]) != record["frame_index"]):
                        raise ValueError("Out-of-fold prediction identity/coverage mismatch")
                    values = torch.tensor([float(row[f"logit_{label}"]) for label in LABELS])
                    if not torch.isfinite(values).all() or LABELS[values.argmax().item()] != row["predicted_phase_id"]:
                        raise ValueError("Invalid out-of-fold logits")
                    logits[idx] = values
                    seen.add(idx)
                    writer.writerow({"fold": k, **row})
                    count += 1
            metrics = result["validation"]
            summaries.append({"fold": k, "validation_samples": count,
                              "validation_videos": ",".join(sorted(metrics["per_video"])),
                              "best_epoch": result["best_epoch"], "last_epoch": result["last_epoch"],
                              "total_steps": result["total_steps"], "stopped_early": result["stopped_early"],
                              "reached_epoch_limit": result["last_epoch"] >= options.epochs,
                              **{key: metrics[key] for key in ("accuracy", "macro_f1", "nll", "ece")}})
    if len(results) != len(partitions) or seen != set(range(len(records))):
        raise ValueError("Incomplete out-of-fold predictions")
    targets = torch.tensor([r["target"] for r in records])
    pooled = classification_metrics(logits, targets, records)
    pooled.update(probability_metrics(logits, targets))
    if sum(sum(row) for row in pooled["confusion"]) != len(records):
        raise ValueError("Pooled confusion coverage mismatch")
    stats = {key: {"mean": statistics.mean(r[key] for r in summaries),
                   "std": statistics.stdev(r[key] for r in summaries)}
             for key in ("accuracy", "macro_f1", "nll", "ece")}
    write_csv(output / "fold_metrics.csv", summaries)
    write_csv(output / "video_metrics.csv", [
        {"video_id": video, **{key: m[key] for key in ("accuracy", "macro_f1", "nll", "ece")},
         "samples": sum(r["support"] for r in m["per_class"])}
        for video, m in pooled["per_video"].items()])
    result = {"status": "passed", "scope": "development cross-validation with per-fold early stopping",
              "folds": summaries, "fold_statistics": stats, "pooled": pooled, "labels": LABELS,
              "samples": len(records), "split_directory": str(split_directory), "options": asdict(options),
              "coverage_exactly_once": True,
              "aggregation": "equal-fold mean and sample std; pooled metrics recomputed from all held-out predictions",
              "note": "Fold validation chooses checkpoints and early stopping. These are development scores, "
                      "not an independent or unbiased final test. No final deployment model is selected."}
    write_json(output / "result.json", result)
    return result


def crossvalidate(args, config, output, cache):
    from .gpu import CPUControl, GPUControl
    from .head_training import fit
    from .runtime import resolve_device
    from .training import LABELS, environment, file_sha256, profile_batch, resources_for, write_json

    records = cache["split"]["train"]["records"] + cache["split"]["validation"]["records"]
    groups = grouped_folds(records, args.folds, config.seed, LABELS)
    options = resolve_options(args)
    checksum = file_sha256(args.features)
    split_directory = args.split_output_dir.resolve()
    records, partitions = write_folds(cache, groups, split_directory, checksum, config.seed)
    write_json(output / "folds.json", {"split_directory": str(split_directory), "validation_videos": groups,
                                       "folds_sha256": file_sha256(split_directory / "folds.json")})
    device = resolve_device(config.device)
    resources = resources_for(args)
    control = GPUControl if device.type == "cuda" else CPUControl
    results = []
    for k, indices in enumerate(partitions, 1):
        child = output / f"fold_{k}"
        child.mkdir()
        train_ids, val_ids = indices["train"], indices["validation"]
        print(f"Fold {k}/{len(groups)}: validation videos {groups[k - 1]}", flush=True)
        try:
            with control(device, resources, child / "gpu.csv") as guard:
                env = environment(device, cache["feature_identity"])
                if device.type == "cuda":
                    profile_batch(args, env, resources)
                metadata = {"feature_identity": cache["feature_identity"], "model_config": cache["model_config"],
                            "feature_cache_sha256": checksum, "environment": env,
                            "scope": "development cross-validation; held-out fold selects checkpoint",
                            "split": {"path": str(split_directory / f"fold_{k}" / "split.json"),
                                      "sha256": file_sha256(split_directory / f"fold_{k}" / "split.json")}}
                result = fit(cache["features"][train_ids], cache["targets"][train_ids],
                             cache["features"][val_ids], cache["targets"][val_ids],
                             [records[i] for i in train_ids], [records[i] for i in val_ids],
                             options, config.seed, guard, child, metadata)
                results.append(result)
        except BaseException as exc:
            write_json(child / "failure.json", {"error": str(exc), "type": type(exc).__name__})
            raise
    return summarize(output, records, partitions, results, options, split_directory)


def run_experiment(args, config, output):
    from .training import LABELS, load_features, write_json

    if args.command == "crossval-head" and args.split_output_dir.exists():
        raise FileExistsError(args.split_output_dir)
    cache = (load_features(args.features, inspect_images=True) if args.command == "audit-features"
             else load_features(args.features))
    contract = cache["feature_identity"]["contract"]
    if config.name != contract["model"] or (
            config.sha256 and config.sha256.lower() != contract.get("checkpoint_sha256")):
        raise ValueError("Requested model does not match the feature cache")
    if args.command == "audit-features":
        train = cache["split"]["train"]["records"]
        validation = cache["split"]["validation"]["records"]
        audit = dataset_audit(train, validation, cache["features"][:len(train)],
                              cache["features"][len(train):], LABELS)
        write_json(output / "dataset_audit.json", audit)
        result = {"status": "passed", "audit": audit, "training_performed": False}
        write_json(output / "result.json", result)
        return result
    return crossvalidate(args, config, output, cache)
