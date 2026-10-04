"""CPU audit of saved JSCAS out-of-fold predictions; no models, fitting, or GPU work.

Advanced scores retain the existing evaluator's perfect-x assumption. The paired
video bootstrap describes these development videos, not an independent test.
"""

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from surgenet_dino.temporal.advanced import OUT_OF_BODY, frame_predictions, read_annotation
from surgenet_dino.temporal.timeline import segments
from surgenet_dino.temporal.workflows import f1_score, segment_counts
from surgenet_dino.training import LABELS, file_sha256


def read_predictions(path):
    videos = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            videos[row["video_id"]].append(
                (int(row["frame_index"]), LABELS.index(row["predicted_phase_id"]),
                 LABELS.index(row["phase_id"]), int(row["fold"]))
            )
    for video, rows in videos.items():
        rows.sort()
        if len({r[0] for r in rows}) != len(rows) or len({r[3] for r in rows}) != 1:
            raise ValueError(f"Duplicate frames or inconsistent fold: {video}")
    return dict(videos)


def best_iou(segment, others):
    label, start, stop = segment
    return max((max(0, min(stop, b) - max(start, a)) / (max(stop, b) - min(start, a))
                for other, a, b in others if label == other), default=0.0)


def pooled(rows, non_x=False):
    prefix = "non_x_" if non_x else ""
    return f1_score(*(sum(r[prefix + key] for r in rows) for key in ("tp", "fp", "fn")))


def evaluate(path, annotations):
    rows, phases, anatomy = [], defaultdict(Counter), Counter()
    for video, samples in sorted(read_predictions(path).items()):
        truth = read_annotation(annotations / f"{video}.csv")
        frames = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        if any(f < 0 or f >= len(truth) or truth[f] != target for f, _, target, _ in samples):
            raise ValueError(f"Prediction samples and annotations disagree: {video}")
        identity = hashlib.sha256("\n".join(f"{f},{t}" for f, _, t, _ in samples).encode()).hexdigest()
        found = segments(frame_predictions(truth, frames, labels))
        actual = segments(truth)
        tp, fp, fn = segment_counts(found, actual, 0.75)
        nx = segment_counts([s for s in found if s[0] != OUT_OF_BODY],
                            [s for s in actual if s[0] != OUT_OF_BODY], 0.75)
        rows.append({"video_id": video, "fold": samples[0][3], "tp": tp, "fp": fp, "fn": fn,
                     "non_x_tp": nx[0], "non_x_fp": nx[1], "non_x_fn": nx[2],
                     "correct_frames": sum(p == t for _, p, t, _ in samples),
                     "frames": len(samples), "sample_identity_sha256": identity,
                     "advanced_f1_perfect_x": f1_score(tp, fp, fn)})
        for label in range(len(LABELS) + 1):
            counts = segment_counts([s for s in found if s[0] == label],
                                    [s for s in actual if s[0] == label], 0.75)
            phases[LABELS[label] if label < len(LABELS) else "x"].update(dict(zip(("tp", "fp", "fn"), counts)))
        for segment in actual:
            if segment[0] == OUT_OF_BODY:
                continue
            iou = best_iou(segment, found)
            if iou < 0.75:
                key = "no_same_label_overlap" if iou == 0 else "iou_below_0_5" if iou < 0.5 else "iou_0_5_to_0_75"
                anatomy[key] += 1
        for segment in found:
            if segment[0] != OUT_OF_BODY and best_iou(segment, actual) < 0.75:
                duration = (segment[2] - segment[1]) / 30
                anatomy["fp_below_30s" if duration < 30 else "fp_30_to_120s" if duration < 120 else "fp_at_least_120s"] += 1
    totals = {key: sum(r[key] for r in rows) for key in ("tp", "fp", "fn", "correct_frames", "frames")}
    return rows, phases, {**totals, "accuracy": totals["correct_frames"] / totals["frames"],
                         "advanced_f1_perfect_x": pooled(rows), "non_x_f1": pooled(rows, True),
                         "mean_video_f1": sum(r["advanced_f1_perfect_x"] for r in rows) / len(rows),
                         "miss_anatomy": dict(anatomy)}


def paired_bootstrap(baseline, candidate, repetitions, seed):
    base, other = ({r["video_id"]: r for r in rows} for rows in (baseline, candidate))
    videos = sorted(base)
    if set(videos) != set(other):
        raise ValueError("Paired runs must contain the same videos")
    for video in videos:
        keys = ("frames", "fold", "sample_identity_sha256")
        if any(base[video][key] != other[video][key] for key in keys):
            raise ValueError("Paired runs must contain the same frames, targets, and folds")
    rng, differences = random.Random(seed), []
    for _ in range(repetitions):
        draw = rng.choices(videos, k=len(videos))
        differences.append(pooled([other[v] for v in draw]) - pooled([base[v] for v in draw]))
    differences.sort()
    folds = sorted({r["fold"] for r in baseline})
    return {"f1_difference": pooled(candidate) - pooled(baseline),
            "descriptive_video_bootstrap_95_interval": [differences[int(0.025 * repetitions)],
                                                       differences[min(repetitions - 1, int(0.975 * repetitions))]],
            "resamples_with_positive_difference": sum(d > 0 for d in differences) / repetitions,
            "fold_f1_differences": {str(f): pooled([r for r in candidate if r["fold"] == f]) -
                                          pooled([r for r in baseline if r["fold"] == f]) for f in folds},
            "videos_improved": sum(other[v]["advanced_f1_perfect_x"] > base[v]["advanced_f1_perfect_x"] for v in videos),
            "videos_worse": sum(other[v]["advanced_f1_perfect_x"] < base[v]["advanced_f1_perfect_x"] for v in videos)}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_check():
    assert segment_counts([(0, 0, 100)], [(0, 0, 100)], 0.75) == (1, 0, 0)
    assert segment_counts([(0, 0, 70)], [(0, 0, 100)], 0.75) == (0, 1, 1)
    assert best_iou((0, 0, 70), [(0, 0, 100)]) == 0.7
    assert best_iou((0, 0, 70), [(1, 0, 100)]) == 0
    rows = [{"video_id": "01", "fold": 1, "tp": 1, "fp": 0, "fn": 0,
             "frames": 10, "sample_identity_sha256": "same", "advanced_f1_perfect_x": 1.0}]
    identical = paired_bootstrap(rows, rows, 100, 42)
    assert identical["f1_difference"] == 0
    assert identical["descriptive_video_bootstrap_95_interval"] == [0, 0]
    assert pooled([{"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 9, "fn": 9}]) == 0.1


def context_offset_diagnostic(path):
    """The historical context experiment uses +/-30 catalogue rows, not seconds."""
    counts, maximum = Counter(), 0.0
    for samples in read_predictions(path).values():
        frames = [s[0] for s in samples]
        for i, frame in enumerate(frames):
            for offset in (-30, 30):
                j = max(0, min(len(frames) - 1, i + offset))
                if abs(j - i) < 30:
                    counts["clamped_end"] += 1
                    continue
                seconds = abs(frames[j] - frame) / 30
                maximum = max(maximum, seconds)
                counts["exact_30s" if seconds == 30 else "longer_than_30s"] += 1
                counts["non_clamped_pairs"] += 1
                if seconds > 60:
                    counts["above_60s"] += 1
                if seconds > 300:
                    counts["above_300s"] += 1
    return {**counts, "maximum_seconds": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=["advanced_cv_01", "context_cv_01", "motion_fixed_cv_01", "splice_cv_01"])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100 or args.output_dir.exists():
        parser.error("Use at least 100 bootstrap samples and a new output directory")
    import torch
    torch.set_num_threads(2)
    self_check()
    started, metrics, video_rows, phase_rows, run_rows = time.perf_counter(), {}, [], [], {}
    for run in args.runs:
        rows, phases, result = evaluate(args.experiments / run / "validation_predictions.csv", args.annotations)
        run_rows[run], metrics[run] = rows, result
        video_rows.extend({"run": run, **row} for row in rows)
        phase_rows.extend({"run": run, "phase": label, **counts, "f1": f1_score(counts["tp"], counts["fp"], counts["fn"])}
                          for label, counts in phases.items())
        print(json.dumps({"run": run, **result}, ensure_ascii=False), flush=True)
    baseline = args.runs[0]
    comparisons = {run: paired_bootstrap(run_rows[baseline], run_rows[run], args.bootstrap, args.seed)
                   for run in args.runs[1:]}
    source_files = {run: args.experiments / run / "validation_predictions.csv" for run in args.runs}
    source_files.update({f"annotation_{video}": args.annotations / f"{video}.csv"
                         for video in sorted(run_rows[baseline][i]["video_id"] for i in range(len(run_rows[baseline])))})
    source_files["audit_script"] = Path(__file__)
    for name in ("context_cv.py", "splice_cv.py"):
        path = args.experiments / "review_20260926" / name
        if path.is_file():
            source_files[name] = path
    report = {"scope": "Saved out-of-fold predictions only; no training or model inference",
              "metric_version": "JSCAS_FAQ_20260927_pooled_segment_F1_75_perfect_x",
              "score_assumption": "Perfect x taken from annotations; not a verified black-frame detector",
              "uncertainty": "Descriptive video bootstrap on 12 repeatedly used development videos; method-selection optimism and training-seed variance are not covered",
              "baseline": baseline, "bootstrap_repetitions": args.bootstrap, "seed": args.seed,
              "runs": metrics, "comparisons": comparisons,
              "context_row_offset_diagnostic": context_offset_diagnostic(source_files[baseline]),
              "source_files": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                               for name, path in source_files.items()},
              "elapsed_seconds": time.perf_counter() - started}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "per_video.csv", video_rows)
    write_csv(args.output_dir / "per_phase.csv", phase_rows)
    print(json.dumps({"comparisons": comparisons, "elapsed_seconds": report["elapsed_seconds"],
                      "output": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
