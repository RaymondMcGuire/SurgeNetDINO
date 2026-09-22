"""Plot the learning-curve diagnostic of `python -m surgenet_dino.temporal curves`.

Requires matplotlib only; run it in a separate environment, as the other plot scripts do:
uv run --isolated --no-project --with matplotlib==3.11.2 python scripts/plot_temporal.py RUN_DIR [--reference-result CV_RESULT_JSON]

Plots only epochs that were actually evaluated. Training and held-out losses are the same
quantity (ordinary cross-entropy, dropout and augmentation off), so they share one axis; the
augmented training objective is a different quantity and is deliberately not drawn with them.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TRAIN, HELD = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the validated default palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{**row, "fold": int(row["fold"]), "seed": int(row["seed"]), "epoch": int(row["epoch"]),
                 **{k: float(row[k]) for k in ("train_loss", "train_accuracy", "heldout_loss", "heldout_accuracy")}}
                for row in csv.DictReader(stream)]


def by_epoch(rows, key):
    """epoch -> (mean, min, max) over seeds."""
    values = defaultdict(list)
    for row in rows:
        values[row["epoch"]].append(row[key])
    epochs = sorted(values)
    return (epochs, [sum(values[e]) / len(values[e]) for e in epochs],
            [min(values[e]) for e in epochs], [max(values[e]) for e in epochs])


def style(axis, title, ylabel, first_column):
    axis.set_title(title, fontsize=10, color=INK, loc="left")
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)
    if first_column:
        axis.set_ylabel(ylabel, fontsize=9, color=MUTED)


def draw(axis, rows, key, marked, reference=None):
    for name, column, colour in (("training videos", f"train_{key}", TRAIN), ("held-out videos", f"heldout_{key}", HELD)):
        epochs, mean, low, high = by_epoch(rows, column)
        axis.fill_between(epochs, low, high, color=colour, alpha=0.15, linewidth=0)
        axis.plot(epochs, mean, color=colour, linewidth=2, marker="o", markersize=3, label=name)
        axis.annotate(f"{mean[-1]:.3f}", (epochs[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                      fontsize=8, color=INK, va="center")
    if marked is not None:
        axis.axvline(marked, color=MUTED, linewidth=1, linestyle="--")
    if reference is not None:
        axis.axhline(reference, color=MUTED, linewidth=1, linestyle=":")


def figure(plt, rows, folds, stage, marks, path, title, note, reference_key=None):
    rows = [r for r in rows if r["stage"] == stage]
    if not rows:
        return False
    numbers = sorted({r["fold"] for r in rows})
    fig, axes = plt.subplots(2, len(numbers), figsize=(3.6 * len(numbers), 6.2), sharex=True, squeeze=False)
    for column, fold in enumerate(numbers):
        part = [r for r in rows if r["fold"] == fold]
        info = next(f for f in folds if f["fold"] == fold)
        videos = ",".join(info["validation_videos"])
        reference = info.get(reference_key) if reference_key else None
        draw(axes[0][column], part, "loss", marks.get(fold))
        draw(axes[1][column], part, "accuracy", marks.get(fold), reference)
        style(axes[0][column], f"Fold {fold}: held-out {videos}", "cross-entropy", column == 0)
        style(axes[1][column], "", "accuracy", column == 0)
        axes[1][column].set_xlabel("epoch", fontsize=9, color=MUTED)
        axes[1][column].set_ylim(0.5, 1.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-result", type=Path,
                        help="result.json of a finished cv run: marks the epochs its inner folds had chosen")
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if not result.get("diagnostic_only"):
        raise SystemExit("Not a `temporal curves` result directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, folds = read_rows(root / "curves.csv"), result["folds"]
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    fixed = {f["fold"]: f["head_fixed_epochs"] for f in folds}
    chosen = {f["fold"]: f["network_common_epoch"] for f in folds if f.get("network_common_epoch")}
    if args.reference_result is not None:
        reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
        chosen = {s["fold"]: s["common_epoch"] for s in reference["selections"]}
    made = []
    if figure(plt, rows, folds, "head", fixed, plots / "head_curves.png",
              "Frame head (single image): training vs held-out videos",
              "Dashed line: the fixed 5-epoch budget. Ordinary cross-entropy, dropout off. One seed.", None):
        made.append("head_curves.png")
    if figure(plt, rows, folds, "network", chosen, plots / "network_curves.png",
              "Temporal network on cross-fitted inputs: training vs held-out videos",
              "Dashed line: epoch chosen beforehand by inner folds. Dotted line: held-out frame-head accuracy. "
              f"Line = mean of {result['network_seeds']} seeds, band = min to max. Augmentation and dropout off.",
              "heldout_frame_accuracy"):
        made.append("network_curves.png")
    print(json.dumps({"plots": [str(plots / name) for name in made]}, indent=2))


if __name__ == "__main__":
    main()
