"""Read-only plots for original run/train or new head-only directories.

Each chart has its own file. Accepts fractional epochs, any selection metric and
variable video counts. Does not load a model, select a different checkpoint, fit
calibration, change metrics, or assume that best_epoch is an integer/minimum CE.
"""

import argparse
import json
from pathlib import Path


def plot_run(run: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run = run.resolve()
    direct = run / "result.json"
    nested = run / "train/result.json"
    path = direct if direct.is_file() else nested
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "passed" or not data.get("history"):
        raise ValueError("Expected one completed head run; for cross-validation pass a fold directory")
    history = data["history"]
    best = data["best_epoch"]
    labels = data["labels"]
    matrix = np.asarray(data["validation"]["confusion"])
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Unexpected confusion matrix shape")
    if not np.isclose(np.trace(matrix) / matrix.sum(), data["validation"]["accuracy"]):
        raise ValueError("Confusion matrix and accuracy disagree")
    output = run / "plots"
    output.mkdir(exist_ok=False)
    epochs = [row["epoch"] for row in history]
    selection = data.get("selection", "minimum validation cross-entropy")

    def curve(filename, series, title, ylabel, percent=False):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for key, legend in series:
            if not any(row.get(key) is not None for row in history):
                continue
            values = [np.nan if row.get(key) is None else row[key] * (100 if percent else 1)
                      for row in history]
            ax.plot(epochs, values, label=legend, marker=".")
        ax.axvline(best, linestyle="--", label=f"Selected epoch {best:.3f}")
        ax.set(title=title, xlabel="Epoch (0 = untrained)", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)

    curve("loss_curve.png", [("train_loss", "Training: ordinary CE"),
                             ("validation_loss", "Validation: ordinary CE")],
          "Frozen-head loss (not smoothed training objective)", "Cross-entropy")
    curve("accuracy_curve.png", [("train_accuracy", "Training"),
                                 ("validation_accuracy", "Validation")], "Classification accuracy", "Accuracy (%)", True)
    curve("macro_f1_curve.png", [("validation_macro_f1", "Validation macro-F1, all 13 classes")],
          "Validation macro-F1", "F1 (%)", True)
    if any("validation_wrong_confidence" in r for r in history):
        curve("confidence_curve.png", [("validation_confidence", "All validation predictions"),
                                       ("validation_wrong_confidence", "Wrong predictions")],
              "Confidence diagnostics (wrong-sample membership can change)", "Confidence (%)", True)
        curve("weight_norm_curve.png", [("weight_norm", "Classifier weight Frobenius norm")],
              "Classifier weight magnitude", "Norm")
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    image = ax.imshow(normalized, vmin=0, vmax=1)
    ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)),
           yticklabels=labels, xlabel="Predicted phase", ylabel="True phase",
           title=f"Selected checkpoint, epoch {best:.3f}\n{selection}")
    fig.colorbar(image, ax=ax, label="Fraction within true class")
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(plot_run(args.run))


if __name__ == "__main__":
    main()
