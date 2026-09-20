"""Plot completed cross-validation without changing metrics, models, or training."""

import argparse
import json
from pathlib import Path


def plot_crossval(run: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .plot_training_result import plot_run

    run = run.resolve()
    data = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if data.get("status") != "passed" or not data.get("coverage_exactly_once"):
        raise ValueError("Cross-validation must finish with complete held-out coverage")
    folds = data["folds"]
    matrix = np.asarray(data["pooled"]["confusion"])
    if matrix.sum() != data["samples"] or not np.isclose(
            np.trace(matrix) / matrix.sum(), data["pooled"]["accuracy"]):
        raise ValueError("Pooled metrics disagree with sample counts")
    output = run / "plots"
    output.mkdir(exist_ok=False)
    histories = []
    for fold in folds:
        child = run / f"fold_{fold['fold']}"
        result = json.loads((child / "result.json").read_text(encoding="utf-8"))
        histories.append(result)
        if not (child / "plots").exists():
            plot_run(child)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
    fig, axes = plt.subplots(len(folds), 2, figsize=(13, 3 * len(folds)), squeeze=False, layout="constrained")
    fig.suptitle("Fixed regularized head: grouped development cross-validation", fontsize=17)
    for row, fold, result in zip(axes, folds, histories):
        h = [r for r in result["history"] if r["epoch"] > 0]
        for ax, metric, ylabel, scale in [(row[0], "loss", "Ordinary CE (lower is better)", 1),
                                          (row[1], "accuracy", "Accuracy (%)", 100)]:
            for role, color in [("train", "#2876A6"), ("validation", "#D67827")]:
                ax.plot([r["epoch"] for r in h], [r[f"{role}_{metric}"] * scale for r in h],
                        label=role, color=color)
            ax.axvline(result["best_epoch"], color="#62717B", ls="--", label="Selected checkpoint")
            ax.set(title=f"Fold {fold['fold']} | validation videos {fold['validation_videos']}",
                   xlabel="Epoch (curves end at actual stop)", ylabel=ylabel)
            ax.grid(alpha=.18)
            ax.legend(fontsize=8)
    fig.savefig(output / "fold_curves.png", dpi=150)
    plt.close(fig)
    videos = data["pooled"]["per_video"]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    bars = ax.bar(list(videos), [m["accuracy"] * 100 for m in videos.values()], color="#2876A6")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set(ylim=(0, 105), xlabel="Held-out video", ylabel="Accuracy (%)",
           title="Each video evaluated once; selected using its fold's validation loss")
    ax.grid(axis="y", alpha=.18)
    fig.savefig(output / "video_accuracy.png", dpi=150)
    plt.close(fig)
    labels = data["labels"]
    values = np.asarray([[c["f1"] * 100 for c in result["validation"]["per_class"]]
                         for result in histories] + [[c["f1"] * 100 for c in data["pooled"]["per_class"]]])
    fig, ax = plt.subplots(figsize=(13, 4.5), layout="constrained")
    im = ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                color="white" if values[i, j] > 60 else "black", fontsize=9)
    ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(folds) + 1),
           yticklabels=[f"Fold {f['fold']}" for f in folds] + ["Pooled"],
           xlabel="Phase ID", title="Per-class F1 (%) | pooled row recomputed from all held-out predictions")
    fig.colorbar(im, ax=ax, label="F1 (%)")
    fig.savefig(output / "phase_f1.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    im = ax.imshow(matrix / matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels,
           xlabel="Predicted phase", ylabel="True phase", title="Pooled held-out confusion: row-normalized")
    fig.colorbar(im, ax=ax, label="Fraction within true class")
    fig.savefig(output / "pooled_confusion.png", dpi=150)
    plt.close(fig)
    lines = ["# 固定组合方案四折交叉验证", "", f"实验目录：{run}", "",
             "每折验证集用于早停和模型选择；以下是开发阶段交叉验证结果，不是独立最终测试成绩。", "",
             "| 折 | 验证视频 | 最佳轮 | 停止轮 | 准确率 | macro-F1 | NLL | ECE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for f in folds:
        lines.append(f"| {f['fold']} | {f['validation_videos']} | {f['best_epoch']:.3f} | "
                     f"{f['last_epoch']:.3f} | {f['accuracy']:.2%} | {f['macro_f1']:.2%} | "
                     f"{f['nll']:.4f} | {f['ece']:.2%} |")
    lines += ["", "各折等权均值 ± 样本标准差（不是置信区间）："]
    for key, stat in data["fold_statistics"].items():
        lines.append(f"- {key}: {stat['mean']:.5f} ± {stat['std']:.5f}")
    pooled = data["pooled"]
    lines += ["", f"合并 {data['samples']:,} 张逐帧验证预测：准确率 {pooled['accuracy']:.2%}，"
              f"macro-F1 {pooled['macro_f1']:.2%}，NLL {pooled['nll']:.4f}，ECE {pooled['ece']:.2%}。",
              "每张有效图片恰好验证一次；合并指标不等于各折指标的简单平均。", "",
              "各折的最佳与最后模型分别保留，没有自动选择一折作为最终提交模型。",
              f"达到训练上限的折：{[f['fold'] for f in folds if f['reached_epoch_limit']]}。本轮不自动延长训练。",
              "", "## 图表", "", "![训练曲线](plots/fold_curves.png)", "",
              "![逐视频准确率](plots/video_accuracy.png)", "", "![逐类别 F1](plots/phase_f1.png)", "",
              "![混淆矩阵](plots/pooled_confusion.png)"]
    with (run / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(plot_crossval(args.run))
