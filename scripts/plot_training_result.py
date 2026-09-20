"""Plot saved training metrics only; does not load models or alter results."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run.resolve()
d = json.loads((run / "train/result.json").read_text(encoding="utf-8"))
assert d["status"] == "passed"
h = d["history"]
best = d["best_epoch"]
labels = d["labels"]
assert best == min(h[1:], key=lambda r: r["validation_loss"])["epoch"]
cm = np.asarray(d["validation"]["confusion"])
assert cm.shape == (len(labels), len(labels))
assert np.isclose(np.trace(cm) / cm.sum(), d["validation"]["accuracy"])
out = run / "plots"
out.mkdir(exist_ok=False)
font = Path("C:/Windows/Fonts/msyh.ttc")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    }
)
blue, orange = "#2374AB", "#D96B27"
epoch = [r["epoch"] for r in h]
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
fig.suptitle(f"冻结 DINOv2 分类训练 | 9 个训练视频 / 3 个验证视频\n{run.name}", fontsize=17)
for key, color, label in [("train_loss", blue, "训练集"), ("validation_loss", orange, "验证集")]:
    ax[0].plot(epoch, [r[key] for r in h], color=color, label=label, marker=".", lw=2)
ax[0].set(title="交叉熵损失（越低越好）", ylabel="损失", ylim=(0, None))
for key, color, label in [
    ("train_accuracy", blue, "训练集"),
    ("validation_accuracy", orange, "验证集"),
]:
    ax[1].plot(epoch, [100 * r[key] for r in h], color=color, label=label, marker=".", lw=2)
ax[1].set(title="分类准确率", ylabel="准确率 (%)", ylim=(0, 103))
ax[2].plot(
    epoch,
    [100 * r["validation_macro_f1"] for r in h],
    color=orange,
    marker=".",
    lw=2,
    label="验证集 macro-F1",
)
ax[2].set(title="各类别等权平均的 F1", ylabel="macro-F1 (%)", ylim=(0, 103))
for a in ax:
    a.axvline(best, color="#52616B", ls="--", alpha=0.7, label=f"选中模型：第 {best} 轮")
    a.set(
        xlabel="训练轮次（0 = 尚未训练）", xticks=range(0, max(epoch) + 1, max(1, max(epoch) // 10))
    )
    a.grid(alpha=0.18)
    a.legend(fontsize=9, loc="best")
fig.savefig(out / "learning_curves.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(
    1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.3, 1]}, layout="constrained"
)
fig.suptitle(
    f"最佳模型验证表现 | 第 {best} 轮（按最低验证损失选择）\n准确率 {d['validation']['accuracy']:.2%}  ·  macro-F1 {d['validation']['macro_f1']:.2%}  ·  {cm.sum():,} 张图片",
    fontsize=16,
)
norm = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
im = axes[0].imshow(norm, vmin=0, vmax=100, cmap="Blues")
axes[0].set(
    xticks=range(len(labels)),
    xticklabels=labels,
    yticks=range(len(labels)),
    yticklabels=labels,
    xlabel="预测阶段编号",
    ylabel="真实阶段编号",
    title="混淆矩阵：每行归一化 (%)",
)
for i in range(len(labels)):
    for j in range(len(labels)):
        if norm[i, j] >= 1 or i == j:
            axes[0].text(
                j,
                i,
                f"{norm[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 55 else "#22313F",
            )
fig.colorbar(im, ax=axes[0], shrink=0.75, label="同一真实类别中的占比 (%)")
classes = d["validation"]["per_class"]
values = [100 * r["f1"] for r in classes]
bars = axes[1].barh(range(len(labels)), values, color=[orange if v < 50 else blue for v in values])
axes[1].set(
    yticks=range(len(labels)),
    yticklabels=[f"{r['phase_id']}  (n={r['support']:,})" for r in classes],
    xlim=(0, 105),
    xlabel="F1 (%)",
    title="逐类别 F1 与验证样本量",
)
axes[1].invert_yaxis()
axes[1].bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
axes[1].grid(axis="x", alpha=0.18)
fig.savefig(out / "validation_diagnostics.png", dpi=160)
plt.close(fig)
print(out)
