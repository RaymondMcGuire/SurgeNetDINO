# 冻结 DINOv2：按视频训练和验证

在项目根目录运行，或直接双击 scripts/train.bat：

```powershell
.\scripts\train.bat
# 可选负载档位
.\scripts\train.bat medium
.\scripts\train.bat high
```

默认 low。脚本先检查数据并缓存冻结特征，再训练线性分类头；成功或失败都显示退出码并暂停。
默认读取 `H:\dataset\JSCAS_experiments\splits\video_holdout_9_3_v1\split.json`：

| 分组 | 视频 | 图片数 |
|---|---|---:|
| 训练 | 01,03,05,06,09,12,13,14,16 | 55,779 |
| 验证 | 04,08,10 | 18,651 |

两组视频完全分离。验证图片只提取特征、计算指标，不参与梯度更新。
无公开标签的官方测试视频不参与本次实验；内部验证指标也不等于最终比赛成绩。

## 输入、训练和结果

入口必须是版本化 split.json，引用同目录的七列 train.csv / validation.csv 和预处理目录的 dataset.json。
程序检查清单哈希、类别覆盖、视频和图片内容互斥、两组并集等于预处理允许清单，
并逐张检查图片可读、哈希一致、没有整图 RGB 最大值 <= 1 的黑帧。不会扫描图片目录补入样本。
读取缓存训练时也重新核对数据来源，因此启动阶段需要磁盘读取时间。
数据约定与重新划分方法见 [DATASET.md](DATASET.md)。

模型配置为 configs/frozen_dinov2.toml：固定已验证的 SurgeNet DINOv2-Base 权重、336 输入、float32、GPU 0、seed 42。
DINO 全部冻结，先生成按训练行、验证行依次排列的 768 维特征缓存，再释放编码器。
分类头为 Linear(768,13)，仅训练集打乱；默认 20 轮、batch 64、AdamW、学习率 0.001、权重衰减 0.0001、普通交叉熵。
第 0 轮记录未训练基线；最佳模型从第 1～20 轮按验证交叉熵最低选取。

每次新建 `H:\dataset\JSCAS_experiments\frozen_dinov2\<时间戳>`，拒绝覆盖已有目录：

- cache/：features.pt、来源与划分快照、GPU 监测、运行参数和结果。
- train/head_best.pt、head_last.pt：验证损失最佳及最后一轮分类头。
- train/learning_curve.csv：每轮训练/验证损失、准确率、验证 macro-F1。
- train/result.json：最佳模型的总体、逐类、逐视频指标及混淆矩阵，保存重载一致性检查和 GPU 统计。
- train/validation_predictions.csv：最佳模型在验证集上的逐帧预测。top_score 为未经校准的 softmax 分数。

混淆矩阵行为真实类别、列为预测类别；macro-F1 固定对 13 类取平均，单视频缺少的类别记 0。
脚本没有断点续训；失败记录留在原目录，重跑请使用新目录。无需每次重新校准 GPU。

## 自定义入口

BAT 参数依次为负载档位、新实验目录、split.json、profiles.json，后三项可省略：

```powershell
.\scripts\train.bat low "H:\dataset\JSCAS_experiments\frozen_dinov2\run02" "H:\dataset\JSCAS_experiments\splits\video_holdout_9_3_v1\split.json" "H:\dataset\JSCAS_experiments\phase1_video01\20260920_phase1\calibration\profiles.json"
```

已有本版本特征缓存时可直接重新训练分类头，并用不同的新输出目录记录实验：

```powershell
uv run --locked --extra cu128 python -m surgenet_dino train-head --features "H:\dataset\JSCAS_experiments\frozen_dinov2\run02\cache\features.pt" --output-dir "H:\dataset\JSCAS_experiments\frozen_dinov2\head02" --gpu-profile low --epochs 20
uv run --locked --extra cu128 python -m surgenet_dino predict-head --head "H:\dataset\JSCAS_experiments\frozen_dinov2\run02\train\head_best.pt" --image "H:\path\image.png" --output-dir "H:\dataset\JSCAS_experiments\frozen_dinov2\prediction02"
```

四个 CLI 入口为 calibrate-gpu、cache-features --split、train-head --features、predict-head --head。
各自 --help 列出设备、batch、显存预算、计算节奏、温度阈值等覆盖参数。
旧单视频特征缓存不兼容新的划分契约，需要重新提取。

## 负载与历史校准

保留原先实测校准配置，其历史路径含 phase1_video01，不代表程序仍使用单视频训练。
当前 GPU 0 RTX 5090 的三个档位提取 batch 均为 8；预算/计算节奏分别为 low 20%/30%、medium 50%/60%、high 80%/80%。
这些是显存上限与批间节奏，不保证实际利用率或显存达到该比例。历史提取约 61 张/秒，
程序峰值已分配显存约 0.5 GiB；全量实验的磁盘速度与耗时需要本次运行测量。
默认 75°C 停止；监测异常、显存不足和非有限数值会停止。不会修改系统功率、频率或风扇。

换 GPU、模型、输入尺寸、精度或解冻编码器后，应重新校准。冻结提取的重新校准示例：

```powershell
uv run --locked --extra cu128 python -m surgenet_dino calibrate-gpu --manifest "H:\dataset\JSCAS_experiments\splits\video_holdout_9_3_v1\train.csv" --output-dir "H:\dataset\JSCAS_experiments\calibration_new"
```

解冻训练不属于当前实现，需要新的训练和校准流程。
历史单视频 4719 张的 99.81% 是训练集准确率，不是本次跨视频验证结果；历史实验产物保留以便追溯。

## 本次入口验证（2026-09-20）

116 项自动测试与 Ruff 通过。完整 74,430 行划分通过读取与身份检查。
实际运行新 BAT：从视频 01、04 各取 26 张真实图片，完成冻结特征、20 轮训练/验证、最佳模型保存重载一致性及单图预测，退出码 0。
记录位于项目 outputs/training_smoke_20260920_102247；这些少量图片只验证程序流程，不作为正式效果评估。
完整 9/3 视频训练现已完成，结果见下节。

## 完整训练结果与绘图（2026-09-20）

运行目录：`H:\dataset\JSCAS_experiments\frozen_dinov2\20260920_122914_522`。
20 轮完成；按验证损失最低选中的第 1 轮：训练准确率 94.96%，验证准确率 75.73%，验证 macro-F1 68.61%。
最后一轮训练准确率 98.25%、验证准确率 73.85%，验证损失由最佳 0.900 升至 2.318，存在过拟合迹象。

默认训练保存 CSV/JSON，不自动画图。已生成的 learning_curves.png 和 validation_diagnostics.png 位于该运行目录的 plots 下。
旧单视频 training_curve.png 保留在 phase1_video01/20260920_phase1 中。

绘图脚本已移至 scripts，使用独立环境，不修改训练依赖。对尚无 plots 目录的新实验运行：

```powershell
uv run --isolated --no-project --with matplotlib==3.11.2 python scripts/plot_training_result.py "H:\dataset\JSCAS_experiments\frozen_dinov2\新实验目录"
```

已有 plots 时拒绝覆盖；仅读取结果，不重新训练。
