# 当前组合方案与四折交叉验证

## 固定训练规则

冻结 SurgeNet DINOv2-Base，不修改图像处理或 768 维特征。头为 Dropout(0.1) → Linear(768,13)。
AdamW 学习率 0.0003、weight decay 0.05（bias 不衰减）、标签平滑 0.05、batch 64、seed 42。
配置见 configs/head_regularized.toml；覆盖优先级为显式 CLI > --head-config TOML > 相同的程序默认值。
不再有配方注册表或 --recipe / --candidates。当前只按验证交叉熵选择最佳模型。

训练上限 20 轮，每轮验证 4 次；连续 8 次未改善至少 0.0001 后早停，至少训练 1 轮。
停止时恢复最佳模型；严格更低的验证损失都可更新最佳模型，min_delta 只影响停止计数。
第 0 轮仅记录未训练基线，不参与最佳模型选择。
验证使用普通、无平滑、无类别加权的交叉熵，并关闭 Dropout；验证图片不参加梯度更新。

原组合实验第 6 轮停止，是早停触发，并非将训练上限设成了 6。当前不增加训练上限。

## 日常运行

```powershell
# 提取特征，再按原 9/3 划分训练
.\scripts\train.bat low
# 直接复用已有完整特征缓存
.\scripts\retrain_head.bat
# 全部 12 个已标注视频，四折训练并出图
.\scripts\crossval.bat
```

train.bat 参数依次为 PROFILE、NEW_RUN_DIR、SPLIT_JSON、PROFILES_JSON。
retrain_head.bat 参数为 FEATURES_PT、PROFILE、NEW_RUN_DIR。
crossval.bat 参数为 FEATURES_PT、PROFILE、NEW_RUN_DIR、NEW_SPLIT_DIR、PROFILES_JSON；均可省略。
默认特征为 `H:\dataset\JSCAS_experiments\frozen_dinov2\20260920_122914_522\cache\features.pt`。
默认 low，支持 medium/high；GPU 校准文件仍复用历史 phase1_video01 目录的 profiles.json。
显存预算和批间休息不是精确利用率目标；75°C、遥测异常、非有限数值、显存不足会停止计算。

四折新实验默认保存到 `H:\dataset\JSCAS_experiments\frozen_dinov2\crossval_regularized_<时间戳>`，
划分另存到 `H:\dataset\JSCAS_experiments\splits` 下的同名新目录。
BAT 成功或失败都会暂停，输出退出码。读取缓存只检查清单、标签、划分和模型身份，不逐张读取、探测或哈希原图。
各折共用已加载的缓存，不重新抽帧、复制图片或运行 DINO。图片完整检查已独立为下述手动操作。

## 独立数据检查

已预处理并检查过的数据可以直接训练；不要求每次训练前生成新的检查报告。
新数据、搬移或修改图片后，或者需要排查损坏/黑帧时，按需执行：

```powershell
.\scripts\validate_data.bat "H:\dataset\JSCAS_preprocessed\labeled_1fps_v1"
```

默认双击使用上面的目录。脚本读取 dataset.json / samples.csv，只检查清单内的图片，
逐张核对可读性、黑帧和 SHA256；报告写入该数据目录的 validation/<时间戳>。
第二个参数可指定新的报告目录。检查失败会保存 failure.json，成功或失败均暂停。
Linux 对应 `validate_data.sh DATASET_DIR [NEW_REPORT_DIR] [cpu|cu128]`。

日常训练仍保留格式、标签映射、视频互斥和缓存身份等轻量检查，避免错用数据；不提前扫描图片。
图像训练遇到缺失或损坏图片时，会在实际读到该图片时失败。缓存分类头训练无需原图在线。
原 audit-features 命令仍是显式完整审计入口，训练脚本不会自动调用它。

## 四折约定

使用缓存中的原训练与验证部分，共 12 个视频、74,430 张有效图片。原 9/3 划分保持原样，官方无公开标签的视频不参加。
固定 seed=42，按完整视频的类别计数平衡分组，不使用模型预测选组：

| 折 | 验证视频 | 训练图片 | 验证图片 |
|---|---|---:|---:|
| 1 | 01,10,14 | 55,507 | 18,923 |
| 2 | 04,06,13 | 56,374 | 18,056 |
| 3 | 05,12,16 | 55,251 | 19,179 |
| 4 | 03,08,09 | 56,158 | 18,272 |

各折其余 9 个视频训练，两组均覆盖全部 13 类。每个视频验证一次，所有帧的验证预测完整覆盖且不重复。
每折重新初始化头、优化器和随机种子，依次运行，不继承上一折权重。每折独立早停，曲线在实际停止处结束。
新划分目录包含 folds.json 和各折 train.csv、validation.csv、split.json；沿用 DATASET.md 的七列清单格式、来源身份和哈希核验。

任一折失败立即停止，留下 failure.json，不执行后续折，不发布成功汇总；不自动重试或调参。
当前无断点续训。重新执行使用新目录，不覆盖失败记录或历史实验。

## 输出与解读

每折保留 head_best.pt、head_last.pt、learning_curve.csv、result.json、training_options.json、
逐帧最佳与末轮验证预测、GPU 记录及数据诊断。原始 Linear 和已实际训练的无特征归一化 v2 分类头仍可预测。
已移除归一化的旧实验模型会明确报错，不会静默跳过变换。

总目录保存：

- fold_metrics.csv：各折准确率、macro-F1、NLL、ECE、最佳/停止轮次、更新数、是否达到上限。
- video_metrics.csv：每个视频作为验证集时的指标。
- validation_predictions.csv：各帧恰好一次的验证预测，含 fold 编号。
- result.json：各折等权均值和样本标准差，以及依据合并预测重新计算的指标与混淆矩阵。
- REPORT.md、plots：训练曲线、逐视频准确率、各类 F1、合并混淆矩阵；每折另有详细曲线。

macro-F1 固定对 13 类等权平均；macro_f1_present 仅统计该视频有真实样本的类别，两者分开标明。
折均值不等于合并样本指标，标准差也不是置信区间。
**每折验证集参与早停和模型选择，所以这里是开发阶段交叉验证，不能称作无偏的最终测试。**
不自动把某一折当作最佳部署模型，也不自动全量重训。到上限仍改善的折只在报告中标注，本轮不延长。

## CLI 与绘图

```powershell
uv run --locked --extra cu128 python -m surgenet_dino crossval-head --features "H:\dataset\JSCAS_experiments\frozen_dinov2\20260920_122914_522\cache\features.pt" --output-dir "H:\dataset\JSCAS_experiments\frozen_dinov2\cv_new" --split-output-dir "H:\dataset\JSCAS_experiments\splits\cv_new" --folds 4 --gpu-profile low
uv run --isolated --no-project --with matplotlib==3.11.2 python -m scripts.plot_crossval "H:\dataset\JSCAS_experiments\frozen_dinov2\cv_new"
```

BAT 自动执行第二步；直接 CLI 训练后须显式绘图。绘图使用独立环境，不修改训练依赖。plots 已存在时拒绝覆盖。
单个训练结果仍使用 scripts/plot_training_result.py，包括历史对比目录中的单个结果。
CPU 缓存训练可明确指定 --device cpu，此时不要提供 CUDA profiles。
Linux：retrain_head.sh FEATURES_PT PROFILE NEW_RUN_DIR [cpu|cu128]；
crossval.sh FEATURES_PT NEW_RUN_DIR NEW_SPLIT_DIR [PROFILE] [PROFILES_JSON] [cpu|cu128]。

预处理与数据版本约定见 [DATASET.md](DATASET.md)。历史单视频、9/3 对比的模型和图片均留在原 H 盘目录。

## 已完成的四折实验（2026-09-20）

结果与图表：[REPORT.md](H:/dataset/JSCAS_experiments/frozen_dinov2/crossval_regularized_20260920_212640/REPORT.md)。
独立划分目录：`H:\dataset\JSCAS_experiments\splits\video_cv4_20260920_212640`。

| 折 | 准确率 | macro-F1 | 最佳轮 | 停止轮 |
|---|---:|---:|---:|---:|
| 1 | 76.50% | 71.58% | 5.25 | 7.25 |
| 2 | 78.69% | 72.52% | 5.25 | 7.25 |
| 3 | 74.74% | 66.91% | 3.25 | 5.25 |
| 4 | 76.95% | 71.03% | 4.50 | 6.50 |

各折准确率均值 76.72%，样本标准差 1.62 个百分点；macro-F1 均值 70.51%，样本标准差 2.48 个百分点。
合并 74,430 张验证预测的准确率为 76.69%、macro-F1 为 70.97%、普通交叉熵为 0.7456、ECE 为 1.42%。
已独立核对每张有效图片恰好验证一次、标签与来源一致、各折独立初始化以及模型保存重载一致。
四折均提前停止，没有触及 20 轮上限；本轮不增加训练轮数。low 档 GPU 最高温度为 61°C。
这些仍是用于早停和模型选择的开发阶段交叉验证结果。

代码验证：Ruff 通过；测试套件 162 项通过、7 项跳过（Windows 上的 POSIX 脚本检查及训练环境缺少绘图库的检查）。
提交前清理后再次通过上述检查；另用 Git Bash 实际通过全部 SH 语法检查，以及 train、retrain_head、crossval 的 6 个成功/失败启动用例。
绘图已在独立 Matplotlib 环境实际执行并检查结果。

## 离线时序网络与顺序规则约束解码（最终方案，2026-09-22，约束解码 2026-09-25）

单帧方案停在 75% 到 77% 的原因是单帧信息上限，不是读出头的结构。`surgenet_dino/temporal/` 在冻结的 DINO 之上加入时间信息，不改主干。
网络输出最后按每折训练视频读出的规则做精确约束解码（`constraints.py`）：顺序规则，例如阶段 3 只有一段且之后不再出现 2，81 开始后不再出现 0 到 7；以及最短块长（2026-09-27 加入），每个阶段的每一段都不短于训练视频里它最短的一段。没有转移概率、时长分布或可调参数。
给 `cv` 加 `--annotations <官方 annotation_csv 目录>` 时，另外按 Advanced 官方口径评分：30 fps 帧上的段级 F1@0.75，x（体外，视频中涂黑）作为一个类并按精确检测处理；按组委会 FAQ（2026-09-27）Q7，x 段计入，所有视频的 TP/FP/FN 合并后算一个 F1，不是逐视频平均。
方法、无泄漏的评估协议、已知局限和做过但未采用的方案见 [TEMPORAL_DESIGN_中文.md](TEMPORAL_DESIGN_中文.md)。范围只限 12 个有标注视频的训练加验证。

```powershell
# 1. 从同一个冻结编码器提取 [最后一层 CLS | patch 均值] 特征（约 20 分钟，与原缓存校验通过才写入）
.\scripts\temporal.bat cache-tokens --features <features.pt> --output-dir <特征目录>
# 2. 四折交叉验证（约 45 分钟，单卡）；--annotations 可选，加上后给出 Advanced 官方口径分数
.\scripts\temporal.bat cv --features <features.pt> --tokens <特征目录> --annotations <annotation_csv 目录> --baseline <基线 csv> --output-dir <新目录>
# 2b. 可选，只限 Advanced（FAQ Q5(d)：Basic 不得使用原视频）：逐秒运动特征（独立 OpenCV 环境从 30 fps 原视频算，CPU 约 30 分钟），作为时序网络的附加输入
uv run --isolated --no-project --python 3.12 --with opencv-python-headless==5.0.0.93 python tools\motion_features.py extract --catalogue <labeled_1fps_v1\samples.csv> --source "<原视频根目录>" --output <motion 目录> --workers 6
.\scripts\temporal.bat cv --features <features.pt> --tokens <特征目录> --motion <motion 目录>\motion.csv --baseline <基线 csv> --output-dir <新目录>
# 3. 学习曲线诊断与出图（只画曲线，不做任何选择）
.\scripts\temporal.bat curves --features <features.pt> --tokens <特征目录> --output-dir <新目录>
uv run --isolated --no-project --with matplotlib==3.11.2 python scripts\plot_temporal.py <新目录> --reference-result <第 2 步的 result.json>
```

<!-- TEMPORAL_RESULTS:BEGIN -->
固定四折、完全嵌套（验证折不参与任何拟合、轮数选择或规则读取）的结果，`final_cv_02`：

| 折 | 验证视频 | 准确率（约束解码） | macro-F1 | 网络直接 argmax | 选定轮数 |
|---|---|---:|---:|---:|---:|
| 1 | 01,10,14 | 90.48% | 86.12% | 89.54% | 50 |
| 2 | 04,06,13 | 89.54% | 85.80% | 89.54% | 60 |
| 3 | 05,12,16 | 88.85% | 83.43% | 88.48% | 60 |
| 4 | 03,08,09 | 87.77% | 85.59% | 87.77% | 60 |

四折合并：逐帧分类头 76.97%，网络直接 argmax 88.83%，约束解码 **89.17%**（macro-F1 85.25%，最差视频 82.0%，每视频约 18 段，真实 16.8 段，段级 F1@0.75 从 0.528 升到 0.564）。相对已提交基线 76.69% 的配对提升为 +12.5 个点（+9.7 到 +15.4）。约束解码在三份独立的网络输出上分别提高 0.30、0.31、0.34 个点。
学习曲线显示验证集准确率约 20 轮后进入平台、到 120 轮没有下降，训练轮数由训练视频内层交叉验证选定，不需要再加 early stop。

加入最短块长后的当前最终方案，`advanced_cv_01`（种子 42，同一份网络输出上对比）：帧准确率网络直接 argmax 88.97%、只用顺序规则 89.32%、再加最短块长 **89.38%**；Advanced 官方口径 F1@0.75（所有视频合并计数）网络直接 argmax 0.690、最终 **0.729**（逐视频平均依次为 0.687、0.706、0.725）。各折 90.44 / 89.77 / 89.07 / 88.23%。仓库原有的 `f1_at_75` 跳过体外段，不等于官方口径。细节见 [TEMPORAL_DESIGN_中文.md](TEMPORAL_DESIGN_中文.md) 4.4 与第六节第六轮。
<!-- TEMPORAL_RESULTS:END -->

可选的运动特征（`--motion`，2026-09-25）：`tools/motion_features.py` 对每个已编目的帧取原视频前后各 15 帧，算帧差和 DIS 光流的 16 个幅值统计，log1p 后按视频标准化，只拼进时序网络的输入，分类头和 PCA 不变。按种子配对：种子 42 为 89.17% → 89.41%（+0.24），种子 43 为 89.51% → 90.03%（+0.52）；两种子均值 89.34% → 89.72%。基线换种子本身相差 0.34，所以这是约 +0.4 ± 0.3 的小而一致的提升：工程 8（81/82/83）相关错误减少约 15%，阶段 0/1/2/7/9 有一致的新增错误。细节见 [TEMPORAL_DESIGN_中文.md](TEMPORAL_DESIGN_中文.md) 4.3。
**2026-09-27 更正：** 以上数字是在增广 bug 下测得的（运动特征接在标志位后面，训练时逐帧概率几乎全被清零）。修复后种子 42 配对重测：最终 89.38% → 89.43%，Advanced F1 0.725 → 0.720（合并计数 0.729 → 0.724），运动特征只是很小、不确定的输入。
**2026-09-28：** 运动特征要解码原视频，组委会 FAQ Q5(d) 规定 Basic 不得使用评价视频文件本身，所以它只能用于 Advanced；Basic 的最终方案不带 `--motion`。

预测是离线的：输入是同一个视频全部帧的序列，标准化用整段视频的统计量，时序网络会读取未来帧，不能对孤立的单张图片预测。
这些仍是同样 12 个视频上的开发估计：嵌套保护了训练轮数的选择和顺序规则的读取，但不保护方法本身的设计。
