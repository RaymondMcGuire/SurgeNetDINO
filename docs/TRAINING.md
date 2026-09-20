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
BAT 成功或失败都会暂停，输出退出码。读取缓存时仍完整检查来源图片和清单，因此启动核验有读盘耗时。
只核验缓存一次，各折共用核验结果，不重新抽帧、复制图片或运行 DINO。

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
