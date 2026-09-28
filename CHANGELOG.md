## 2026-09-28：对照组委会 FAQ（2026-09-27）

- Advanced 分数的汇总改为官方口径（FAQ Q7）：x 段计入，所有视频的 TP、FP、FN 先合并再算一个 F1，不再是逐视频 F1 的平均。`workflows.segment_counts` 返回 (TP, FP, FN)，`advanced.advanced_counts` / `pooled_f1` 取代 `advanced_scores`；`summary.csv` 的 `advanced_f1_at_75` 是合并值，`segment_metrics.csv` 仍给逐视频 F1。单元测试检查合并值 14/19 不等于平均值。
- 用保存的逐帧输出重算（不重跑训练）：`advanced_cv_01` 网络 0.687 → 0.690，最终 0.725 → **0.729**；其它运行高 0.000 到 0.007，先后不变。
- `cv --motion` 的帮助与文档注明只限 Advanced：运动特征要解码 30 fps 原视频，FAQ Q5(d) 禁止 Basic 使用评价视频文件本身。Basic 的最终方案不带 `--motion`，其余输入属于 Q5 (b)(c)。
- 核对新的 `submission_format/basic_challenge/submission.csv`：视频 02、07、11、15 共 24,507 行，表头 `filename,phase-id`，帧号都是 30 的倍数（四个视频都是 30.0 fps），每个视频恰有一段不少于 300 秒的空洞。顺序解码四个视频对照：列出的帧都是正常画面，未列出的 1 fps 帧全部是黑帧（最大像素不超过 7），即模板正好是非 x 帧。

## 2026-09-27：最短块长规则与 Advanced 官方口径评分

- `constraints.py`：每折除顺序规则外，再从训练视频的标签序列读出每个阶段的最短块长（按有图片的秒计），解码时每一段都不能更短。仍是同一个精确 Viterbi：进入阶段时一次走完它的最短块长（累积和），不增加状态；最短块长全为 1 时与原解码逐秒相同。太短的块会被补长或删掉，取总代价小的一种，所以标签对、只是偏短的块不会被删。
- 新增 `surgenet_dino/temporal/advanced.py` 与 `cv --annotations <annotation_csv 目录>`：按 Advanced 官方口径评分，30 fps 帧、x 作为一个类（x 帧在视频中涂黑，按精确检测处理），其余帧取最近的 1 fps 预测。`summary.csv` 多一列 `advanced_f1_at_75`，`segment_metrics.csv` 给出逐视频分数。仓库原有的 `f1_at_75` 跳过体外段（`81 | x | 81` 算一段），不等于官方口径。
- `cv` 的 `summary.csv` 增加"只用顺序规则"一行（`ordered`），`fold_metrics.csv` 增加 `ordered_accuracy`；新写出 `network_predictions.csv`（网络直接输出），以后的解码实验不必重跑网络。主输出仍是 `validation_predictions.csv`。
- `advanced_cv_01`（种子 42）：帧准确率网络 88.97%、只用顺序规则 89.32%、再加最短块长 **89.38%**；Advanced F1@0.75 依次 0.687、0.706、**0.725**（当天按逐视频平均；次日按 FAQ Q7 改为合并计数，最终为 0.729，见上一条）。
- 同期实测或判断的一份外部建议（CORAL 式全协方差白化无效、贪心合并短块降低帧准确率、训练视频数学习曲线等）见 [TEMPORAL_DESIGN_中文.md](docs/TEMPORAL_DESIGN_中文.md) 第六节第六轮；这些对比脚本不进仓库。
- 测试：最短块长的穷举精确性对照、Advanced 评分的单元测试，端到端测试覆盖 `--annotations`。
- 修复 `cv --motion` 的增广 bug（2026-09-25 引入）：运动特征接在标志位之后，`augment()` 却按"标志位在最后"取"这一秒没有图片"的标志，结果训练时几乎所有秒的逐帧概率被清零（保留 0%，不加运动时 92%）。现在运动特征放在标志位之前；不加运动时输入不变。`tests/test_temporal_motion.py` 新增两个在旧代码上失败的回归测试。修复后重测（种子 42）：最终 89.43%，基线 89.38%，Advanced F1 0.720 对 0.725；此前记录的运动特征 +0.4 ± 0.3 作废。

## 2026-09-25：可选的逐秒运动特征输入

- 新增 `tools/motion_features.py`：在与抽帧相同的独立 OpenCV 环境里顺序解码 30 fps 原视频，对每个已编目的帧取前后各 15 帧，写出 16 个帧差与 DIS 光流的幅值统计（含镜头整体运动与局部残差运动），每帧一行 CSV；带合成视频的 `self-test`。不读标签，训练环境不加依赖。
- 新增 `surgenet_dino/temporal/motion.py` 与 `cv --motion <csv>`：读入后 log1p、按视频无标签标准化，只拼进时序网络的输入（51 → 67 维），分类头与 PCA 不变；`result.json` 记录该文件的路径、sha256 与列名。缺省不启用，原有结果不受影响。新增 `tests/test_temporal_motion.py`。
- 按种子配对实测：种子 42 为 89.17% → 89.41%，种子 43 为 89.51% → 90.03%，两种子均值 89.34% → 89.72%；工程 8（81/82/83）相关错误一致减少约 15%，阶段 0/1/2/7/9 一致新增错误。基线换种子相差 0.34，同种子重跑只差 0.01。见 [TEMPORAL_DESIGN_中文.md](docs/TEMPORAL_DESIGN_中文.md) 4.3 与第六节第五轮（含未做的更高 fps、工程 8 独立网络、音频等思路及理由）。

## 2026-09-25：顺序规则约束解码

- 新增 `surgenet_dino/temporal/constraints.py`：每一折只从训练视频的标签序列读出两类规则（先后：a 在 b 开始前结束；单段：阶段只有连续一段），对验证视频做精确 Viterbi 约束解码，最大化网络逐秒对数概率之和。没有转移概率、时长或可调参数；网络输出不违反规则时等于 argmax。
- `cv` 的主输出改为约束解码结果。`summary.csv` 同时给出逐帧头、网络 argmax、约束解码三行；`fold_metrics.csv` 增加网络 argmax 准确率和规则改动秒数；`result.json` 记录每折读出的规则。规则改动标签的那些秒，网络最高类与解码类的概率互换，每行仍是一个分布。
- 新增 `tests/test_temporal_constraints.py`（含小规模穷举对照）；端到端测试证明规则只从训练视频读出。
- `final_cv_02`：网络 argmax 88.83% → 约束解码 **89.17%**，宏 F1 85.25%，最差视频 82.0%，段级 F1@0.75 0.528 → 0.564；在三份独立网络输出上分别 +0.30、+0.31、+0.34。
- 同期实测但未采用的方案（保持宽高比输入、活动辅助头、无标签 patch 组成与 VLAD、时长先验，以及器械时间线的前提检查）见 [TEMPORAL_DESIGN_中文.md](docs/TEMPORAL_DESIGN_中文.md) 第六节第四轮；这些对比代码未进入仓库。

## 2026-09-22：离线时序网络（最终方案）与代码清理

- 新增 `surgenet_dino/temporal/`：从同一个冻结 DINO 提取 [最后一层 CLS ‖ patch 均值] 特征并与原缓存逐行校验、按视频无标签标准化、留一视频交叉拟合的固定轮数线性头、带一层全局自注意力的膨胀时序卷积网络（直接 argmax）、由训练视频内层交叉验证选公共轮数的四折流程，以及只作诊断的学习曲线。
- 入口 `python -m surgenet_dino.temporal`（`cache-tokens`、`cv`、`curves`）与 `scripts/temporal.bat/.sh`、`scripts/plot_temporal.py`；配置 `configs/temporal.toml`，固定四折 `configs/cv4_reference.json`。原有 CLI、训练逻辑和特征缓存契约未改动。
- 结果与做过但未采用的方案（马尔可夫链、显式时长、时序 U-Net、合并分类头与时序网络、多层 CLS、序数辅助头等）见 [TEMPORAL_DESIGN_中文.md](docs/TEMPORAL_DESIGN_中文.md)。这些对比代码已全部移除。
- 新增本机只读的数据与预测查看器 `tools/viewer/`（`scripts/viewer.bat` 或 `python -m tools.viewer`）。它是独立的可视化辅助工具，只用标准库，核心训练代码不依赖它；Ruff 检查路径加入 `tools`。
- 移除此前未提交的 ImageAdapt（LoRA 单图适配）分支及其配置、脚本、示例、测试和文档：实测没有带来提升。

## 2026-09-21：独立数据审计与真实模型检查

- 完整图片解码、黑帧与 SHA256 检查移到 validate_data.bat/.sh；训练启动仅校验清单、标签与缓存身份，不逐张访问图片文件。
- 冻结特征训练、交叉验证、图像适配和特征提取共用新的元数据读取方式；不要求训练前重复生成审计报告。
- 真实 DINOv2 + LoRA 前向/反向检查通过。74,430 条真实缓存清单加载约 3.3 秒；新增自动检查覆盖无原图访问、独立审计和 BAT 暂停。

## 2026-09-20：固定组合方案与四折交叉验证

- 冻结 DINOv2，复用特征训练 Dropout → Linear 分类头；固定组合参数、轮内验证和早停见 [训练说明](docs/TRAINING.md)。
- 12 个已标注视频按整段视频进行四折，每张有效图片恰好作为验证样本一次；划分独立保存，输出各折与合并指标、模型和图表。
- 保留 train、retrain_head、crossval 启动脚本；Windows 成功或失败均暂停并保留退出码。
- 移除旧 compare、多配方、实验采样、特征归一化、显式 L2、梯度裁剪和交付验包入口。
- 清除旧交付包的重复文档、合成实验记录与缓存注入分支；参数校验统一使用 HeadOptions，修正 Bash 测试参数顺序，绘图脚本纳入 Ruff 检查。
- 原始 Linear 与实际使用过的历史组合头兼容预测；H 盘数据、特征、模型和历史结果全部保留。

四折实测结果及报告位置见 [TRAINING.md](docs/TRAINING.md)。当前结果是开发阶段交叉验证，不是独立最终测试成绩。
