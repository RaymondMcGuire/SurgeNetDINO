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
