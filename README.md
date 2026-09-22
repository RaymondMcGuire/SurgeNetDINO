> **手术阶段识别的最终方案**：冻结的 DINO 特征，加按视频标准化、交叉拟合的线性头和一个带全局自注意力的小型离线时序网络，直接 argmax 输出。方法、无泄漏的评估协议和实测结果见 [docs/TEMPORAL_DESIGN_中文.md](docs/TEMPORAL_DESIGN_中文.md)，入口为 `scripts\temporal.bat`。

# SurgeNetDINO

SurgeNetDINO 的模型加载、冻结图像特征和手术阶段分类工程，使用 uv 管理环境。
支持八种 DINO 架构；当前分类基线使用已验证的 SurgeNet DINOv2-Base 预训练权重。

训练采用固定组合：Dropout(0.1) → Linear(768,13)，AdamW lr=0.0003、weight decay=0.05（仅 weight）、label smoothing=0.05、batch=64、seed=42。
最多 20 轮，每轮验证 4 次，连续 8 次无足够改善后早停。按普通验证交叉熵保存最佳模型。

## Windows 入口

```powershell
.\scripts\init.bat cu128
.\scripts\check.bat cu128
.\scripts\train.bat low
.\scripts\retrain_head.bat
.\scripts\crossval.bat
```

- train：检查原 9/3 划分，提取冻结特征后训练。
- retrain_head：复用已有特征，训练同一组合方案。
- crossval：复用全部 12 个已标注视频的特征，依次进行四折（每折 9 训练、3 验证），汇总并绘图。

三个训练入口默认 GPU 0、low 档；成功或失败均暂停并显示退出码。新运行和划分使用新目录，不覆盖历史结果。
训练启动只检查清单和缓存元数据，不再逐张核验原图。需要完整数据审计时，单独运行 `scripts\validate_data.bat`；默认检查已预处理数据目录并保存报告，详见训练说明。
当前只保留一个训练方案，旧 compare、多配方、实验采样、归一化与显式 L2 入口已移除。

[训练及交叉验证说明](docs/TRAINING.md) · [数据与划分](docs/DATASET.md) · [基础架构](docs/PROJECT_GUIDE.md) · [时序解码设计](docs/TEMPORAL_DESIGN_中文.md)

交叉验证中每折验证数据用于早停和模型选择，结果属于开发阶段验证，不能称作独立最终测试。
原始图片、缓存、模型和历史结果图保存在 H 盘。早期对比实验结果保留，清理代码不删除实验数据。

## 数据可视化查看

双击 `scripts\viewer.bat`，或运行 `python -m tools.viewer`，打开本机手术阶段查看器（独立的可视化辅助工具，位于 `tools/viewer/`，不属于核心代码）。
「人工标注」与「模型预测」分开浏览，支持阶段时间轴、原始帧号／时间定位、排除帧查看和预测真值对照。
只读 H 盘数据，无需 GPU 或新增依赖；启动与预测接入方式见 [查看器说明](tools/viewer/README.md)。

## 环境与特征检查

```powershell
.\scripts\demo.bat cu128 dinov2_vitb
.\scripts\pretrained.bat cu128 dinov2_vitb
uv run --locked --extra cu128 python -m surgenet_dino encode --config configs/dinov2_vitb.toml --image data/example.jpg --output outputs/example_features.pt
```

demo 使用随机输入和随机权重，pretrained 检查作者权重，两者都不是分类训练。
encode 保存图像特征；predict-head 加载已训练分类头预测阶段。

## 来源

上游：[rlpddejong/SurgeNetDINO](https://github.com/rlpddejong/SurgeNetDINO)。本工程是重构版本。
代码和模型许可分别见 [LICENSE.txt](LICENSE.txt)、[LICENSE_MODELS.txt](LICENSE_MODELS.txt)。
论文与预训练资料见 [UPSTREAM_REFERENCES.md](docs/UPSTREAM_REFERENCES.md)。
