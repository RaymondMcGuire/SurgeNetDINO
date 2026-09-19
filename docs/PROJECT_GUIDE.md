# SurgeNetDINO 项目说明

更新：2026-09-19；工程版本：0.3.0。

## 1. 项目现在在做什么

这是 SurgeNetDINO 的运行工程重构：把上游模型创建与权重加载整理成统一的注册表、TOML 配置、Python API 和命令行，使用 uv 固定依赖。
模型计算仍使用 timm 的原生实现，没有重写 Transformer、attention 或 DINO 训练算法。

已实现：八种模型选择、作者权重下载、本地权重读取与校验、CPU/CUDA 检查、合成前向/反向检查、单图特征提取、参数冻结开关。
未实现：数据集读取与划分、监督分类训练循环、时序网络、分类指标评价、比赛提交文件、大规模自监督预训练。

## 2. DINO、特征与分类的关系

这里的 DINOv1/v2/v3 是通过自监督学习获得图像表征的模型系列；SurgeNetDINO 提供面向手术图像的预训练权重。
自监督预训练让模型学习视觉特征，并不自动赋予模型你的手术阶段名称。
相关背景见[原论文](https://proceedings.mlr.press/v315/jong26a.html)和 [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)。

当前默认流程：**图片 → DINO 编码器 → 特征向量**。
未来的阶段分类流程：**图片 → DINO 编码器 → 用阶段标签训练的分类头 → 各阶段分数 / 阶段名称**。

- `num_classes=0`：输出 `[B,D]` 特征。B 是图片数量，D 是特征维数。
- `num_classes=C>0`：创建 C 类分类头，输出 C 个原始分数（logits），不是自动得到已训练分类器，也不是直接得到概率。
- 使用作者的主干权重时，新分类头仍然随机，必须训练。
- 已有兼容的完整分类器权重时，可用 `weights="local"`、`weights_scope="full"` 恢复主干和分类头；当前 CLI 仍不提供阶段名称映射。

默认 Base 模型的 `[1,768]` 表示一张图的 768 个特征数值，不是 768 类或预测未来 768 帧。
当前程序不输出检测框、分割掩码或手术阶段标签；这些需要对应任务头及训练/评价代码。

## 3. demo 的真实执行过程

```bat
.\scripts\demo.bat cu128 dinov2_vitb
```

1. 检查后端参数与 `uv.lock`，用 uv 启动项目环境。
2. 通过 `smoke --weights none` 创建一个随机初始化的 DINOv2-Base；不下载模型权重。
3. 生成随机张量 `[1,3,336,336]`，即一张三通道、336×336 的合成输入，不读取图片目录。
4. 切换到 eval 模式，执行一次前向计算，不计算梯度，不更新参数。
5. 检查输出为 `[1,768]`，且没有 NaN 或无穷大，记录耗时和 CUDA 峰值已分配显存。
6. 写入 `outputs/dinov2_vitb/environment_smoke.json`，打印结果、退出码并暂停。

成功只说明环境和网络计算正常。随机权重的输出不应当用作手术识别特征。
`backward: false` 表示没有反向传播；`challenge_accuracy_measured: false` 表示没有计算分类准确率。
报告内的预处理标识用于记录真实图像编码约定；smoke 的随机输入不会读取图片或执行图片预处理。

| 命令 | 使用的输入 | 权重 | 结果 |
|---|---|---|---|
| `demo.bat` | 随机张量 | 随机 | 环境检查 JSON |
| `pretrained.bat` | 随机张量 | 作者预训练权重 | 权重加载/前向检查 JSON |
| `encode` | 指定真实图片 | 作者或本地权重 | 包含特征与元数据的 `.pt` |
| `smoke --backward` | 随机张量 | 所选权重 | 额外检查合成损失的梯度，没有 optimizer step |

`pretrained` 也不测识别准确率。快速脚本会覆盖同一模型的同名报告；正式实验应另设输出目录。

## 4. 环境和常用命令

固定起点：Python 3.12（允许 >=3.12,<3.14）、torch 2.10.0、torchvision 0.25.0、timm 1.0.24、uv >=0.10.0。
`cu128` 与 `cpu` 是互斥的依赖选项，每次 uv sync/run 都选同一个，不使用 `--all-extras`。

Windows，在项目根目录运行：

```bat
.\scripts\init.bat cu128
.\scripts\check.bat cu128
.\scripts\demo.bat cu128 dinov2_vitb
.\scripts\pretrained.bat cu128 dinov2_vitb
```

`init` 安装 Python、同步依赖并检查环境；已有 `uv.lock` 时使用现有锁，缺失时才联网生成。
修改依赖后显式运行 `uv lock` 并审查差异，不创建空锁文件。安装 uv 见[官方说明](https://docs.astral.sh/uv/getting-started/installation/)。
`check` 检查依赖导入、Ruff 和测试；要求 CUDA 执行时另外使用 `doctor --require-cuda`（CUDA 版 init 已包含）。

四个 `.bat` 在成功、参数错误、依赖错误等正常退出路径都会暂停，显示 `Script finished with exit code ...`。
`0` 表示成功，非零表示失败，按键后保留原退出码。
Linux 使用对应的 `bash scripts/init.sh cu128`、`check.sh`、`demo.sh`、`pretrained.sh`，不暂停。
CPU 开发机器将脚本的后端参数改成 `cpu`。

```bash
# 配置预览：命令本身不构造模型、不下载权重
uv run --locked --extra cu128 python -m surgenet_dino list
uv run --locked --extra cu128 python -m surgenet_dino plan --config configs/dinov3_vitb.toml

# CUDA 运算检查
uv run --locked --extra cu128 python -m surgenet_dino doctor --require-cuda

# 合成反向检查，不是数据集训练
uv run --locked --extra cu128 python -m surgenet_dino smoke --config configs/dinov2_vitb.toml --backward --report outputs/v2_backward01/report.json

# 把真实图片编码为特征，需先提供 data/example.jpg
uv run --locked --extra cu128 python -m surgenet_dino encode --config configs/dinov2_vitb.toml --image data/example.jpg --output outputs/v2_run01/features.pt
```

图片预处理：按 EXIF 转正、转 RGB、bicubic 缩放至模型输入尺寸、转换张量、按 ImageNet mean/std 归一化。
`.pt` 包含 `features` 和 `metadata`，其中记录架构、权重 SHA-256、timm 版本、预处理和 `feature_identity`。
命令默认拒绝覆盖已有文件，需要覆盖时显式添加 `--overwrite`。
`encode` 拒绝随机权重和非零 `num_classes`，避免混淆随机输出、特征和分类分数。

## 5. 模型切换和配置

| 模型名 | 输入尺寸 | patch | 特征维数 |
|---|---:|---:|---:|
| `dinov1_vits` | 224 | 16 | 384 |
| `dinov1_vitb` | 224 | 16 | 768 |
| `dinov2_vits` | 336 | 14 | 384 |
| `dinov2_vitb` | 336 | 14 | 768 |
| `dinov2_vitl` | 336 | 14 | 1024 |
| `dinov3_vits` | 336 | 16 | 384 |
| `dinov3_vitb` | 336 | 16 | 768 |
| `dinov3_vitl` | 336 | 16 | 1024 |

八种模型分别有 `configs/<模型名>.toml`。`offline_smoke.toml` 是 CPU 随机权重检查配置。
切换 v3：`scripts\demo.bat cu128 dinov3_vitb`；加载 v3 作者权重：`scripts\pretrained.bat cu128 dinov3_vitb`。
无需复制代码或另建环境，但必须使用所选模型对应的权重。

配置优先级：显式 CLI 参数 > TOML > 程序默认值。

```toml
[model]
name = "dinov3_vitb"
weights = "surgenet"
num_classes = 0
weights_scope = "backbone"
device = "cuda"
cache_dir = ".cache/huggingface"
revision = "main"
offline = false
seed = 0
```

`weights` 三种选择：`surgenet` 下载作者权重，`local` 读取 `checkpoint`，`none` 随机初始化。
`weights_scope="backbone"` 丢弃旧分类头，要求主干键及形状完整匹配；`full` 要求兼容的完整本地任务权重。
下载失败不会退回随机权重；读取采用 `torch.load(weights_only=True)`；可配置 `sha256` 校验与 `checkpoint_key` 容器路径。
更换模型时不会自动复用旧配置中绑定的 checkpoint、SHA-256 或 checkpoint_key。

v2/v3 的 768 维特征含义不同。换编码器、权重或预处理后，需要重新提取特征并训练/验证下游头。
正式实验固定报告中的 HF commit 和 SHA-256，并记录数据划分、源码版本、依赖锁、采样策略及 seed。

## 6. 代码结构

| 路径 | 职责 |
|---|---|
| `surgenet_dino/registry.py` | 八种架构、输入尺寸、特征维数、权重文件名 |
| `surgenet_dino/config.py` | 配置读取与合法性检查 |
| `surgenet_dino/model.py` | 统一 timm 工厂与参数冻结开关 |
| `surgenet_dino/weights.py` | 下载、本地读取、哈希和权重兼容性校验 |
| `surgenet_dino/runtime.py` | 设备检查与图片预处理 |
| `surgenet_dino/provenance.py` | 特征空间来源标识 |
| `surgenet_dino/cli.py` | list / plan / doctor / smoke / encode |
| `load_models.py` | 保留旧文件入口，转调新 CLI |
| `examples/feature_api.py` | 直接使用 Python API 提取单图特征的示例 |
| `scripts/` | Windows / Linux 启动脚本 |
| `tests/` | 配置、加载、CLI、真实 timm、Windows 批处理回归检查 |

Python API 通过 `build_model(config)` 返回 CPU 模型和加载报告；调用方负责 seed、device 和 train/eval 状态。
`set_trainable(model, "none"/"head"/"all")` 只设置梯度开关，不包含训练循环。
示例：

```bash
uv run --locked --extra cu128 python -m examples.feature_api --config configs/dinov3_vitb.toml --image data/example.jpg
```

## 7. 本轮清理与验证

清理了九个一次性交付文件：源码哈希清单及其验包脚本、机器可读的旧交付状态、重复测试报告，以及五份旧环境验证产物。
来源归档哈希和论文归属保留在 [UPSTREAM_REFERENCES.md](UPSTREAM_REFERENCES.md)。
删除验包测试、旧迁移文件存在性测试，以及重复的模型构造/配置加载测试；保留实际加载、错误处理、特征、真实 timm 验证。
删除脚本中未使用的 DEVICE/device 变量与 smoke 之前重复的 plan 调用；同步更新 CI 的 Ruff 路径。

2026-09-19 本机实际验证：

- Windows、Python 3.12.13、torch 2.10.0+cu128、torchvision 0.25.0+cu128、timm 1.0.24。
- `python -m pytest -q`：**102 passed**；其中 Windows 批处理用替代外部命令验证成功、非法后端、缺少锁、找不到 uv、外部命令失败的暂停和退出码行为。
- `ruff check surgenet_dino tests examples load_models.py`：通过。
- `doctor --require-cuda`：通过，检测到两张 RTX 5090。demo 默认使用第一张，不是多卡训练。
- 实际运行 `demo.bat cu128 dinov2_vitb` 和 `demo.bat cu128 dinov3_vitb`：均输出 `[1,768]`，退出码 0，并显示暂停提示。
- 三种 Small 模型的真实 timm 前向等价测试和八种模型注册检查通过；不代表八种架构均做了 CUDA/预训练权重验证。

本轮未重新安装完整环境，未运行真实官方权重下载，未验证真实图片的官方权重特征质量、训练准确率、RTX 6000 Ada 或远程 GitHub Actions。
批处理中的 init/pretrained 外部操作通过替代命令检查控制流程，不声称实际下载/安装已经执行。
历史交付环境的 `90 passed, 11 skipped` 是旧记录，不能代表当前机器；当前结果也不等于模型识别准确率。

下一步若要做阶段分类，需要先确定数据、类别和按手术划分的训练/验证集，再实现分类头训练与评价；时序识别还需要连续帧输入和时序模型。
原始大规模预训练配方与许可说明见 [UPSTREAM_REFERENCES.md](UPSTREAM_REFERENCES.md)。
