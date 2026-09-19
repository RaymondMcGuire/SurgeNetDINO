# SurgeNetDINO

基于上游 SurgeNetDINO 重构的模型加载与图像特征提取工程，使用 uv 管理环境。
支持 DINOv1-S/B、DINOv2-S/B/L、DINOv3-S/B/L，共八种模型。

**当前输出主要是图像特征向量，还没有手术阶段分类训练器、视频时序网络或比赛提交器。**
部署、代码结构、模型切换与验证结果见 [项目说明](docs/PROJECT_GUIDE.md)。

## Windows 快速运行

在项目根目录执行（默认后端 `cu128`，默认模型 `dinov2_vitb`）：

```bat
.\scripts\init.bat cu128
.\scripts\demo.bat cu128 dinov2_vitb
.\scripts\pretrained.bat cu128 dinov2_vitb
.\scripts\check.bat cu128
```

四个批处理无论成功还是失败，都会显示退出码并暂停，按任意键后退出。`0` 表示成功。
CPU 机器把第一个参数改为 `cpu`；切换 v3 把第二个参数改为 `dinov3_vitb`。
Linux 对应使用 `bash scripts/init.sh cu128` 等 `.sh` 脚本。

## demo 究竟做什么

| 入口 | 输入与权重 | 用途 |
|---|---|---|
| `demo` | 随机图像张量 + 随机权重 | 检查环境和网络能否前向计算 |
| `pretrained` | 随机图像张量 + 作者预训练权重 | 检查权重下载、兼容性和前向计算 |
| `encode` | 你的真实图片 + 预训练权重 | 保存图像特征，供后续分类等任务使用 |

默认 demo 生成一张 `[1,3,336,336]` 的随机输入，运行所选 Base 模型，检查 `[1,768]` 输出的形状和数值是否正常。
它不读取手术图片，不训练模型，也不会输出阶段名称或分类准确率。

提取真实图片特征：

```bash
uv run --locked --extra cu128 python -m surgenet_dino encode --config configs/dinov2_vitb.toml --image data/example.jpg --output outputs/example_features.pt
```

768 维表示一张图片被编码成 768 个数，不代表 768 个类别。
要预测手术阶段，还需要带标签的数据、经过训练的分类头和验证流程。

## 当前验证

2026-09-19 本机 Windows / Python 3.12.13 / torch 2.10.0+cu128 / timm 1.0.24：
**102 项测试通过**，Ruff 通过，DINOv2-Base、DINOv3-Base 的 RTX 5090 CUDA 随机权重 demo 通过。
官方预训练权重和下游分类准确率本轮未验证。具体边界见项目说明。
仓库已有真实 `uv.lock`；初始化优先使用现有锁文件，仅在缺失时生成。

## 来源

上游：[rlpddejong/SurgeNetDINO](https://github.com/rlpddejong/SurgeNetDINO)。
本工程是重构版本，不是作者的官方新版本。
代码与模型权重分别遵循 [LICENSE.txt](LICENSE.txt) 和 [LICENSE_MODELS.txt](LICENSE_MODELS.txt)。
论文归属与预训练配方链接保留在 [上游资料](docs/UPSTREAM_REFERENCES.md)。
