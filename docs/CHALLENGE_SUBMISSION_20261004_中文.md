# Challenge 提交准备

日期：2026 年 10 月 4 日。研究依据见 [实验记录](CHALLENGE_EXPERIMENTS_20261004_中文.md)。算法决策、已完成的产物与尚未执行的步骤分别按真实状态记录；本文不代表已提交比赛。

本文引用的 `outputs/` 产物和隔离 OpenCV 环境位于本机，由 Git 忽略；代码与文档的提交不包含模型、数据、缓存或正式比赛提交。其他机器需准备相应模型与数据路径，并使用已有的 OpenCV 环境运行像素扫描。

## 当前准备状态

### 2026-10-22：Basic 双击入口

双击 `scripts\basic_challenge.bat`，顺序执行：环境检查 → 模板指定帧抽取 → 冻结特征提取 →
加载最终模型推理 → 按模板原序生成完整 `submission.csv`。脚本不重训、不发送提交。
默认源目录为 `H:\dataset\JSCAS AI Challenge`，模板位于其下
`jscas_ai_challenge_2026_data1\jscas_ai_challenge_2026_data1\submission_format\basic_challenge\submission.csv`。
最终模型使用 `outputs/challenge_final_20261004/checkpoint.pt`；结果位于
`outputs/basic_challenge/<时间戳>/predictions/submission.csv`，窗口会打印具体路径和退出码。
仅保存模板列出的 24,507 张原尺寸 PNG，不读取标注，也不计算更密帧的特征；需预留图片存储空间。
抽帧复用本机已有 `.cache/uv/archive-v0/SruTPdM1WCWkIu0S/Scripts/python.exe` 的 OpenCV，
模型使用 `.venv/Scripts/python.exe`；缺少环境或文件会报错停止，不自动安装依赖。
GPU 沿用 low 档、温度上限 70°C、CPU 2 线程。已有目录不覆盖；失败后保留产物便于排查。
也可运行 `scripts\basic_challenge.bat "源目录" "新结果目录"` 指定路径。
此入口已做合成抽帧及启动流程检查；尚未执行真实四视频预测。

### 迁移到 Linux

Git 保存代码、配置、依赖锁、测试和文档，不包含 `.venv`、`.cache`、`outputs` 或 H 盘数据。
需另外复制最终模型 `outputs/challenge_final_20261004/checkpoint.pt`、DINO 权重缓存
`.cache/huggingface`，以及四个评价视频和官方模板；仅克隆仓库不会恢复这些文件。
Windows 虚拟环境不能复用，在 Linux 用 `bash scripts/init.sh cu128` 重建模型环境；
OpenCV 仍需单独准备。`basic_challenge.bat` 是 Windows 入口，Linux 可依次运行相同的
`python -m tools.basic_extract`、`python -m surgenet_dino.temporal.submission cache-images` 和
`python -m surgenet_dino.temporal.submission predict`，将 H 盘路径替换为 Linux 实际路径。
已有 `features.pt` 内的训练图片绝对路径也需单独处理；仅做最终模型推理无需重用训练缓存。

以下保留 2026-10-04 的准备记录：

现有 H 盘档案和仓库没有找到这版时序算法的正式 `submission.csv` / `submission.zip`。官方 Basic `submission.csv` 是 24,507 行的空白模板，不能当成已完成预测。评价视频为 02、07、11、15；它们尚未形成预处理图片和时序特征缓存。

新增 [submission.py](../surgenet_dino/temporal/submission.py) 复用现有标准化、交叉拟合头、PCA、时序网络和精确约束解码，提供三个入口：

| 入口 | 实际功能 |
| --- | --- |
| `fit-final` | 仅在 12 个有标注视频上拟合；保存全训练分类头、训练侧 PCA、全部 5 个时序网络、解码规则、编码器身份与选择来源 |
| `cache-images` | 使用同一冻结编码器，仅提取官方 Basic 模板列出的 1 fps 图片；单独保存无标签缓存 |
| `predict` | 从已保存组件恢复无标签推理；按模板精确输出 Basic；提供真实像素 x 掩码后输出 Advanced zip |

轮数规则固定为：按各折拟合帧数加权合并既有训练侧内层准确率曲线，选择最大值，平分时取更早轮数。来源为 seed 42 的 `advanced_cv_01/result.json`，选择 **60 轮**；不读取外层得分或官方评价标签来选择。记录见 [training_selection.json](../outputs/submission_preparation_20261004/training_selection.json)。最终重训默认仍用原始损失和网络骨架；Advanced 导出额外应用固定后验边界校正；两组完整四折各有约 1.4～1.7 个 F1 百分点的开发增益，仍有统计与迁移不确定性。Basic 不应用边界校正。

5 项 CPU 检查覆盖：所有训练视频上的小规模拟合与组件保存；模型重载后预测逐值相同且没有标签输入；Basic 行序与完整覆盖；Advanced 最近帧时间轴、真实 x 覆盖、严格起始帧格式和不合格掩码拒绝；边界规则不跨空洞且保持训练侧最短段长。使用合成小模型，不把检查称为实际 12 视频最终模型训练或官方推理。可运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_temporal_submission.py -q
```

最终模型已生成：[checkpoint.pt](../outputs/challenge_final_20261004/checkpoint.pt)，约 6.4 MiB；训练覆盖全部 12 个有标注视频、74,430 个有效样本，seed 42、60 轮。全训练头、PCA、5 个网络和规则已严格重载，5 个网络状态不同，有限参数、PCA 正交性、交叉拟合隔离和 SHA256 检查均通过。[运行记录](../outputs/challenge_final_20261004/result.json)、[重载核验](../outputs/challenge_final_20261004/verification.json)。GPU low 约 573.21 秒、平均利用率 12.77%、最高 62°C；没有官方评价数据进入拟合。仍没有生成或发送正式的评价预测文件。

## 顺序执行

以下都是本地准备操作。`fit-final` 已执行；原样再次执行会拒绝已有目录，需要重训时另取新目录。其余评价步骤尚未执行。输出目录必须是新目录。脚本默认 CPU 2 线程、GPU low、duty cycle 0.3、memory fraction 0.2、温度上限 70°C；不同时运行多项 GPU 作业。

```powershell
.\.venv\Scripts\python.exe -m surgenet_dino.temporal.submission fit-final `
  --features 'H:\dataset\JSCAS_experiments\frozen_dinov2\20260920_122914_522\cache\features.pt' `
  --tokens 'H:\dataset\JSCAS_experiments\temporal\tokens_01' `
  --selection-report 'H:\dataset\JSCAS_experiments\temporal\advanced_cv_01\result.json' `
  --output-dir 'outputs\challenge_final_20261004'
```

模型得到以后，先用官方抽帧脚本在工作区生成评价视频的 1 fps 原图，保持 `data/<video>/images/<video>_<frame>.png`。不在 H 盘覆盖原数据，不把评价图像加入任何拟合、伪标签、自监督训练或模型选择。这里只生成主办方允许的 1 fps 图片；Basic 特征不读更密视频帧。

```powershell
.\.venv\Scripts\python.exe -m surgenet_dino.temporal.submission cache-images `
  --checkpoint 'outputs\challenge_final_20261004\checkpoint.pt' `
  --template 'H:\dataset\JSCAS AI Challenge\jscas_ai_challenge_2026_data1\jscas_ai_challenge_2026_data1\submission_format\basic_challenge\submission.csv' `
  --image-root 'outputs\challenge_evaluation_1fps\data' `
  --output-dir 'outputs\challenge_evaluation_tokens_20261004'
```

Advanced 的 x 由 [challenge_black_scan.py](../tools/challenge_black_scan.py) 从完整原视频像素提取：最大 RGB ≤16、至少 60 帧的连续黑段。检测器不读取标注或 Basic 模板；仅在检测完成后用训练标签核验。复用已安装的隔离 OpenCV 环境，没有为主训练环境增加依赖。训练视频 01 的完整扫描已核验 10 个 x 段全部匹配；还有 9 个非 x 帧被黑段末尾覆盖。扫描约 492.96 秒，没有读取标注做检测。实际评价视频尚未扫描。证据见 [整视频核验](../outputs/submission_preparation_20261004/x_video01_verification.json)。

```powershell
& '.cache\uv\archive-v0\SruTPdM1WCWkIu0S\Scripts\python.exe' -m tools.challenge_black_scan `
  --videos 'H:\dataset\JSCAS AI Challenge\jscas_ai_challenge_2026_data2\jscas_ai_challenge_2026_data2\HD_videos\02.mp4' `
           'H:\dataset\JSCAS AI Challenge\jscas_ai_challenge_2026_data4\jscas_ai_challenge_2026_data4\HD_videos\07.mp4' `
           'H:\dataset\JSCAS AI Challenge\jscas_ai_challenge_2026_data6\jscas_ai_challenge_2026_data6\HD_videos\11.mp4' `
           'H:\dataset\JSCAS AI Challenge\jscas_ai_challenge_2026_data9\jscas_ai_challenge_2026_data9\HD_videos\15.mp4' `
  --output 'outputs\challenge_evaluation_x_20261004.json'
```

```powershell
.\.venv\Scripts\python.exe -m surgenet_dino.temporal.submission predict `
  --checkpoint 'outputs\challenge_final_20261004\checkpoint.pt' `
  --images 'outputs\challenge_evaluation_tokens_20261004\images.pt' `
  --template 'H:\dataset\JSCAS AI Challenge\jscas_ai_challenge_2026_data1\jscas_ai_challenge_2026_data1\submission_format\basic_challenge\submission.csv' `
  --x-masks 'outputs\challenge_evaluation_x_20261004.json' `
  --output-dir 'outputs\challenge_submissions_20261004'
```

省略 `--x-masks` 时只输出 Basic，不用标注或模板空洞代替真实 x 检测。Advanced 从相邻 1 fps 预测的中点扩展到 30 fps，等距帧取较早预测，再覆盖像素黑段；输出包含 `x`，不将参考标注中的 `e` 当成预测阶段。

## 交付要求和方法说明草稿

Basic 使用 `filename,phase-id`，按官方模板原序输出 24,507 行。Advanced 是 `submission.zip`，包含四个 `submission_<video>.csv`，表头为 `start-frame,phase-id`，起始帧严格递增。两项还需要简短方法稿和代表人信息，格式以主办方要求为准。来源：[Basic 官方提交说明](https://github.com/modafone/jscaschallenge2026/blob/main/basic_challenge/readme.md)、[Advanced 官方提交说明](https://github.com/modafone/jscaschallenge2026/blob/main/advanced_challenge/readme.md)。

方法正文可从以下已经实现的事实整理：冻结公开手术预训练 DINOv2，在 336×336 输入上提取 CLS 与 patch 均值；同视频无标签标准化；训练视频留一交叉拟合线性分类头生成时序训练输入；训练侧 PCA 32 维、分类对数概率和 6 个时间标志输入小型卷积与全局注意力时序网络；5 个种子集成、训练侧顺序和最短块长精确解码。最终模型只拟合 12 个有标注视频。Basic 只用允许的图片；Advanced 加入像素黑段与段起始帧导出。未采用的 ROI、光流、频率校准、F1 选轮数和短阶段损失不能写成最终算法组成。

本地四折分数是反复使用的开发估计，Advanced 研究数字仍需明确 perfect x 假设；方法稿不能称为官方测试分数或稳定上限。Advanced 的边界调整只是固定的后验变化峰校正，不是器具动作识别；额外 CPU 时间约 0.03 秒 / 12 视频，不含原有推理与解码。

当前缺少代表人姓名、单位、邮箱、电话和参会信息，尚未套入主办方 1～2 页模板。没有发送邮件或上传提交文件。产物状态以 [readiness.json](../outputs/submission_preparation_20261004/readiness.json) 为准。
