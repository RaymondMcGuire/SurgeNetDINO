# JSCAS Basic：预处理数据与实验划分

预处理数据属于可复用的数据资产；训练/验证划分属于实验配置。更换划分或改做交叉验证时，
新建划分目录并生成清单，不重新抽帧，不复制图片，不改已有清单。

## 保存位置与职责

| 内容 | 位置 | 用途 |
|---|---|---|
| 主办方原始视频、分段 CSV、脚本 | `H:\dataset\JSCAS AI Challenge` | 只读来源 |
| 每个视频的图片和分类清单 | `H:\dataset\JSCAS_preprocessed\videoXX_1fps\basic_challenge\data\XX` | 与任何训练/验证划分无关 |
| 本批数据索引、检查报告与复现脚本 | `H:\dataset\JSCAS_preprocessed\labeled_1fps_v1` | 汇总 12 个已标注视频，引用原位图片 |
| 本次固定划分 | `H:\dataset\JSCAS_experiments\splits\video_holdout_9_3_v1` | 9 个训练视频、3 个验证视频的独立清单 |
| 之后的训练产物 | `H:\dataset\JSCAS_experiments\<experiment>\<run_id>` | 模型、指标、曲线、实际参数，以及引用的划分身份 |

2026-09-20 的历史批次复用了视频 01 的原图；现在仓库中的重建脚本从原视频处理全部 12 个案例，
不再要求预先准备视频 01。官方无公开标签的视频 02、07、11、15 不参加本批处理或划分。

## 固定的预处理规则

- 调用主办方原始 `extract_frames.py` 的 `process_video`，`frame_step_sec=1`、`ignore_phase=['x']`。
- 当前视频均为 30 fps，保存原视频零基索引 0、30、60……对应的 PNG，文件名如 `03_000030.png`。
- 原分段 CSV 的区间为 `[本行 start-frame, 下一行 start-frame)`；不重写人工阶段标签。
- 图片保持 1280×720 RGB，不提前缩放、归一化或增强。DINO 输入变换仍由模型运行阶段完成。
- `annotation_XX.csv` 保留官方脚本输出。`classification_XX.csv` 是后续分类允许使用的清单，
  仍为 `filename,phase-id` 两列，文件名不含 `.png`。
- 排除 x、e，以及整张图片所有 RGB 像素均不超过 1 的图片；`excluded_XX.csv` 记录原标签、
  文件名、原因和最大像素值。排除清单中的 PNG 保留，读取训练数据时禁止扫描目录补入样本。
- 类别顺序固定为 `0,1,2,3,4,5,6,7,81,82,83,9,13`，类别编号不是连续的 0～12。
- 本规则面向 Basic 单图分类；Advanced 时序任务需要另行定义 x 与连续时间轴的处理。

检查包含：预期图片数量和文件名、每张 PNG 完整解码、每行标签与原始区间一致、黑帧筛选、
保留与排除清单互斥且覆盖全部抽帧、固定抽样原视频帧与 PNG 逐像素一致、来源与图片 SHA-256。
这些检查验证文件完整性与标注对应关系，不等同于重新人工审核所有阶段的医学标注准确性。

视频 09 的 e 标记比视频帧数少 1，但不落在 1 fps 抽帧点上；保留原标注并在报告记录偏差。
单个视频允许缺少某阶段，类别覆盖在训练集和验证集各自整体范围检查。

## 清单格式

汇总 `samples.csv` 及实验 `train.csv` / `validation.csv` 使用相同列：

| 列 | 含义 |
|---|---|
| `video_id` | 保留前导零的视频编号，如 `03`，也是当前分组键 |
| `filename` | 官方格式的图片名，不含扩展名 |
| `frame_index` | 原视频零基帧号 |
| `time_seconds` | 帧号 / 30，便于回查原视频 |
| `phase-id` | 主办方原始阶段编号 |
| `image_path` | 原位 PNG 的绝对路径，不复制图片 |
| `image_sha256` | PNG 内容校验值 |

`dataset.json` 记录抽帧/筛选规则、版本、来源校验值、逐视频统计和完成状态。
`split.json` 记录数据版本校验值、分组规则、随机种子、视频名单、样本数量、各类别数量以及两份清单的校验值。
移动数据目录后，路径需要重新生成并形成新版本；不能静默修改旧实验正在引用的清单。

`scripts/train.bat` 显式读取 split.json 和两份清单，保存划分快照及校验值，并分别训练和验证。
使用方法见 [TRAINING.md](TRAINING.md)；完整 9/3 视频训练现已完成，结果及图片位置见训练说明。

## 本次划分规则

按完整视频选择 9 个训练、3 个验证，固定 seed=42。视频 01 已用于调试和训练，固定放入训练组。
从其余视频的三视频组合中，以固定种子打乱顺序，选取第一个满足以下条件的组合：

- 验证图片数占全量的 20%～30%。
- 每个阶段在验证组的图片数占该阶段全量的 10%～40%，确保两组都覆盖全部 13 类。
- 两组视频、样本名、PNG 文件内容校验值不相交，合并后覆盖全部有效样本。

只根据标注数量选择划分，不参考任何模型的预测结果。它是一次固定的内部验证方案，
不是官方测试集，也不保证覆盖所有患者或手术差异。当前没有患者关联信息，因此暂以视频代表一台手术；
若以后发现多个视频属于同一患者/手术，须将它们整体分组并生成新版本。

将来交叉验证只需要生成多组训练/验证清单。模型选择完成后，可以另建“全部 12 个视频训练”的实验清单；
此前的验证指标保留其原始划分含义，不能当作全量重训模型的独立测试指标。

## 复现

换电脑只带原始数据也可以重新准备；入口是仓库内的 `scripts/prepare_data.bat` / `prepare_data.sh`，
核心代码为 [tools/prepare_labeled.py](../tools/prepare_labeled.py)。先安装 uv；Python 3.12 和
固定版本的 OpenCV/pandas/Pillow 由 uv 自动准备在独立环境，不加入训练环境。预处理不需要 GPU，
预处理盘至少留 120 GiB 空间（历史 PNG 合计约 77 GiB，另需保留原始视频）。首次准备依赖需要联网。

Windows 默认原始数据路径为 `H:\dataset\JSCAS AI Challenge`，双击 `scripts\prepare_data.bat` 即可。
它递归寻找根目录下面各处 `HD_videos/*.mp4`，按视频编号匹配唯一的 `annotation_csv`，
并调用该标注目录旁的官方 `extract_frames.py`。保留各数据包的解压结构即可，**不用将视频集中到 data1**。
缺视频、重复编号或多个标注目录都会报错；原始视频、标注、官方脚本不作修改。

默认输出是原始数据根目录的同级目录：`JSCAS_preprocessed` 和 `JSCAS_experiments`。
也可指定三个路径（第二、三个参数可省略）：

```powershell
.\scripts\prepare_data.bat "H:\dataset\JSCAS AI Challenge" "H:\dataset\JSCAS_preprocessed_new" "H:\dataset\JSCAS_experiments_new"
```

Linux 在项目根目录运行：

```bash
bash scripts/prepare_data.sh "/data/JSCAS AI Challenge"
# 自定义输出根目录：
# bash scripts/prepare_data.sh "/data/JSCAS AI Challenge" "/data/preprocessed_new" "/data/experiments_new"
```

一次执行顺序为：发现 12 个有标注视频 → 官方 1 fps 抽帧 → 全量 PNG/标签/黑帧检查及抽样像素比对
→ 保存 `labeled_1fps_v1/dataset.json`、`samples.csv`、逐视频报告和预览 → 生成独立 9/3 划分
`JSCAS_experiments/splits/video_holdout_9_3_v1/split.json`。默认两个 CPU 工作进程，不会启动模型训练。

所有输出索引、视频案例目录和划分目录必须尚不存在；失败产物会保留，拒绝覆盖和自动删除。
在当前已有旧数据的电脑试重建时，请像上面的示例一样换用新的输出根目录。
换系统/盘符后应重新运行预处理以生成当前机器的绝对路径，不直接拿 Windows 的旧清单在 Linux 上训练。

只做规则自检可运行 `uv run --no-project --python 3.12 --script tools/prepare_labeled.py self-test`。
已有合格数据只想另建划分时，同一入口使用 `split --dataset <labeled_1fps_v1> --output <新划分目录> --seed 42`；
不重新抽帧，也不在训练中隐式重新随机划分。

### 准备好后如何训练

1. 建立训练环境：Windows 运行 `scripts\init.bat cu128`；Linux 运行 `bash scripts/init.sh cu128`。
2. 准备冻结 DINO 权重：可以连同旧项目的 `.cache/huggingface` 一起复制。若只有原始数据，
   在项目根目录运行下面的命令下载并校验同一版本；训练配置仍固定版本及 SHA-256，并离线读取它：

   ```text
   uv run --locked --extra cu128 python -m surgenet_dino smoke --config configs/dinov2_vitb.toml --revision 9b2c75b7d469850b750a715b90cb91d6319f7e30 --sha256 cd6e73e692074f1d58d2e4125998818121acc61748ff941849784616b03c2def --device cuda:0
   ```

3. 按新生成的划分提取特征和训练单帧分类头。Windows 默认 H 盘布局用 `scripts\train.bat low`；
   自定义路径用 `scripts\train.bat low "<新的运行目录>" "<split.json>"`。
   Linux 用 `bash scripts/train.sh "/data/JSCAS_experiments/splits/video_holdout_9_3_v1/split.json" "/data/JSCAS_experiments/frozen_dinov2/new_run"`。
   新电脑可先用默认 low 限制；历史 GPU 校准文件不是必需项，若提供则仍会严格校验硬件和模型身份。

这一步得到 `<新的运行目录>/cache/features.pt` 和单帧分类结果。若要继续复现我们讨论的时序网络，
把这个**新缓存**传入 `scripts\temporal.bat cache-tokens --features <features.pt> --output-dir <新token目录>`，
然后 `scripts\temporal.bat cv --features <features.pt> --tokens <新token目录> --output-dir <新验证目录>`。
Linux 将入口替换为 `bash scripts/temporal.sh`。Advanced 官方口径评分再加 `--annotations <annotation_csv>`；
Basic 不加 `--motion`。完整训练/验证选项见 [TRAINING.md](TRAINING.md)。

`JSCAS_experiments` 内的旧缓存和历史结果并非废文件：复制它们能保留旧实验；没有它们也能从原始数据重建。
`retrain_head.bat` / `crossval.bat` 默认仍指向历史缓存，新电脑需显式传入刚生成的 `features.pt`。

### 逐秒运动特征（`motion_v1`，可选，只限 Advanced）

时序网络的可选输入，同样在独立 OpenCV 环境里生成，训练环境不变。它要解码 30 fps 原视频，组委会 FAQ（2026-09-27）Q5(d) 规定 Basic 推理时不得使用评价视频文件本身，所以只能用于 Advanced。只读清单里的 `video_id` 和 `frame_index`，不读标签；对每个已编目的帧 f 顺序解码原视频的 [f−15, f+15) 共 30 帧，写出 16 个非负的帧差和 DIS 光流统计量，每帧一行（`motion.csv`），参数与各视频信息记录在 `motion.json`。输出目录必须尚不存在。

```powershell
uv run --isolated --no-project --python 3.12 --with opencv-python-headless==5.0.0.93 python tools\motion_features.py extract --catalogue "H:\dataset\JSCAS_preprocessed\labeled_1fps_v1\samples.csv" --source "H:\dataset\JSCAS AI Challenge" --output "H:\dataset\JSCAS_preprocessed\motion_v1" --workers 6
```

本次生成：74,430 行，6 进程 1,750 秒。`tools\motion_features.py self-test --directory <临时目录>` 用合成视频检查静止与运动两秒的特征大小关系。加载时每个已编目的帧都必须有且只有一行，否则报错。

## 2026-09-20 完成结果

12 个已标注视频全部完成核验；视频 01 复用原图，其余 11 个新抽帧。
共 92,061 张 PNG，保留 74,430 张有效分类图片；
排除阶段 x 的 17,627 张，以及带有效阶段标签但整图 RGB 最大值不超过 1 的 4 张。
这些额外排除的近黑图已逐张回查原视频，像素完全一致，见数据目录中的 verification.json。

| 用途 | 视频 | 图片数 |
|---|---|---:|
| 训练 | 01, 03, 05, 06, 09, 12, 13, 14, 16 | 55,779 |
| 验证 | 04, 08, 10 | 18,651 |

两组均覆盖全部 13 类；验证图片占比 25.06%。
样本清单精确覆盖全部有效图片，两组视频和 PNG 内容校验值互不重叠。
本次仅完成数据准备与固定划分，未启动新的分类训练或计算独立验证准确率。

[数据报告](H:/dataset/JSCAS_preprocessed/labeled_1fps_v1/README.md) ·
[划分记录](H:/dataset/JSCAS_experiments/splits/video_holdout_9_3_v1/split.json)

## 固定方案四折验证

当前 crossval-head 会合并原 9/3 缓存的全部 12 个已标注视频，按完整视频生成四折，
每折 9 个训练、3 个验证。各折 train.csv、validation.csv、split.json 与总 folds.json
保存到独立的新划分目录，不修改这里记录的原始数据版本或原 9/3 清单。
每张有效图片恰好作为验证样本一次；分组 seed=42，只依据标签数量平衡，不参考模型预测。
具体名单、入口和开发验证的解释见 [TRAINING.md](TRAINING.md)。
