# 手术阶段数据查看器

位于 `tools/viewer/`，是独立的可视化辅助工具，不属于训练与评估的核心代码，核心代码不依赖它。

本机只读查看 H 盘预处理图片及已有预测，无需 GPU、模型加载、联网或额外 Python 包。

## 启动

在项目根目录运行，或双击 `scripts/viewer.bat`：

```powershell
scripts\viewer.bat
```

默认打开 <http://127.0.0.1:8765>。窗口保持运行；Ctrl+C 退出。脚本优先使用项目已有
`.venv`，否则使用系统 Python，不执行 `uv sync`，不会切换训练环境。

自定义位置或端口：

```powershell
python -m tools.viewer --data-root "H:\dataset\JSCAS_preprocessed" --experiments-root "H:\dataset\JSCAS_experiments" --port 8765
```

`--no-browser` 仅启动服务，不自动打开网页。服务固定监听本机地址。
目录与结果元数据在本次运行中缓存；新增或修改数据后重启查看器并刷新网页。

## 使用

- **人工标注**：选择视频；左侧列出每次出现的阶段片段。点击片段，跳转到该区间内首张可用图片。
  默认只显示有效分类帧，勾选「显示排除帧」可查看体外画面、黑帧及排除原因。
- **模型预测**：选择实验结果和视频。预测与人工标注各有一条时间轴，差异轨道标出不一致位置。
  没有真值时显示「未标注」，不借用预测文件的标签充当人工标注。
- 输入原始帧号或 `HH:MM:SS`／秒数跳转；支持小数秒。未保存的帧会提示最近可用图片和实际帧号。
  按阶段筛选后，跳转、上下帧、播放和差异导航均限于筛选后的图片。
- 上下帧按钮或左右方向键切换；播放速度为每秒 1、2、5 张抽帧，图片读取较慢时等待图片加载。
  点击图片或「放大原图」查看原始尺寸画面，Escape 关闭。
- 时间轴保留完整时间跨度；标注区间为 `[起始帧, 结束帧)`，结束帧不属于该区间。
  预测只合并相邻抽帧的相同结果，中间缺帧保留空白，不进行平滑或补标签。
- 中文名称是官方日文描述的辅助译文，详情同时显示日文原文。
  [官方编号与阶段定义](https://github.com/modafone/jscaschallenge2026/blob/main/basic_challenge/readme.md)
  中编号为 `0,1,2,3,4,5,6,7,81,82,83,9,13`；`x` 是体外，`e` 是结束标记。

示例：视频 01 的阶段 0 从原始帧 **615（20.5 秒）** 开始，首张对应的 1 fps 图片为
**630（21 秒）**。界面分别显示阶段边界和实际图片帧号。

## 数据和预测接入

已标注数据使用 `labeled_1fps_v1/dataset.json`、`samples.csv`、逐视频的 `source/XX.csv`
（或元数据记录的原始标注）以及 `excluded_XX.csv`。只按清单读取分类图片，不扫描补入训练样本。
图片绝对路径必须位于配置的预处理根目录内；移动数据后应按项目数据版本规则生成新清单。

未标注视频使用相同图片目录：

```text
<data-root>/video02_1fps/
  processing.json
  basic_challenge/data/02/images/02_000030.png
```

`processing.json` 沿用现有预处理元数据；读取 `fps`、`source_frame_count` 和可选的
`sample_every_frames`。缺少元数据时页面明确提示，按本项目当前 30 fps、每 30 帧一张的约定显示，
时间轴终点以最后图片加抽帧间隔估算。当前 H 盘的 02、07、11、15 尚未预处理；查看器不负责抽帧。

预测按视频编号和原始帧号关联，支持：

| 来源 | 读取方式 |
| --- | --- |
| `validation_predictions*.csv` | 自动发现；兼容旧版缺少 `frame_index` 的文件，以及新版带 `fold`、概率列的结果 |
| `predict-head` 的 `result.json` | 自动发现带 `predictions` 数组的文件；从 `image` 文件名取身份，从 `phase_id` 和 `scores` 读取预测 |
| 官方两列 `filename,phase-id` CSV | 必须以 `--prediction` 显式指定，避免误把人工标注当预测 |
| 实验目录外的上述结果 | 用 `--prediction` 指定，可重复使用该参数 |

```powershell
scripts\viewer.bat --prediction "H:\my-results\submission.csv" --prediction "H:\my-results\predict-head\result.json"
```

两列预测 CSV 示例（文件名可含或不含 `.png`）：

```csv
filename,phase-id
02_000030,0
02_000060,1
```

文件只加载已有结果；缺少概率或置信度显示「未提供」，不会伪造。缺少预测的图片显示「无预测」。
实验结果名称显示相对路径，区分不同实验、折次、最佳／末轮及不同解码版本；不会自动选择“最佳”模型。
标签冲突、重复帧、错误格式、无法对应的预测帧会报告具体来源；磁盘缺图则在当前画面提示。

## 验证与接口

```powershell
python -m unittest discover -s tests -p test_viewer.py -v
```

查看器的只读 GET 接口：`/api/catalog`、`/api/prediction?source=<id>`、
`/api/video?video=01&source=<id>`（标注模式省略 source）、`/api/image?video=01&frame=630`。
来源 ID 由 catalog 返回；图片仅可通过已登记的视频和帧号请求，不支持任意文件路径。
查看器不会更改 H 盘数据，不写标签或审核状态，也不运行推理。
