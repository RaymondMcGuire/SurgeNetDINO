# 来源、许可证与上游参考资料

本文件区分“本包确实包含的源码”和“只提供链接的上游资料”，不将引用链接冒充已打包的源文件。

## 重构来源

本版基于本次对话上传的 `SurgeNetDINO_uv_migration.zip` 中的完整 overlay 源码继续开发。
原交付记录中的来源归档 SHA-256：`bd7032efbc8d3cd00099fc0141004c311b44d8de273960e6d0a4e726c05a3b06`。
该值保留来源记录，本轮未重新取得归档核验。旧交付状态 JSON 已清理，当前运行状态见 `PROJECT_GUIDE.md`。
所参考的原始模型加载器是：

<https://github.com/rlpddejong/SurgeNetDINO/blob/main/load_models.py>

保留其八种 `timm` 模型注册名、图像尺寸、patch size 与权重文件对应关系。
不重写 `timm` 的数学实现；运行时通过声明的依赖取得 `timm`。
用户 fork 当前 HEAD 未能读取，本版不声称包含未提供的自定义改动。

## 大规模预训练参考（不在本包内，也不是运行依赖）

- <https://github.com/rlpddejong/SurgeNetDINO/tree/main/dinov1_configs>
- <https://github.com/rlpddejong/SurgeNetDINO/tree/main/dinov2_configs>
- <https://github.com/rlpddejong/SurgeNetDINO/tree/main/dinov3_configs>
- <https://github.com/facebookresearch/dino>
- <https://github.com/facebookresearch/dinov2>
- <https://github.com/facebookresearch/dinov3>

本包没有取得并复制这些原始配方；上面的链接不是下载恢复脚本。当前项目无需它们即可进行模型加载与本版命令。
未来确实要复现自监督预训练时，应取得相应版本的原始配置与原始训练实现，保存 commit，并另行管理其训练依赖。
`configs/*.toml` 是本次编写的下游加载配置，不是上游预训练配置的替代复现。

## 权重及许可

官方 SurgeNetDINO 权重：
<https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3>

本包只包含权重解析与下载代码，不附带任何 `.pth` 文件。
官方仓库将代码标注 MIT、预训练权重标注 CC BY-NC-SA；请按各自许可使用，不能将模型权重理解为随代码转成 MIT。

`LICENSE.txt` 保留所查上游 MIT 文本和 copyright：
<https://github.com/rlpddejong/SurgeNetDINO/blob/main/LICENSE.txt>
该文件的 Git blob SHA-1 为 `8d001c93db502c22dfd0a456669de30391088dd3`。

`LICENSE_MODELS.txt` 包含 CC BY-NC-SA 4.0 正文，由 CC 官方纯文本内容重新录入；不宣称它的空行/字节与上游文件完全相同。
权威文本：<https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt>
上游说明：<https://github.com/rlpddejong/SurgeNetDINO/blob/main/LICENSE_MODELS.txt>

## 论文引用

```bibtex
@inproceedings{jong2026towards,
  title={Towards Effective Surgical Representation Learning with DINO Models},
  author={Ronald L.P.D. de Jong and Yiping Li and Tim J.M. Jaspers and
          Romy C. van Jaarsveld and Gino M. Kuiper and Franco Badaloni and
          Richard van Hillegersberg and Jelle P. Ruurda and Fons van der Sommen and
          Josien P.W. Pluim and Marcel Breeuwer},
  booktitle={Medical Imaging with Deep Learning},
  year={2026},
  url={https://proceedings.mlr.press/v315/jong26a.html}
}
```

本重构工程不获得作者官方背书；使用模型做研究时，请保留上述论文归属。
