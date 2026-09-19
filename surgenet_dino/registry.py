"""The eight original timm architectures and official weight filenames, unchanged."""

from dataclasses import dataclass
from types import MappingProxyType

HF_REPO_ID = "rlpddejong/SurgeNetXL_DINOv1-v3"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    timm_name: str
    image_size: int
    patch_size: int
    feature_dim: int
    filename: str


_MODELS = (
    ModelSpec("dinov1_vits", "vit_small_patch16_224.dino", 224, 16, 384,
              "DINOv1_ViTs16_size224_SurgeNetXL.pth"),
    ModelSpec("dinov1_vitb", "vit_base_patch16_224.dino", 224, 16, 768,
              "DINOv1_ViTb16_size224_SurgeNetXL.pth"),
    ModelSpec("dinov2_vits", "vit_small_patch14_dinov2", 336, 14, 384,
              "DINOv2_ViTs14_size336_SurgeNetXL.pth"),
    ModelSpec("dinov2_vitb", "vit_base_patch14_dinov2", 336, 14, 768,
              "DINOv2_ViTb14_size336_SurgeNetXL.pth"),
    ModelSpec("dinov2_vitl", "vit_large_patch14_dinov2", 336, 14, 1024,
              "DINOv2_ViTl14_size336_SurgeNetXL.pth"),
    ModelSpec("dinov3_vits", "vit_small_patch16_dinov3.lvd1689m", 336, 16, 384,
              "DINOv3_ViTs16_size336_SurgeNetXL.pth"),
    ModelSpec("dinov3_vitb", "vit_base_patch16_dinov3.lvd1689m", 336, 16, 768,
              "DINOv3_ViTb16_size336_SurgeNetXL.pth"),
    ModelSpec("dinov3_vitl", "vit_large_patch16_dinov3.lvd1689m", 336, 16, 1024,
              "DINOv3_ViTl16_size336_SurgeNetXL.pth"),
)
MODELS = MappingProxyType({spec.name: spec for spec in _MODELS})


def get_spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODELS)}") from exc
