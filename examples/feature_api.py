"""One image, one selected backbone. Run from project root with `python -m examples.feature_api`.

This exports visual features, not surgical phase predictions. DINOv2/v3 use the
same build_model API, but their checkpoints and feature caches are not interchangeable.
"""

import argparse
from dataclasses import replace

import torch

from surgenet_dino.config import load_config
from surgenet_dino.model import build_model, set_trainable
from surgenet_dino.registry import get_spec
from surgenet_dino.runtime import preprocess_image, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dinov2_vitb.toml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg = replace(cfg, device=args.device)
    if cfg.num_classes != 0 or cfg.weights == "none":
        raise ValueError("Choose pretrained weights and num_classes=0 for feature extraction")
    spec = get_spec(cfg.name)
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    backbone, report = build_model(cfg)
    set_trainable(backbone, "none")
    backbone = backbone.to(device).eval()
    x = preprocess_image(args.image, spec.image_size).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = backbone(x)
    if tuple(features.shape) != (1, spec.feature_dim):
        raise RuntimeError(f"Unexpected features: {features.shape}")
    print(f"{cfg.name}: features={tuple(features.shape)}")
    print(f"Weights SHA-256: {report['checkpoint_sha256']}")
    print("A later classifier/temporal head should use spec.feature_dim, not a hard-coded 768.")


if __name__ == "__main__":
    main()
