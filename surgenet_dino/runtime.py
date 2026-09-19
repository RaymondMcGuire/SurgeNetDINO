"""Device checks and a deterministic image-input example (not a training augmentation policy)."""

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Check the driver and the cu128 environment. "
                "For an intentional CPU run, use --device cpu and the cpu extra."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is out of range")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This workstation configuration supports CPU and CUDA only")
    return device


def preprocess_image(path: str | Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return transform(rgb)
