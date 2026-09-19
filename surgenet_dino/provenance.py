"""Feature-space identity: equal vector length does NOT mean interchangeable features.

This covers the encoder/weight/preprocessing contract. A sequence-cache implementation
must ALSO identify its dataset, split, sampling rate and original timestamps.
"""

import hashlib
import json
from typing import Any

from .config import ModelConfig
from .registry import get_spec


def feature_identity(config: ModelConfig, report: dict[str, Any], timm_version: str) -> dict:
    # Keep this consistent with runtime.preprocess_image. Import here only when needed.
    from .runtime import MEAN, STD

    spec = get_spec(config.name)
    contract = {
        "schema_version": 1,
        "model": config.name,
        "timm_architecture": spec.timm_name,
        "timm_version": timm_version,
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "random_init_seed": config.seed if config.weights == "none" else None,
        "output_dimension": config.num_classes or spec.feature_dim,
        "num_classes": config.num_classes,
        "pooling": "native_timm_forward_no_custom_token_reduction",
        "preprocessing": {
            "exif_transpose": True,
            "color": "RGB",
            "resize": [spec.image_size, spec.image_size],
            "interpolation": "bicubic",
            "antialias": True,
            "mean": list(MEAN),
            "std": list(STD),
        },
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"id": hashlib.sha256(encoded).hexdigest(), "contract": contract}
