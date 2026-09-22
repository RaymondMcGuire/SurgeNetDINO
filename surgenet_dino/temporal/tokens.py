"""Frame tokens from the FROZEN encoder: [final CLS | mean of the final patch tokens].

The base cache keeps only the final CLS token and is left untouched. This module writes a
separate token cache that refers to the base cache and is verified against it: its CLS half must
reproduce the base rows, so both caches provably describe the same frames, encoder, weights and
preprocessing. Nothing here trains or changes the encoder.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..training import digest_json, file_sha256, save_tensor, write_json

FORMAT = "surgenet-temporal-tokens-v1"


class FrameImages(torch.utils.data.Dataset):
    """Top-level so Windows worker processes can import it. Same deterministic preprocessing."""

    def __init__(self, paths, size):
        self.paths, self.size = list(paths), size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from ..runtime import preprocess_image
        return preprocess_image(self.paths[index], self.size)


def check_architecture(model):
    """The slicing below assumes exactly one prefix token (CLS) and token pooling."""
    if (getattr(model, "num_prefix_tokens", None) != 1 or getattr(model, "num_reg_tokens", 0) != 0
            or not getattr(model, "has_class_token", False) or getattr(model, "global_pool", None) != "token"):
        raise ValueError("Tokens need a ViT with one class token, no register tokens and token pooling")


@torch.inference_mode()
def frame_tokens(model, images):
    """[B, 2D]: the final CLS token, then the mean of the final (normalised) patch tokens."""
    tokens = model.forward_features(images)
    return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], 1)


def record_keys(records):
    return digest_json([[r["video_id"], r["filename"], r["frame_index"]] for r in records])


def extract(model, records, base_rows, image_size, guard, batch, workers, tolerance):
    """Preallocated extraction; every batch is compared with the base cache and aborts on mismatch."""
    check_architecture(model)
    dimension = base_rows.shape[1]
    tokens = torch.empty((len(records), 2 * dimension), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(FrameImages([r["image_path"] for r in records], image_size),
                                         batch_size=batch, shuffle=False, num_workers=workers)
    start, worst, equal = 0, 0.0, 0
    for step, images in enumerate(loader):
        with guard.batch(len(images)):
            part = frame_tokens(model, images.to(guard.device)).float().cpu()
        if not torch.isfinite(part).all():
            raise RuntimeError("Non-finite tokens")
        stop = start + len(images)
        difference = (part[:, :dimension] - base_rows[start:stop]).abs().max().item()
        if difference > tolerance:
            raise RuntimeError(f"CLS differs from the base cache at rows {start}:{stop} (max abs "
                               f"{difference:.3g}); wrong weights, preprocessing or record order")
        worst = max(worst, difference)
        equal += int((part[:, :dimension] == base_rows[start:stop]).all(1).sum())
        tokens[start:stop] = part
        start = stop
        if step % 250 == 0:
            print(f"Tokens {stop}/{len(records)}", flush=True)
    if start != len(records):
        raise RuntimeError("Extraction did not cover every record")
    return tokens, {"max_abs_cls_difference": worst, "bitwise_equal_rows": equal}


def cache_tokens(base_features, config, resources, output, *, batch_size=None, workers=8, tolerance=1e-5):
    from ..gpu import CPUControl, GPUControl
    from ..registry import get_spec
    from ..runtime import resolve_device
    from ..training import assert_identity, encoder, environment, load_features, state_digest
    output, base_features = Path(output), Path(base_features)
    output.mkdir(parents=True, exist_ok=False)
    base = load_features(base_features)
    records = base["split"]["train"]["records"] + base["split"]["validation"]["records"]
    if batch_size is None:  # the batch size of the base run gives the same kernels, hence equal values
        report = base_features.parent / "result.json"
        batch_size = json.loads(report.read_text(encoding="utf-8"))["batch_size"] if report.is_file() else 8
    device = resolve_device(config.device)
    control = GPUControl if device.type == "cuda" else CPUControl
    with control(device, resources, output / "gpu.csv") as guard:
        model, identity = encoder(config, device)
        assert_identity(identity, base["feature_identity"], "Token cache encoder")
        before = state_digest(model)
        started = guard.mark()
        tokens, check = extract(model, records, base["features"], get_spec(config.name).image_size, guard,
                                batch_size, workers, tolerance)
        if before != state_digest(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen backbone changed or received gradients")
        usage = guard.summary(started)
        env = environment(device, identity)
    save_tensor(output / "tokens.pt", {
        "format": FORMAT, "tokens": tokens, "targets": base["targets"].clone(),
        "base_features_sha256": file_sha256(base_features), "records_sha256": record_keys(records),
        "feature_identity": identity, "model_config": asdict(config),
        "layout": "final CLS token, then the mean of the final patch tokens"})
    result = {"status": "passed", "samples": len(records), "token_shape": list(tokens.shape),
              "batch_size": batch_size, "workers": workers, "verification": {**check, "tolerance": tolerance},
              "frozen_unchanged": True, "environment": env, "resources": usage,
              "base_features": str(base_features.resolve())}
    write_json(output / "result.json", result)
    return result


def load_tokens(directory, base, base_features, tolerance=1e-5):
    """Tokens re-verified against the base cache on every load."""
    directory, base_features = Path(directory), Path(base_features)
    payload = torch.load(directory / "tokens.pt", map_location="cpu", weights_only=True)
    records = base["split"]["train"]["records"] + base["split"]["validation"]["records"]
    tokens = payload.get("tokens")
    if (payload.get("format") != FORMAT or payload.get("base_features_sha256") != file_sha256(base_features)
            or payload.get("records_sha256") != record_keys(records)
            or payload.get("feature_identity") != base["feature_identity"]):
        raise ValueError("Token cache does not belong to this base feature cache")
    dimension = base["features"].shape[1]
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.float32
            or tokens.shape != (len(records), 2 * dimension) or not torch.isfinite(tokens).all()
            or not torch.equal(payload["targets"], base["targets"])):
        raise ValueError("Malformed or misaligned tokens")
    difference = (tokens[:, :dimension] - base["features"]).abs().max().item()
    if difference > tolerance:
        raise ValueError(f"Token cache CLS differs from the base cache (max abs {difference:.3g})")
    return tokens, {"tokens_sha256": file_sha256(directory / "tokens.pt"), "max_abs_cls_difference": difference}
