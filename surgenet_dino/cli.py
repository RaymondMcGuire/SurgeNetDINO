"""Explicit one-model commands. No model is constructed during import or --help."""

import argparse
import importlib
import importlib.metadata
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from .config import ModelConfig, load_config
from .registry import HF_REPO_ID, MODELS, get_spec


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", dest="name", choices=tuple(MODELS), default=argparse.SUPPRESS)
    parser.add_argument("--weights", choices=("surgenet", "local", "none"), default=argparse.SUPPRESS)
    parser.add_argument("--num-classes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-key", default=argparse.SUPPRESS)
    parser.add_argument("--weights-scope", choices=("backbone", "full"), default=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", default=argparse.SUPPRESS)
    parser.add_argument("--revision", default=argparse.SUPPRESS)
    parser.add_argument("--offline", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--sha256", default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SurgeNetDINO: uv environment and model utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List the eight model definitions; no downloads/import of torch")
    listing.add_argument("--json", action="store_true")
    plan = sub.add_parser("plan", help="Show resolved model/configuration; no network or torch import")
    _common(plan)
    doctor = sub.add_parser("doctor", help="Check versions, binary imports and optionally CUDA execution")
    doctor.add_argument("--require-cuda", action="store_true")
    validation = sub.add_parser("validate-data", help="Explicit full image audit; not run during training")
    validation.add_argument("--dataset", type=Path, required=True, help="Dataset directory or dataset.json")
    validation.add_argument("--output-dir", type=Path, required=True, help="New audit report directory")
    smoke = sub.add_parser("smoke", help="One-model synthetic forward/backward check, not an accuracy test")
    _common(smoke)
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--backward", action="store_true", help="Also backpropagate a synthetic loss; no optimizer")
    smoke.add_argument("--report", type=Path)
    smoke.add_argument("--overwrite", action="store_true")
    encode = sub.add_parser("encode", help="Encode one real image to a feature .pt (not phase predictions)")
    _common(encode)
    encode.add_argument("--image", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--overwrite", action="store_true")
    for command, help_text in (
        ("calibrate-gpu", "Measure three GPU profiles on real labeled images"),
        ("cache-features", "Export frozen pretrained features from a classification manifest"),
        ("train-head", "Train a 13-class head with video-disjoint validation"),
        ("predict-head", "Predict phase labels with a trained head and its matching encoder"),
        ("crossval-head", "Grouped cross-validation across all labeled cached videos"),
        ("audit-features", "Validate cache sources and report video/class coverage; no model training"),
    ):
        phase = sub.add_parser(command, help=help_text)
        _common(phase)
        phase.set_defaults(config=Path(__file__).resolve().parents[1] / "configs/frozen_dinov2.toml")
        phase.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
        if command != "calibrate-gpu":
            phase.add_argument("--gpu-profile", choices=("low", "medium", "high"), default="low")
        else:
            phase.set_defaults(gpu_profile="low")
        phase.add_argument("--gpu-memory-fraction", type=float)
        phase.add_argument("--gpu-duty-cycle", type=float)
        phase.add_argument("--max-gpu-temperature", type=float, help="Default 75 C, or calibrated value")
        phase.add_argument("--cpu-threads", type=int, help="Default 2, or calibrated value")
        phase.set_defaults(profiles=None, batch_size=None)
        if command in {"cache-features", "train-head", "crossval-head"}:
            phase.add_argument("--profiles", type=Path, help="profiles.json from calibrate-gpu")
        if command == "cache-features":
            phase.add_argument("--batch-size", type=int, help="Feature extraction batch override")
            phase.add_argument("--split", type=Path, required=True, help="Audited split.json")
        if command == "calibrate-gpu":
            phase.add_argument("--manifest", type=Path, required=True)
        if command in {"train-head", "crossval-head"}:
            from .head_options import add_head_arguments
            add_head_arguments(phase)
            if command == "crossval-head":
                phase.add_argument("--folds", type=int, default=4)
                phase.add_argument("--split-output-dir", type=Path, required=True,
                                   help="New independent directory for fold manifests")
        if command == "audit-features":
            phase.add_argument("--features", type=Path, required=True)
        if command == "predict-head":
            phase.add_argument("--head", type=Path, required=True)
            phase.add_argument("--image", type=Path, required=True, nargs="+")
    return parser


def _configuration(args: argparse.Namespace) -> ModelConfig:
    config = load_config(args.config) if args.config else ModelConfig()
    allowed = set(asdict(config))
    overrides = {k: v for k, v in vars(args).items() if k in allowed}
    if overrides.get("name", config.name) != config.name:
        # Do not accidentally carry a v2 local checkpoint/checksum into a v3 config.
        for key in ("checkpoint", "sha256", "checkpoint_key"):
            if getattr(config, key) is not None and key not in overrides:
                raise ValueError(
                    f"Changing model would reuse pinned {key}. Use a separate model-specific config."
                )
    return replace(config, **overrides)


def _plan(args: argparse.Namespace) -> int:
    config = _configuration(args)
    spec = get_spec(config.name)
    info = {
        "config": asdict(config),
        "spec": asdict(spec),
        "expected_output_dimension": config.num_classes or spec.feature_dim,
        "checkpoint_source": {
            "repo_id": HF_REPO_ID if config.weights == "surgenet" else None,
            "filename": spec.filename if config.weights == "surgenet" else config.checkpoint,
        },
        "downloads_performed": False,
        "network_constructed": False,
        "checkpoint_compatibility_verified": False,
        "note": "Selection preview only. Run smoke to test the actual model and weight compatibility.",
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def _doctor(require_cuda: bool) -> int:
    import platform

    info = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {}, "errors": []}
    pins = {"torch": "2.10.0", "torchvision": "0.25.0", "timm": "1.0.24"}
    for package in ("torch", "torchvision", "timm", "pillow", "huggingface-hub"):
        try:
            version = importlib.metadata.version(package)
            info["packages"][package] = version
            if package in pins and version.split("+")[0] != pins[package]:
                info["errors"].append(f"{package}: expected {pins[package]}, got {version}")
        except importlib.metadata.PackageNotFoundError:
            info["errors"].append(f"Missing dependency: {package}")
    try:
        import torch
        import torchvision

        importlib.import_module("timm")
        # Catch torch/torchvision binary mismatches as well as simple import problems.
        torchvision.ops.nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5)
        info["binary_import_check"] = "passed"
        info["torch_cuda_runtime"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["gpus"] = []
        if info["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(index)
                info["gpus"].append({"index": index, "name": prop.name,
                                     "memory_bytes": prop.total_memory,
                                     "compute_capability": f"{prop.major}.{prop.minor}"})
        if require_cuda:
            if not info["cuda_available"]:
                raise RuntimeError("CUDA required but unavailable; check cu128 wheels and NVIDIA driver")
            x = torch.ones((8, 8), device="cuda")
            value = (x @ x).sum().item()
            torch.cuda.synchronize()
            if value != 512.0:
                raise RuntimeError("Unexpected CUDA matrix multiplication result")
            info["cuda_execution_check"] = "passed"
    except Exception as exc:
        info["errors"].append(f"{type(exc).__name__}: {exc}")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 2 if info["errors"] else 0


def _write_json(path: Path, data: dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _run_model(args: argparse.Namespace) -> int:
    import torch

    from .model import build_model
    from .runtime import MEAN, STD, preprocess_image, resolve_device
    from .provenance import feature_identity

    config = _configuration(args)
    spec = get_spec(config.name)
    if args.command == "smoke" and args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    if args.command == "encode":
        if config.num_classes != 0:
            raise ValueError("encode requires num_classes=0: export features, not arbitrary task logits")
        if config.weights == "none":
            raise ValueError("encode refuses random features. Use smoke for a weights=none test")
        if not args.image.is_file():
            raise FileNotFoundError(f"Image not found: {args.image}")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}; use --overwrite intentionally")
    if args.command == "smoke" and args.report and args.report.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists: {args.report}; use --overwrite intentionally")
    device = resolve_device(config.device)  # fail before a download on a broken CUDA environment
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, load_report = build_model(config)
    model = model.to(device)
    metadata = {"config": asdict(config), "loading": load_report,
                "torch_version": str(torch.__version__), "timm_version": importlib.metadata.version("timm"),
                "device": str(device)}
    metadata["feature_identity"] = feature_identity(config, load_report, metadata["timm_version"])
    if args.command == "smoke":
        images = torch.randn(args.batch_size, 3, spec.image_size, spec.image_size, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if args.backward:
            model.train()
            outputs = model(images)
            loss = outputs.float().square().mean()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(grad).all().item() for grad in gradients):
                raise RuntimeError("Synthetic backward produced missing or non-finite gradients")
            metadata["synthetic_loss"] = loss.item()
            metadata["parameters_with_grad"] = len(gradients)
            metadata["finite_gradients"] = True
        else:
            model.eval()
            with torch.inference_mode():
                outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        expected_dim = config.num_classes or spec.feature_dim
        if list(outputs.shape) != [args.batch_size, expected_dim]:
            raise RuntimeError(f"Unexpected output shape: {tuple(outputs.shape)}")
        if not torch.isfinite(outputs).all().item():
            raise RuntimeError("Output contains NaN or infinity")
        metadata.update(kind="synthetic_smoke_only", input_shape=list(images.shape),
                        output_shape=list(outputs.shape), backward=args.backward,
                        single_step_seconds=time.perf_counter() - start,
                        challenge_accuracy_measured=False)
        if device.type == "cuda":
            metadata["peak_process_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        if args.report:
            _write_json(args.report, metadata, args.overwrite)
    else:
        images = preprocess_image(args.image, spec.image_size).unsqueeze(0).to(device)
        model.eval()
        with torch.inference_mode():
            outputs = model(images).float().cpu()
        if list(outputs.shape) != [1, spec.feature_dim] or not torch.isfinite(outputs).all().item():
            raise RuntimeError("Invalid feature output")
        metadata.update(kind="single_image_features", image=str(args.image),
                        output_shape=list(outputs.shape),
                        preprocessing={"color": "RGB", "resize": [spec.image_size, spec.image_size],
                                       "interpolation": "bicubic", "mean": MEAN, "std": STD})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation is intentional: do not overwrite an experiment silently.
        with args.output.open("wb" if args.overwrite else "xb") as stream:
            torch.save({"features": outputs, "metadata": metadata}, stream)
        metadata["saved_to"] = str(args.output)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "list":
            if args.json:
                print(json.dumps([asdict(spec) for spec in MODELS.values()], indent=2))
            else:
                for spec in MODELS.values():
                    print(f"{spec.name:14} {spec.image_size:3}px  patch={spec.patch_size:2}  "
                          f"features={spec.feature_dim:4}  {spec.timm_name}")
            return 0
        if args.command == "plan":
            return _plan(args)
        if args.command == "doctor":
            return _doctor(args.require_cuda)
        if args.command == "validate-data":
            from .training import validate_data
            print(json.dumps(validate_data(args.dataset, args.output_dir), indent=2, ensure_ascii=False))
            return 0
        if args.command in {"calibrate-gpu", "cache-features", "train-head", "predict-head",
                            "crossval-head", "audit-features"}:
            from .training import run

            return run(args, _configuration(args))
        return _run_model(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
