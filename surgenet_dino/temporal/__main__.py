"""CLI: python -m surgenet_dino.temporal --help. The original CLIs remain unchanged."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="Offline temporal phase network on frozen per-frame tokens")
    subs = root.add_subparsers(dest="command", required=True)
    tokens = subs.add_parser("cache-tokens", help="Extract [CLS | mean patch] tokens from the frozen encoder, "
                                                  "verified against the base feature cache")
    tokens.add_argument("--batch-size", type=int, help="Default: the batch size of the base extraction")
    tokens.add_argument("--num-workers", type=int, default=8)
    cv = subs.add_parser("cv", help="Grouped cross-validation: cross-fitted heads + temporal network, decoded "
                                    "under the order rules of the training videos")
    cv.add_argument("--only-fold", type=int, help="Pilot only; does not produce complete held-out results")
    cv.add_argument("--baseline", type=Path, help="Existing out-of-fold CSV for the paired video bootstrap")
    curves = subs.add_parser("curves", help="Diagnostic learning curves (train vs held-out); selects nothing")
    curves.add_argument("--head-epochs", type=int, default=12, help="Epochs plotted for the frame head")
    curves.add_argument("--every", type=int, default=2, help="Evaluate the temporal network every N epochs")
    curves.add_argument("--seeds", type=int, default=3)
    curves.add_argument("--network-epochs", type=int,
                        help="Train the diagnostic networks longer than the configured budget")
    curves.add_argument("--reference-result", type=Path,
                        help="result.json of a finished cv run; only marks its inner-chosen epochs")
    for p in (tokens, cv, curves):
        p.add_argument("--features", type=Path, required=True, help="Base feature cache (features.pt)")
        p.add_argument("--config", type=Path, default=Path("configs/frozen_dinov2.toml"))
        p.add_argument("--device")
        p.add_argument("--seed", type=int)
        p.add_argument("--gpu-profile", choices=("low", "medium", "high"), default="medium")
        p.add_argument("--gpu-memory-fraction", type=float)
        p.add_argument("--gpu-duty-cycle", type=float)
        p.add_argument("--max-gpu-temperature", type=float, default=75)
        p.add_argument("--cpu-threads", type=int, default=2)
        p.add_argument("--output-dir", type=Path, required=True)
    for p in (cv, curves):
        p.add_argument("--tokens", type=Path, required=True, help="Directory written by cache-tokens")
        p.add_argument("--head-config", type=Path, default=Path("configs/head_regularized.toml"))
        p.add_argument("--temporal-config", type=Path, default=Path("configs/temporal.toml"))
        p.add_argument("--folds-json", type=Path, default=Path("configs/cv4_reference.json"))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists; choose a new directory: {args.output_dir}")
    from ..config import load_config
    from ..gpu import PRESETS, Resources
    config = load_config(args.config)
    config = replace(config, **{k: getattr(args, k) for k in ("device", "seed") if getattr(args, k) is not None})
    memory, duty = PRESETS[args.gpu_profile]
    resources = Resources(args.gpu_profile,
                          memory if args.gpu_memory_fraction is None else args.gpu_memory_fraction,
                          duty if args.gpu_duty_cycle is None else args.gpu_duty_cycle,
                          args.max_gpu_temperature, args.cpu_threads)
    if args.command == "cache-tokens":
        from .tokens import cache_tokens
        result = cache_tokens(args.features, config, resources, args.output_dir, batch_size=args.batch_size,
                              workers=args.num_workers)
        summary = {"token_shape": result["token_shape"], "verification": result["verification"]}
    else:
        from ..head_options import resolve_options
        from .options import load_options
        options, head_options = load_options(args.temporal_config), resolve_options(args)
        if args.command == "cv":
            from .workflows import run_cv
            result = run_cv(args.features, args.tokens, args.folds_json, options, head_options, config, resources,
                            args.output_dir, baseline=args.baseline, only_fold=args.only_fold)
            keep = ("variant", "accuracy", "macro_f1", "min_video_accuracy", "mean_predicted_segments")
            summary = {"summary": [{k: row[k] for k in keep} for row in result["summary"]],
                       "folds": [{k: f[k] for k in ("fold", "accuracy", "common_epoch")} for f in result["folds"]]}
            if "paired_comparison_vs_baseline" in result:
                summary["paired_comparison_vs_baseline"] = result["paired_comparison_vs_baseline"]
        else:
            from .curves import run_curves
            run_curves(args.features, args.tokens, args.folds_json, options, head_options, config, resources,
                       args.output_dir, head_epochs=args.head_epochs, every=args.every, seeds=args.seeds,
                       reference=args.reference_result, network_epochs=args.network_epochs)
            summary = {}
    print(json.dumps({"status": "passed", "output": str(args.output_dir), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
