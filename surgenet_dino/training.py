"""Frozen features, video-disjoint head training/validation, and GPU calibration."""

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from .config import ModelConfig
from .gpu import GPUControl, PRESETS, Resources, gpu_identity, query_gpu
from .model import build_model, set_trainable
from .provenance import feature_identity
from .registry import get_spec
from .runtime import preprocess_image, resolve_device
from .weights import file_sha256

LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "81", "82", "83", "9", "13"]
CANDIDATES = [1, 4, 8, 16, 32, 64, 128, 256]
MANIFEST_FIELDS = ["video_id", "filename", "frame_index", "time_seconds", "phase-id",
                   "image_path", "image_sha256"]


def write_json(path: Path, data: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_tensor(path: Path, data: dict):
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        torch.save(data, stream)
    if path.exists():
        raise FileExistsError(path)
    partial.rename(path)


def digest_json(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_manifest(path: Path, inspect_images: bool = True) -> dict:
    path = path.resolve(strict=True)
    records, names = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError("Manifest needs the versioned multi-video CSV columns")
        for row in reader:
            if None in row or any(not value for value in row.values()):
                raise ValueError("Missing or extra manifest fields")
            name, label = row["filename"], row["phase-id"]
            video = row["video_id"]
            if (not re.fullmatch(r"\d{2}", video or "")
                    or not re.fullmatch(rf"{video}_\d{{6}}", name or "")):
                raise ValueError(f"Invalid image filename: {name!r}")
            filename = name + ".png"
            if filename in names or label not in LABELS or None in row:
                raise ValueError(f"Duplicate filename or invalid phase: {row}")
            names.add(filename)
            frame = int(row["frame_index"])
            image = Path(row["image_path"])
            if (frame < 0 or frame % 30 or name != f"{video}_{frame:06d}"
                    or float(row["time_seconds"]) != frame / 30
                    or not image.is_absolute() or image.name != filename
                    or not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"] or "")):
                raise ValueError(f"Invalid frame identity/path: {name}")
            record = {"filename": filename, "video_id": video, "frame_index": frame,
                      "image_path": str(image.resolve()), "phase_id": label,
                      "target": LABELS.index(label), "sha256": row["image_sha256"]}
            if inspect_images:
                data = image.read_bytes()
                try:
                    with Image.open(io.BytesIO(data)) as loaded:
                        rgb = loaded.convert("RGB")
                        if max(hi for lo, hi in rgb.getextrema()) <= 1:
                            raise ValueError(f"Black image must be excluded from manifest: {filename}")
                except OSError as exc:
                    raise ValueError(f"Unreadable image {image}: {exc}") from exc
                if hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError(f"Image identity mismatch: {image}")
            elif not image.is_file():
                raise FileNotFoundError(image)
            records.append(record)
            if inspect_images and len(records) % 500 == 0:
                print(f"Validated {len(records)} images", flush=True)
    if not records:
        raise ValueError("Empty classification manifest")
    identity = {"manifest_sha256": file_sha256(path), "records": records, "labels": LABELS}
    return {"manifest": str(path), "identity": digest_json(identity), **identity}


def read_split(path: Path, inspect_images: bool = True) -> dict:
    path = path.resolve(strict=True)
    definition = json.loads(path.read_text(encoding="utf-8"))
    if (definition["status"] != "passed" or definition["label_order"] != LABELS
            or definition["group_key"] != "video_id"):
        raise ValueError("Invalid split status, labels or grouping")
    partitions = {}
    for role in ("train", "validation"):
        manifest = path.parent / f"{role}.csv"
        if file_sha256(manifest) != definition[f"{role}_csv_sha256"]:
            raise ValueError(f"{role} manifest identity mismatch")
        data = read_manifest(manifest, inspect_images)
        rows = data["records"]
        if (len(rows) != definition[f"{role}_samples"]
                or sorted({r["video_id"] for r in rows}) != sorted(definition[f"{role}_video_ids"])
                or {r["phase_id"] for r in rows} != set(LABELS)):
            raise ValueError(f"{role} split count, video IDs or class coverage mismatch")
        partitions[role] = data
    train, validation = (partitions[k]["records"] for k in ("train", "validation"))
    for key in ("video_id", "filename", "image_path", "sha256"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f"Training/validation leakage: overlapping {key}")
    dataset_path = Path(definition["dataset_json"])
    if file_sha256(dataset_path) != definition["dataset_json_sha256"]:
        raise ValueError("Preprocessed dataset identity mismatch")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    catalog = read_manifest(dataset_path.parent / "samples.csv", inspect_images=False)
    if (dataset["status"] != "passed"
            or catalog["manifest_sha256"] != dataset["samples_csv_sha256"]
            or catalog["manifest_sha256"] != definition["samples_csv_sha256"]
            or sorted(train + validation, key=lambda r: r["filename"])
            != sorted(catalog["records"], key=lambda r: r["filename"])):
        raise ValueError("Split does not exactly partition the preprocessed allowlist")
    return {"path": str(path), "sha256": file_sha256(path), "definition": definition, **partitions}


def subset(records: list, per_class: int, seed: int) -> list:
    rng = random.Random(seed)
    result = []
    for label in LABELS:
        group = [r for r in records if r["phase_id"] == label]
        if len(group) < per_class:
            raise ValueError(f"Phase {label} needs at least {per_class} images")
        result.extend(rng.sample(group, per_class))
    rng.shuffle(result)
    return result


def encoder(config: ModelConfig, device):
    if config.weights == "none" or config.num_classes != 0:
        raise ValueError("Frozen training needs pretrained features with num_classes=0")
    model, report = build_model(config)
    set_trainable(model, "none")
    model.eval().to(device)
    identity = feature_identity(config, report, importlib.metadata.version("timm"))
    return model, identity


def state_digest(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def environment(device, identity) -> dict:
    return {"hardware": gpu_identity(device), "feature_identity": identity,
            "torch": str(torch.__version__), "timm": importlib.metadata.version("timm"),
            "torchvision": importlib.metadata.version("torchvision"), "cuda": torch.version.cuda,
            "python": platform.python_version(), "platform": platform.platform(),
            "precision": "float32", "frozen": True, "num_workers": 0,
            "lock_sha256": file_sha256(Path(__file__).resolve().parents[1] / "uv.lock")}


def assert_identity(actual: dict, expected: dict, what: str):
    if actual != expected:
        raise ValueError(f"{what} identity mismatch; regenerate/recalibrate for this environment")


def extract_batch(model, records, size, guard):
    # ponytail: serial PIL loading; add loader workers only if measured CPU/I/O time warrants it.
    images = torch.stack([preprocess_image(Path(r["image_path"]), size) for r in records])
    with guard.batch(len(records)), torch.inference_mode():
        output = model(images.to(guard.device)).float().cpu()
        if not torch.isfinite(output).all().item() or output.ndim != 2:
            raise RuntimeError("Non-finite or malformed features")
    return output


def resources_for(args, profile=None) -> Resources:
    profile = profile or args.gpu_profile
    memory, duty = PRESETS[profile]
    temperature, threads = 75, 2
    if getattr(args, "profiles", None):
        saved = json.loads(args.profiles.read_text(encoding="utf-8"))["profiles"][profile]["resources"]
        memory, duty = saved["memory_fraction"], saved["duty_cycle"]
        temperature, threads = saved["max_temperature"], saved["cpu_threads"]
    return Resources(profile, memory if args.gpu_memory_fraction is None else args.gpu_memory_fraction,
                     duty if args.gpu_duty_cycle is None else args.gpu_duty_cycle,
                     temperature if args.max_gpu_temperature is None else args.max_gpu_temperature,
                     threads if args.cpu_threads is None else args.cpu_threads)


def profile_batch(args, env: dict, resources: Resources) -> int:
    batch = args.batch_size or 4
    if args.profiles:
        profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
        assert_identity(env, profiles["environment"], "Calibration")
        chosen = profiles["profiles"][resources.profile]
        if all(getattr(args, field) is None for field in
               ("gpu_memory_fraction", "gpu_duty_cycle", "max_gpu_temperature", "cpu_threads")):
            assert_identity(asdict(resources), chosen["resources"], "Resource settings")
        batch = args.batch_size or chosen["batch_size"]
    return batch


def cache_features(args, config, output):
    split = read_split(args.split)
    write_json(output / "dataset.json", split)
    for name in ("split.json", "train.csv", "validation.csv"):
        with (output / name).open("xb") as stream:
            stream.write((args.split if name == "split.json" else args.split.parent / name).read_bytes())
    device = resolve_device(config.device)
    resources = resources_for(args)
    with GPUControl(device, resources, output / "gpu.csv") as guard:
        model, identity = encoder(config, device)
        env = environment(device, identity)
        batch = profile_batch(args, env, resources)
        original = state_digest(model)
        rows = split["train"]["records"] + split["validation"]["records"]
        chunks = []
        start = guard.mark()
        for i in range(0, len(rows), batch):
            chunks.append(extract_batch(model, rows[i:i + batch], get_spec(config.name).image_size, guard))
            if i // batch % 25 == 0:
                print(f"Features {min(i + batch, len(rows))}/{len(rows)}", flush=True)
        features = torch.cat(chunks)
        if features.shape != (len(rows), get_spec(config.name).feature_dim):
            raise RuntimeError("Feature shape mismatch")
        if original != state_digest(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen backbone changed or received gradients")
        result = {"status": "passed", "shape": list(features.shape), "batch_size": batch,
                  "environment": env, "frozen_unchanged": True, "gpu": guard.summary(start)}
        save_tensor(output / "features.pt", {
            "features": features, "targets": torch.tensor([r["target"] for r in rows]),
            "split": split, "feature_identity": identity, "model_config": asdict(config),
            "environment": env, "labels": LABELS,
        })
        del model, chunks
        torch.cuda.empty_cache()
        result["allocated_after_release_bytes"] = torch.cuda.memory_allocated(device)
        write_json(output / "result.json", result)
    return result


def load_features(path):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if "split" not in cache:
        raise ValueError("Legacy single-video cache; regenerate with cache-features --split")
    x, y = cache["features"], cache["targets"]
    rows = cache["split"]["train"]["records"] + cache["split"]["validation"]["records"]
    if (cache["labels"] != LABELS or x.dtype != torch.float32 or x.ndim != 2
            or y.dtype != torch.int64 or y.shape != (len(rows),) or len(x) != len(rows)
            or x.shape[1] != cache["feature_identity"]["contract"]["output_dimension"]
            or not torch.isfinite(x).all().item() or not len(rows)
            or any(type(r["target"]) is not int or not 0 <= r["target"] < len(LABELS) for r in rows)
            or y.tolist() != [r["target"] for r in rows]
            or any(r["phase_id"] != LABELS[r["target"]] for r in rows)):
        raise ValueError("Invalid feature cache/label alignment")
    actual = read_split(Path(cache["split"]["path"]))
    assert_identity(actual, cache["split"], "Dataset cache")
    return cache


def new_head(dim, device, seed):
    torch.manual_seed(seed)
    return nn.Linear(dim, len(LABELS)).to(device)


def head_step(head, optimizer, x, y, guard):
    with guard.batch(len(x)):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x.to(guard.device))
        loss = nn.functional.cross_entropy(logits, y.to(guard.device))
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite head loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in head.parameters()):
            raise RuntimeError("Missing or non-finite head gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all().item() for p in head.parameters()):
            raise RuntimeError("Non-finite head parameter")
        value = loss.item()
    return value


def evaluate(head, x, y, guard, batch):
    logits = []
    head.eval()
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            with guard.batch(len(x[i:i + batch])):
                logits.append(head(x[i:i + batch].to(guard.device)).cpu())
    result = torch.cat(logits)
    if not torch.isfinite(result).all().item():
        raise RuntimeError("Non-finite head evaluation")
    return result, nn.functional.cross_entropy(result, y).item()


def classification_metrics(logits, targets, records=None):
    predicted = logits.argmax(1)
    confusion = torch.bincount(targets * len(LABELS) + predicted,
                               minlength=len(LABELS)**2).reshape(len(LABELS), len(LABELS))
    per_class = []
    for i, label in enumerate(LABELS):
        tp = confusion[i, i].item()
        support, positive = confusion[i].sum().item(), confusion[:, i].sum().item()
        per_class.append({"phase_id": label, "support": support, "precision": tp / max(1, positive),
                          "recall": tp / max(1, support), "f1": 2 * tp / max(1, support + positive)})
    result = {"accuracy": (predicted == targets).float().mean().item(),
              "macro_f1": statistics.mean(r["f1"] for r in per_class),
              "confusion": confusion.tolist(), "per_class": per_class, "labels": LABELS}
    if records:
        result["per_video"] = {}
        for video in sorted({r["video_id"] for r in records}):
            indices = [i for i, r in enumerate(records) if r["video_id"] == video]
            result["per_video"][video] = classification_metrics(logits[indices], targets[indices])
    return result


def train_head(args, config, output):
    cache = load_features(args.features)
    contract = cache["feature_identity"]["contract"]
    if config.name != contract["model"] or (
        config.sha256 and config.sha256.lower() != contract["checkpoint_sha256"]
    ):
        raise ValueError("Requested model does not match the feature cache")
    split = cache["split"]
    train_rows, val_rows = split["train"]["records"], split["validation"]["records"]
    count = len(train_rows)
    x, y = cache["features"][:count], cache["targets"][:count]
    vx, vy = cache["features"][count:], cache["targets"][count:]
    write_json(output / "split.json", split["definition"])
    provenance = {"path": split["path"], "sha256": split["sha256"],
                  "train_identity": split["train"]["identity"],
                  "validation_identity": split["validation"]["identity"]}
    device = resolve_device(config.device)
    resources = resources_for(args)
    batch = args.head_batch_size
    with GPUControl(device, resources, output / "gpu.csv") as guard:
        env = environment(device, cache["feature_identity"])
        profile_batch(args, env, resources)
        head = new_head(x.shape[1], device, config.seed)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(config.seed)
        full_before = state_digest(head)
        start = guard.mark()
        history, best_loss, best_state, best_epoch = [], float("inf"), None, None
        fields = ["epoch", "train_loss", "train_accuracy", "validation_loss",
                  "validation_accuracy", "validation_macro_f1"]
        with (output / "learning_curve.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(args.epochs + 1):
                if epoch:
                    head.train()
                    order = torch.randperm(len(x), generator=generator)
                    for indices in order.split(batch):
                        head_step(head, optimizer, x[indices], y[indices], guard)
                logits, loss = evaluate(head, x, y, guard, batch)
                val_logits, val_loss = evaluate(head, vx, vy, guard, batch)
                val_metrics = classification_metrics(val_logits, vy)
                row = {"epoch": epoch, "train_loss": loss,
                       "train_accuracy": (logits.argmax(1) == y).float().mean().item(),
                       "validation_loss": val_loss, "validation_accuracy": val_metrics["accuracy"],
                       "validation_macro_f1": val_metrics["macro_f1"]}
                history.append(row)
                writer.writerow(row)
                stream.flush()
                if epoch and val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    best_logits = val_logits.clone()
                print(f"Epoch {epoch}/{args.epochs}: train loss={loss:.5f}, "
                      f"accuracy={row['train_accuracy']:.4f}; validation loss={val_loss:.5f}, "
                      f"accuracy={val_metrics['accuracy']:.4f}, macro-F1={val_metrics['macro_f1']:.4f}",
                      flush=True)
        payload = {"labels": LABELS, "feature_identity": cache["feature_identity"],
                   "model_config": cache["model_config"], "split": provenance,
                   "feature_cache_sha256": file_sha256(args.features), "environment": env,
                   "selection": "minimum validation cross-entropy over trained epochs"}
        save_tensor(output / "head_last.pt", {**payload, "epoch": args.epochs,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}})
        save_tensor(output / "head_best.pt", {**payload, "epoch": best_epoch, "state_dict": best_state})
        changed = full_before != state_digest(head)
        restored = torch.load(output / "head_best.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(restored["state_dict"])
        reloaded, _ = evaluate(head, vx, vy, guard, batch)
        torch.testing.assert_close(best_logits, reloaded, rtol=0, atol=0)
        best_train, _ = evaluate(head, x, y, guard, batch)
        result = {"status": "passed", "scope": "video-disjoint training and internal validation",
                  "history": history, "best_epoch": best_epoch, "selection": payload["selection"],
                  "train": classification_metrics(best_train, y, train_rows),
                  "validation": classification_metrics(reloaded, vy, val_rows),
                  "split": provenance, "parameters_changed": changed,
                  "train_loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
                  "labels": LABELS, "reload_identical": True, "gpu": guard.summary(start),
                  "head_batch_size": batch, "head_parameters": sum(p.numel() for p in head.parameters())}
        write_json(output / "result.json", result)
        with (output / "validation_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "filename", "phase_id",
                                                        "predicted_phase_id", "top_score"])
            writer.writeheader()
            probabilities, predictions = reloaded.softmax(1).max(1)
            for record, pred, score in zip(val_rows, predictions.tolist(), probabilities.tolist()):
                writer.writerow({"video_id": record["video_id"], "filename": record["filename"],
                                 "phase_id": record["phase_id"], "predicted_phase_id": LABELS[pred],
                                 "top_score": score})
    return result


def predict_head(args, config, output):
    checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    if checkpoint["labels"] != LABELS:
        raise ValueError("Head label mapping mismatch")
    device = resolve_device(config.device)
    with GPUControl(device, resources_for(args), output / "gpu.csv") as guard:
        model, identity = encoder(config, device)
        assert_identity(identity, checkpoint["feature_identity"], "Head encoder")
        head = new_head(get_spec(config.name).feature_dim, device, config.seed)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        predictions = []
        for path in args.image:
            features = extract_batch(model, [{"image_path": str(path)}],
                                     get_spec(config.name).image_size, guard)
            with guard.batch(1), torch.inference_mode():
                logits = head(features.to(device)).cpu()
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Non-finite prediction")
            predictions.append({"image": str(path.resolve()), "phase_id": LABELS[logits.argmax(1).item()],
                                "scores": dict(zip(LABELS, logits.softmax(1)[0].tolist()))})
        result = {"status": "passed", "predictions": predictions, "gpu": guard.summary(),
                  "feature_identity": identity, "scores_are_uncalibrated": True}
        write_json(output / "result.json", result)
    return result


def benchmark_worker(request: dict, output: Path) -> dict:
    config = ModelConfig(**request["config"])
    device = resolve_device(config.device)
    resources = Resources(**request["resources"])
    batch, mode = request["batch_size"], request["mode"]
    with GPUControl(device, resources, output / "gpu.csv") as guard:
        if mode == "head":
            cache = torch.load(request["features"], weights_only=True, map_location="cpu")
            x, y = cache["features"], cache["targets"]
            head = new_head(x.shape[1], device, config.seed)
            optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
            env = environment(device, cache["feature_identity"])
        else:
            model, identity = encoder(config, device)
            env = environment(device, identity)
        if request.get("environment"):
            assert_identity(env, request["environment"], "Benchmark")
        records = request["records"]
        offset = 0

        def step():
            nonlocal offset
            indices = [(offset + i) % len(records) for i in range(batch)]
            offset = (offset + batch) % len(records)
            if mode == "head":
                head_step(head, optimizer, x[indices], y[indices], guard)
            else:
                extract_batch(model, [records[i] for i in indices],
                              get_spec(config.name).image_size, guard)

        if mode == "probe":
            original = state_digest(model)
            start = guard.mark()
            features = torch.cat([
                extract_batch(model, records[i:i + batch],
                              get_spec(config.name).image_size, guard)
                for i in range(0, len(records), batch)
            ])
            if original != state_digest(model) or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen probe changed the backbone")
            if list(features.shape) != [len(records), get_spec(config.name).feature_dim]:
                raise RuntimeError("Probe feature shape mismatch")
            save_tensor(output / "probe_features.pt", {
                "features": features, "targets": torch.tensor([r["target"] for r in records]),
                "feature_identity": identity,
            })
            return {"status": "passed", "environment": env, "gpu": guard.summary(start),
                    "batch_size": batch, "frozen_unchanged": True}
        for _ in range(3):
            step()
        guard.idle(2)
        before = query_gpu(guard.hardware["uuid"])
        reserved_before = torch.cuda.memory_reserved(device) / 1024**2
        start = guard.mark()
        while time.perf_counter() - start < request["seconds"]:
            step()
        result = {"status": "passed", "environment": env, "batch_size": batch,
                  "mode": mode, "gpu": guard.summary(start)}
        guard.idle(2)
        after = query_gpu(guard.hardware["uuid"])
        reserved_after = torch.cuda.memory_reserved(device) / 1024**2
        drift = (after["used_mib"] - reserved_after) - (before["used_mib"] - reserved_before)
        result["possible_interference"] = abs(drift) > 512 or after["utilization"] > 20
        result["idle_after"] = after
        result["external_memory_drift_mib"] = drift
        return result


def run_trial(output, name, request):
    """Isolate OOM and allocator state; a failed probe cannot poison the next trial."""
    for attempt in range(2):
        trial = output / (name if attempt == 0 else name + "_repeat")
        trial.mkdir()
        write_json(trial / "request.json", request)
        print(f"GPU trial: {trial.name}, batch={request['batch_size']}, "
              f"seconds={request['seconds']}", flush=True)
        with (trial / "console.log").open("x", encoding="utf-8") as stream:
            completed = subprocess.run(
                [sys.executable, "-m", "surgenet_dino.training", "--worker", str(trial / "request.json")],
                stdout=stream, stderr=subprocess.STDOUT,
                timeout=max(300, request["seconds"] + 180),
            )
        report_path = trial / "result.json"
        if not report_path.is_file():
            raise RuntimeError(f"Benchmark worker failed without report: {trial}")
        result = json.loads(report_path.read_text(encoding="utf-8"))
        result["trial"] = str(trial)
        if completed.returncode:
            if result.get("oom") or "memory budget" in result.get("error", ""):
                return result
            raise RuntimeError(f"GPU trial stopped: {result.get('error')}; see {trial}")
        if not result.get("possible_interference"):
            print(f"  {result['gpu']['images_per_second']:.2f} samples/s, "
                  f"peak allocated {result['gpu']['peak_allocated_gib']:.3f} GiB", flush=True)
            return result
        print("  Possible interference; repeating once with the same settings", flush=True)
    raise RuntimeError(f"Repeated interference in {name}; rerun calibration when GPU is idle")


def select_profile(trials: list) -> dict:
    eligible = [t for t in trials if t["status"] == "passed"]
    if not eligible:
        raise RuntimeError("No GPU batch passed this profile")
    fastest = max(t["score"] for t in eligible)
    return min((t for t in eligible if t["score"] >= .95 * fastest), key=lambda t: t["batch_size"])


def calibrate(args, config, output):
    dataset = read_manifest(args.manifest)
    write_json(output / "dataset.json", dataset)
    device = resolve_device(config.device)
    hardware = gpu_identity(device)
    baseline = []
    for _ in range(15):
        baseline.append(query_gpu(hardware["uuid"]))
        time.sleep(1)
    write_json(output / "idle_baseline.json", {"samples": baseline})
    if statistics.mean(s["utilization"] for s in baseline) > 10:
        raise RuntimeError("Selected GPU is busy before calibration; stop other compute jobs first")
    tiny = subset(dataset["records"], 2, config.seed)
    representative = subset(dataset["records"], 20, config.seed)
    base = {"config": asdict(config), "records": tiny,
            "resources": asdict(resources_for(args, "low")), "mode": "probe", "seconds": 0}
    probe = run_trial(output, "probe_1", {**base, "batch_size": 1})
    if probe["status"] != "passed":
        raise RuntimeError("Batch 1 feasibility probe failed")
    env = probe["environment"]
    base["environment"] = env
    probe4 = run_trial(output, "probe_4", {**base, "batch_size": 4})
    if probe4["status"] != "passed":
        raise RuntimeError("Batch 4 feasibility probe failed")
    profiles = {}
    for profile in PRESETS:
        resources = resources_for(args, profile)
        request = {**base, "resources": asdict(resources), "records": representative,
                   "mode": "extract", "seconds": 30}
        trials, flat = [], 0
        for batch in CANDIDATES:
            trial = run_trial(output, f"{profile}_batch{batch}", {**request, "batch_size": batch})
            if trial["status"] != "passed":
                break
            speed = trial["gpu"]["images_per_second"]
            if trials:
                flat = flat + 1 if speed < trials[-1]["gpu"]["images_per_second"] * 1.05 else 0
            trials.append(trial)
            if flat >= 2:
                break
        finalists = sorted(trials, key=lambda t: t["gpu"]["images_per_second"], reverse=True)[:2]
        for trial in finalists:
            repeat = run_trial(output, f"{profile}_batch{trial['batch_size']}_confirm",
                               {**request, "batch_size": trial["batch_size"], "seconds": 60})
            if repeat["status"] != "passed":
                trial["status"] = "failed"
                continue
            speeds = [trial["gpu"]["images_per_second"], repeat["gpu"]["images_per_second"]]
            trial["score"] = statistics.median(speeds)
            trial["repeat_relative_range"] = (max(speeds) - min(speeds)) / trial["score"]
        chosen = select_profile(finalists)
        soak = run_trial(output, f"{profile}_sustained",
                         {**request, "batch_size": chosen["batch_size"], "seconds": 180})
        if soak["status"] != "passed":
            raise RuntimeError(f"{profile} sustained check failed")
        head = run_trial(output, f"{profile}_head", {
            **request, "records": tiny, "mode": "head", "batch_size": 64, "seconds": 30,
            "features": str(Path(probe["trial"]) / "probe_features.pt"),
        })
        if head["status"] != "passed":
            raise RuntimeError(f"{profile} head check failed")
        profiles[profile] = {"resources": asdict(resources), "batch_size": chosen["batch_size"],
                             "repeat_relative_range": chosen["repeat_relative_range"],
                             "extraction": soak["gpu"], "head": head["gpu"],
                             "estimated_dataset_seconds": len(dataset["records"]) / soak["gpu"]["images_per_second"]}
        write_json(output / f"{profile}.json", profiles[profile])
    result = {"status": "passed", "environment": env, "profiles": profiles,
              "dataset_identity": dataset["identity"], "seed": config.seed}
    write_json(output / "profiles.json", result)
    return result


def run(args, config):
    started = time.perf_counter()
    if config.weights == "none" or config.num_classes != 0:
        raise ValueError("Frozen training requires pretrained features with num_classes=0")
    for name in ("batch_size", "head_batch_size", "epochs"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("learning_rate", "weight_decay"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value < 0
                                  or (name == "learning_rate" and value == 0)):
            raise ValueError(f"Invalid {name}")
    torch.set_num_threads(resources_for(args).cpu_threads)
    if resolve_device(config.device).type != "cuda":
        raise ValueError("Training requires a CUDA device; no CPU fallback")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"command": args.command, "model": asdict(config),
                                    "arguments": {k: str(v) if isinstance(v, Path) else v
                                                  for k, v in vars(args).items() if k != "image"},
                                    "images": [str(p) for p in getattr(args, "image", [])]})
    try:
        if args.command == "calibrate-gpu":
            result = calibrate(args, config, output)
        elif args.command == "cache-features":
            result = cache_features(args, config, output)
        elif args.command == "train-head":
            result = train_head(args, config, output)
        else:
            result = predict_head(args, config, output)
        write_json(output / "timing.json", {"command_seconds": time.perf_counter() - started,
                                            "includes_input_validation_and_model_loading": True})
        print(f"Completed {args.command}: {output}", flush=True)
        return 0 if result["status"] == "passed" else 2
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "failed", "error": str(exc),
                                            "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Use python -m surgenet_dino --help")
    request_path = Path(sys.argv[2])
    try:
        response = benchmark_worker(json.loads(request_path.read_text(encoding="utf-8")),
                                    request_path.parent)
    except Exception as exc:
        response = {"status": "failed", "error": str(exc), "type": type(exc).__name__,
                    "oom": isinstance(exc, torch.cuda.OutOfMemoryError)}
    write_json(request_path.parent / "result.json", response)
    raise SystemExit(0 if response["status"] == "passed" else 2)
