"""Single-GPU budgets, sampled telemetry and cooperative pacing (not a hardware cap)."""

import csv
import math
import os
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PRESETS = {"low": (0.20, 0.30), "medium": (0.50, 0.60), "high": (0.80, 0.80)}
GIB = 1024**3


@dataclass(frozen=True)
class Resources:
    profile: str = "low"
    memory_fraction: float = 0.20
    duty_cycle: float = 0.30
    max_temperature: float = 75.0
    cpu_threads: int = 2

    def __post_init__(self):
        if self.profile not in PRESETS:
            raise ValueError("Unknown GPU profile")
        for name in ("memory_fraction", "duty_cycle"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if (type(self.max_temperature) not in (int, float)
                or not math.isfinite(self.max_temperature) or not 0 < self.max_temperature <= 100):
            raise ValueError("max_temperature must be in (0, 100]")
        if type(self.cpu_threads) is not int or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")


def query_gpu(uuid: str) -> dict:
    fields = "uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw"
    result = subprocess.run(
        ["nvidia-smi", "-i", uuid, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(f"GPU monitoring failed: {result.stderr.strip()}")
    row = next(csv.reader(result.stdout.splitlines()))
    if len(row) != 8:
        raise RuntimeError("Incomplete GPU telemetry")
    values = [float(v.strip()) for v in row[3:]]  # N/A is an error, never a zero reading.
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("Non-finite GPU telemetry")
    return dict(zip(
        ("uuid", "name", "driver", "total_mib", "used_mib", "utilization", "temperature", "watts"),
        [v.strip() for v in row[:3]] + values,
    ))


def gpu_identity(device: torch.device) -> dict:
    if device.type != "cuda":
        raise ValueError("GPU resource control requires an explicit CUDA device")
    prop = torch.cuda.get_device_properties(device)
    sample = query_gpu("GPU-" + str(prop.uuid).removeprefix("GPU-"))
    return {k: sample[k] for k in ("uuid", "name", "driver", "total_mib")}


def sleep_seconds(busy: float, elapsed: float, duty: float) -> float:
    """CPU loading since the preceding batch already counts as idle time."""
    return max(0.0, busy / duty - elapsed)


def check_sample(sample: dict, resources: Resources, now: float) -> None:
    if now - sample["monotonic"] > 5:
        raise RuntimeError("GPU telemetry stale for more than 5 seconds")
    if sample["temperature"] >= resources.max_temperature:
        raise RuntimeError("GPU temperature stop threshold reached")
    if sample["used_mib"] >= sample["total_mib"] * resources.memory_fraction:
        raise RuntimeError("GPU board memory budget reached")


class GPUControl:
    """One monitor per process; all GPU batches share this pacing/stop boundary."""

    def __init__(self, device: torch.device, resources: Resources, log: Path):
        self.device, self.resources, self.log = device, resources, log
        self.hardware = gpu_identity(device)
        self.samples, self.batches = [], []
        self.error = None
        self.stop = threading.Event()
        self.thread = None
        self.last_end = time.perf_counter()

    def __enter__(self):
        torch.set_num_threads(self.resources.cpu_threads)
        torch.cuda.set_device(self.device)
        baseline = query_gpu(self.hardware["uuid"])
        total = torch.cuda.get_device_properties(self.device).total_memory
        reserve = max(GIB, 0.05 * total)
        self.allocator_bytes = int(total * self.resources.memory_fraction
                                   - baseline["used_mib"] * 1024**2 - reserve)
        if self.allocator_bytes <= 0:
            raise RuntimeError("Insufficient free memory within selected GPU budget")
        torch.cuda.set_per_process_memory_fraction(self.allocator_bytes / total, self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.log.open("x", newline="", encoding="utf-8")
        fields = ["time", "monotonic", "used_mib", "total_mib", "utilization", "temperature",
                  "watts", "allocated_mib", "reserved_mib"]
        self.writer = csv.DictWriter(self.stream, fieldnames=fields, extrasaction="ignore")
        self.writer.writeheader()
        try:
            self._sample()
            self.check()
        except BaseException:
            self.stream.close()
            raise
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        sample = query_gpu(self.hardware["uuid"])
        sample.update(time=time.time(), monotonic=time.perf_counter(),
                      allocated_mib=torch.cuda.memory_allocated(self.device) / 1024**2,
                      reserved_mib=torch.cuda.memory_reserved(self.device) / 1024**2)
        self.samples.append(sample)
        self.writer.writerow(sample)
        self.stream.flush()

    def _monitor(self):
        while not self.stop.wait(1):
            try:
                self._sample()
                check_sample(self.samples[-1], self.resources, time.perf_counter())
            except Exception as exc:
                self.error = str(exc)
                return

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        check_sample(self.samples[-1], self.resources, time.perf_counter())

    def idle(self, seconds: float):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.check()
            time.sleep(min(0.25, max(0, end - time.perf_counter())))

    @contextmanager
    def batch(self, count: int):
        self.check()
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        done = time.perf_counter()
        busy = done - started
        pause = sleep_seconds(busy, done - self.last_end, self.resources.duty_cycle)
        self.idle(pause)
        end = time.perf_counter()
        self.batches.append({"monotonic": end, "count": count, "busy_seconds": busy,
                             "sleep_seconds": end - done, "wall_seconds": end - self.last_end})
        self.last_end = end
        self.check()

    def mark(self):
        self.last_end = time.perf_counter()
        return self.last_end

    def summary(self, since: float = 0) -> dict:
        samples = [s for s in self.samples if s["monotonic"] >= since]
        batches = [b for b in self.batches if b["monotonic"] >= since]
        wall = sum(b["wall_seconds"] for b in batches)
        result = {"hardware": self.hardware, "resources": asdict(self.resources),
                  "allocator_budget_gib": self.allocator_bytes / GIB,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / GIB,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / GIB,
                  "batches": len(batches), "images": sum(b["count"] for b in batches),
                  "seconds": wall, "samples": len(samples)}
        result["images_per_second"] = result["images"] / wall if wall else 0
        for key in ("busy_seconds", "sleep_seconds"):
            result["mean_" + key] = statistics.mean(b[key] for b in batches) if batches else 0
        if samples:
            result.update(mean_utilization=statistics.mean(s["utilization"] for s in samples),
                          mean_watts=statistics.mean(s["watts"] for s in samples),
                          max_temperature=max(s["temperature"] for s in samples),
                          max_board_used_gib=max(s["used_mib"] for s in samples) / 1024)
        return result

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)
        self.stream.close()
