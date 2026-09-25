"""Per-second motion features of the catalogued 1 fps frames, read from the 30 fps source videos.

Runs in the isolated preprocessing environment, never in the training environment (docs/DATASET.md):

  uv run --isolated --no-project --python 3.12 --with opencv-python-headless==5.0.0.93 python tools/motion_features.py
      extract --catalogue H:/dataset/JSCAS_preprocessed/labeled_1fps_v1/samples.csv
      --source "H:/dataset/JSCAS AI Challenge" --output H:/dataset/JSCAS_preprocessed/motion_v1

Label-free: only video_id and frame_index of the catalogue are read. For the catalogued frame f the
window is the 30 frames [f-15, f+15): frame-difference energy at 128x72 and DIS optical flow between
every third frame at 320x180 (global = camera-like coherent motion, local = residual, e.g. instruments).
One CSV row per catalogued frame; magnitudes are raw (non-negative), compressed and standardised by
the consumer (surgenet_dino/temporal/motion.py).
"""

import argparse
import csv
import json
from collections import deque
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HALF = 15  # window [f-HALF, f+HALF) around the catalogued frame f
WINDOW = 2 * HALF
FLOW_STEP = 3
DIFF_SIZE = (128, 72)
FLOW_SIZE = (320, 180)
COLUMNS = ["diff_mean", "diff_max", "diff_std", "flow_mean", "flow_max", "flow_global", "flow_local",
           *[f"grid_{r}{c}" for r in range(3) for c in range(3)]]


def features(frames, dis):
    """16 non-negative floats from WINDOW consecutive gray frames."""
    small = [cv2.resize(f, DIFF_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames]
    diffs = np.array([np.abs(a - b).mean() for a, b in zip(small, small[1:])])
    coarse = [cv2.resize(f, FLOW_SIZE, interpolation=cv2.INTER_AREA) for f in frames[::FLOW_STEP]]
    means, globals_, locals_, grids = [], [], [], []
    for a, b in zip(coarse, coarse[1:]):
        flow = dis.calc(a, b, None)
        mx, my = float(flow[..., 0].mean()), float(flow[..., 1].mean())
        magnitude = np.hypot(flow[..., 0], flow[..., 1])
        means.append(magnitude.mean())
        globals_.append(np.hypot(mx, my))
        locals_.append(np.hypot(flow[..., 0] - mx, flow[..., 1] - my).mean())
        grids.append(cv2.resize(magnitude, (3, 3), interpolation=cv2.INTER_AREA).ravel())
    return [float(v) for v in (diffs.mean(), diffs.max(), diffs.std(), np.mean(means), np.max(means),
                               np.mean(globals_), np.mean(locals_), *np.mean(grids, 0))]


def process_video(job):
    """Sequential decode; frames outside every window are grabbed without conversion."""
    video, path, wanted = job
    started = time.monotonic()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"{video}: cannot open {path}")
    total, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    last = max(wanted) + HALF - 1
    if min(wanted) < HALF or last >= total:
        raise ValueError(f"{video}: catalogued frames {min(wanted)}..{max(wanted)} do not fit {total} frames")
    needed = np.zeros(last + 1, dtype=bool)
    for f in wanted:
        needed[f - HALF:f + HALF] = True
    wanted, buffer, run, rows = set(wanted), deque(maxlen=WINDOW), 0, []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    for i in range(last + 1):
        if not needed[i]:
            if not capture.grab():
                raise RuntimeError(f"{video}: decode failed at frame {i}")
            run = 0
            continue
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"{video}: decode failed at frame {i}")
        buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        run += 1
        f = i - HALF + 1
        if f in wanted:
            if run < WINDOW:
                raise RuntimeError(f"{video}: incomplete window for frame {f}")
            rows.append([video, f, *features(list(buffer), dis)])
        if i % 60000 == 0:
            print(f"{video}: frame {i}/{last}", flush=True)
    if len(rows) != len(wanted):
        raise RuntimeError(f"{video}: {len(rows)} rows for {len(wanted)} catalogued frames")
    return rows, {"video_id": video, "source": str(path), "bytes": path.stat().st_size, "frame_count": total,
                  "fps": fps, "rows": len(rows), "seconds": round(time.monotonic() - started, 1)}


def read_catalogue(path):
    wanted = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            wanted.setdefault(row["video_id"], set()).add(int(row["frame_index"]))
    if not wanted:
        raise ValueError(f"Empty catalogue: {path}")
    return {v: sorted(f) for v, f in sorted(wanted.items())}


def extract(catalogue, source, output, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    wanted = read_catalogue(catalogue)
    jobs = []
    for video, frames in wanted.items():
        found = sorted(Path(source).glob(f"*/*/HD_videos/{video}.mp4"))
        if len(found) != 1:
            raise FileNotFoundError(f"{video}: expected one HD_videos/{video}.mp4 under {source}, found {found}")
        jobs.append((video, found[0], frames))
    started = time.monotonic()
    with Pool(workers) as pool:
        results = pool.map(process_video, jobs, chunksize=1)
    rows = sorted((row for part, _ in results for row in part), key=lambda r: (r[0], r[1]))
    with (output / "motion.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "frame_index", *COLUMNS])
        writer.writerows(rows)
    meta = {"format": "surgenet-motion-v1", "catalogue": str(Path(catalogue).resolve()), "columns": COLUMNS,
            "window_frames": [-HALF, HALF], "flow_step": FLOW_STEP, "diff_size": DIFF_SIZE, "flow_size": FLOW_SIZE,
            "flow": "DIS FAST", "opencv": cv2.__version__, "videos": [info for _, info in results],
            "rows": len(rows), "wall_seconds": round(time.monotonic() - started, 1)}
    (output / "motion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(rows)} rows -> {output / 'motion.csv'} in {meta['wall_seconds']} s", flush=True)


def self_test(directory):
    """A static second must show less motion than a second with a moving square."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(120):
        frame = np.full((240, 320, 3), 90, dtype=np.uint8)
        cv2.circle(frame, (160, 120), 40, (30, 30, 30), -1)
        if i >= 60:
            cv2.rectangle(frame, (20 + 3 * (i - 60), 60), (60 + 3 * (i - 60), 100), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()
    rows, info = process_video(("99", path, [30, 90]))
    static, moving = rows[0][2:], rows[1][2:]
    assert info["rows"] == 2 and all(np.isfinite(v) and v >= 0 for v in static + moving)
    assert moving[0] > static[0] and moving[3] > static[3] and moving[6] > static[6], (static, moving)
    print("self-test passed", dict(zip(COLUMNS, moving)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("extract")
    run.add_argument("--catalogue", type=Path, required=True, help="samples.csv with video_id and frame_index")
    run.add_argument("--source", type=Path, required=True, help="Root containing */*/HD_videos/XX.mp4")
    run.add_argument("--output", type=Path, required=True, help="New directory for motion.csv and motion.json")
    run.add_argument("--workers", type=int, default=4)
    test = subs.add_parser("self-test")
    test.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract(args.catalogue, args.source, args.output, args.workers)
    else:
        self_test(args.directory)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, AssertionError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
