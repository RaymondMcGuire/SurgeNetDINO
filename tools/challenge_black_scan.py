"""Advanced only: detect anonymised x intervals directly from every video's pixels.

Run with the existing isolated OpenCV environment; no training dependencies.
Intervals are zero-based and half-open. No annotations or Basic template enter.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import tempfile


def scan(path, threshold=16, minimum_frames=60, cpu_threads=2):
    import cv2
    cv2.setNumThreads(cpu_threads)
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise ValueError(f'Cannot decode video: {path}')
    fps, expected = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(fps - 30) > 1e-6:
        cap.release()
        raise ValueError('Challenge frame indices require 30 fps')
    intervals, frame, start = [], 0, None
    try:
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            black = int(pixels.max()) <= threshold
            if black and start is None:
                start = frame
            elif not black and start is not None:
                intervals.append([start, frame])
                start = None
            frame += 1
            if frame % 30000 == 0:
                print(f'{path.stem}: decoded {frame}/{expected} frames', flush=True)
    finally:
        cap.release()
    if frame != expected:
        raise RuntimeError(f'Incomplete video decode: {frame} versus {expected}')
    if start is not None:
        intervals.append([start, frame])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'total_frames': frame, 'fps': fps,
            'intervals': [p for p in intervals if p[1] - p[0] >= minimum_frames],
            'discarded_short_intervals': [p for p in intervals if p[1] - p[0] < minimum_frames],
            'source_path': str(path.resolve()), 'source_sha256': digest.hexdigest()}


def self_check():
    import cv2
    import numpy as np
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'test.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (32, 32))
        if not writer.isOpened():
            raise RuntimeError('Cannot encode the synthetic scan check')
        for i in range(90):
            # A short black run is rejected; a final 60-frame run reaches EOF.
            value = 0 if 10 <= i < 20 or i >= 30 else 255
            writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
        writer.release()
        result = scan(path)
        assert result['total_frames'] == 90
        assert result['intervals'] == [[30, 90]]
        assert result['discarded_short_intervals'] == [[10, 20]]
    print('Synthetic pixel-scan check passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_check()
        return
    if not args.videos or args.output is None:
        parser.error('Provide --videos and --output, or --self-test')
    if args.output.exists() or len({p.stem for p in args.videos}) != len(args.videos):
        raise ValueError('Choose a new output and unique video identities')
    started = time.perf_counter()
    result = {'source': 'pixel_scan', 'annotation_used': False, 'threshold_max_rgb': 16,
              'minimum_frames': 60, 'cpu_threads': 2, 'decoder_threads': 1, 'gpu_used': False,
              'videos': {p.stem: scan(p) for p in args.videos}, 'seconds': time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'seconds': result['seconds'], 'videos': {v: len(r['intervals']) for v, r in result['videos'].items()}}))


if __name__ == '__main__':
    main()
