"""Extract only Basic template images, without annotations or dense-frame features.

Run with the existing isolated OpenCV Python; the training environment stays unchanged.
Original zero-based frame indices are preserved, matching the official extractor.
"""

import argparse
import json
from pathlib import Path

from surgenet_dino.temporal.template import read_template


def extract(source, template, output):
    import cv2

    names = read_template(template)
    frames = {}
    for name in names:
        frames.setdefault(name[:2], []).append(int(name[3:]))
    if set(frames) != {'02', '07', '11', '15'}:
        raise ValueError('Expected all four official Basic evaluation videos')
    videos = {}
    for video in sorted(frames):
        matches = sorted(source.rglob(f'HD_videos/{video}.mp4'))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one source video {video}.mp4; found {len(matches)}')
        videos[video] = matches[0]
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    counts = {}
    for video, path in videos.items():
        wanted = set(frames[video])
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open video: {path}')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if abs(fps - 30) > 1e-6 or not fps > 0:
                raise ValueError(f'Expected 30 fps, got {fps}: {path}')
            directory = output / video / 'images'
            directory.mkdir(parents=True)
            saved = 0
            # 顺序定位原始帧；只取模板要求的 1 fps 像素，不计算更密帧的图像特征。
            for frame in range(max(wanted) + 1):
                if not cap.grab():
                    raise RuntimeError(f'Video ended before required frame {frame}: {path}')
                if frame not in wanted:
                    continue
                ok, pixels = cap.retrieve()
                if not ok:
                    raise RuntimeError(f'Cannot decode required frame {frame}: {path}')
                image = directory / f'{video}_{frame:06d}.png'
                if not cv2.imwrite(str(image), pixels):
                    raise OSError(f'Cannot write image: {image}')
                saved += 1
                if saved % 500 == 0:
                    print(f'{video}: saved {saved}/{len(wanted)} images', flush=True)
            counts[video] = saved
            print(f'{video}: completed {saved} images', flush=True)
        finally:
            cap.release()
    report = {'status': 'passed', 'source_videos': {v: str(p.resolve()) for v, p in videos.items()},
              'template': str(template.resolve()), 'images': counts,
              'total_images': len(names), 'annotation_used': False,
              'sampling': 'official template only, zero-based original frames, 30 fps / 1 fps'}
    with (output / 'extraction.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    extract(args.source, args.template, args.output_dir)


if __name__ == '__main__':
    main()
