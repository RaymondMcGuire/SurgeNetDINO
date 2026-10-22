# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "opencv-python-headless==5.0.0.93",
#   "pandas==3.0.6",
#   "pillow==12.3.0",
# ]
# ///
"""JSCAS: official 1 fps extraction, audited allowlists, independent splits.

Run with Python 3.12, opencv-python-headless 5.0.0.93, pandas 3.0.6,
pillow 12.3.0. All 12 labeled videos are extracted from scratch, including 01.
Use the `prepare` command to extract, audit and create a training split in one run.
Sources and existing outputs are never overwritten; evaluation videos are excluded.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import shutil
import sys
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '81', '82', '83', '9', '13']
LABELED = ['01', '03', '04', '05', '06', '08', '09', '10', '12', '13', '14', '16']
OFFICIAL_TEST = ['02', '07', '11', '15']
SAMPLE_FIELDS = ['video_id', 'filename', 'frame_index', 'time_seconds', 'phase-id',
                 'image_path', 'image_sha256']
POLICY = {'sample_fps': 1, 'frame_index_base': 0, 'excluded_labels': ['x', 'e'],
          'whole_image_rgb_max_at_most': 1, 'resize': False, 'normalize': False,
          'augment': False, 'preserve_excluded_images': True}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path, fields=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if fields is not None:
            check(reader.fieldnames == fields, f'Unexpected CSV fields: {path}')
        return list(reader)


def write_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    # Only status reports are replaceable; image/manifest outputs are exclusive.
    tmp = Path(str(path) + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def exclusion(label, maximum):
    if label in {'x', 'e'}:
        return 'phase_x' if label == 'x' else 'end_marker'
    check(label in LABELS, f'Unknown phase: {label}')
    return 'all_pixels_at_most_1' if maximum <= 1 else ''


def ranges(annotation, frame_count):
    rows = read_csv(annotation, ['start-frame', 'phase-id'])
    starts = [int(row['start-frame']) for row in rows]
    check(len(rows) >= 2 and starts[0] == 0, 'Annotation must start at frame 0')
    check(all(a < b for a, b in zip(starts, starts[1:])), 'Unordered annotation')
    check(rows[-1]['phase-id'] == 'e', 'Missing final end marker')
    check(all(row['phase-id'] in LABELS + ['x'] for row in rows[:-1]), 'Unknown phase')
    check(starts[-1] in (frame_count, frame_count - 1), 'Unexpected end marker offset')
    return rows, starts


def official_module(path):
    spec = importlib.util.spec_from_file_location('official_extract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_video(job):
    video_id, video, annotation, extractor, case, audit_dir = map(Path, job)
    video_id = str(video_id)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    report_path = audit_dir / 'report.json'
    report = {'video_id': video_id, 'status': 'running', 'source_video': str(video),
              'source_annotation': str(annotation), 'case_dir': str(case)}
    write_json(report_path, report)
    try:
        cap = cv2.VideoCapture(str(video))
        check(cap.isOpened(), f'Cannot open {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        check(fps == 30.0 and size == (1280, 720), f'Unexpected video format: {video}')
        rows, starts = ranges(annotation, frame_count)
        expected = {f'{video_id}_{frame:06d}': rows[bisect_right(starts, frame) - 1]['phase-id']
                    for frame in range(0, frame_count, 30)}
        check('e' not in expected.values(), 'An end marker falls on the sampling grid')
        raw_expected = {name: label for name, label in expected.items() if label != 'x'}
        data = case / 'basic_challenge' / 'data' / video_id
        case.mkdir(parents=True, exist_ok=False)
        (case / 'source').mkdir()
        shutil.copy2(annotation, case / 'source' / annotation.name)
        shutil.copy2(extractor, case / 'source' / 'extract_frames.py')
        print(f'[{video_id}] extracting {len(expected)} frames with official script', flush=True)
        official_module(extractor).process_video(video, annotation.parent,
                                                case / 'basic_challenge' / 'data',
                                                frame_step_sec=1, ignore_phase=['x'])
        images = sorted((data / 'images').glob('*.png'))
        check({path.stem for path in images} == set(expected), f'Image inventory mismatch: {video_id}')
        raw = read_csv(data / f'annotation_{video_id}.csv', ['filename', 'phase-id'])
        check(len(raw) == len(raw_expected), 'Raw manifest count mismatch')
        check({r['filename']: r['phase-id'] for r in raw} == raw_expected, 'Raw labels mismatch')
        accepted, excluded, samples = [], [], []
        image_bytes = 0
        preview = Image.new('RGB', (320 * 4, 210 * 4), 'white')
        draw = ImageDraw.Draw(preview)
        preview_names = {}
        preview_targets = {}
        for label in LABELS:
            names = [name for name, phase in raw_expected.items() if phase == label]
            if names:
                preview_targets[label] = names[len(names) // 2]
        for i, path in enumerate(images, 1):
            label = expected[path.stem]
            with Image.open(path) as image:
                image.load()
                check(image.mode == 'RGB' and image.size == size, f'Bad image: {path}')
                maximum = max(high for low, high in image.getextrema())
                reason = exclusion(label, maximum)
                if preview_targets.get(label) == path.stem:
                    index = LABELS.index(label)
                    x, y = index % 4 * 320, index // 4 * 210
                    preview.paste(image.resize((320, 180)), (x, y))
                    draw.text((x + 4, y + 184), f'phase={label} {path.stem}', fill='black')
                    preview_names[label] = path.stem
            image_bytes += path.stat().st_size
            if reason:
                excluded.append({'filename': path.stem, 'phase-id': label, 'reason': reason,
                                 'max_pixel': maximum})
            else:
                accepted.append({'filename': path.stem, 'phase-id': label})
                frame = int(path.stem.split('_')[1])
                samples.append({'video_id': video_id, 'filename': path.stem, 'frame_index': frame,
                                'time_seconds': frame // 30, 'phase-id': label,
                                'image_path': str(path.resolve()), 'image_sha256': sha(path)})
            if i % 1000 == 0:
                print(f'[{video_id}] decoded/checked {i}/{len(images)} PNGs', flush=True)
        check(len(accepted) + len(excluded) == len(images), 'Incomplete partition')
        check(not ({r['filename'] for r in accepted} & {r['filename'] for r in excluded}), 'Overlap')
        classification = data / f'classification_{video_id}.csv'
        excluded_csv = data / f'excluded_{video_id}.csv'
        write_csv(classification, ['filename', 'phase-id'], accepted)
        write_csv(excluded_csv, ['filename', 'phase-id', 'reason', 'max_pixel'], excluded)
        write_csv(audit_dir / 'samples.csv', SAMPLE_FIELDS, samples)
        preview.save(audit_dir / 'preview.jpg', quality=90)
        # Independently seek a fixed, deterministic sample of original frames and compare every pixel.
        indices = sorted(set([0, frame_count // 2 // 30 * 30, (frame_count - 1) // 30 * 30]
                             + [((n + 29) // 30) * 30 for n in starts[1:-1]][::5]))
        cap = cv2.VideoCapture(str(video))
        for frame in indices:
            check(cap.set(cv2.CAP_PROP_POS_FRAMES, frame), f'Cannot seek frame {frame}')
            ok, original = cap.read()
            check(ok, f'Cannot decode original frame {frame}')
            with Image.open(data / 'images' / f'{video_id}_{frame:06d}.png') as image:
                check(image.tobytes() == cv2.cvtColor(original, cv2.COLOR_BGR2RGB).tobytes(),
                      f'Frame/pixel alignment mismatch: {video_id}:{frame}')
        cap.release()
        counts = Counter(row['phase-id'] for row in accepted)
        report.update(status='passed', reused_existing_case=False, fps=fps,
                      source_frame_count=frame_count, image_size=list(size), policy=POLICY,
                      end_marker_offset=starts[-1] - frame_count, saved_images=len(images),
                      raw_annotation_rows=len(raw), accepted_images=len(accepted),
                      excluded_images=len(excluded), excluded_reasons=dict(Counter(r['reason'] for r in excluded)),
                      accepted_phase_counts={label: counts[label] for label in LABELS},
                      missing_phases=[label for label in LABELS if not counts[label]],
                      source_video_bytes=video.stat().st_size, source_video_sha256=sha(video),
                      source_annotation_sha256=sha(annotation), source_script_sha256=sha(extractor),
                      classification_csv=str(classification), classification_csv_sha256=sha(classification),
                      excluded_csv=str(excluded_csv), excluded_csv_sha256=sha(excluded_csv),
                      samples_csv_sha256=sha(audit_dir / 'samples.csv'), image_bytes=image_bytes,
                      exact_pixel_check_frames=indices, preview_samples=preview_names,
                      checks=['all expected frames present', 'all PNGs decoded', 'all labels matched',
                              'all accepted images hashed', 'accepted/excluded form complete disjoint partition',
                              'sampled original frames pixel-identical'],
                      elapsed_seconds=round(time.perf_counter() - started, 3))
        write_json(case / 'processing.json', report)
        print(f'[{video_id}] PASSED: {len(accepted)} accepted, {len(excluded)} excluded', flush=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(report_path, report)
    return report


def extract(args):
    source = args.source.resolve()
    args.output = args.output.resolve()
    check(source.is_dir(), f'Source directory not found: {source}')
    check(not args.output.parent.is_relative_to(source)
          and not source.is_relative_to(args.output.parent), 'Keep output roots separate from source data')
    annotation_dirs = [path for path in source.rglob('annotation_csv') if path.is_dir()]
    check(len(annotation_dirs) == 1, 'Expected exactly one annotation_csv directory')
    annotation_dir = annotation_dirs[0]
    annotations = {p.stem: p for p in annotation_dir.glob('*.csv')}
    check(sorted(annotations) == LABELED, 'Unexpected labeled video list')
    extractor = annotation_dir.parent / 'extract_frames.py'
    check(extractor.is_file(), f'Official extract_frames.py not found: {extractor}')
    video_paths = list(source.rglob('HD_videos/*.mp4'))
    videos = {p.stem: p for p in video_paths}
    check(len(video_paths) == len(videos), 'Duplicate video IDs')
    check(all(video_id in videos for video_id in LABELED), 'Missing labeled video')
    check(1 <= args.workers <= 3, 'Use 1 to 3 workers')
    check(not args.output.exists(), f'Dataset output already exists: {args.output}')
    for video_id in LABELED:
        check(not (args.output.parent / f'video{video_id}_1fps').exists(), f'Case {video_id} already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    check(shutil.disk_usage(args.output.parent).free > 120 * 1024**3, 'Need 120 GiB free space')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / 'prepare_labeled.py')
    shutil.copy2(extractor, args.output / 'extract_frames.py')
    for video_id in LABELED:
        (args.output / video_id).mkdir()
    metadata = {'status': 'running', 'created_at': datetime.now(timezone.utc).isoformat(),
                'source_root': str(source), 'label_order': LABELS, 'policy': POLICY,
                'labeled_video_ids': LABELED, 'official_test_ids_not_processed': OFFICIAL_TEST,
                'script_sha256': sha(__file__), 'official_script_sha256': sha(extractor),
                'workers': args.workers, 'python': sys.version,
                'versions': {name: importlib.metadata.version(name) for name in
                             ['opencv-python-headless', 'pandas', 'pillow']}}
    write_json(args.output / 'dataset.json', metadata)
    try:
        jobs = [[v, str(videos[v]), str(annotations[v]), str(extractor),
                 str(args.output.parent / f'video{v}_1fps'), str(args.output / v)] for v in LABELED]
        reports = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare_video, job) for job in jobs]
            for future in as_completed(futures):
                reports.append(future.result())
        reports.sort(key=lambda row: row['video_id'])
        samples = [row for v in LABELED for row in read_csv(args.output / v / 'samples.csv', SAMPLE_FIELDS)]
        check(len({r['filename'] for r in samples}) == len(samples), 'Duplicate sample IDs')
        write_csv(args.output / 'samples.csv', SAMPLE_FIELDS, samples)
        count_rows = [{'video_id': r['video_id'], 'phase-id': label, 'image_count': r['accepted_phase_counts'][label]}
                      for r in reports for label in LABELS]
        write_csv(args.output / 'class_counts.csv', ['video_id', 'phase-id', 'image_count'], count_rows)
        metadata.update(status='passed', videos=reports, samples_csv_sha256=sha(args.output / 'samples.csv'),
                        accepted_images=len(samples), saved_images=sum(r['saved_images'] for r in reports),
                        excluded_images=sum(r['excluded_images'] for r in reports),
                        excluded_reasons=dict(sum((Counter(r['excluded_reasons']) for r in reports), Counter())),
                        image_bytes=sum(r['image_bytes'] for r in reports))
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(args.output / 'dataset.json', metadata)


def choose_split(samples, seed):
    ids = sorted({r['video_id'] for r in samples})
    check(ids == LABELED, 'Split expects the 12 labeled videos')
    totals = Counter(r['phase-id'] for r in samples)
    check(set(totals) == set(LABELS), 'Dataset does not cover all classes')
    candidates = list(combinations([v for v in ids if v != '01'], 3))
    random.Random(seed).shuffle(candidates)
    for validation_ids in candidates:
        val = Counter(r['phase-id'] for r in samples if r['video_id'] in validation_ids)
        if (0.20 <= sum(val.values()) / len(samples) <= 0.30
                and all(0.10 <= val[label] / totals[label] <= 0.40 for label in LABELS)):
            return [v for v in ids if v not in validation_ids], list(validation_ids)
    raise ValueError('No split satisfies coverage/balance constraints; inspect counts')


def split(args):
    dataset_file = args.dataset / 'dataset.json'
    dataset = json.loads(dataset_file.read_text(encoding='utf-8'))
    check(dataset['status'] == 'passed', 'Dataset audit has not passed')
    check(sha(args.dataset / 'samples.csv') == dataset['samples_csv_sha256'], 'Dataset manifest changed')
    samples = read_csv(args.dataset / 'samples.csv', SAMPLE_FIELDS)
    check(len(samples) == dataset['accepted_images'], 'Dataset count mismatch')
    check(len({r['filename'] for r in samples}) == len(samples), 'Duplicate sample IDs')
    for video in dataset['videos']:
        check(sha(video['classification_csv']) == video['classification_csv_sha256'], 'Allowlist changed')
    for row in samples:
        check(Path(row['image_path']).is_file(), f'Missing image: {row["image_path"]}')
    train_ids, val_ids = choose_split(samples, args.seed)
    train = [r for r in samples if r['video_id'] in train_ids]
    val = [r for r in samples if r['video_id'] in val_ids]
    check(not ({r['image_sha256'] for r in train} & {r['image_sha256'] for r in val}),
          'Identical PNG contents cross training/validation; inspect before splitting')
    check(not (set(train_ids) & set(val_ids)), 'Video leakage')
    check(len(train) + len(val) == len(samples), 'Partition incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name, data in [('train', train), ('validation', val)]:
        path = args.output / f'{name}.csv'
        write_csv(path, SAMPLE_FIELDS, data)
        check(read_csv(path) == data, f'{name} CSV readback mismatch')
        counts[name] = dict(Counter(r['phase-id'] for r in data))
    result = {'status': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(),
              'dataset_json': str(dataset_file.resolve()), 'dataset_json_sha256': sha(dataset_file),
              'samples_csv_sha256': dataset['samples_csv_sha256'], 'label_order': LABELS,
              'script_sha256': sha(__file__), 'seed': args.seed, 'group_key': 'video_id',
              'patient_grouping': 'No patient linkage metadata supplied; each video is treated as one surgery.',
              'selection': 'First eligible tuple from seeded shuffled combinations; 01 fixed in train; '
                           'validation has 3 videos, 20-30% of frames and 10-40% of each class; no model scores used.',
              'train_video_ids': train_ids, 'validation_video_ids': val_ids,
              'train_samples': len(train), 'validation_samples': len(val), 'class_counts': counts,
              'train_csv_sha256': sha(args.output / 'train.csv'),
              'validation_csv_sha256': sha(args.output / 'validation.csv'),
              'official_test_ids_not_used': OFFICIAL_TEST,
              'checks': ['all 13 classes in both partitions', 'disjoint video IDs',
                         'disjoint sample IDs and PNG hashes', 'complete partition', 'CSV readback matched']}
    write_json(args.output / 'split.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def self_test():
    import tempfile
    check(exclusion('x', 255) == 'phase_x', 'x filtering')
    check(exclusion('e', 255) == 'end_marker', 'e filtering')
    check(exclusion('81', 1) and not exclusion('81', 2), 'black threshold')
    with tempfile.TemporaryDirectory() as root:
        annotation = Path(root) / 'annotation.csv'
        write_csv(annotation, ['start-frame', 'phase-id'], [
            {'start-frame': 0, 'phase-id': 'x'}, {'start-frame': 30, 'phase-id': '81'},
            {'start-frame': 60, 'phase-id': 'e'}])
        rows, starts = ranges(annotation, 61)
        check(rows[bisect_right(starts, 29) - 1]['phase-id'] == 'x', 'left boundary')
        check(rows[bisect_right(starts, 30) - 1]['phase-id'] == '81', 'right boundary')
        try:
            ranges(annotation, 65)
        except ValueError:
            pass
        else:
            raise AssertionError('Unexpected end offset accepted')
    samples = [{'video_id': v, 'phase-id': label} for v in LABELED for label in LABELS]
    train, val = choose_split(samples, 42)
    check(len(train) == 9 and len(val) == 3 and '01' in train and not set(train) & set(val), 'group split')
    check((train, val) == choose_split(samples, 42), 'split reproducibility')
    print('self-test passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    extract_parser = sub.add_parser('extract')
    prepare_parser = sub.add_parser('prepare', help='Extract all labeled videos, audit and create a 9/3 split')
    for command in (extract_parser, prepare_parser):
        command.add_argument('--source', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True, help='New dataset index directory')
        command.add_argument('--workers', type=int, default=2)
    prepare_parser.add_argument('--split-output', type=Path, required=True)
    prepare_parser.add_argument('--seed', type=int, default=42)
    split_parser = sub.add_parser('split')
    split_parser.add_argument('--dataset', type=Path, required=True)
    split_parser.add_argument('--output', type=Path, required=True)
    split_parser.add_argument('--seed', type=int, default=42)
    sub.add_parser('self-test')
    args = parser.parse_args()
    if args.command == 'self-test':
        self_test()
    elif args.command == 'extract':
        extract(args)
    elif args.command == 'prepare':
        check(not args.split_output.exists(), f'Split output already exists: {args.split_output}')
        check(not args.split_output.resolve().is_relative_to(args.source.resolve()),
              'Split output must be outside source data')
        extract(args)
        split(argparse.Namespace(dataset=args.output, output=args.split_output, seed=args.seed))
        print(f'Dataset ready: {args.output}\nTraining split: {args.split_output.resolve() / "split.json"}', flush=True)
    else:
        split(args)


if __name__ == '__main__':
    main()
