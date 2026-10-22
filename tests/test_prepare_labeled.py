"""Portable data preparation with dispersed source videos and fresh video 01."""

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from surgenet_dino.training import read_dataset, read_split


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    def color(video_id, frame):
        if frame == 0:
            return (0, 0, 0)
        if video_id == '01' and frame == 30:
            return (1, 1, 1)
        return (int(video_id) * 10 + 10, frame // 30 * 10, 30)

    class Capture:
        def __init__(self, path):
            self.video_id = Path(path).stem
            assert self.video_id not in {'02', '07', '11', '15'}
            self.frame = 0

        def isOpened(self):
            return True

        def get(self, key):
            return {1: 30.0, 2: 420, 3: 1280, 4: 720}[key]

        def set(self, key, frame):
            self.frame = frame
            return True

        def read(self):
            rgb = color(self.video_id, self.frame)
            return True, np.full((720, 1280, 3), rgb[::-1], dtype=np.uint8)

        def release(self):
            pass

    cv2 = SimpleNamespace(VideoCapture=Capture, CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2,
                          CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                          CAP_PROP_POS_FRAMES=5, COLOR_BGR2RGB=6,
                          setNumThreads=lambda _: None, cvtColor=lambda data, _: data[:, :, ::-1])
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    path = Path(__file__).resolve().parents[1] / 'tools/prepare_labeled.py'
    spec = importlib.util.spec_from_file_location('prepare_labeled_test', path)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    monkeypatch.setattr(prep, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(prep.shutil, 'disk_usage', lambda _: SimpleNamespace(free=121 * 1024**3))
    monkeypatch.setattr(prep.importlib.metadata, 'version', lambda _: 'test')
    source = tmp_path / 'source with spaces'
    annotations = source / 'annotation_csv'
    annotations.mkdir(parents=True)
    (source / 'extract_frames.py').write_text('# official extractor placeholder')
    for video_id in prep.LABELED + prep.OFFICIAL_TEST:
        video = source / f'package{video_id}' / 'nested' / 'HD_videos' / f'{video_id}.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(f'original video {video_id}'.encode())
    for video_id in prep.LABELED:
        prep.write_csv(annotations / f'{video_id}.csv', ['start-frame', 'phase-id'],
                       [{'start-frame': i * 30, 'phase-id': label}
                        for i, label in enumerate(['x', *prep.LABELS, 'e'])])

    def process_video(video, annotation_dir, output, frame_step_sec, ignore_phase):
        assert frame_step_sec == 1 and ignore_phase == ['x']
        data = output / video.stem
        (data / 'images').mkdir(parents=True)
        rows = []
        for i, label in enumerate(['x', *prep.LABELS]):
            filename = f'{video.stem}_{i * 30:06d}'
            Image.new('RGB', (1280, 720), color(video.stem, i * 30)).save(data / 'images' / f'{filename}.png')
            if label != 'x':
                rows.append({'filename': filename, 'phase-id': label})
        prep.write_csv(data / f'annotation_{video.stem}.csv', ['filename', 'phase-id'], rows)

    monkeypatch.setattr(prep, 'official_module', lambda _: SimpleNamespace(process_video=process_video))
    args = argparse.Namespace(source=source, output=tmp_path / 'prepared' / 'labeled_1fps_v1',
                              split_output=tmp_path / 'experiments' / 'splits' / 'new', workers=2, seed=42)
    return prep, args


def test_prepare_dispersed_sources_are_training_compatible(preparation, monkeypatch):
    prep, args = preparation
    originals = {path: path.read_bytes() for path in args.source.rglob('*') if path.is_file()}
    monkeypatch.setattr(sys, 'argv', ['prepare_labeled.py', 'prepare', '--source', str(args.source),
                                    '--output', str(args.output), '--split-output', str(args.split_output)])
    prep.main()
    dataset = json.loads((args.output / 'dataset.json').read_text())
    assert dataset['status'] == 'passed'
    assert (dataset['saved_images'], dataset['accepted_images'], dataset['excluded_images']) == (168, 155, 13)
    assert dataset['excluded_reasons'] == {'phase_x': 12, 'all_pixels_at_most_1': 1}
    assert len(read_dataset(args.output, inspect_images=True)['records']) == 155
    split = read_split(args.split_output / 'split.json', inspect_images=True)
    assert len(split['definition']['train_video_ids']) == 9
    assert len(split['definition']['validation_video_ids']) == 3
    assert '01' in split['definition']['train_video_ids']
    assert all(not row['reused_existing_case'] for row in dataset['videos'])
    assert not any((args.output.parent / f'video{video_id}_1fps').exists() for video_id in prep.OFFICIAL_TEST)
    assert originals == {path: path.read_bytes() for path in args.source.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='Split output already exists'):
        prep.main()


@pytest.mark.parametrize('case', ['existing01', 'duplicate', 'missing', 'source_output'])
def test_preflight_refuses_before_extraction(preparation, case):
    prep, args = preparation
    if case == 'existing01':
        sentinel = args.output.parent / 'video01_1fps' / 'keep.txt'
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text('keep existing data')
        message = 'Case 01 already exists'
    elif case == 'duplicate':
        duplicate = args.source / 'other' / 'HD_videos' / '01.mp4'
        duplicate.parent.mkdir(parents=True)
        duplicate.touch()
        message = 'Duplicate video IDs'
    elif case == 'missing':
        next(args.source.rglob('16.mp4')).unlink()
        message = 'Missing labeled video'
    else:
        args.output = args.source / 'generated' / 'index'
        message = 'Keep output roots separate'
    with pytest.raises(ValueError, match=message):
        prep.extract(args)
    assert not args.output.exists()
    if case == 'existing01':
        assert sentinel.read_text() == 'keep existing data'


def test_failed_extraction_does_not_publish_passed_dataset(preparation, monkeypatch):
    prep, args = preparation

    def broken_extractor(*unused, **kwargs):
        raise ValueError('test decode failure')

    monkeypatch.setattr(prep, 'official_module', lambda _: SimpleNamespace(process_video=broken_extractor))
    with pytest.raises(ValueError, match='test decode failure'):
        prep.extract(args)
    assert json.loads((args.output / 'dataset.json').read_text())['status'] == 'failed'
    assert json.loads((args.output / '01' / 'report.json').read_text())['status'] == 'failed'
    assert not (args.output / 'samples.csv').exists()
    assert not args.split_output.exists()


def test_filter_boundaries_and_split_self_check(preparation):
    prep, _ = preparation
    prep.self_test()
