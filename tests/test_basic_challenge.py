"""Basic extraction identity and one-click launcher failure paths; no real evaluation inference."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.basic_extract import extract


@pytest.fixture
def basic_inputs(tmp_path):
    source = tmp_path / 'source with spaces'
    names = []
    for video in ('02', '07', '11', '15'):
        path = source / video / 'HD_videos' / f'{video}.mp4'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'original video')
        names.extend([f'{video}_000060', f'{video}_000000'])
    template = tmp_path / 'submission.csv'
    template.write_text('filename,phase-id\n' + ''.join(f'{n},\n' for n in names), encoding='utf-8-sig')
    return source, template, tmp_path / 'images'


@pytest.fixture
def decoder(monkeypatch):
    settings = SimpleNamespace(fps=30, total=61, write_ok=True, retrieved=[], released=[])

    class Capture:
        def __init__(self, path, *args):
            self.path, self.frame = path, -1

        def isOpened(self):
            return True

        def get(self, key):
            return settings.fps

        def grab(self):
            self.frame += 1
            return self.frame < settings.total

        def retrieve(self):
            settings.retrieved.append(self.frame)
            return True, self.frame

        def release(self):
            settings.released.append(self.path)

    def write(path, frame):
        if settings.write_ok:
            Path(path).write_text(str(frame))
        return settings.write_ok

    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(
        setNumThreads=lambda n: None, VideoCapture=Capture, imwrite=write,
        CAP_FFMPEG=0, CAP_PROP_N_THREADS=1, CAP_PROP_FPS=2))
    return settings


def test_extracts_only_template_frames_without_changing_sources(basic_inputs, decoder):
    source, template, output = basic_inputs
    original = template.read_bytes()
    report = extract(source, template, output)
    assert report['total_images'] == 8 and report['annotation_used'] is False
    assert decoder.retrieved == [0, 60] * 4  # no pixels retrieved for non-template frames
    assert len(decoder.released) == 4
    assert len(list(output.rglob('*.png'))) == 8
    assert all(p.read_text() == str(int(p.stem[3:])) for p in output.rglob('*.png'))
    assert all(p.read_bytes() == b'original video' for p in source.rglob('*.mp4'))
    assert template.read_bytes() == original
    assert json.loads((output / 'extraction.json').read_text())['status'] == 'passed'
    with pytest.raises(FileExistsError):
        extract(source, template, output)


@pytest.mark.parametrize('case', ['fps', 'truncated', 'write'])
def test_extraction_rejects_bad_video_or_write_without_success_report(basic_inputs, decoder, case):
    source, template, output = basic_inputs
    if case == 'fps':
        decoder.fps = 29.97
    elif case == 'truncated':
        decoder.total = 60
    else:
        decoder.write_ok = False
    with pytest.raises((ValueError, RuntimeError, OSError)):
        extract(source, template, output)
    assert len(decoder.released) == 1 and not (output / 'extraction.json').exists()


def test_missing_source_is_rejected_before_creating_output(basic_inputs, decoder):
    source, template, output = basic_inputs
    next(source.rglob('15.mp4')).unlink()
    with pytest.raises(ValueError, match='exactly one source'):
        extract(source, template, output)
    assert not output.exists()


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows cmd.exe')
@pytest.mark.parametrize('failure', [None, 'double_click', 0, 1, 2, 3, 'missing_model', 'existing_output'])
def test_basic_batch_stops_at_failure_and_keeps_template(tmp_path, failure):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'scripts' / 'basic_challenge.bat'
    script.parent.mkdir()
    source = (root / 'scripts/basic_challenge.bat').read_text()
    data = tmp_path / 'source with spaces'
    template = data / ('jscas_ai_challenge_2026_data1/jscas_ai_challenge_2026_data1/'
                       'submission_format/basic_challenge/submission.csv')
    files = [tmp_path / '.venv/Scripts/python.exe',
             tmp_path / '.cache/uv/archive-v0/SruTPdM1WCWkIu0S/Scripts/python.exe',
             tmp_path / 'outputs/challenge_final_20261004/checkpoint.pt', template]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('unchanged')
    if failure == 'missing_model':
        files[2].unlink()
    output = tmp_path / 'run with spaces'
    if failure == 'existing_output':
        output.mkdir()
        (output / 'keep.txt').write_text('previous result')
    if failure == 'double_click':
        source = source.replace('H:\\dataset\\JSCAS AI Challenge', str(data))
    calls = []

    def stub(match):
        index = len(calls)
        calls.append(index)
        return f'echo BASIC_STAGE_{index}\ncmd /d /c exit {7 if failure == index else 0}'

    script.write_text(re.sub(r'(?m)^"%(?:MODEL|VIDEO)_PYTHON%"[^\n]+', stub, source))
    assert len(calls) == 4
    args = [] if failure == 'double_click' else [str(data), str(output)]
    result = subprocess.run(['cmd', '/d', '/c', str(script), *args],
                            input=b'\n', capture_output=True, timeout=10)
    early = failure in ('missing_model', 'existing_output')
    expected = 1 if early else 7 if isinstance(failure, int) else 0
    assert result.returncode == expected, result.stdout + result.stderr
    expected_calls = 0 if early else failure + 1 if isinstance(failure, int) else 4
    assert result.stdout.count(b'BASIC_STAGE_') == expected_calls
    assert result.stdout.count(f'Script finished with exit code {expected}'.encode()) == 1
    assert template.read_text() == 'unchanged'
    if failure == 'existing_output':
        assert (output / 'keep.txt').read_text() == 'previous result'
    if failure == 'double_click':
        assert re.search(rb'outputs\\basic_challenge\\\d{8}_\d{6}_\d{3}\\predictions\\submission.csv',
                         result.stdout)
