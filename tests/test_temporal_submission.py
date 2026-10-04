"""Submission integrity and complete restored-model inference without held-out labels."""

import csv
import json
import zipfile
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from surgenet_dino.gpu import CPUControl, Resources
from surgenet_dino.config import load_config
from surgenet_dino.head import RegularizedLinearHead
from surgenet_dino.head_options import HeadOptions
from surgenet_dino.temporal import submission, tcn
from surgenet_dino.temporal.boundary import snap_boundaries
from surgenet_dino.temporal.options import TemporalOptions
from surgenet_dino.training import LABELS


def test_final_selection_uses_only_inner_curves_and_rejects_eval_training():
    options = replace(TemporalOptions(), epochs=20, epoch_checks=2)
    head = HeadOptions()
    records = [{'video_id': v} for v in sorted(submission.TRAINING_VIDEOS)]
    provenance = {'features_sha256': 'base', 'tokens_sha256': 'tokens', 'feature_identity': {'id': 'encoder'}}
    selections = []
    for held in [sorted(submission.TRAINING_VIDEOS)[i:i + 3] for i in range(0, 12, 3)]:
        selections.append({'validation_videos': held,
                           'fit_videos': sorted(submission.TRAINING_VIDEOS - set(held)),
                           'inner_frames': 9, 'inner_accuracy_by_epoch': {'10': .7, '20': .8}})
    report = {'complete': True, 'options': asdict(options), 'head_options': asdict(head),
              'provenance': {**provenance, 'seed': 42}, 'selections': selections,
              'pooled': {'accuracy': -1}}  # deliberately invalid outer score: never consulted
    epoch, scores = submission.final_epoch(report, records, options, head, provenance, 42)
    assert epoch == 20 and scores[20] == pytest.approx(.8)
    selections[0]['inner_accuracy_by_epoch']['10'] = .8
    for s in selections:
        s['inner_accuracy_by_epoch']['10'] = .8
    assert submission.final_epoch(report, records, options, head, provenance, 42)[0] == 10
    with pytest.raises(ValueError, match='12 annotated'):
        submission.final_epoch(report, [*records, {'video_id': '02'}], options, head, provenance, 42)


def test_final_fit_saves_every_component_using_all_training_videos(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    options = replace(TemporalOptions(), channels=4, refine_channels=4, layers=1, attention_layers=0,
                      pca=2, seeds=1, epochs=1, epoch_checks=1, head_epochs=1, augment=False)
    head = HeadOptions()
    config = replace(load_config('configs/frozen_dinov2.toml'), device='cpu')
    identity = {'contract': {'model': config.name, 'checkpoint_sha256': config.sha256}}
    records = [{'video_id': v, 'filename': f'{v}_{i * 30:06d}.png', 'frame_index': i * 30}
               for v in sorted(submission.TRAINING_VIDEOS) for i in range(130)]
    targets = torch.tensor([i % len(LABELS) for _ in submission.TRAINING_VIDEOS for i in range(130)])
    features = torch.randn(len(records), 4)
    provenance = {'feature_identity': identity, 'features_sha256': 'base', 'tokens_sha256': 'tokens'}
    selections = [{'validation_videos': held, 'fit_videos': sorted(submission.TRAINING_VIDEOS - set(held)),
                   'inner_frames': 9 * 130, 'inner_accuracy_by_epoch': {'1': .7}}
                  for held in [sorted(submission.TRAINING_VIDEOS)[i:i + 3] for i in range(0, 12, 3)]]
    report = {'complete': True, 'options': asdict(options), 'head_options': asdict(head),
              'provenance': {**provenance, 'seed': config.seed}, 'selections': selections}
    source = tmp_path / 'selection.json'
    source.write_text(json.dumps(report))
    output = tmp_path / 'final'
    output.mkdir()
    args = SimpleNamespace(temporal_config=None, features=None, tokens=None, selection_report=source, output_dir=output)
    monkeypatch.setattr(submission, 'load_inputs', lambda *a: (None, records, targets, features, provenance))
    monkeypatch.setattr(submission, 'load_options', lambda *a: options)
    monkeypatch.setattr(submission, 'resolve_options', lambda *a: head)
    submission.fit_final(args, config, Resources(cpu_threads=1))
    saved, restored_options = submission.load_final(output / 'checkpoint.pt')
    assert restored_options == options and saved['selection']['epoch'] == 1
    assert saved['selection']['evaluation_videos_used'] is False
    assert saved['pca']['basis'].shape == (4, 2) and saved['inputs'] == 21
    result = json.loads((output / 'result.json').read_text())
    assert result['training_videos'] == sorted(submission.TRAINING_VIDEOS)
    assert len(result['emission_ledger']) == 12
    assert all(e['video_id'] not in e['trained_on'] for e in result['emission_ledger'])


def test_restored_model_is_label_free_and_basic_preserves_template_order(tmp_path):
    torch.set_num_threads(1)
    options = replace(TemporalOptions(), channels=4, refine_channels=4, layers=1,
                      attention_layers=0, pca=2, seeds=2)
    identity = {'id': 'test-encoder'}
    head = RegularizedLinearHead(4, len(LABELS), HeadOptions())
    models = [tcn.PhaseTCN(21, options).eval() for _ in range(2)]
    checkpoint = {'format': submission.FORMAT, 'labels': LABELS, 'trained_on': sorted(submission.TRAINING_VIDEOS),
                  'feature_identity': identity, 'options': asdict(options), 'inputs': 21,
                  'pca': {'mean': torch.zeros(4, dtype=torch.float64), 'basis': torch.eye(4, dtype=torch.float64)[:, :2],
                          'scale': torch.ones(2, dtype=torch.float64)},
                  'head': {'head_spec': head.specification(), 'state_dict': submission.cpu_state(head)},
                  'networks': [submission.cpu_state(m) for m in models],
                  'rules': {'precedence': [], 'single': [], 'minimum': [1] * len(LABELS)}}
    path = tmp_path / 'checkpoint.pt'
    torch.save(checkpoint, path)
    restored, _ = submission.load_final(path)
    records = [{'video_id': '02', 'filename': f'02_{i * 30:06d}.png', 'frame_index': i * 30} for i in range(120)]
    # A missing second tests inference timeline gaps; there are no targets or phase fields.
    records.pop(45)
    cache = {'format': submission.IMAGE_FORMAT, 'records': records,
             'tokens': torch.randn(len(records), 4), 'feature_identity': identity}
    with CPUControl(torch.device('cpu'), Resources(cpu_threads=1)) as guard:
        _, first, labels, _ = submission.infer(checkpoint, cache, guard)
        _, second, restored_labels, _ = submission.infer(restored, cache, guard)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(labels, restored_labels) and torch.isfinite(first).all()
    template = tmp_path / 'template.csv'
    names = [r['filename'][:-4] for r in reversed(records)]
    template.write_text('filename,phase-id\n' + '\n'.join(f'{n},' for n in names) + '\n')
    output = tmp_path / 'submission.csv'
    submission.write_basic(template, records, labels, output)
    with output.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['filename', 'phase-id']
        rows = list(reader)
    assert [r['filename'] for r in rows] == names
    assert all(r['phase-id'] in LABELS for r in rows)
    with pytest.raises(ValueError, match='exactly once'):
        submission.write_basic(template, records[:-1], labels[:-1], tmp_path / 'missing.csv')
    cache['records'][0]['video_id'] = '01'
    with CPUControl(torch.device('cpu'), Resources(cpu_threads=1)) as guard, pytest.raises(ValueError, match='overlapping'):
        submission.infer(restored, cache, guard)


def test_advanced_export_respects_pixel_masks_and_nearest_frame_ties(tmp_path):
    records = [{'video_id': '02', 'filename': f'02_{f:06d}.png', 'frame_index': f} for f in [30, 60]]
    predicted = torch.tensor([0, 1])
    masks = {'source': 'pixel_scan', 'annotation_used': False,
             'videos': {'02': {'total_frames': 90, 'intervals': [[0, 10], [70, 80]]}}}
    path = tmp_path / 'submission.zip'
    submission.write_advanced(records, predicted, masks, path)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ['submission_02.csv']
        text = archive.read('submission_02.csv').decode()
    assert text == 'start-frame,phase-id\n0,x\n10,0\n46,1\n70,x\n80,1\n'
    masks['annotation_used'] = True
    with pytest.raises(ValueError, match='pixel-derived'):
        submission.write_advanced(records, predicted, masks, tmp_path / 'oracle.zip')
    masks['annotation_used'] = False
    masks['videos']['02']['intervals'] = [[0, 10], [9, 20]]
    with pytest.raises(ValueError, match='overlapping'):
        submission.write_advanced(records, predicted, masks, tmp_path / 'invalid.zip')
    # JSON round-trip preserves zero-based, half-open integer intervals.
    assert json.loads(json.dumps(masks)) == masks


def test_boundary_recipe_preserves_order_minimum_and_gaps():
    from surgenet_dino.temporal.timeline import segments
    margin = torch.tanh((torch.arange(80).double() - 40) / 2)
    logp = torch.stack([-margin, margin], dim=1)
    labels = torch.tensor([0] * 35 + [1] * 45)
    frames = list(range(0, 2400, 30))
    found = snap_boundaries(logp, labels, frames, (1, 1))
    assert abs(segments(found)[0][2] - 40) <= 1
    assert [s[0] for s in segments(found)] == [0, 1]
    frames[35:] = [f + 90 for f in frames[35:]]
    assert torch.equal(snap_boundaries(logp, labels, frames, (1, 1)), labels)
    assert torch.equal(snap_boundaries(logp, labels, list(range(0, 2400, 30)), (35, 45)), labels)
