"""Final training, label-free image inference and official challenge CSV export.

Usage: python -m surgenet_dino.temporal.submission --help.
The evaluation image cache is deliberately separate from all training caches.
"""

import argparse
import csv
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ..config import load_config
from ..gpu import CPUControl, GPUControl, Resources
from ..head import head_from_checkpoint
from ..head_options import resolve_options
from ..runtime import resolve_device
from ..training import LABELS, assert_identity, file_sha256, save_tensor, write_json
from . import constraints, tcn
from .boundary import snap_boundaries
from .emissions import fit_head_fixed, head_logits, per_video_zscore, stacked_emissions
from .options import TemporalOptions, load_options
from .timeline import build_timelines, segments
from .template import read_template
from .tokens import FrameImages, check_architecture, frame_tokens
from .workflows import load_inputs

TRAINING_VIDEOS = {'01', '03', '04', '05', '06', '08', '09', '10', '12', '13', '14', '16'}
EVALUATION_VIDEOS = {'02', '07', '11', '15'}
FORMAT = 'surgenet-temporal-final-v1'
IMAGE_FORMAT = 'surgenet-temporal-unlabeled-v1'


def final_epoch(report, records, options, head_options, provenance, seed):
    """Use only previously recorded INNER accuracy curves, weighted by fit-frame counts."""
    if (not report['complete'] or report['options'] != asdict(options) or
            report['head_options'] != asdict(head_options) or report['provenance']['seed'] != seed or
            any(report['provenance'][k] != provenance[k] for k in
                ('features_sha256', 'tokens_sha256', 'feature_identity'))):
        raise ValueError('Selection report does not match this complete training recipe')
    videos, rows = sorted({r['video_id'] for r in records}), tcn.video_rows(records)
    if set(videos) != TRAINING_VIDEOS:
        raise ValueError('Final training requires exactly the 12 annotated challenge videos')
    selections = report['selections']
    held = [v for s in selections for v in s['validation_videos']]
    if sorted(held) != videos:
        raise ValueError('Selection folds must cover each training video exactly once')
    epochs = tcn.checkpoint_epochs(options)
    totals = {e: 0.0 for e in epochs}
    weight = 0
    for s in selections:
        fit = sorted(set(videos) - set(s['validation_videos']))
        n = sum(len(rows[v]) for v in fit)
        if (s['fit_videos'] != fit or s['inner_frames'] != n or
                set(map(int, s['inner_accuracy_by_epoch'])) != set(epochs)):
            raise ValueError('Malformed training-side selection fold')
        for e in epochs:
            value = float(s['inner_accuracy_by_epoch'][str(e)])
            if not 0 <= value <= 1:
                raise ValueError('Invalid inner accuracy')
            totals[e] += n * value
        weight += n
    scores = {e: value / weight for e, value in totals.items()}
    return max(epochs, key=lambda e: (scores[e], -e)), scores


def control(config, resources, output):
    device = resolve_device(config.device)
    return (GPUControl if device.type == 'cuda' else CPUControl)(device, resources, output / 'gpu.csv')


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit_final(args, config, resources):
    options, head_options = load_options(args.temporal_config), resolve_options(args)
    _, records, targets, tokens, provenance = load_inputs(args.features, args.tokens)
    identity = provenance['feature_identity']
    if config.name != identity['contract']['model'] or config.sha256 != identity['contract']['checkpoint_sha256']:
        raise ValueError('Encoder configuration differs from the verified training cache')
    report = json.loads(args.selection_report.read_text(encoding='utf-8'))
    epoch, scores = final_epoch(report, records, options, head_options, provenance, config.seed)
    videos = sorted(TRAINING_VIDEOS)
    x = per_video_zscore(tokens, records)
    timelines = build_timelines(records, options.long_gap_seconds)
    rules = constraints.training_rules(targets, timelines, videos)
    selection = {'source': str(args.selection_report.resolve()),
                 'source_sha256': file_sha256(args.selection_report), 'epoch': epoch,
                 'weighted_inner_accuracy': scores, 'outer_scores_used': False,
                 'evaluation_videos_used': False}
    write_json(args.output_dir / 'selection.json', selection)
    with control(config, resources, args.output_dir) as guard:
        logits, ledger = stacked_emissions(x, targets, records, videos, [], head_options,
                                          options.head_epochs, config.seed, guard)
        pca = tcn.fit_pca(x, options.pca)
        sequences = tcn.make_sequences(timelines, videos, logits, x, targets, pca)
        confusion = tcn.confusion_rows(logits, targets, torch.arange(len(records)))
        models = []
        for s in range(options.seeds):
            print(f'Final temporal network {s + 1}/{options.seeds}, {epoch} epochs', flush=True)
            model, _ = tcn.fit_tcn(sequences, options, config.seed + 1000 * (s + 1), guard, epoch, confusion)
            models.append(cpu_state(model))
        head = fit_head_fixed(x, targets, head_options, config.seed, options.head_epochs, guard)
        save_tensor(args.output_dir / 'checkpoint.pt', {
            'format': FORMAT, 'labels': LABELS, 'trained_on': videos,
            'feature_identity': identity, 'options': asdict(options), 'seed': config.seed,
            'inputs': sequences[videos[0]][0].shape[0], 'pca': pca,
            'head': {'head_spec': head.specification(), 'state_dict': cpu_state(head)},
            'networks': models, 'rules': {'precedence': sorted(rules.precedence),
                                        'single': sorted(rules.single), 'minimum': list(rules.minimum)},
            'selection': selection})
        usage = guard.summary()
    write_json(args.output_dir / 'result.json', {'status': 'passed', 'training_videos': videos,
               'samples': len(records), 'epoch': epoch, 'selection': selection,
               'provenance': provenance, 'emission_ledger': ledger, 'resources': usage,
               'checkpoint_sha256': file_sha256(args.output_dir / 'checkpoint.pt'),
               'note': 'Final fit on all labeled videos. No independent accuracy claimed.'})


def load_final(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['format'] != FORMAT or payload['labels'] != LABELS or
            set(payload['trained_on']) != TRAINING_VIDEOS):
        raise ValueError('Invalid final temporal checkpoint')
    options = TemporalOptions(**payload['options'])
    if len(payload['networks']) != options.seeds:
        raise ValueError('Incomplete temporal ensemble')
    return payload, options


def cache_images(args, config, resources):
    from ..registry import get_spec
    from ..training import encoder, state_digest
    checkpoint, _ = load_final(args.checkpoint)
    names = read_template(args.template)
    records = [{'video_id': n[:2], 'filename': n + '.png', 'frame_index': int(n[3:]),
                'image_path': str((args.image_root / n[:2] / 'images' / (n + '.png')).resolve())} for n in names]
    if {r['video_id'] for r in records} != EVALUATION_VIDEOS:
        raise ValueError('Expected all four official evaluation videos')
    missing = [r['image_path'] for r in records if not Path(r['image_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} template images missing; first: {missing[0]}')
    with control(config, resources, args.output_dir) as guard:
        model, identity = encoder(config, guard.device)
        assert_identity(identity, checkpoint['feature_identity'], 'Evaluation encoder')
        check_architecture(model)
        before = state_digest(model)
        dataset = FrameImages([r['image_path'] for r in records], get_spec(config.name).image_size)
        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size, shuffle=False, num_workers=0)
        parts = []
        for batch in loader:
            with guard.batch(len(batch)):
                parts.append(frame_tokens(model, batch.to(guard.device)).float().cpu())
        values = torch.cat(parts)
        if not torch.isfinite(values).all() or state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Nonfinite features or modified frozen encoder')
        usage = guard.summary()
    save_tensor(args.output_dir / 'images.pt', {'format': IMAGE_FORMAT, 'records': records,
                'tokens': values, 'feature_identity': identity, 'template_sha256': file_sha256(args.template)})
    write_json(args.output_dir / 'result.json', {'status': 'passed', 'samples': len(records),
               'resources': usage, 'input': 'Only official template 1fps images', 'training_performed': False,
               'template_sha256': file_sha256(args.template), 'cache_sha256': file_sha256(args.output_dir / 'images.pt')})


def infer(checkpoint, image_cache, guard):
    options = TemporalOptions(**checkpoint['options'])
    if image_cache['format'] != IMAGE_FORMAT:
        raise ValueError('Expected the separate unlabeled image cache')
    assert_identity(image_cache['feature_identity'], checkpoint['feature_identity'], 'Inference cache')
    records, values = image_cache['records'], image_cache['tokens']
    if (set(r['video_id'] for r in records) & set(checkpoint['trained_on']) or
            values.ndim != 2 or len(values) != len(records) or not torch.isfinite(values).all()):
        raise ValueError('Invalid or training-overlapping inference cache')
    x = per_video_zscore(values, records)
    head = head_from_checkpoint(checkpoint['head'], values.shape[1], len(LABELS), guard.device)
    logits = head_logits(head, x, guard)
    timelines = build_timelines(records, options.long_gap_seconds)
    models = []
    for state in checkpoint['networks']:
        model = tcn.PhaseTCN(checkpoint['inputs'], options)
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Nonfinite temporal checkpoint')
        models.append(model.to(guard.device).eval())
    sequences = {v: (tcn.build_inputs(t, logits, x, checkpoint['pca']), None) for v, t in timelines.items()}
    logp = tcn.predict_tcn(models, sequences, timelines, guard, len(records))
    saved = checkpoint['rules']
    rules = constraints.Rules(frozenset(tuple(p) for p in saved['precedence']),
                              frozenset(saved['single']), tuple(saved['minimum']))
    decoded, _ = constraints.apply_rules(logp, timelines, sorted(timelines), rules)
    return records, logp, decoded.argmax(1), rules


def write_basic(template, records, predicted, path):
    names = read_template(template)
    lookup = {Path(r['filename']).stem: LABELS[int(p)] for r, p in zip(records, predicted, strict=True)}
    if len(lookup) != len(records) or set(lookup) != set(names):
        raise ValueError('Predictions must match the template exactly once')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filename', 'phase-id'])
        writer.writerows((n, lookup[n]) for n in names)


def write_advanced(records, predicted, masks, path):
    """Export real pixel-derived x intervals; never borrow x from annotations or Basic gaps."""
    if masks.get('source') != 'pixel_scan' or masks.get('annotation_used') is not False:
        raise ValueError('Advanced export needs a pixel-derived mask without annotation input')
    videos = sorted({r['video_id'] for r in records})
    if set(videos) != set(masks['videos']):
        raise ValueError('x masks must cover exactly the predicted videos')
    contents = {}
    for video in videos:
        items = sorted((r['frame_index'], int(p)) for r, p in zip(records, predicted, strict=True)
                       if r['video_id'] == video)
        frames, labels = zip(*items)
        total = masks['videos'][video]['total_frames']
        if (type(total) is not int or total <= frames[-1] or frames[0] < 0 or len(set(frames)) != len(frames)
                or any(not 0 <= c < len(LABELS) for c in labels)):
            raise ValueError('Invalid total-frame count, labels or frame identities')
        edges = [0, *((a + b) // 2 + 1 for a, b in zip(frames, frames[1:])), total]
        track = torch.empty(total, dtype=torch.int64)
        for a, b, label in zip(edges[:-1], edges[1:], labels, strict=True):
            track[a:b] = label
        last = 0
        for a, b in masks['videos'][video]['intervals']:
            if type(a) is not int or type(b) is not int or not last <= a < b <= total:
                raise ValueError('Invalid, unsorted or overlapping half-open x intervals')
            track[a:b] = len(LABELS)
            last = b
        rows = ['start-frame,phase-id', *(f'{start},{"x" if c == len(LABELS) else LABELS[c]}'
                                        for c, start, _ in segments(track))]
        # Official submitted-label list includes x; e is a reference end marker, not a predicted phase.
        contents[f'submission_{video}.csv'] = '\n'.join(rows) + '\n'
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)


def main(argv=None):
    """最终训练/提交入口；评价图像提取走 main → cache_images → FrameImages / frame_tokens。

    cache-images 与训练侧共用图片预处理及视觉编码，但仅按评价模板读取图片，
    不要求训练侧 features.pt，也不使用评价标签；predict 直接读取已保存的 images.pt。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit-final')
    fit.add_argument('--features', type=Path, required=True)
    fit.add_argument('--tokens', type=Path, required=True)
    fit.add_argument('--selection-report', type=Path, required=True)
    fit.add_argument('--head-config', type=Path, default=Path('configs/head_regularized.toml'))
    fit.add_argument('--temporal-config', type=Path, default=Path('configs/temporal.toml'))
    cache = sub.add_parser('cache-images')
    cache.add_argument('--image-root', type=Path, required=True, help='Official data/<video>/images layout')
    cache.add_argument('--batch-size', type=int, default=8)
    predict = sub.add_parser('predict')
    predict.add_argument('--images', type=Path, required=True, help='images.pt from cache-images')
    predict.add_argument('--x-masks', type=Path, help='Optional complete pixel-scan JSON; required for Advanced export')
    for p in (cache, predict):
        p.add_argument('--checkpoint', type=Path, required=True)
        p.add_argument('--template', type=Path, required=True)
    for p in (fit, cache, predict):
        p.add_argument('--config', type=Path, default=Path('configs/frozen_dinov2.toml'))
        p.add_argument('--device')
        p.add_argument('--cpu-threads', type=int, default=2)
        p.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    resources = Resources(max_temperature=70, cpu_threads=args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.command == 'fit-final':
        fit_final(args, config, resources)
    elif args.command == 'cache-images':
        if args.batch_size < 1:
            raise ValueError('Batch size must be positive')
        cache_images(args, config, resources)
    else:
        checkpoint, _ = load_final(args.checkpoint)
        images = torch.load(args.images, map_location='cpu', weights_only=True)
        if images['template_sha256'] != file_sha256(args.template):
            raise ValueError('Inference cache was built with a different template')
        with control(config, resources, args.output_dir) as guard:
            records, logp, predicted, rules = infer(checkpoint, images, guard)
            usage = guard.summary()
        write_basic(args.template, records, predicted, args.output_dir / 'submission.csv')
        advanced = predicted.clone()
        if args.x_masks:
            timelines = build_timelines(records)
            for timeline in timelines.values():
                ids = timeline.rows[timeline.present]
                advanced[ids] = snap_boundaries(logp[ids], predicted[ids],
                                                [records[i]['frame_index'] for i in ids], rules.minimum)
            masks = json.loads(args.x_masks.read_text(encoding='utf-8'))
            write_advanced(records, advanced, masks, args.output_dir / 'submission.zip')
        save_tensor(args.output_dir / 'predictions.pt', {'records': records, 'log_probabilities': logp,
                                                      'predicted': predicted, 'advanced_predicted': advanced})
        write_json(args.output_dir / 'result.json', {'status': 'passed', 'samples': len(records),
                   'checkpoint_sha256': file_sha256(args.checkpoint), 'images_sha256': file_sha256(args.images),
                   'template_sha256': file_sha256(args.template), 'resources': usage,
                   'advanced_exported': bool(args.x_masks),
                   'advanced_boundary_recipe': '5-sample mean, 30s max, quarter adjacent runs, no gap crossing',
                   'x_mask_sha256': file_sha256(args.x_masks) if args.x_masks else None,
                   'training_performed': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
