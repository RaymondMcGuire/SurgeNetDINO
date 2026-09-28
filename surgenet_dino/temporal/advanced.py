"""Advanced-challenge score: segmental F1@0.75 over the 30 fps frames, out-of-body phase x as a class.

The official annotation CSV gives each phase from its start frame to the next row's start frame; the
last row ('e') marks the end and is not scored. Frames of phase x are painted black in the videos, so they
are taken as detected exactly; their segments count like any other. Every other frame takes the
prediction of the nearest catalogued 1 fps frame. Organisers' FAQ of 2026-09-27, Q7: x segments are
counted, and TP, FP and FN are summed over the videos before the one F1 is computed.
"""

import csv
from pathlib import Path

from ..training import LABELS
from .timeline import segments
from .workflows import f1_score, segment_counts

OUT_OF_BODY = len(LABELS)  # class index of x


def read_annotation(path):
    """Class index of every frame [0, end) of one official annotation CSV; x is OUT_OF_BODY."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [(int(r["start-frame"]), r["phase-id"]) for r in csv.DictReader(stream)]
    if len(rows) < 2 or rows[-1][1] != "e":
        raise ValueError(f"Annotation must end with an 'e' row: {path}")
    labels = []
    for (start, phase), (stop, _) in zip(rows, rows[1:]):
        labels += [OUT_OF_BODY if phase == "x" else LABELS.index(phase)] * (stop - start)
    return labels


def frame_predictions(truth, frames, predicted):
    """30 fps labels: x where the truth is x, else the class of the nearest catalogued frame."""
    result = []
    edges = [0, *((a + b) // 2 + 1 for a, b in zip(frames, frames[1:])), len(truth)]  # a tie goes to the earlier
    for start, stop, label in zip(edges, edges[1:], predicted):
        result += [label] * (min(stop, len(truth)) - min(start, len(truth)))
    return [OUT_OF_BODY if t == OUT_OF_BODY else p for t, p in zip(truth, result)]


def advanced_counts(predicted, records, directory, threshold=0.75):
    """{video: (TP, FP, FN)} at IoU `threshold`; predicted [N] class indices of the catalogued frames `records`."""
    by_video = {}
    for label, record in zip(predicted.tolist(), records, strict=True):
        by_video.setdefault(record["video_id"], []).append((record["frame_index"], label))
    counts = {}
    for video, items in sorted(by_video.items()):
        truth = read_annotation(Path(directory) / f"{video}.csv")
        frames, labels = zip(*sorted(items))
        found = frame_predictions(truth, frames, labels)
        counts[video] = segment_counts(segments(found), segments(truth), threshold)
    return counts


def pooled_f1(counts):
    """The official score: one F1 from TP, FP and FN summed over all videos, not a mean of per-video F1."""
    return f1_score(*map(sum, zip(*counts.values())))
