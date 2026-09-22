"""One-second timelines per video. Seconds without a catalogued frame are out-of-body gaps."""

from dataclasses import dataclass

import torch

FRAMES_PER_SECOND = 30  # extraction step of the preprocessed 1 fps catalogue (docs/DATASET.md)


@dataclass(frozen=True)
class Timeline:
    video_id: str
    start_second: int
    rows: torch.Tensor  # int64 [T]: index into the record list, -1 where no frame exists
    long_gap: tuple | None  # [start, stop) in timeline coordinates; observable without labels
    extra_long_gaps: int = 0  # further gaps above the threshold; the longest one is the anchor

    def __len__(self):
        return len(self.rows)

    @property
    def present(self):
        return self.rows >= 0

    def chains(self):
        """(name, start, stop) pieces that are conditionally independent given the clamped gap."""
        if self.long_gap is None:
            return [("whole", 0, len(self))]
        return [("pre", 0, self.long_gap[0]), ("post", self.long_gap[1], len(self))]


def gaps(present):
    """Maximal runs of missing seconds as (start, stop)."""
    result, t, total = [], 0, len(present)
    while t < total:
        if present[t]:
            t += 1
            continue
        u = t
        while u < total and not present[u]:
            u += 1
        result.append((t, u))
        t = u
    return result


def build_timelines(records, long_gap_seconds=300):
    """Group catalogue rows by video. Uses video_id and frame_index only, never labels."""
    if long_gap_seconds < 1:
        raise ValueError("long_gap_seconds must be positive")
    by_video = {}
    for index, record in enumerate(records):
        frame = record["frame_index"]
        if type(frame) is not int or frame < 0 or frame % FRAMES_PER_SECOND:
            name = record.get("filename")
            raise ValueError(f"frame_index is not a whole 1 fps second: {name}")
        by_video.setdefault(record["video_id"], []).append((frame // FRAMES_PER_SECOND, index))
    timelines = {}
    for video, items in sorted(by_video.items()):
        items.sort()
        seconds = [s for s, _ in items]
        if len(set(seconds)) != len(seconds):
            raise ValueError(f"Duplicate second in video {video}")
        start = seconds[0]
        rows = torch.full((seconds[-1] - start + 1,), -1, dtype=torch.int64)
        rows[torch.tensor(seconds) - start] = torch.tensor([i for _, i in items])
        long = sorted((g for g in gaps((rows >= 0).tolist()) if g[1] - g[0] >= long_gap_seconds),
                      key=lambda g: (g[0] - g[1], g[0]))
        timelines[video] = Timeline(video, start, rows, long[0] if long else None, max(0, len(long) - 1))
    return timelines


def label_track(timeline, targets):
    """Training-only label per second. A gap is filled when both sides agree, else left at -1."""
    track = torch.full((len(timeline),), -1, dtype=torch.int64)
    present = timeline.present
    track[present] = targets[timeline.rows[present]]
    for start, stop in gaps(present.tolist()):
        if start > 0 and stop < len(track) and track[start - 1] == track[stop]:
            track[start:stop] = track[stop]
    if timeline.long_gap is not None:
        track[timeline.long_gap[0]:timeline.long_gap[1]] = -1  # clamped gap: never part of a chain
    return track


def segments(labels):
    """Run-length encoding [(label, start, stop)] of a label sequence; -1 seconds are skipped."""
    result = []
    for t, value in enumerate(labels.tolist() if hasattr(labels, "tolist") else labels):
        if value < 0:
            continue
        if result and result[-1][0] == value and result[-1][2] == t:
            result[-1][2] = t + 1
        else:
            result.append([value, t, t + 1])
    return [tuple(s) for s in result]
