"""Fixed local posterior-change boundary correction, used only for Advanced export."""

from torch.nn import functional as F

from .timeline import segments


def snap_boundaries(logp, labels, frames, minimum):
    """Move existing boundaries to the local rise in right-versus-left class margin.

    Fixed five-sample smoothing; at most 30 seconds and a quarter of each
    adjacent run. Preserve phase order, segment count and training-side minimum
    lengths. Neither the search nor its smoothing crosses a catalogue gap.
    """
    runs = segments(labels)
    edges = [0, *(a for _, a, _ in runs[1:]), len(labels)]
    phases = [c for c, _, _ in runs]
    gaps = [i for i in range(1, len(frames)) if frames[i] - frames[i - 1] != 30]
    for j in range(1, len(phases)):
        centre = edges[j]
        if centre in gaps:
            continue
        left, right = phases[j - 1], phases[j]
        width = min(30, (centre - edges[j - 1]) // 4, (edges[j + 1] - centre) // 4)
        start = max([0, *(g for g in gaps if g < centre)])
        stop = min([len(labels), *(g for g in gaps if g > centre)])
        low = max(centre - width, edges[j - 1] + minimum[left], start + 3)
        high = min(centre + width, edges[j + 1] - minimum[right], stop - 3)
        if low > high:
            continue
        margin = logp[:, right] - logp[:, left]
        smooth = F.avg_pool1d(margin[None, None], 5, stride=1, padding=2)[0, 0]
        rise = smooth[low:high + 1] - smooth[low - 1:high]
        peak = int(rise.argmax()) + low
        if float(rise.max()) > 1e-6:
            edges[j] = peak
    result = labels.clone()
    for c, start, stop in zip(phases, edges, edges[1:]):
        result[start:stop] = c
    assert [s[0] for s in segments(result)] == phases
    assert all(stop - start >= minimum[c] for c, start, stop in segments(result))
    return result
