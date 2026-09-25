"""Order rules learned from the training videos, and exact decoding of held-out videos under them.

Some facts hold in every labeled operation: IMA treatment (3) forms a single block and phase 2 never
follows it; no phase 0-7 appears once the rectal division scenes (81) have begun. The temporal
network respects such facts most of the time, not always. For every fold, the rules that hold in ALL
of its training videos are read from their label sequences, and each held-out video is decoded by an
exact Viterbi search that maximises the summed network log-probabilities subject to those rules.
There are no transition probabilities, durations or tunable parameters: where no rule is violated,
the result is the plain argmax.
"""

from dataclasses import dataclass

import torch

from ..training import LABELS

MAX_BACK_POINTERS = 2 ** 31  # bytes: seconds x (masks of begun constrained phases) x phases


@dataclass(frozen=True)
class Rules:
    precedence: frozenset  # (a, b): every a precedes the first b, so a is impossible once b has begun
    single: frozenset  # phases forming one contiguous block: once left, they cannot come back

    def summary(self):
        return {"precedence_rules": len(self.precedence),
                "single_block_phases": [LABELS[m] for m in sorted(self.single)]}


def blocks(sequence):
    return [c for i, c in enumerate(sequence) if i == 0 or c != sequence[i - 1]]


def learn_rules(sequences, classes=len(LABELS)):
    """Rules that EVERY sequence (class indices in time order, one per video) agrees with.

    A rule also needs one sequence that contains its phases, so it is never vacuous.
    """
    spans = [({c: s.index(c) for c in set(s)}, {c: len(s) - 1 - s[::-1].index(c) for c in set(s)})
             for s in sequences]
    precedence = {(a, b) for a in range(classes) for b in range(classes) if a != b
                  and any(a in first and b in first for first, _ in spans)
                  and all(a not in last or b not in first or last[a] < first[b] for first, last in spans)}
    single = {m for m in range(classes) if any(m in s for s in sequences)
              and all(blocks(s).count(m) <= 1 for s in sequences)}
    return Rules(frozenset(precedence), frozenset(single))


def training_rules(targets, timelines, videos):
    """Rules of the given (training) videos, from their catalogued labels in time order."""
    return learn_rules([targets[t.rows[t.present]].tolist() for t in (timelines[v] for v in videos)])


def decode(scores, rules):
    """Labels [T] that maximise the summed log-probabilities scores [T,K] and obey every rule.

    Exact Viterbi over (current phase, set of constrained phases already begun). Back-pointers are one
    byte: the previous phase in the low four bits, bit 4 set when the current phase has just begun.
    """
    steps, classes = scores.shape
    tracked = sorted({b for _, b in rules.precedence} | set(rules.single))
    if classes > 16 or steps * classes * (1 << len(tracked)) > MAX_BACK_POINTERS:
        raise ValueError("Too many phases or constrained phases for the decoder state")
    bit = [0] * classes
    for i, phase in enumerate(tracked):
        bit[phase] = 1 << i
    forbid = torch.zeros(classes, dtype=torch.long)
    for a, b in rules.precedence:
        forbid[a] |= bit[b]
    masks = torch.arange(1 << len(tracked))
    allowed = (masks[:, None] & forbid[None, :]) == 0  # a begun phase forbids nothing in this mask
    scores = scores.double()
    value = torch.full((len(masks), classes), float("-inf"), dtype=torch.float64)
    for q in range(classes):
        if allowed[bit[q], q]:
            value[bit[q], q] = scores[0, q]
    back = torch.zeros((steps, len(masks), classes), dtype=torch.uint8)
    stay = torch.arange(classes, dtype=torch.uint8).expand(len(masks), classes)
    for t in range(1, steps):
        top, arg = value.topk(2, dim=1)
        new, code = value + scores[t], stay.clone()
        for q in range(classes):
            other = arg[:, 0] == q  # the best predecessor must be a different phase
            best, best_arg = torch.where(other, top[:, 1], top[:, 0]), torch.where(other, arg[:, 1], arg[:, 0])
            source, begun = masks, torch.zeros(len(masks), dtype=torch.bool)
            candidate = best
            if bit[q]:
                begun = (masks & bit[q]) != 0
                source = torch.where(begun, masks ^ bit[q], masks)  # first entry: from the mask without q
                candidate = torch.where(begun, best[source], float("-inf"))
                if q not in rules.single:  # a phase that has begun before may come back
                    again = begun & (best > candidate)
                    candidate, source = torch.where(again, best, candidate), torch.where(again, masks, source)
                    begun = begun & ~again
            entered = torch.where(allowed[:, q], candidate + scores[t, q], float("-inf"))
            take = entered > new[:, q]
            new[:, q] = torch.where(take, entered, new[:, q])
            code[:, q] = torch.where(take, (best_arg[source] | begun.long() << 4).to(torch.uint8), code[:, q])
        value, back[t] = new, code
    if not torch.isfinite(value.max()):
        raise ValueError("No label sequence satisfies the rules")
    mask, phase = divmod(int(value.argmax()), classes)
    path = [phase]
    for t in range(steps - 1, 0, -1):
        step = int(back[t, mask, phase])
        if step >> 4:
            mask ^= bit[phase]
        phase = step & 15
        path.append(phase)
    return torch.tensor(path[::-1])


def apply_rules(log_probs, timelines, videos, rules):
    """(log-probabilities [N,K] whose argmax is the constrained decoding, number of changed seconds).

    Where a rule changes a label, the probabilities of the network's top class and of the decoded
    class are swapped, so every row stays a proper distribution. Rows of other videos are untouched.
    """
    result, changed = log_probs.clone(), 0
    for video in videos:
        timeline = timelines[video]
        rows = timeline.rows[timeline.present]
        decoded, top = decode(log_probs[rows], rules), log_probs[rows].argmax(1)
        moved = decoded != top
        index, a, b = rows[moved], top[moved], decoded[moved]
        result[index, a], result[index, b] = log_probs[index, b], log_probs[index, a]
        changed += int(moved.sum())
    return result, changed
