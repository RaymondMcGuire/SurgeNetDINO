"""Order rules read from training label sequences, and exact decoding under them."""

import itertools

import torch

from surgenet_dino.temporal.constraints import Rules, apply_rules, blocks, decode, learn_rules
from surgenet_dino.temporal.timeline import build_timelines
from surgenet_dino.training import LABELS
from tests.test_temporal_data import records_for

CLASSES = len(LABELS)


def obeys(sequence, rules):
    for a, b in rules.precedence:
        if b in sequence and a in sequence[sequence.index(b):]:
            return False
    if any(len(list(g)) < rules.minimum[c] for c, g in itertools.groupby(sequence) if rules.minimum):
        return False
    return all(blocks(sequence).count(m) <= 1 for m in rules.single)


def test_rules_hold_in_every_sequence_and_are_never_vacuous():
    sequences = [[0, 0, 1, 2, 2, 3], [0, 1, 1, 2, 3, 3], [0, 1, 2, 1, 3]]  # 1 comes back after 2 once
    rules = learn_rules(sequences, classes=5)
    assert rules.precedence == {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}  # no rule mentions the absent 4
    assert rules.single == {0, 2, 3}
    assert all(obeys(s, rules) for s in sequences)
    assert rules.summary() == {"precedence_rules": 5, "single_block_phases": ["0", "2", "3"],
                               "minimum_block_seconds": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1}}


def test_minimum_block_length_is_the_shortest_block_of_any_training_sequence():
    rules = learn_rules([[0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 1, 1, 1, 2, 2, 0, 0]], classes=4)
    assert rules.minimum == (2, 2, 2, 1)  # every block counts, also a phase coming back; absent 3 is free


def test_decoding_is_exact_under_minimum_block_lengths():
    rules = Rules(frozenset({(1, 2)}), frozenset({2}), minimum=(1, 3, 2, 1))
    generator = torch.Generator().manual_seed(4)
    for _ in range(5):
        scores = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        best = max((s for s in itertools.product(range(4), repeat=7) if obeys(list(s), rules)),
                   key=lambda s: scores[range(7), list(s)].sum().item())
        found = decode(scores, rules)
        assert obeys(found.tolist(), rules)
        torch.testing.assert_close(scores[range(7), found].sum(), scores[range(7), list(best)].sum())
    blip = torch.tensor([0] * 5 + [1] + [0] * 6)  # a 1 s block of 1 where every block lasts >= 3 s
    scores = (torch.nn.functional.one_hot(blip, 4) * 2.0).log_softmax(1)
    assert decode(scores, Rules(frozenset(), frozenset())).tolist() == blip.tolist()
    assert decode(scores, Rules(frozenset(), frozenset(), (1, 3, 1, 1))).tolist() == [0] * 12  # cheaper than growing
    short = torch.tensor([0] * 5 + [1] * 2 + [0] * 5)
    scores = (torch.nn.functional.one_hot(short, 4) * 2.0).log_softmax(1)
    found = decode(scores, Rules(frozenset(), frozenset(), (1, 3, 1, 1))).tolist()
    assert found.count(1) == 3 and found[5:7] == [1, 1]  # a 2 s block grows by one second, cheaper than deletion


def test_decoding_is_the_plain_argmax_when_no_rule_is_broken():
    path = torch.tensor([0, 0, 1, 1, 2, 2, 3])
    scores = (torch.nn.functional.one_hot(path, CLASSES) * 2 + torch.rand(7, CLASSES)).log_softmax(1)
    assert torch.equal(decode(scores, learn_rules([[0, 1, 2, 3]])), path)


def test_decoding_is_the_best_sequence_that_obeys_the_rules():
    rules = Rules(frozenset({(1, 2), (0, 3)}), frozenset({2}))
    generator = torch.Generator().manual_seed(3)
    for _ in range(5):
        scores = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        best = max((s for s in itertools.product(range(4), repeat=7) if obeys(list(s), rules)),
                   key=lambda s: scores[range(7), list(s)].sum().item())
        found = decode(scores, rules)
        assert obeys(found.tolist(), rules)
        torch.testing.assert_close(scores[range(7), found].sum(), scores[range(7), list(best)].sum())


def test_apply_rules_changes_only_the_held_out_videos_and_keeps_distributions():
    records = records_for("a", range(8), [0] * 8) + records_for("b", range(4), [0] * 4)
    timelines = build_timelines(records, 300)
    wanted = torch.tensor([2, 2, 3, 3, 2, 2, 4, 4] + [5, 5, 5, 5])  # video a: 2 comes back after 3
    log_probs = (torch.nn.functional.one_hot(wanted, CLASSES) * 3.0).double().log_softmax(1)
    rules = Rules(frozenset({(2, 3)}), frozenset())
    result, changed = apply_rules(log_probs, timelines, ["a"], rules)
    decoded = result[:8].argmax(1)
    assert changed == int((decoded != wanted[:8]).sum()) > 0 and obeys(decoded.tolist(), rules)
    torch.testing.assert_close(result.exp().sum(1), torch.ones(12, dtype=torch.float64))
    assert torch.equal(result[8:], log_probs[8:])  # video b was not decoded
    same = decoded == wanted[:8]
    assert torch.equal(result[:8][same], log_probs[:8][same])
    assert torch.equal(result[:8].sort(1).values, log_probs[:8].sort(1).values)  # swapped, not rescaled
