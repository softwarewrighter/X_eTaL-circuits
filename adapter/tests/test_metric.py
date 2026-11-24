"""The logit difference at the second copy's positions (docs/design.md, D14)."""

import numpy as np

from circuits_adapter import metric, prompts

BOS = 50256


def test_positions_are_the_second_copy_whose_next_token_induction_predicts():
    assert metric.positions(4).tolist() == [5, 6, 7]


def test_logit_difference_reads_the_right_and_wrong_next_tokens():
    clean, corrupt = prompts.twins(seed=0, length=4, bos=BOS)
    logits = np.zeros((9, 10000), dtype=np.float32)
    for q in (5, 6, 7):
        logits[q, clean[q + 1]] = 3.0 + q
        logits[q, corrupt[q + 1]] = 1.0
    got = metric.logit_difference(logits, clean, corrupt, 4)
    assert got.dtype == np.float64
    assert got.tolist() == [7.0, 8.0, 9.0]


def test_equal_logits_give_zero():
    clean, corrupt = prompts.twins(seed=1, length=4, bos=BOS)
    assert not metric.logit_difference(np.ones((9, 10000)), clean, corrupt, 4).any()
