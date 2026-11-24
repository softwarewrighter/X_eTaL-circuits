"""The metric (docs/design.md, D14): the logit difference at the second copy's positions.

At position q of the second copy (q = length + 1 .. 2 length - 1, 0-based), the
clean prompt's next token is the induction answer and the corrupted twin's next
token is a fresh one. The metric is logit[q, clean next] - logit[q, twin next],
per position, and its mean.
"""

import numpy as np


def positions(length: int) -> np.ndarray:
    """The positions whose next token induction predicts in the clean prompt."""
    return np.arange(length + 1, 2 * length)


def logit_difference(logits: np.ndarray, clean: list[int], corrupt: list[int], length: int):
    """Per-position logit differences (float64) from logits of shape position x vocab."""
    q = positions(length)
    right = np.asarray(clean)[q + 1]
    wrong = np.asarray(corrupt)[q + 1]
    lg = logits.astype(np.float64)
    return lg[q, right] - lg[q, wrong]
