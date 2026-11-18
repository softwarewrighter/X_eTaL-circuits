"""Prompts for the experiments."""

import numpy as np


def repeated_random(seed: int, length: int, bos: int, low: int = 1000, high: int = 10000):
    """BOS, then `length` distinct random token ids from [low, high), then the same ids again.

    Distinct ids make every token in the second copy occur exactly once before,
    so "the token after its earlier occurrence" is unambiguous.
    """
    rng = np.random.default_rng(seed)
    seq = rng.choice(np.arange(low, high), size=length, replace=False).tolist()
    return [bos, *seq, *seq]
