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


def twins(seed: int, length: int, bos: int, low: int = 1000, high: int = 10000):
    """The clean prompt (repeated_random) and its corrupted twin.

    The twin keeps BOS and the first copy and replaces the second copy with
    `length` fresh distinct ids that occur nowhere in the clean prompt, so the
    two line up position by position and the twin gives induction nothing to copy.
    """
    clean = repeated_random(seed, length, bos, low, high)
    used = set(clean[1 : 1 + length])
    pool = np.array([t for t in range(low, high) if t not in used])
    rng = np.random.default_rng([seed, 1])
    fresh = rng.choice(pool, size=length, replace=False).tolist()
    return clean, [bos, *clean[1 : 1 + length], *fresh]
