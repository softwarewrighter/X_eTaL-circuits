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


def key_twin(seed: int, length: int, bos: int, low: int = 1000, high: int = 10000):
    """The twin that differs where an induction head's keys look: BOS, the fresh
    tokens of `twins`'s twin, then the clean prompt's second copy. Its first copy
    holds nothing the second copy repeats, so previous-token information at the
    keys is what changes; the metric's tokens are the same as for `twins`."""
    clean, corrupt = twins(seed, length, bos, low, high)
    return [bos, *corrupt[1 + length :], *clean[1 + length :]]
