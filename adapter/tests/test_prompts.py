"""Seeded repeated-random-token induction prompts."""

from circuits_adapter import prompts


def test_shape_bos_then_sequence_twice():
    ids = prompts.repeated_random(seed=0, length=8, bos=50256)
    assert len(ids) == 1 + 2 * 8
    assert ids[0] == 50256
    assert ids[1:9] == ids[9:]


def test_same_seed_same_prompt_other_seed_differs():
    a = prompts.repeated_random(seed=3, length=20, bos=0)
    assert a == prompts.repeated_random(seed=3, length=20, bos=0)
    assert a != prompts.repeated_random(seed=4, length=20, bos=0)


def test_tokens_in_range_and_distinct_within_a_copy():
    ids = prompts.repeated_random(seed=1, length=50, bos=0, low=1000, high=10000)
    first = ids[1:51]
    assert all(1000 <= t < 10000 for t in first)
    assert len(set(first)) == 50
