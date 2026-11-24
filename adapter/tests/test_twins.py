"""Twin prompts line up token by token (rule 4), and the check refuses any that do not."""

import pytest

from circuits_adapter import prompts, twins

BOS = 50256


def test_twins_line_up_and_differ_only_in_the_second_copy():
    clean, corrupt = prompts.twins(seed=0, length=16, bos=BOS)
    assert clean == prompts.repeated_random(0, 16, BOS)
    assert len(clean) == len(corrupt) == 33
    assert clean[:17] == corrupt[:17]
    assert all(a != b for a, b in zip(clean[17:], corrupt[17:], strict=True))
    twins.check(clean, corrupt, 16)


def test_twins_are_deterministic_per_seed():
    assert prompts.twins(3, 16, BOS) == prompts.twins(3, 16, BOS)
    assert prompts.twins(3, 16, BOS) != prompts.twins(4, 16, BOS)


def test_the_fresh_copy_shares_no_token_with_the_first():
    clean, corrupt = prompts.twins(seed=7, length=16, bos=BOS)
    assert not set(corrupt[17:]) & set(clean[1:17])
    assert len(set(corrupt[17:])) == 16


@pytest.mark.parametrize(
    "edit,says",
    [
        (lambda c, k: (c, k[:-1]), "expected 33"),
        (lambda c, k: (c, [k[0] + 1, *k[1:]]), "differ before"),
        (lambda c, k: ([*c[:-1], c[1]], k), "not the first copy"),
        (lambda c, k: (c, [*k[:-1], c[1]]), "not fresh"),
        (lambda c, k: (c, [*k[:-1], k[-2]]), "not fresh"),
    ],
)
def test_misaligned_twins_are_refused(edit, says):
    clean, corrupt = edit(*prompts.twins(seed=0, length=16, bos=BOS))
    with pytest.raises(twins.TwinError, match=says):
        twins.check(clean, corrupt, 16)


def test_a_first_copy_with_a_repeat_is_refused():
    clean, corrupt = prompts.twins(seed=0, length=4, bos=BOS)
    first = [clean[1], clean[1], clean[3], clean[4]]
    with pytest.raises(twins.TwinError, match="ambiguous"):
        twins.check([BOS, *first, *first], [BOS, *first, *corrupt[5:]], 4)
