"""Activation patching on GPT-2 Small (needs the model extra and the weights)."""

import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def twins0():
    from circuits_adapter import model, patch

    return patch.Twins(model.load("gpt2"), 16, 0)


def test_patching_the_whole_input_recovers_each_side_exactly(twins0):
    from circuits_adapter import patch

    name = patch.resid_name(0)
    assert twins0.fraction("denoise", name) == pytest.approx(1.0, abs=1e-6)
    assert twins0.fraction("noise", name) == pytest.approx(1.0, abs=1e-6)


def test_patching_a_position_both_prompts_share_changes_nothing(twins0):
    from circuits_adapter import patch

    # Positions 0..16 (BOS and the first copy) are the same tokens in both twins,
    # and the residual entering layer 0 there depends only on them.
    for pos in (0, 5, 16):
        assert twins0.fraction("denoise", patch.resid_name(0), position=pos) == 0.0


def test_the_known_induction_heads_restore_the_most_when_patched_in(twins0):
    """Seed 0: a sanity check of the method, not the evidence (the controls')."""
    from circuits_adapter import patch

    top = set(patch.leading(patch.heads(twins0, "denoise"), 6))
    assert len(top & {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}) >= 4
