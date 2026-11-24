"""Head ablation on GPT-2 Small (needs the model extra and the weights)."""

from pathlib import Path

import pytest

from circuits_adapter import experiment, prompts

pytestmark = pytest.mark.model
INDUCTION = Path(__file__).resolve().parents[2] / "experiments" / "induction.toml"


@pytest.fixture(scope="module")
def setup():
    from circuits_adapter import model

    m = model.load("gpt2")
    exp = experiment.load(INDUCTION)
    clean, corrupt = prompts.twins(0, exp.length, m.bos)
    return m, exp, clean, corrupt


def test_a_hook_that_changes_nothing_has_no_effect(setup):
    from circuits_adapter import ablate

    m, exp, clean, corrupt = setup
    base = ablate.mean_ld(m, clean, clean, corrupt, exp.length)
    same = ablate.mean_ld(
        m, clean, clean, corrupt, exp.length, [(ablate.z_name(5), lambda z, hook: z)]
    )
    assert same == base


def test_zeroing_every_head_of_a_layer_leaves_only_its_output_bias(setup):
    import torch

    from circuits_adapter import ablate

    m, exp, clean, corrupt = setup
    heads = [(ablate.z_name(5), ablate.set_head(h)) for h in range(12)]
    by_heads = ablate.mean_ld(m, clean, clean, corrupt, exp.length, heads)
    bias = m.bridge.b_O[5].detach()

    def only_bias(out, hook):
        return torch.zeros_like(out) + bias

    by_output = ablate.mean_ld(
        m, clean, clean, corrupt, exp.length, [("blocks.5.hook_attn_out", only_bias)]
    )
    assert by_heads == pytest.approx(by_output, rel=1e-5, abs=1e-4)


def test_ablating_5_1_hurts_more_than_a_typical_layer_0_head(setup):
    """A sanity check of the method, not the evidence (that is the controls' job)."""
    from circuits_adapter import ablate

    m, exp, _, _ = setup
    r = ablate.effects(m, exp, 0, ablate.twin_means(m, exp))
    for method in ablate.METHODS:
        assert r[method][5, 1] > r[method][0].mean() + 1.0, method
