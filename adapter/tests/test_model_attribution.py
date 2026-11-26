"""Edge attribution on GPT-2 Small (needs the model extra and the weights)."""

import numpy as np
import pytest

from circuits_adapter import graph

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def a():
    from circuits_adapter import attribution, model

    return attribution.Attribution(model.load("gpt2"), 16, 0)


def test_scores_and_logit_difference_by_hand_equal_the_models(a):
    import torch

    for twin in ("second", "first"):
        for layer in (0, 5, 11):
            with torch.enable_grad():
                s, _, _ = a.scores(layer, twin)
            assert np.allclose(s.detach().numpy(), a.s.base[twin][layer][0], atol=1e-4)
    assert a.ld == pytest.approx(a.s.paths["second"].ld_corrupt, abs=1e-4)


def test_same_seed_same_attributions(a):
    from circuits_adapter import attribution

    again = attribution.Attribution(a.m, 16, 0)
    assert np.array_equal(again.matrix()[2], a.matrix()[2])


def test_a_handful_of_edges_agree_with_path_patching_in_sign_and_rank(a):
    """Seed 0, measured: the 4.11 key edges within 0.02 of path patching, ranked alike;
    the logit edges of 10.7, 9.6, 7.2 and 11.10 likewise."""
    senders, receivers, e = a.matrix()
    s = graph.Sender(4, 11)
    rs = [graph.receiver(r, 12, 12) for r in ("5.1.k", "5.5.k", "6.9.k", "7.2.k", "7.10.k")]
    got = [e[senders.index(s), receivers.index(r)] for r in rs]
    want = [a.s.effects(r, [s])[0] for r in rs]
    assert np.allclose(got, want, atol=0.02)
    assert list(np.argsort(got)) == list(np.argsort(want))
    logit = graph.Receiver(12)
    ss = [graph.sender(t, 12, 12) for t in ("10.7", "9.6", "7.2", "11.10")]
    got = [e[senders.index(s), receivers.index(logit)] for s in ss]
    want = list(a.s.effects(logit, ss))
    assert np.allclose(got, want, atol=0.02)
    assert list(np.argsort(got)) == list(np.argsort(want))
    assert (np.sign(got) == np.sign(want)).all()
