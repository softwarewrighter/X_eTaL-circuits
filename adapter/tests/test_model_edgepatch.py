"""Edge path patching with controls on GPT-2 Small (needs the model extra and the weights)."""

import numpy as np
import pytest

from circuits_adapter import graph

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def run():
    from circuits_adapter import edgepatch, model

    return edgepatch.Run(model.load("gpt2"), 16, 0, k=4)


def test_effects_are_path_patchings_and_controls_are_seeded(run):
    r = graph.receiver("5.1.k", 12, 12)
    senders = [graph.sender(t, 12, 12) for t in ("4.11", "3.7", "embed")]
    e, c = run.receiver(r, senders)
    assert np.allclose(e, run.s.effects(r, senders), atol=1e-6)
    assert c.shape == (4, 3)
    e2, c2 = run.receiver(r, senders)
    assert np.array_equal(c, c2)


def test_the_4_11_key_edge_beats_its_controls_on_seed_0(run):
    e, c = run.receiver(graph.receiver("5.1.k", 12, 12), [graph.Sender(4, 11)])
    assert e[0] > 0.3 and np.abs(c).max() < 0.1
