"""Path patching on GPT-2 Small (needs the model extra and the weights)."""

import pytest

from circuits_adapter import graph

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def m():
    from circuits_adapter import model

    return model.load("gpt2")


def test_q_from_the_residual_stream_by_hand_equals_the_models(m):
    import torch

    from circuits_adapter import pathpatch

    p = pathpatch.Paths(m, 16, 0)
    names = ["blocks.5.hook_resid_pre", "blocks.5.attn.hook_q"]
    with torch.no_grad():
        _, c = m.bridge.run_with_cache(torch.tensor([p.clean]), names_filter=lambda h: h in names)
    ln = m.bridge.blocks[5].ln1._original_component
    q = pathpatch.layer_norm(c[names[0]][0], ln) @ m.bridge.W_Q[5, 1] + m.bridge.b_Q[5, 1]
    assert torch.allclose(q, c[names[1]][0][:, 1], atol=1e-4)


def test_patching_a_sender_into_every_reader_equals_patching_its_output(m):
    from circuits_adapter import patch, pathpatch

    p = pathpatch.Paths(m, 16, 0)
    d = p.difference(graph.Sender(5, 1))[None]
    readers = [
        (graph.Receiver(la, h, i), d) for la in range(6, 12) for h in range(12) for i in "qkv"
    ]
    readers += [(("mlp", la), d) for la in range(5, 12)] + [("logits", d)]
    whole = patch.Twins(m, 16, 0).fraction("denoise", patch.z_name(5), head=1)
    assert p.effect(readers, 1)[0] == pytest.approx(whole, abs=1e-5)


def test_a_sender_that_writes_the_same_on_both_twins_has_no_effect(m):
    import torch

    from circuits_adapter import pathpatch

    s = pathpatch.Seed(m, 16, 0)
    zero = torch.zeros(1, 33, 768)
    for twin in ("second", "first"):
        p = s.paths[twin]
        # q is recomputed by hand, equal to the model's within float32 rounding (1e-5)
        assert p.effect([(graph.Receiver(6, 9, "q"), zero)], 1)[0] == pytest.approx(0, abs=1e-5)
        assert p.effect([("logits", zero)], 1)[0] == pytest.approx(0, abs=1e-9)


def test_4_11_leads_the_key_of_each_induction_head(m):
    """Seeds 0 and 1: a sanity check of the method (the evidence is the controls')."""
    import numpy as np

    from circuits_adapter import pathpatch

    senders = [graph.Sender(-1)] + [graph.Sender(la, h) for la in range(5) for h in range(12)]
    for seed in (0, 1):
        s = pathpatch.Seed(m, 16, seed)
        e = s.effects(graph.Receiver(5, 1, "k"), senders)
        assert senders[int(np.argmax(e))] == graph.Sender(4, 11)
        assert e.max() > 0.2
