"""Captures from GPT-2 Small through TransformerLens (needs the model extra and the weights)."""

import numpy as np
import pytest

from circuits_adapter import capture, prompts

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def gpt2():
    from circuits_adapter import model

    return model.load("gpt2")


def test_capture_is_a_valid_causal_attention_pattern(gpt2):
    from circuits_adapter import model

    ids = prompts.repeated_random(seed=0, length=8, bos=gpt2.bos)
    c = model.capture(gpt2, ids, seed=0)
    assert c.attn.shape == (12, 12, 17, 17)
    assert c.attn.dtype == np.float32
    assert c.logits.shape == (50257,)
    capture.check_attention(c.attn)
    assert c.info["model"] == ["gpt2"]
    assert c.info["revision"][0] == model.GPT2_REVISION
    assert c.info["token_ids"] == [str(t) for t in ids]
    assert len(c.info["tokens"]) == len(ids)


def test_two_runs_are_byte_identical(gpt2, tmp_path):
    from circuits_adapter import model

    ids = prompts.repeated_random(seed=1, length=8, bos=gpt2.bos)
    for name in ("a", "b"):
        capture.write(tmp_path / name, model.capture(gpt2, ids, seed=1))
    for f in ("attn.xta", "logits.xta", "capture.toml"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_layer_subset(gpt2):
    from circuits_adapter import model

    ids = prompts.repeated_random(seed=0, length=4, bos=gpt2.bos)
    c = model.capture(gpt2, ids, seed=0, layers=[0, 5])
    assert c.attn.shape == (2, 12, 9, 9)
    assert c.attn_layers == [0, 5]


def test_known_induction_heads_lead_the_scores(gpt2):
    """Attention-pattern scores only (candidates), as in docs/reference.md."""
    from circuits_adapter import model

    t = 50
    ids = prompts.repeated_random(seed=0, length=t, bos=gpt2.bos)
    attn = model.capture(gpt2, ids, seed=0).attn
    q = np.arange(1 + t, 1 + 2 * t)
    score = attn[:, :, q, q - t + 1].mean(-1)
    top = {(int(i) // 12, int(i) % 12) for i in np.argsort(-score.ravel())[:5]}
    assert top == {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}
